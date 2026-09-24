"""
Align LOTUSDIS TextGrid session files to the reconstructed parquet
conversations, verify the two agree, and attach real start/end
timestamps to each parquet row once alignment is confirmed.

Usage:
    python align_textgrid_to_parquet.py /path/to/textgrid/dir

Assumes:
- TextGrid files are one-per-session, single tier, sequential,
  non-overlapping intervals, with text formatted as
  "SPEAKER_ID, <tags> sentence text <tags>" (e.g. "F01, <n> สวัสดี ค่ะ <n>"),
  and standalone <n>/<sil>-only intervals with no speaker prefix.
- The parquet dataset ("nectec/LOTUSDIS") has speaker_id unique per
  conversation, letting conversations be reconstructed via union-find
  (as established earlier in this project).
"""

import re
import os
import csv
import glob
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from datasets import load_dataset

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the current working directory (or nearest parent)
except ImportError:
    raise SystemExit(
        "python-dotenv is required to load the .env file. Install it with:\n"
        "    pip install python-dotenv --break-system-packages"
    )

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit(
        "No HF_TOKEN found. Add a line like the following to your .env file:\n"
        "    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )


# ---------- Step 1: TextGrid parsing ----------

INTERVAL_RE = re.compile(
    r'xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*"(.*?)"',
    re.DOTALL,
)
SPEAKER_PREFIX_RE = re.compile(r'^([\w&]+),\s*(.*)$')


def parse_textgrid(path):
    """Return a list of speaker-tagged, content-bearing intervals: {xmin,
    xmax, speakers, raw_text, clean_text}. Two kinds of intervals are
    dropped, since neither appears to survive into the parquet transcript:
    (1) standalone noise/silence intervals with no speaker prefix at all
    (e.g. leading silence before anyone speaks), and (2) speaker-tagged
    intervals whose content is PURELY tags (e.g. 'M38, <n>' — a cough or
    noise with no actual words) — these still have a speaker prefix so
    they'd otherwise pass the first filter, but they leave no trace in the
    parquet row count, causing an index-shifting mismatch if kept."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    intervals = []
    for xmin, xmax, text in INTERVAL_RE.findall(content):
        text = text.strip()
        if not text:
            continue
        m = SPEAKER_PREFIX_RE.match(text)
        if not m:
            continue  # e.g. a bare "<n>" interval with no speaker
        speaker_field, clean_text = m.groups()
        if not strip_tags(clean_text):
            continue  # speaker-tagged but purely noise/silence tags, no words
        intervals.append({
            "xmin": float(xmin),
            "xmax": float(xmax),
            "speakers": [s for s in speaker_field.split("&") if s],
            "raw_text": text,
            "clean_text": clean_text.strip(),
        })
    return intervals


def strip_tags(text):
    """Normalize for text-content comparison: remove <n>/<unk>/<td>/<sil>
    tags (including combined forms like <n>+<unk>), unwrap bracket markup
    like [M]/[F]/[TOPIC] to their bare word, strip # and % markers
    (emphasis/disfluency — not reliably used in matched pairs, so
    stripped outright rather than unwrapped as spans), strip ALL
    whitespace (TextGrid keeps syllable-level spaces the parquet
    transcript doesn't), and lowercase (parquet uses 'Topic', TextGrid
    uses '[TOPIC]')."""
    text = re.sub(r'(<[^>]+>)(\+<[^>]+>)*', '', text)
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    text = text.replace('#', '').replace('%', '')
    text = re.sub(r'\s+', '', text)
    return text.strip().lower()


# ---------- Thai number-word parsing (for topic-number disambiguation) ----------

_DIGIT_VAL = {"ศูนย์": 0, "หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "สาม": 3, "สี่": 4,
              "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9}
_SCALE_VAL = {"สิบ": 10, "ร้อย": 100, "พัน": 1000, "หมื่น": 10000,
              "แสน": 100000, "ล้าน": 1000000}
_NUM_VOCAB_SORTED = sorted(
    list(_DIGIT_VAL) + list(_SCALE_VAL) + ["ยี่สิบ"], key=len, reverse=True
)


def parse_thai_number_prefix(s):
    """Parse a leading contiguous run of Thai number words at the start of
    `s` (e.g. 'ห้าสิบเจ็ด...' -> 57). Stops at the first character that
    isn't part of a recognized number word, so trailing non-number text
    (the rest of a topic description) doesn't get misread. Returns None
    if `s` doesn't start with a number word at all."""
    total, current, i, consumed_any = 0, 0, 0, False
    while i < len(s):
        matched = next((w for w in _NUM_VOCAB_SORTED if s.startswith(w, i)), None)
        if matched is None:
            break
        consumed_any = True
        i += len(matched)
        if matched == "ยี่สิบ":
            total += 20
        elif matched in _SCALE_VAL:
            total += (current or 1) * _SCALE_VAL[matched]
            current = 0
        else:
            current = _DIGIT_VAL[matched]
    total += current
    return total if consumed_any else None


def find_topic_number(rows):
    """Scan a conversation's rows for a 'Topic ...' announcement and parse
    the Thai number that follows it (optionally after a 'ที่' connector,
    e.g. 'Topic ที่ห้าสิบเจ็ด...' or 'Topicหกสิบเก้า...'). Returns the
    parsed integer, or None if no topic announcement is found/parseable."""
    for row in rows:
        norm = strip_tags(row["sentence"])
        idx = norm.find("topic")
        if idx == -1:
            continue
        rest = norm[idx + len("topic"):]
        if rest.startswith("ที่"):
            rest = rest[len("ที่"):]
        value = parse_thai_number_prefix(rest)
        if value is not None:
            return value
    return None


def extract_topic_number_from_filename(filename):
    """Extract the topic number from a filename like
    'Hijack_S081_T069_Con123' -> 69."""
    m = re.search(r"_T(\d+)_", filename)
    return int(m.group(1)) if m else None


def extract_mic_from_filename(filename):
    """Extract the trailing mic-condition token from a filename like
    'Hijack_S081_T069_Con123' -> 'con123' (lowercased, to compare against
    the parquet's `mic` field, which uses mixed case for at least one
    value: 'BT3m')."""
    m = re.search(r"_([A-Za-z]+\d+)$", filename)
    return m.group(1).lower() if m else None


# ---------- Step 2: Parquet-side conversation reconstruction ----------

def get_speakers(speaker_id_str):
    speaker_id_str = speaker_id_str.replace("฿", "&")
    return [s for s in speaker_id_str.split("&") if s]

TEXT_SIMILARITY_THRESHOLD = 0.6  # see text_similar(); calibrated against
# S050/S058's confirmed content-shifted rows, whose ratios topped out at
# 0.571 (row 88) — genuine matches are expected well above this since
# strip_tags already normalizes tokenization differences.

def text_similar(tg_text, pq_text, threshold=TEXT_SIMILARITY_THRESHOLD):
    """Coarse content-similarity check, used only as a tiebreaker inside
    speaker_sets_match's overlap branch — never as a standalone match
    criterion. Distinguishes a genuine overlap-crediting disagreement
    (same underlying utterance, different speaker credit) from an indel
    that happens to preserve speaker overlap with its neighbors (different
    utterance entirely, same speakers still active nearby). Compares on
    strip_tags-normalized text since TextGrid and parquet tokenize
    differently (syllable-spaced vs. continuous)."""
    a, b = strip_tags(tg_text), strip_tags(pq_text)
    if not a or not b:
        return True  # nothing to compare against; don't penalize on missing text
    return SequenceMatcher(None, a, b).ratio() >= threshold


def speaker_sets_match(a, b, tg_text=None, pq_text=None):
    """Compare two speaker sets for one row-position. If both sides are
    single-speaker, require exact equality (unambiguous — should never
    legitimately differ). If either side reflects an overlap (more than
    one speaker), the sets must share at least one speaker, since
    TextGrid and parquet are known to sometimes disagree on which
    speakers get credited during overlapping speech (e.g. dominant
    speaker only vs. all overlapping speakers) — but that alone used to
    be treated as a free match regardless of content, which let
    fitting_align silently walk through indels whenever the shifted rows
    still happened to share a speaker with their neighbors (near-certain
    in a 2-3 person conversation; confirmed empirically in S050/S058,
    where 13/13 and 28/29 of the resulting discrepancy rows would have
    passed the old check). When text is supplied, it's used as a
    tiebreaker so a real content shift isn't masked by coincidental
    speaker overlap; with no text supplied, this falls back to the old,
    overlap-only behavior (used by the coarse window-search functions
    below, which don't have per-position text lined up to compare)."""
    if len(a) == 1 and len(b) == 1:
        return a == b
    if not (a & b):
        return False
    if tg_text is not None and pq_text is not None:
        return text_similar(tg_text, pq_text)
    return True

def find_matching_window(rows_ordered, tg_intervals, filename_mic=None):
    if filename_mic is not None:
        rows_ordered = [r for r in rows_ordered if r["mic"].lower() == filename_mic]
    tg_seq = [frozenset(iv["speakers"]) for iv in tg_intervals]
    n = len(tg_seq)
    row_seq = [frozenset(get_speakers(r["speaker_id"])) for r in rows_ordered]

    for start in range(len(row_seq) - n + 1):
        if all(speaker_sets_match(row_seq[start + i], tg_seq[i]) for i in range(n)):
            return rows_ordered[start:start + n]
    return None


def find_best_partial_window(rows_ordered, tg_intervals, filename_mic=None):
    if filename_mic is not None:
        rows_ordered = [r for r in rows_ordered if r["mic"].lower() == filename_mic]
    tg_seq = [frozenset(iv["speakers"]) for iv in tg_intervals]
    n = len(tg_seq)
    row_seq = [frozenset(get_speakers(r["speaker_id"])) for r in rows_ordered]

    best_start, best_mismatches = None, n + 1
    for start in range(len(row_seq) - n + 1):
        mismatches = sum(1 for a, b in zip(row_seq[start:start + n], tg_seq)
                          if not speaker_sets_match(a, b))
        if mismatches < best_mismatches:
            best_start, best_mismatches = start, mismatches
    return best_start, best_mismatches


def find_first_divergence(rows_ordered, tg_intervals, filename_mic=None):
    if filename_mic is not None:
        rows_ordered = [r for r in rows_ordered if r["mic"].lower() == filename_mic]
    tg_seq = [frozenset(iv["speakers"]) for iv in tg_intervals]
    n = len(tg_seq)
    row_seq = [frozenset(get_speakers(r["speaker_id"])) for r in rows_ordered]

    best_start, best_mismatches = None, n + 1
    for start in range(len(row_seq) - n + 1):
        mismatches = sum(1 for a, b in zip(row_seq[start:start + n], tg_seq)
                          if not speaker_sets_match(a, b))
        if mismatches < best_mismatches:
            best_start, best_mismatches = start, mismatches

    first_divergence = next(
        (i for i, (a, b) in enumerate(zip(row_seq[best_start:best_start + n], tg_seq))
         if not speaker_sets_match(a, b)),
        None,
    )
    return best_start, best_mismatches, first_divergence

def fitting_align(tg_seq, pq_seq, match_fn=speaker_sets_match, gap_cost=1,
                   tg_text_seq=None, pq_text_seq=None):
    """Semi-global ('fitting') alignment: tg_seq must be fully consumed,
    but pq_seq may be freely skipped at both the start and end with no
    penalty — only substitutions/indels WITHIN the matched region cost
    anything. This avoids the previous bug where padding the candidate
    window artificially inflated the cost floor to the padding size
    regardless of actual match quality. Returns (cost, alignment), where
    alignment is a list of (tg_idx_or_None, pq_idx_or_None) pairs.

    tg_text_seq/pq_text_seq, when supplied, run parallel to tg_seq/pq_seq
    and get passed to match_fn alongside the speaker sets — this is what
    lets an overlap-tolerant match_fn (speaker_sets_match) use text as a
    tiebreaker instead of treating any shared-speaker overlap as free,
    which previously let real indels go undetected whenever they
    preserved speaker overlap with their neighbors."""
    n, m = len(tg_seq), len(pq_seq)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        back[i][0] = "up"
    for j in range(1, m + 1):
        back[0][j] = "free"  # dp[0][j] stays 0: free leading skip
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if tg_text_seq is not None and pq_text_seq is not None:
                is_match = match_fn(tg_seq[i - 1], pq_seq[j - 1],
                                     tg_text_seq[i - 1], pq_text_seq[j - 1])
            else:
                is_match = match_fn(tg_seq[i - 1], pq_seq[j - 1])
            sub_cost = 0 if is_match else 1
            diag, up, left = dp[i - 1][j - 1] + sub_cost, dp[i - 1][j] + gap_cost, dp[i][j - 1] + gap_cost
            best = min(diag, up, left)
            dp[i][j] = best
            back[i][j] = "diag" if best == diag else ("up" if best == up else "left")

    best_j = min(range(m + 1), key=lambda j: dp[n][j])
    best_cost = dp[n][best_j]

    i, j, alignment = n, best_j, []
    while i > 0:
        move = back[i][j]
        if move == "diag":
            alignment.append((i - 1, j - 1)); i -= 1; j -= 1
        elif move == "up":
            alignment.append((i - 1, None)); i -= 1
        else:
            alignment.append((None, j - 1)); j -= 1
    alignment.reverse()
    return best_cost, alignment


def attempt_indel_tolerant_match(rows_ordered, tg_intervals, filename_mic, best_start, margin=30):
    """Fallback used only when a strict fixed-length window fails but the
    best partial match was already close. Free end-gaps (see
    fitting_align) mean the candidate window can be generously padded
    without inflating the cost, so we no longer need to tune margin
    tightly against the threshold."""
    if filename_mic is not None:
        rows_ordered = [r for r in rows_ordered if r["mic"].lower() == filename_mic]
    lo = max(0, best_start - 10)
    hi = best_start + len(tg_intervals) + margin
    candidate_rows = rows_ordered[lo:hi]
    tg_seq = [frozenset(iv["speakers"]) for iv in tg_intervals]
    pq_seq = [frozenset(get_speakers(r["speaker_id"])) for r in candidate_rows]
    tg_text_seq = [iv["clean_text"] for iv in tg_intervals]
    pq_text_seq = [r["sentence"] for r in candidate_rows]
    cost, alignment = fitting_align(tg_seq, pq_seq, tg_text_seq=tg_text_seq, pq_text_seq=pq_text_seq)
    return cost, alignment, candidate_rows

# ---------- Step 3: Match a TextGrid file to its parquet conversation ----------

def match_textgrid_to_conversation(tg_intervals, conversations, filename_topic=None, filename_mic=None):
    """Match by EXACT speaker-set equality, not just overlap — a recurring
    trio of speakers can appear together in more than one topic recording,
    so partial/best overlap is not a reliable fingerprint on its own.

    When multiple blocks share the exact same speaker set, this corpus has
    a specific, expected cause: each real session's transcript appears to
    be duplicated once per recording microphone (the paper's "5-mic total
    duration" figure is ~5x the base session duration). The TextGrid
    filename encodes which mic it corresponds to (e.g. '..._Con123'),
    matching the parquet's `mic` field directly — so mic matching is tried
    first. If that doesn't uniquely resolve it, fall back to the announced
    topic number (parsed from each candidate's own 'Topic ...' utterance)
    against the topic number in the filename. Early-row text similarity is
    a last resort only, since sessions by the same trio tend to open with
    near-identical boilerplate greetings that can't distinguish sessions.

    Returns (best_idx, was_ambiguous, debug_info).
    """
    tg_speakers = set()
    for iv in tg_intervals:
        tg_speakers.update(iv["speakers"])

    candidates = []
    for idx, rows in enumerate(conversations):
        conv_speakers = set()
        for r in rows:
            conv_speakers.update(get_speakers(r["speaker_id"]))
        if conv_speakers == tg_speakers:
            candidates.append(idx)

    if not candidates:
        return None, False, {}
    if len(candidates) == 1:
        return candidates[0], False, {}

    debug = {"filename_topic": filename_topic, "filename_mic": filename_mic}

    if filename_mic is not None:
        mic_matches = []
        for idx in candidates:
            conv_mics = {r["mic"].lower() for r in conversations[idx]}
            if conv_mics == {filename_mic}:
                mic_matches.append(idx)
        debug["mic_matches"] = mic_matches
        if len(mic_matches) == 1:
            return mic_matches[0], False, debug
        if len(mic_matches) > 1:
            candidates = mic_matches  # narrowed but still ambiguous

    if filename_topic is not None:
        candidate_topics = {idx: find_topic_number(conversations[idx]) for idx in candidates}
        debug["candidate_topics"] = candidate_topics
        topic_matches = [idx for idx, t in candidate_topics.items() if t == filename_topic]
        if len(topic_matches) == 1:
            return topic_matches[0], False, debug
        if len(topic_matches) > 1:
            candidates = topic_matches  # narrowed but still ambiguous

    # Fallback: content similarity on early rows, then length closeness.
    # Weakest signal (see note above) — only reached if mic and topic
    # matching didn't uniquely resolve the candidates.
    def score(idx):
        rows = conversations[idx]
        n_check = min(5, len(rows), len(tg_intervals))
        text_matches = sum(
            strip_tags(rows[i]["sentence"]) == strip_tags(tg_intervals[i]["clean_text"])
            for i in range(n_check)
        )
        length_diff = abs(len(rows) - len(tg_intervals))
        return (-text_matches, length_diff)

    best_idx = min(candidates, key=score)
    return best_idx, True, debug



# ---------- Step 4: Alignment verification ----------

def check_alignment(tg_intervals, parquet_rows, session_name, max_report=5):
    issues = []
    if len(tg_intervals) != len(parquet_rows):
        issues.append(
            f"Count mismatch: TextGrid has {len(tg_intervals)} speaker-tagged "
            f"intervals, parquet conversation has {len(parquet_rows)} rows."
        )

    n = min(len(tg_intervals), len(parquet_rows))
    mismatches = 0
    for i in range(n):
        tg_sp = set(tg_intervals[i]["speakers"])
        pq_sp = set(get_speakers(parquet_rows[i]["speaker_id"]))
        if tg_sp != pq_sp:
            mismatches += 1
            if mismatches <= max_report:
                issues.append(f"Row {i}: speaker mismatch — TextGrid={tg_sp}, parquet={pq_sp}")
            continue  # skip text check if speakers already disagree

        tg_text = strip_tags(tg_intervals[i]["clean_text"])
        pq_text = strip_tags(parquet_rows[i]["sentence"])
        if tg_text != pq_text:
            mismatches += 1
            if mismatches <= max_report:
                note = ""
                if tg_text.startswith(pq_text) or pq_text.startswith(tg_text):
                    note = " (one is a prefix of the other — likely truncation, not misalignment)"
                issues.append(
                    f"Row {i}: text differs after tag-stripping.{note}\n"
                    f"    TextGrid: {tg_text[:60]!r}\n"
                    f"    Parquet : {pq_text[:60]!r}"
                )

    return {
        "session": session_name,
        "n_compared": n,
        "n_mismatches": mismatches,
        "issues": issues,
    }

# ---------- Step 4.5: Logging discrepencies ------------
def log_discrepancies_from_alignment(alignment, tg_intervals, candidate_rows, session_name, discrepancy_log):
    """Record every discrepancy in a (tg_idx, pq_idx) alignment — indels
    (either side None) and speaker/text substitutions — into
    discrepancy_log, a shared list of dicts, for later review. This is
    the concrete input for finalizing the overlap-audit rule (project
    methodology section 1.5): every row where TextGrid and parquet
    disagree on who was speaking during overlap shows up here, rather
    than being capped at the first few printed to console."""
    for tg_i, pq_j in alignment:
        if tg_i is None:
            row = candidate_rows[pq_j]
            discrepancy_log.append({
                "session": session_name, "row_index": pq_j,
                "type": "extra_parquet_row",
                "tg_speakers": "", "pq_speakers": row["speaker_id"],
                "tg_text": "", "pq_text": row["sentence"],
            })
            continue
        if pq_j is None:
            iv = tg_intervals[tg_i]
            discrepancy_log.append({
                "session": session_name, "row_index": tg_i,
                "type": "extra_textgrid_interval",
                "tg_speakers": "&".join(iv["speakers"]), "pq_speakers": "",
                "tg_text": iv["clean_text"], "pq_text": "",
            })
            continue

        iv, row = tg_intervals[tg_i], candidate_rows[pq_j]
        tg_sp, pq_sp = set(iv["speakers"]), set(get_speakers(row["speaker_id"]))
        tg_text, pq_text = strip_tags(iv["clean_text"]), strip_tags(row["sentence"])

        if tg_sp != pq_sp:
            dtype = "speaker_attribution"
        elif tg_text != pq_text:
            dtype = "prefix_truncation" if (tg_text.startswith(pq_text) or pq_text.startswith(tg_text)) else "text_mismatch"
        else:
            continue  # exact match at this position, nothing to log

        discrepancy_log.append({
            "session": session_name, "row_index": tg_i, "type": dtype,
            "tg_speakers": "&".join(sorted(tg_sp)), "pq_speakers": "&".join(sorted(pq_sp)),
            "tg_text": iv["clean_text"], "pq_text": row["sentence"],
        })

def reclassify_boundary_shifts(discrepancy_log):
    """Section 4.1 overlap-rule finding: some 'speaker_attribution' rows
    aren't a disagreement about WHO overlapped, only about WHICH of two
    consecutive rows the overlap is attached to. When two adjacent
    speaker_attribution entries in the same session cleanly swap
    (tg[i]==pq[i+1] and pq[i]==tg[i+1]), relabel both as
    'speaker_attribution_boundary_shift' so they're separated from the
    genuine case-by-case remainder that still needs manual audit."""
    def clean_set(s):
        return frozenset(p for p in re.split(r'[&%]', s) if p)

    by_key = {
        (d["session"], d["row_index"]): d
        for d in discrepancy_log if d["type"] == "speaker_attribution"
    }
    for (session, idx), d in by_key.items():
        nxt = by_key.get((session, idx + 1))
        if nxt is None:
            continue
        if (clean_set(d["tg_speakers"]) == clean_set(nxt["pq_speakers"]) and
                clean_set(d["pq_speakers"]) == clean_set(nxt["tg_speakers"])):
            d["type"] = "speaker_attribution_boundary_shift"
            nxt["type"] = "speaker_attribution_boundary_shift"
    return discrepancy_log

def write_discrepancy_log(discrepancy_log, path="discrepancy_log.csv"):
    if not discrepancy_log:
        print("No discrepancies logged.")
        return
    fieldnames = ["session", "row_index", "type", "tg_speakers", "pq_speakers", "tg_text", "pq_text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(discrepancy_log)

    from collections import Counter
    counts = Counter(d["type"] for d in discrepancy_log)
    print(f"\nWrote {len(discrepancy_log)} discrepancy rows to {path}")
    for dtype, n in counts.most_common():
        print(f"  {dtype}: {n}")

# ---------- Step 5: Attach real timestamps once alignment is confirmed ----------

def attach_timestamps(parquet_rows, tg_intervals):
    """Copy start/end times onto parquet rows. NOTE: intervals observed so
    far are perfectly contiguous (no gap between xmax and the next xmin) —
    pause information lives inside leading/trailing <n>/<sil> tags within
    the interval text, not as a separate silent span. This attaches a
    categorical padding flag rather than a continuous pause duration;
    getting an actual pause length requires a VAD pass on the audio
    within each interval (see run_vad_on_interval stub below)."""
    for i, row in enumerate(parquet_rows):
        iv = tg_intervals[i]
        row["start_time"] = iv["xmin"]
        row["end_time"] = iv["xmax"]
        row["has_leading_pad"] = bool(re.match(r"^\s*<(n|sil)>", iv["raw_text"]))
        row["has_trailing_pad"] = bool(re.search(r"<(n|sil)>\s*$", iv["raw_text"]))
    return parquet_rows


def run_vad_on_interval(audio_path, xmin, xmax):
    """Stub for a follow-up step: given the full session audio and an
    interval's [xmin, xmax], slice that span and run VAD (e.g. webrtcvad
    or a torchaudio-based energy/VAD method) to find the actual last
    voiced frame before xmax and first voiced frame after xmin. The gap
    between consecutive intervals' voiced boundaries is your real,
    continuous pause duration. Not implemented here — a reasonable next
    step once alignment (this script) is confirmed correct."""
    raise NotImplementedError


# ---------- Main ----------

def load_all_splits(parquet_source, token, splits=("train", "validation", "test")):
    """Load and concatenate all named splits into one row list, tagging
    each row with its originating split (_split). TextGrid filenames
    don't indicate which split a session belongs to, so the search needs
    the full combined row space rather than guessing per-file. The split
    tag is kept on each row afterward for later use — e.g. verifying no
    single session ends up split across train/validation/test."""
    all_rows = []
    for split in splits:
        try:
            ds = load_dataset(parquet_source, split=split, token=token)
        except Exception as e:
            print(f"  (skipping split '{split}': {e})")
            continue
        if "audio" in ds.column_names:
            ds = ds.remove_columns(["audio"])
        for i, r in enumerate(ds):
            all_rows.append(dict(r, _orig_index=i, _split=split))
        print(f"  loaded {len(ds)} rows from split '{split}'")
    return all_rows

def main(textgrid_dir, parquet_source="nectec/LOTUSDIS"):
    print(f"Loading parquet dataset ({parquet_source}), all splits...")
    rows = load_all_splits(parquet_source, HF_TOKEN)
    print(f"Loaded {len(rows)} rows total.\n")

    tg_paths = sorted(glob.glob(os.path.join(textgrid_dir, "*.TextGrid")))
    print(f"Found {len(tg_paths)} TextGrid files in {textgrid_dir}.\n")

    reports = []
    discrepancy_log = []
    for tg_path in tg_paths:
        session_name = os.path.splitext(os.path.basename(tg_path))[0]
        intervals = parse_textgrid(tg_path)
        filename_mic = extract_mic_from_filename(session_name)

        parquet_rows = find_matching_window(rows, intervals, filename_mic=filename_mic)

        if parquet_rows is None:
            best_start, best_mismatches, first_divergence = find_first_divergence(
                rows, intervals, filename_mic=filename_mic
            )
            cost, alignment, candidate_rows = attempt_indel_tolerant_match(
                rows, intervals, filename_mic, best_start
            )
            n_indels = sum(1 for tg_i, pq_j in alignment if tg_i is None or pq_j is None)
            if cost <= 5:  # low residual cost -> genuine match modulo a few indels
                aligned_rows = [candidate_rows[pq_j] for tg_i, pq_j in alignment if pq_j is not None]
                extra_parquet = [pq_j for tg_i, pq_j in alignment if tg_i is None]
                extra_textgrid = [tg_i for tg_i, pq_j in alignment if pq_j is None]
                log_discrepancies_from_alignment(alignment, intervals, candidate_rows, session_name, discrepancy_log)
                reports.append({
                    "session": session_name, "n_compared": len(alignment), "n_mismatches": 0,
                    "issues": [f"Matched via indel-tolerant alignment: cost={cost}. "
                               f"Extra parquet row(s) with no TextGrid counterpart at local "
                               f"window index {extra_parquet}. Extra TextGrid interval(s) with "
                               f"no parquet counterpart at TextGrid row {extra_textgrid}. "
                               f"Row-for-row timestamp attachment needs these positions "
                               f"accounted for — do not assume a simple 1:1 index mapping."],
                })
            else:
                reports.append({
                    "session": session_name, "n_compared": 0, "n_mismatches": 0,
                    "issues": [f"No exact-match window found. Best partial match: "
                               f"start_row={best_start}, mismatches={best_mismatches}/{len(intervals)}, "
                               f"first_divergence_at_row={first_divergence}. Indel-tolerant alignment "
                               f"also failed (cost={cost}) — needs manual inspection."],
                })
            continue

        report = check_alignment(intervals, parquet_rows, session_name)
        report["split"] = parquet_rows[0]["_split"]
        discrepancy_log = reclassify_boundary_shifts(discrepancy_log)
        write_discrepancy_log(discrepancy_log)
        log_discrepancies_from_alignment(
            [(i, i) for i in range(len(parquet_rows))], intervals, parquet_rows,
            session_name, discrepancy_log,
        )
 
        if report["n_mismatches"] == 0 and not any(
            "Count mismatch" in issue for issue in report["issues"]
        ):
            attach_timestamps(parquet_rows, intervals)
 
    print("=== Alignment summary ===")
    ok_count = 0
    for r in reports:
        status = "OK" if not r["issues"] else f"ISSUES ({r['n_mismatches']} mismatches)"
        split_tag = f" [{r['split']}]" if r.get("split") else ""
        print(f"{r['session']}{split_tag}: {status}  ({r['n_compared']} rows compared)")
        for issue in r["issues"]:
            print(f"    {issue}")
 
    print(f"\n{ok_count}/{len(reports)} sessions aligned cleanly.")
    write_discrepancy_log(discrepancy_log)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("textgrid_dir", help="Directory containing .TextGrid files")
    parser.add_argument("--parquet-source", default="nectec/LOTUSDIS")
    args = parser.parse_args()
    main(args.textgrid_dir, parquet_source=args.parquet_source)