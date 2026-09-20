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
import glob
import argparse
from collections import defaultdict
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
            "speakers": speaker_field.split("&"),
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
    # Defensive: at least one row has the Thai Baht sign (฿) in place of
    # the intended '&' delimiter — likely a data-entry slip, since the
    # two characters sit in similar positions on some keyboard layouts.
    speaker_id_str = speaker_id_str.replace("฿", "&")
    return speaker_id_str.split("&")

def speaker_sets_match(a, b):
    """Compare two speaker sets for one row-position. If both sides are
    single-speaker, require exact equality (unambiguous — should never
    legitimately differ). If either side reflects an overlap (more than
    one speaker), only require at least one shared speaker, since
    TextGrid and parquet are known to sometimes disagree on which
    speakers get credited during overlapping speech (e.g. dominant
    speaker only vs. all overlapping speakers)."""
    if len(a) == 1 and len(b) == 1:
        return a == b
    return bool(a & b)

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
    for tg_path in tg_paths:
        session_name = os.path.splitext(os.path.basename(tg_path))[0]
        intervals = parse_textgrid(tg_path)
        filename_mic = extract_mic_from_filename(session_name)

        parquet_rows = find_matching_window(rows, intervals, filename_mic=filename_mic)

        if parquet_rows is None:
            best_start, best_mismatches, first_divergence = find_first_divergence(
                rows, intervals, filename_mic=filename_mic
            )
            reports.append({
                "session": session_name, "n_compared": 0, "n_mismatches": 0,
                "issues": [f"No exact-match window found. Best partial match: "
                           f"start_row={best_start}, mismatches={best_mismatches}/{len(intervals)}, "
                           f"first_divergence_at_row={first_divergence} — needs manual inspection."],
            })
            continue

        report = check_alignment(intervals, parquet_rows, session_name)
        report["split"] = parquet_rows[0]["_split"]
 
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
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("textgrid_dir", help="Directory containing .TextGrid files")
    parser.add_argument("--parquet-source", default="nectec/LOTUSDIS")
    args = parser.parse_args()
    main(args.textgrid_dir, parquet_source=args.parquet_source)