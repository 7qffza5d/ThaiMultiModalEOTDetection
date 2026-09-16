from alignTextGridsToParquet import (
    parse_textgrid, segment_conversations_by_contiguity,
    match_textgrid_to_conversation, extract_topic_number_from_filename,
    extract_mic_from_filename,
)
from datasets import load_dataset
import os

TEXTGRID_PATH = "rawData/text/textgrid/Hijack_S081_T069_Con123.TextGrid"
SPLIT = "validation"  # match whichever split you've been testing on

ds = load_dataset("nectec/LOTUSDIS", split=SPLIT)
if "audio" in ds.column_names:
    ds = ds.remove_columns(["audio"])
rows = [dict(r, _orig_index=i) for i, r in enumerate(ds)]
conversations = segment_conversations_by_contiguity(rows)

session_name = os.path.splitext(os.path.basename(TEXTGRID_PATH))[0]
intervals = parse_textgrid(TEXTGRID_PATH)
conv_idx, was_ambiguous, debug = match_textgrid_to_conversation(
    intervals, conversations,
    filename_topic=extract_topic_number_from_filename(session_name),
    filename_mic=extract_mic_from_filename(session_name),
)
print(f"Matched conversation index: {conv_idx}, ambiguous={was_ambiguous}, debug={debug}\n")

matched_rows = conversations[conv_idx]
print(f"Matched block has {len(matched_rows)} rows. First 12:\n")
for i, r in enumerate(matched_rows[:12]):
    print(f"row {i}: speaker_id={r['speaker_id']!r}  mic={r.get('mic')!r}  orig_index={r['_orig_index']}")
    print(f"    sentence repr: {r['sentence']!r}")
    print() 