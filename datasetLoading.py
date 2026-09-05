#!/usr/bin/env python3
"""
Download the nectec/LOTUSDIS dataset from Hugging Face and save it locally.

For each split (train / validation / test) this script:
  - loads the dataset with the `datasets` library
  - writes each audio sample to a .wav file
  - writes a metadata.csv (and metadata.json) per split with columns:
      file_name, sentence, speaker_id, mic, duration

Usage:
    python download_lotusdis.py --output_dir ./LOTUSDIS_local
    python download_lotusdis.py --output_dir ./LOTUSDIS_local --splits train validation
    python download_lotusdis.py --output_dir ./LOTUSDIS_local --max_samples 200   # quick test run

Requirements:
    pip install datasets soundfile huggingface_hub

Note:
    The full dataset (~161k rows across train/validation/test) can be several
    GB of audio. Use --max_samples for a quick test before downloading everything.
    If the dataset requires accepting terms on Hugging Face, log in first with:
    `huggingface-cli login`, or pass --hf_token.
"""

import argparse
import csv
import json
import os
import sys

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Missing dependency. Install with: pip install datasets soundfile huggingface_hub")

try:
    import soundfile as sf
except ImportError:
    sys.exit("Missing dependency. Install with: pip install soundfile")


DATASET_ID = "nectec/LOTUSDIS"
DEFAULT_SPLITS = ["train", "validation", "test"]


def parse_args():
    parser = argparse.ArgumentParser(description="Download the LOTUSDIS dataset locally.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./LOTUSDIS_local",
        help="Directory to save audio files and metadata into (default: ./LOTUSDIS_local)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=DEFAULT_SPLITS,
        help="Which splits to download (default: all three)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples per split (useful for a quick test run)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face access token, if the dataset requires authentication",
    )
    return parser.parse_args()


def download_split(split_name: str, output_dir: str, max_samples: int, hf_token: str):
    print(f"\n=== Loading split: {split_name} ===")
    ds = load_dataset(DATASET_ID, split=split_name, token=hf_token)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    split_dir = os.path.join(output_dir, split_name)
    audio_dir = os.path.join(split_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    csv_path = os.path.join(split_dir, "metadata.csv")
    json_path = os.path.join(split_dir, "metadata.json")

    total = len(ds)
    print(f"Downloading {total} samples for split '{split_name}' -> {split_dir}")

    records = []
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_name", "sentence", "speaker_id", "mic", "duration"])

        for i, example in enumerate(ds):
            audio = example["audio"]
            array = audio["array"]
            sampling_rate = audio["sampling_rate"]

            file_name = f"{split_name}_{i:06d}.wav"
            file_path = os.path.join(audio_dir, file_name)
            sf.write(file_path, array, sampling_rate)

            row = {
                "file_name": os.path.join("audio", file_name),
                "sentence": example.get("sentence", ""),
                "speaker_id": example.get("speaker_id", ""),
                "mic": example.get("mic", ""),
                "duration": example.get("duration", ""),
            }
            writer.writerow(
                [row["file_name"], row["sentence"], row["speaker_id"], row["mic"], row["duration"]]
            )
            records.append(row)

            if (i + 1) % 500 == 0 or (i + 1) == total:
                print(f"  {i + 1}/{total} samples saved")

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(records, json_file, ensure_ascii=False, indent=2)

    print(f"Finished split '{split_name}': {total} audio files + metadata saved in {split_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Dataset: {DATASET_ID}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Splits: {args.splits}")
    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples per split")

    for split in args.splits:
        download_split(split, args.output_dir, args.max_samples, args.hf_token)

    print("\nAll requested splits downloaded successfully.")


if __name__ == "__main__":
    main()