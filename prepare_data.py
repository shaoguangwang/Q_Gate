"""Q-Gate | Stage 0: Dataset preparation.

Convert a raw benchmark (LongVideoBench or Video-MME) into the unified
Q-Gate QA-index JSON used by every downstream stage. Each output record has:

    video_id        str    unique clip id
    video_path      str    path to the .mp4 (rooted at --video_root)
    question        str
    options         str    "A) ...\nB) ...\n..."  (newline-joined, letter-prefixed)
    answer          str    the correct option letter, e.g. "B"
    duration_group  str/int  length bucket (see per-dataset notes below)
    position        list   optional ground-truth timestamp hints (may be empty)

The result is written to <output_dir>/<dataset>_processed.json and matches the
files shipped under Data/.

Usage:
    # Video-MME (question metadata pulled from the HuggingFace hub; videos local)
    python prepare_data.py --dataset videomme \
        --video_root /path/to/Video-MME/videos --output_dir ./Data

    # LongVideoBench (official val json + local videos)
    python prepare_data.py --dataset longvideobench \
        --input_file /path/to/lvb_val.json \
        --video_root /path/to/LVBench/videos --output_dir ./Data
"""

import json
import os
from typing import List
import argparse

def videomme_to_qgate_json(video_root: str) -> List[dict]:
    """Build records for Video-MME. Question metadata is pulled from the
    HuggingFace hub (`lmms-lab/Video-MME`); the videos must already be local."""
    from datasets import load_dataset
    print("Processing Video-MME from the HuggingFace hub...")
    dataset = load_dataset("lmms-lab/Video-MME")
    testset = dataset["test"]
    records = []
    dropped = 0
    for entry in testset:
        try:
            # Each option arrives as "A. text"; re-emit as "A) text".
            options_list = entry.get("options", [])
            options = ""
            for option in options_list:
                options += option[0] + ') ' + option[3:] + '\n'

            records.append({
                "video_id": entry.get("videoID"),
                "video_path": os.path.join(video_root, f"{entry.get('videoID')}.mp4"),
                "question": entry.get("question"),
                "options": options.rstrip('\n'),
                "answer": entry.get("answer", ""),
                "duration_group": entry.get("duration"),  # "short" | "medium" | "long"
                "position": [],
            })
        except Exception as e:
            dropped += 1
            print(f"Skipping malformed Video-MME entry: {e}")
            continue
    if dropped:
        print(f"Skipped {dropped} malformed Video-MME entries.")
    return records

def longvideobench_to_qgate_json(input_path: str, video_root: str) -> List[dict]:
    """Build records for LongVideoBench from its official validation JSON."""
    with open(input_path, 'r', encoding='utf-8') as file:
        lvb_dataset = json.load(file)
    records = []
    dropped = 0
    num2letter = ['A', 'B', 'C', 'D', 'E']
    for entry in lvb_dataset:
        try:
            options_list = entry.get("candidates", [])
            options = ""
            for i in range(len(options_list)):
                options += num2letter[i] + ') ' + options_list[i] + '\n'
            records.append({
                "video_id": entry.get("video_id"),
                "video_path": os.path.join(video_root, entry.get("video_path")),
                "question": entry.get("question"),
                "options": options.rstrip('\n'),
                "answer": num2letter[entry.get("correct_choice")],
                "duration_group": entry.get("duration_group"),  # seconds bucket: 15/60/600/3600
                "position": entry.get("position", []),
            })
        except Exception as e:
            dropped += 1
            print(f"Skipping malformed LongVideoBench entry: {e}")
            continue
    if dropped:
        print(f"Skipped {dropped} malformed LongVideoBench entries.")
    return records

def main():
    parser = argparse.ArgumentParser(description="Q-Gate dataset preparation: raw benchmark -> unified QA-index JSON.")
    parser.add_argument("--dataset", type=str, required=True, choices=['longvideobench', 'videomme'])
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write <dataset>_processed.json.")
    parser.add_argument("--video_root", type=str, required=True, help="Root directory holding the videos.")
    parser.add_argument("--input_file", type=str, default=None, help="LongVideoBench only: path to the official validation JSON.")

    args = parser.parse_args()

    if args.dataset == 'longvideobench':
        if not args.input_file:
            parser.error("--input_file (the official LongVideoBench val JSON) is required for --dataset longvideobench.")
        records = longvideobench_to_qgate_json(input_path=args.input_file, video_root=args.video_root)
    else:  # videomme
        records = videomme_to_qgate_json(video_root=args.video_root)

    if records:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.dataset}_processed.json")
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(records, file, indent=4, ensure_ascii=False)
        print(f"\nSuccess! Saved {len(records)} entries to {output_path}")
    else:
        print("No data processed.")

if __name__ == "__main__":
    main()
