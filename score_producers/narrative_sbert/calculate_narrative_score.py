"""Compute per-second raw score_narrative values by matching the question (and optional options) against video subtitles via SBERT cosine similarity."""
import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
import math
import cv2
from typing import List

# --- 1. Helper functions ---

def parse_time(time_str: str) -> float:
    time_str = time_str.replace(',', '.')
    try:
        if '.' in time_str:
            parts = time_str.split('.')
            h, m, s = map(int, parts[0].split(':'))
            ms = int(parts[1])
        else:
            h, m, s = map(int, time_str.split(':'))
            ms = 0
        return h * 3600 + m * 60 + s + ms / 1000.0
    except Exception:
        return 0.0

def clean_srt_line(line: str) -> str:
    return re.sub(r'<[^>]+>', '', line).strip()

def get_aligned_duration(video_path: str) -> int:
    """Compute the effective video duration (in whole seconds) with floor rounding, so per-second frame indices stay aligned across all score streams."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps) if fps > 0 else 0
        cap.release()
        return duration
    except Exception:
        return 0

def process_subtitles(subtitle_path: str, dataset_type: str) -> List[dict]:
    subtitles = []
    if not os.path.exists(subtitle_path): return subtitles
    try:
        with open(subtitle_path, 'r', encoding='utf-8', errors='ignore') as f:
            if dataset_type == 'longvideobench':
                sub_data = json.load(f)
                for item in sub_data:
                    subtitles.append({
                        'start': parse_time(item['start']),
                        'end': parse_time(item['end']),
                        'text': item['line'].replace('&nbsp;', ' ').strip()
                    })
            elif dataset_type == 'videomme':
                content = f.read().strip()
                # Handle different SRT line-break conventions.
                if '\r\n\r\n' in content:
                    lines = content.split('\r\n\r\n')
                else:
                    lines = content.split('\n\n')
                    
                for block in lines:
                    parts = block.strip().split('\n')
                    # An SRT block is typically index / timestamp / text lines, but the
                    # layout varies, so locate the timestamp line explicitly rather than
                    # assuming a fixed position.
                    if len(parts) >= 3:
                        time_line_idx = -1
                        for idx, p in enumerate(parts):
                            if '-->' in p:
                                time_line_idx = idx
                                break
                        
                        if time_line_idx != -1 and time_line_idx + 1 < len(parts):
                            time_line = parts[time_line_idx]
                            text_lines = parts[time_line_idx+1:]
                            try:
                                start_str, end_str = time_line.split(' --> ')
                                subtitles.append({
                                    'start': parse_time(start_str),
                                    'end': parse_time(end_str),
                                    'text': clean_srt_line(" ".join(text_lines))
                                })
                            except ValueError: continue
    except Exception as e:
        print(f"Error processing subtitle file {subtitle_path}: {e}")
    return subtitles

# --- 2. Core score computation and processing logic ---

def calculate_narrative_scores(input_json_path: str, output_dir: str, dataset_type: str,
                               st_model_name: str, use_options: bool):
    """Compute the raw score_narrative values (no smoothing or normalization applied)."""
    print(f"Loading Sentence Transformer model: {st_model_name}...")
    model = SentenceTransformer(st_model_name)
    print("Model loaded successfully.")

    # Subtitle roots. Override via env vars; defaults are relative to the repo.
    subtitle_roots = {
        'longvideobench': os.environ.get('LVB_SUBTITLE_ROOT', './Datasets/LVBench/subtitles'),
        'videomme': os.environ.get('VIDEOMME_SUBTITLE_ROOT', './Datasets/Video-MME/subtitle')
    }
    subtitle_root = subtitle_roots.get(dataset_type)
    
    if not subtitle_root:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_data = []
    
    for entry in tqdm(data, desc=f"Processing {dataset_type} data"):
        video_id = entry['video_id']
        question = entry['question']
        options = entry.get('options', '')
        
        query_text = question
        if use_options and options:
            query_text += "\n" + options.replace('\n', ' ')

        video_duration = get_aligned_duration(entry['video_path'])

        # Per-second raw scores, initialized to zero.
        score_narrative_raw = np.zeros(video_duration, dtype=np.float32)

        if video_duration > 0:
            if dataset_type == 'longvideobench':
                subtitle_path = os.path.join(subtitle_root, f"{video_id}_en.json")
            else:  # videomme
                subtitle_path = os.path.join(subtitle_root, f"{video_id}.srt")
                
            subtitles = process_subtitles(subtitle_path, dataset_type)
            
            if subtitles:
                query_embedding = model.encode(query_text, convert_to_tensor=True)
                subtitle_texts = [sub['text'] for sub in subtitles if sub['text']]
                if subtitle_texts:
                    subtitle_embeddings = model.encode(subtitle_texts, convert_to_tensor=True)
                    similarities = util.pytorch_cos_sim(query_embedding, subtitle_embeddings)[0].cpu().numpy()
                    
                    sub_idx = 0
                    for sub in subtitles:
                        if not sub['text']: continue
                        score = similarities[sub_idx]
                        sub_idx += 1

                        # Expand-outward coverage: a subtitle marks every second it
                        # touches (floor the start, ceil the end).
                        start_sec = math.floor(sub['start'])
                        end_sec = math.ceil(sub['end'])

                        start_sec = max(0, start_sec)
                        end_sec = min(video_duration, end_sec)

                        if start_sec < end_sec:
                            # Keep the highest similarity when subtitle spans overlap on a second.
                            score_narrative_raw[start_sec:end_sec] = np.maximum(score_narrative_raw[start_sec:end_sec], score)

        entry['score_narrative'] = score_narrative_raw.tolist()
        updated_data.append(entry)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_json_path)
    file_name_without_ext, _ = os.path.splitext(base_name)
    output_filename = f"{file_name_without_ext}_{st_model_name.split('/')[-1]}_narrative.json"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\nSaving updated data with RAW narrative scores to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and add RAW narrative scores to a processed JSON file.")
    
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input processed JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new JSON file.")
    parser.add_argument("--dataset", type=str, required=True, choices=['longvideobench', 'videomme'], help="Type of the dataset.")
    parser.add_argument("--st_model", type=str, default='all-mpnet-base-v2',
                        help="Name of the Sentence Transformer model to use.")
    parser.add_argument("--use_options", action='store_true',
                        help="Include options in the query text for similarity calculation.")

    args = parser.parse_args()

    calculate_narrative_scores(
        input_json_path=args.input_json,
        output_dir=args.output_dir,
        dataset_type=args.dataset,
        st_model_name=args.st_model,
        use_options=args.use_options
    )
