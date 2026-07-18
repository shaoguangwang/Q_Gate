"""Stage 3: top-K keyframe selection from a fused score distribution, then
subtitle-aware multiple-choice QA with a downstream MLLM. Reads a fused-scores
JSON and writes per-sample predictions plus an accuracy summary.
"""
import os
import logging
import json
from typing import List, Dict, Tuple
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import datetime
from collections import defaultdict
import re
# -----------------------------------------------------------------------------
# 1. Import the project's LLM interface (make sure the path is correct)
# -----------------------------------------------------------------------------
# The qgate package sits alongside this script.
from qgate.interface_llm import QGateGrounder

# -----------------------------------------------------------------------------
# 2. Logging configuration
# -----------------------------------------------------------------------------
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Lightweight check that only tests file existence without parsing; fast.
def check_subtitle_exists(video_id: str, dataset_type: str) -> bool:
    """Check whether the subtitle file for the given video_id and dataset_type exists."""
    # Subtitle roots. Override via env vars; defaults are relative to the repo.
    subtitle_roots = {
        'longvideobench': os.environ.get('LVB_SUBTITLE_ROOT', './Datasets/LVBench/subtitles'),
        'videomme': os.environ.get('VIDEOMME_SUBTITLE_ROOT', './Datasets/Video-MME/subtitle')
    }
    
    if dataset_type == 'longvideobench':
        subtitle_path = os.path.join(subtitle_roots['longvideobench'], f"{video_id}_en.json")
    elif dataset_type == 'videomme':
        subtitle_path = os.path.join(subtitle_roots['videomme'], f"{video_id}.srt")
    else:
        return False # unknown dataset type
        
    return os.path.exists(subtitle_path)

# -----------------------------------------------------------------------------
# 3. Helper functions (video processing, subtitle processing & keyframe selection)
# -----------------------------------------------------------------------------

def get_aligned_duration_and_fps(video_path: str) -> Tuple[int, float]:
    """
    Compute the video's effective duration (total seconds) and FPS, using floor
    rounding — int(duration) — so per-second frame indices stay aligned across
    all score streams.
    Returns: (duration_in_seconds, fps)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0, 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count / fps) if fps > 0 else 0
        cap.release()
        return duration, fps
    except Exception as e:
        logger.error(f"OpenCV Error while reading video properties for {video_path}: {e}")
        return 0, 0.0

def extract_frames_from_video(video_path: str, frame_indices: List[int]) -> List[Image.Image]:
    """Extract the frames at the given frame indices from the video and return a list of PIL Images."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video for frame extraction: {video_path}")
        return frames
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for idx in sorted(list(set(frame_indices))):
        if idx >= total_frames:
            logger.warning(f"Frame index {idx} exceeds total frame count {total_frames}, skipping.")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    cap.release()
    return frames

def parse_time(time_str: str) -> float:
    """Convert an SRT/JSON timestamp string to seconds."""
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
    except Exception as e:
        logger.error(f"Failed to parse timestamp string: '{time_str}', error: {e}")
        return 0.0

def clean_srt_line(line: str) -> str:
    """Strip HTML tags from an SRT subtitle line."""
    return re.sub(r'<[^>]+>', '', line).strip()

def load_and_parse_subtitles(video_id: str, dataset_type: str) -> List[Dict]:
    """Load and parse the subtitle file according to the dataset type."""
    # Subtitle roots. Override via env vars; defaults are relative to the repo.
    subtitle_roots = {
        'longvideobench': os.environ.get('LVB_SUBTITLE_ROOT', './Datasets/LVBench/subtitles'),
        'videomme': os.environ.get('VIDEOMME_SUBTITLE_ROOT', './Datasets/Video-MME/subtitle')
    }
    subtitles = []

    if dataset_type == 'longvideobench':
        subtitle_path = os.path.join(subtitle_roots['longvideobench'], f"{video_id}_en.json")
    elif dataset_type == 'videomme':
        subtitle_path = os.path.join(subtitle_roots['videomme'], f"{video_id}.srt")
    else:  # datasets without subtitles
        return subtitles

    if not os.path.exists(subtitle_path):
        return subtitles
    
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
                lines = f.read().strip().split('\n\n')
                for block in lines:
                    parts = block.split('\n')
                    if len(parts) >= 3:
                        time_line = parts[1]
                        text_lines = parts[2:]
                        try:
                            start_str, end_str = time_line.split(' --> ')
                            subtitles.append({
                                'start': parse_time(start_str),
                                'end': parse_time(end_str),
                                'text': clean_srt_line(" ".join(text_lines))
                            })
                        except ValueError:
                            continue
    except Exception as e:
        logger.error(f"Failed to parse subtitle file {subtitle_path}: {e}")

    return subtitles

def format_time(seconds: float) -> str:
    """Format seconds as a H:MM:SS timestamp string (e.g. 65 -> '0:01:05')."""
    return str(datetime.timedelta(seconds=int(seconds)))

def select_and_format_subtitles(
    selected_timestamps: List[int], 
    all_subtitles: List[Dict], 
    narrative_scores: np.ndarray, 
    top_n_supplement: int = 2
) -> str:
    """
    Select and format subtitles for the selected frames.

    Frame-associated subtitles are timestamped to their triggering frame; any
    remaining narrative-supplementary subtitles keep their own start time.
    """
    if not all_subtitles:
        return ""

    associated_subs = {}
    # 1. Frame-associated subtitles
    for ts in selected_timestamps:
        for sub in all_subtitles:
            if sub['start'] <= ts <= sub['end']:
                # Dedup by the subtitle start time as the key, while also storing the frame timestamp that triggered it
                if sub['start'] not in associated_subs:
                    associated_subs[sub['start']] = {'sub': sub, 'frame_ts': ts}

    # 2. Narrative supplementary subtitles
    if len(narrative_scores) > 0:
        # Build a mapping from subtitle start time to score
        sub_to_score = {}
        for sub in all_subtitles:
            # Use the center of the subtitle interval as its representative time point
            center_ts = int((sub['start'] + sub['end']) / 2)
            if center_ts < len(narrative_scores):
                 # Assign this subtitle its highest score (a subtitle may span multiple seconds)
                 score_in_range = narrative_scores[int(sub['start']):int(sub['end'])+1]
                 if len(score_in_range) > 0:
                    sub_to_score[sub['start']] = np.max(score_in_range)

        if sub_to_score:
            # Sort all subtitles by score
            sorted_by_score = sorted(sub_to_score.items(), key=lambda item: item[1], reverse=True)

            # Add the top_n_supplement highest-scoring subtitles that have not yet been selected
            added_count = 0
            for sub_start_time, score in sorted_by_score:
                if added_count >= top_n_supplement:
                    break
                if sub_start_time not in associated_subs:
                    # Find the corresponding subtitle object
                    for sub in all_subtitles:
                        if sub['start'] == sub_start_time:
                            # This is a supplementary subtitle; it has no associated frame timestamp
                            associated_subs[sub_start_time] = {'sub': sub, 'frame_ts': None}
                            added_count += 1
                            break

    if not associated_subs:
        return ""

    # 3. Final assembly and formatting
    # Sort by the subtitle's original start time
    sorted_subs_data = sorted(associated_subs.values(), key=lambda x: x['sub']['start'])

    formatted_text = "\n[Relevant Subtitles (in chronological order)]\n"
    for data_item in sorted_subs_data:
        sub = data_item['sub']
        frame_ts = data_item['frame_ts']

        if frame_ts is not None:
            # Frame-associated subtitle: use the frame timestamp
            time_str = format_time(frame_ts)
            prefix = f"- [Subtitle for Image at {time_str}]"
        else:
            # Supplementary subtitle: use its own start time
            time_str = format_time(sub['start'])
            prefix = f"- [at {time_str}]"
            
        formatted_text += f"{prefix} {sub['text']}\n"
        
    return formatted_text

def select_keyframes(entry: Dict, k: int, score_key: str) -> List[int]:
    """
    Select the k highest-scoring seconds from `score_key` as keyframe timestamps.

    Returns the selected seconds sorted and de-duplicated. If the score array is
    empty, falls back to k evenly spaced seconds.
    """
    video_duration, _ = get_aligned_duration_and_fps(entry['video_path'])
    if video_duration == 0 or k <= 0:
        return []

    if k >= video_duration:
        return list(range(video_duration))

    scores = np.array(entry.get(score_key, []))
    if len(scores) == 0:
        logger.warning(f"Score list '{score_key}' for video {entry['video_id']} is empty, falling back to uniform sampling.")
        return np.linspace(0, video_duration - 1, k, dtype=int).tolist()

    if len(scores) < video_duration:
        scores = np.pad(scores, (0, video_duration - len(scores)), 'constant')

    indices = np.argsort(scores)[-k:].tolist()
    return sorted(list(set(indices)))

def calculate_and_save_stats(qa_results: List[Dict], output_path: str):
    """Compute detailed accuracy statistics, then print and save them to a file."""
    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
    
    for item in qa_results:
        duration_group = item.get("duration_group", "unknown")
        if item.get('pred_answer') == "ERROR":
            stats[duration_group]['errors'] += 1
            continue
        stats[duration_group]['total'] += 1
        if item.get('correct') is True:
            stats[duration_group]['correct'] += 1
    
    report_lines = ["="*50, "  QA Accuracy Summary Report", "="*50]
    total_correct = sum(v['correct'] for v in stats.values())
    total_valid = sum(v['total'] for v in stats.values())
    overall_accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    
    report_lines.append(f"\nOverall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_valid})")
    report_lines.append("\n--- Per Duration Group ---")
    report_lines.append("{:<15} | {:<11} | {:<15} | {:<15}".format("Duration Group", "Accuracy", "Correct/Total", "Error Samples"))
    report_lines.append("-" * 65)
    
    for group in sorted(stats.keys(), key=str):
        counts = stats[group]
        acc = counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        report_lines.append("{:<15} | {:<11.2%} | {:<15} | {:<15}".format(group, acc, f"{counts['correct']}/{counts['total']}", counts['errors']))
    
    report_lines.append("\n" + "="*50)
    summary_text = "\n".join(report_lines)
    print(summary_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    logger.info(f"Statistics report saved to: {output_path}")

# -----------------------------------------------------------------------------
# 4. Main pipeline function
# -----------------------------------------------------------------------------

def run_qa_pipeline(args):
    """Main execution pipeline."""
    logger.info(f"Initializing LLM backend: {args.backend}...")
    grounder = QGateGrounder(backend=args.backend)

    logger.info(f"Loading input JSON file: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    qa_results, skipped_count = [], 0

    CKPT_EVERY = 50
    _base = f"{nowTime}_qa_{args.dataset}_{args.backend}_k{args.frame_num}_{args.score_key}"
    if args.use_subtitles: _base += "_subtitle"
    if args.filter_with_subtitles: _base += "_filtered"
    os.makedirs(args.output_dir, exist_ok=True)
    output_json_path = os.path.join(args.output_dir, f"{_base}.json")
    output_txt_path = os.path.join(args.output_dir, f"{_base}.txt")

    for entry in tqdm(data, desc=f"Running QA ({args.backend}, k={args.frame_num})"):
        video_id = entry.get('video_id', 'unknown')

        # Filter by subtitle existence only when subtitle filtering is enabled, so
        # subtitle-less datasets are not entirely skipped.
        if args.use_subtitles and args.filter_with_subtitles and not check_subtitle_exists(video_id, args.dataset):
            skipped_count += 1
            continue # go straight to the next iteration
        
        try:
            subtitle_context = ""
            if args.use_subtitles:
                all_subtitles = load_and_parse_subtitles(video_id, args.dataset)
                narrative_scores = np.array(entry.get('score_narrative', []))
            
            timestamps = select_keyframes(entry, args.frame_num, args.score_key)
            
            if args.use_subtitles:
                subtitle_context = select_and_format_subtitles(timestamps, all_subtitles, narrative_scores)

            _, fps = get_aligned_duration_and_fps(entry['video_path'])
            if fps == 0: raise ValueError(f"FPS is 0; the video file '{entry['video_path']}' may be corrupt or invalid.")
            
            frame_indices = [int(ts * fps) for ts in timestamps]
            frames = extract_frames_from_video(entry['video_path'], frame_indices)
            
            if not frames: raise ValueError("Failed to extract any valid frames.")

            pred_answer = grounder.inference_qa(
                frames=frames,
                frame_timestamps=timestamps,
                question=entry['question'],
                options=entry['options'],
                subtitle_info=subtitle_context,
                temperature=0.2
            )

            # A backend/API failure surfaces as an "Error: ..." string; treat it as
            # an ERROR (excluded from accuracy), not a wrong answer.
            if pred_answer.startswith("Error"):
                raise RuntimeError(pred_answer)

            gt_answer_clean = entry['answer'].strip().lower()
            pred_answer_clean = pred_answer.strip().lower()
            entry['pred_answer'] = pred_answer
            entry['correct'] = (pred_answer_clean == gt_answer_clean)
            entry['used_frame_timestamps'] = timestamps

        except Exception as e:
            logger.error(f"Error while processing video {video_id}: {e}")
            entry['pred_answer'] = "ERROR"
            entry['correct'] = False
            entry['used_frame_timestamps'] = []
            
        qa_results.append(entry)

        # Incremental checkpoint: flush to disk every CKPT_EVERY items (overwrite with partial results) to guard against mid-run crashes
        if len(qa_results) % CKPT_EVERY == 0:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(qa_results, f, indent=4, ensure_ascii=False)
            logger.info(f"[checkpoint] Incrementally saved {len(qa_results)} items to {output_json_path}")

    logger.info(f"QA finished. Processed {len(qa_results)} samples in total.")
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} samples in total due to missing subtitle files.")

    # Save results (output_json_path/output_txt_path were computed before the loop; reuse them here for the final full flush)
    logger.info(f"Saving QA results to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(qa_results, f, indent=4, ensure_ascii=False)
        
    calculate_and_save_stats(qa_results, output_txt_path)

# -----------------------------------------------------------------------------
# 5. Main program entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Gate Stage 3: top-K keyframe selection and downstream Video-QA.")
    
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file with all pre-computed scores.")
    parser.add_argument("--dataset", type=str, required=True, choices=['longvideobench', 'videomme'], help="Name of the dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save QA results and summary.")
    parser.add_argument("--backend", type=str, required=True, choices=['gpt4', 'qwen3_32b', 'internvl_26b', 'internvl3_8b'], help="MLLM backend for QA (each self-configures from its endpoint / API key).")
    parser.add_argument("--frame_num", type=int, required=True, help="Number of keyframes (k) to select.")

    parser.add_argument("--score_key", type=str, default='score_fusion',
                        choices=['score_fusion'],
                        help="Fused score distribution to select keyframes from.")

    parser.add_argument("--use_subtitles", action='store_true',
                        help="Enable this flag to load subtitles and include them in the prompt.")

    parser.add_argument("--filter_with_subtitles", action='store_true',
                        help="If set, only run QA on videos that have a subtitle file.")

    args = parser.parse_args()

    run_qa_pipeline(args)

# Example (Q-Gate, LongVideoBench, GPT-4o, K=8). See README for the full pipeline.
#   export OPENAI_API_KEY="..."
#   python run_qa.py \
#       --input_json <fused_scores.json> \
#       --dataset longvideobench \
#       --output_dir ./results \
#       --backend gpt4 --frame_num 8 \
#       --score_key score_fusion \
#       --use_subtitles --filter_with_subtitles
