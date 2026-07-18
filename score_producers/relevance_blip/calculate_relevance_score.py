"""Compute per-second frame-query relevance scores for videos using BLIP image-text matching and write them as score_relevance to JSON."""
import torch
from PIL import Image
# LAVIS is imported lazily inside the scoring function so `--help` works without it.
import json
import cv2
import os
import argparse
from tqdm import tqdm

# --- Helper functions ---

def get_aligned_duration_and_fps(video_path: str) -> tuple[int, float]:
    """Compute the video's effective duration (in whole seconds) and FPS with
    floor rounding, so per-second frame indices stay aligned across all score streams."""
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
        print(f"OpenCV Error while reading video properties for {video_path}: {e}")
        return 0, 0.0

# --- Core relevance score computation ---

def calculate_relevance_scores(
    input_json_path: str,
    output_dir: str,
    dataset_type: str,
    model_name: str,
    use_options: bool,
    device: str
):
    """Compute raw ``score_relevance`` values for each entry and write a new JSON file."""
    print(f"Loading feature extraction model: {model_name}...")

    if model_name != 'blip':
        raise ValueError(f"Unsupported model name '{model_name}' (only 'blip' is supported).")

    from lavis.models import load_model_and_preprocess
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device, is_eval=True
    )
    model.eval()
    print("Model loaded successfully.")

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated_data = []

    for entry in tqdm(data, desc=f"Computing relevance scores for {dataset_type} ({model_name})"):
        question = entry['question']
        options = entry.get('options', '')
        video_path = entry['video_path']
        query_text = question
        if use_options and options:
            query_text += "\n" + options.replace('\n', ' ')
        # The query text is constant per entry, so preprocess it once here rather
        # than re-running the tokenizer for every frame in the hot loop below.
        txt = text_processors["eval"](query_text)

        video_duration, fps = get_aligned_duration_and_fps(video_path)

        if video_duration == 0:
            entry['score_relevance'] = []
            updated_data.append(entry)
            continue

        raw_scores = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            entry['score_relevance'] = [0.0] * video_duration
            updated_data.append(entry)
            continue

        try:
            for i in range(video_duration):
                frame_idx = int(i * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    raw_scores.append(0.0)
                    continue
                
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_image = Image.fromarray(frame_rgb)

                    with torch.no_grad():
                        img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                        # BLIP image-text matching head; take the matching probability (class 1).
                        blip_output = model({"image": img, "text_input": txt}, match_head="itm")
                        score = torch.nn.functional.softmax(blip_output, dim=1)[:, 1].item()

                    raw_scores.append(score)
                except Exception:
                    # A frame that fails to decode/score contributes zero relevance.
                    raw_scores.append(0.0)
        finally:
            cap.release()

        # Store the raw per-second scores without smoothing or normalization.
        entry['score_relevance'] = raw_scores
        updated_data.append(entry)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_json_path)
    file_name_without_ext, _ = os.path.splitext(base_name)
    output_filename = f"{file_name_without_ext}_{model_name}_relevance.json"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\nSaving updated data with RAW relevance scores to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and add RAW frame-query relevance scores.")

    parser.add_argument("--input_json", type=str, required=True, help="Path to the input processed JSON file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new JSON file.")
    parser.add_argument("--dataset", type=str, required=True, choices=['longvideobench', 'videomme'], help="Type of the dataset.")
    parser.add_argument("--device", type=str, default='cuda:0', help="Device to run the models on.")
    parser.add_argument("--model_name", type=str, default='blip', choices=['blip'],
                        help="Vision-language model for image-text matching (BLIP).")
    parser.add_argument("--use_options", action='store_true',
                        help="Include options in the query text for similarity calculation.")

    args = parser.parse_args()

    calculate_relevance_scores(
        input_json_path=args.input_json,
        output_dir=args.output_dir,
        dataset_type=args.dataset,
        model_name=args.model_name,
        use_options=args.use_options,
        device=args.device
    )
