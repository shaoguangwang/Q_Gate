'''
2nd step of the pipeline: perform keyframe search over videos based on the grounded objects.

Driver for the T*-derived grounding library under this producer
(see THIRD_PARTY_LICENSES at the repository root).
'''

import sys
import argparse
import json
import numpy as np
import os

# Resolve paths relative to this file so the producer runs from any cwd.
script_dir = os.path.dirname(os.path.abspath(__file__))          # location of scripts/
root_dir = os.path.dirname(script_dir)                           # grounding producer root directory
yolo_world_dir = os.path.join(root_dir, 'YOLO-World')

sys.path.insert(0, root_dir)          # allow the local lib package to be imported
sys.path.insert(0, yolo_world_dir)    # allow yolo_world to be imported

# Import the local grounding interfaces
from lib.interface_llm import UniversalGrounder
from lib.interface_yolo import YoloInterface
from lib.framework import GroundingFramework, initialize_yolo

np.random.seed(2025)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Searcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--obj_path', type=str, default='./runs/obj/obj_result.json', help='The input data path of grounding objects.')
    parser.add_argument('--kfs_path', type=str, default='./runs/kfs/kfs_VideoMME.json', help='Path to save the key frame searching results.')
    
    # Common arguments
    parser.add_argument('--config_path', type=str, default="./YOLO-World/configs/pretrain/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py", help='Path to the YOLO configuration file.')
    parser.add_argument('--checkpoint_path', type=str, default="./pretrained/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth", help='Path to the YOLO model checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device for model inference (e.g., "cuda:0" or "cpu").')
    parser.add_argument('--search_nframes', type=int, default=8, help='Number of top frames to return.')
    parser.add_argument('--grid_rows', type=int, default=4, help='Number of rows in the image grid.')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of columns in the image grid.')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='YOLO detection confidence threshold.')
    parser.add_argument('--search_budget', type=float, default=1.0, help='Maximum ratio of frames to process during search.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save outputs.')
    parser.add_argument('--prefix', type=str, default='stitched_image', help='Prefix for output filenames.')
    parser.add_argument('--backend', type=str, default='gpt4', help='Grounding VLM backend (gpt4 is the documented default; a local llava server is also supported).')
    parser.add_argument('--save_batch', type=int, default=10, help='Save batch results to output_json every N entries.')
    parser.add_argument('--update_method', type=str, default='spline', help='Update distribution method.')
    parser.add_argument('--relation_alpha', type=float, default=0, help='Weight of relation score.')
    return parser.parse_args()

def process_one_video(args, data_item,
        yolo_scorer: YoloInterface,
        grounder: UniversalGrounder,) -> dict:
    """
    Process a single video search and QA.

    Args:
        args (argparse.Namespace): Parsed arguments.
        data_item (dict): Dictionary containing 'video_path', 'question', and 'options'.
        yolo_scorer: YOLO interface instance.
        grounder (UniversalGrounder): Universal Grounder instance.

    Returns:
        dict: Results containing 'video_path', 'grounding_objects', 'frame_timestamps', 'answer'.
    """
    # Initialize the grounding framework
    framework = GroundingFramework(
        grounder=grounder,
        yolo_scorer=yolo_scorer,
        video_path=data_item['video_path'],
        question=data_item['question'],
        options=data_item['options'],
        search_nframes=args.search_nframes,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence_threshold,
        search_budget=args.search_budget,
        prefix=args.prefix,
        device=args.device,
        update_method=args.update_method
    )

    # load objects from json
    target_objects = data_item['grounding_objects']['target_objects']
    cue_objects = data_item['grounding_objects']['cue_objects']
    relations = data_item['grounding_objects'].get('relations', [])

    # Initialize searching targets in the searcher
    video_searcher = framework.set_searching_targets(target_objects, cue_objects, relations)
    video_searcher.relation_alpha = args.relation_alpha

    # Perform search
    all_frames, time_stamps = framework.perform_search(video_searcher)

    # Output the results
    print("Final Results:")
    print(f"Grounding Objects: {data_item['grounding_objects']}")
    print(f"Frame Timestamps: {framework.results['timestamps']}")

    # Collect the results
    result = {
        "video_path": data_item['video_path'],
        "grounding_objects": data_item['grounding_objects'],
        "keyframe_timestamps": framework.results.get('timestamps', []),
        "frame_distribution": video_searcher.P_history[-1],
        "score_list": video_searcher.Score_history[-1],
        "num_iterations": framework.results.get('num_iterations', 0)
    }

    return result

def main():
    """
    Main function to execute key frame search.
    """
    args = parse_arguments()    

    # Initialize Grounder
    grounder = UniversalGrounder(
        backend=args.backend,
        gpt4_model_name="gpt-4o"
    )

    # Initialize YOLO interface
    yolo_interface = initialize_yolo(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )

    results = []

    with open(args.obj_path, 'r', encoding='utf-8') as f_read:
        dataset = json.load(f_read)
    
    if not os.path.exists(os.path.dirname(args.kfs_path)):
        os.makedirs(os.path.dirname(args.kfs_path))

    for idx, data_item in enumerate(dataset):
        try:
            result = process_one_video(args, data_item=data_item, grounder=grounder, yolo_scorer=yolo_interface)            
            print(f"Completed: {data_item['video_id']}\n")

        except Exception as e:
            print(f"Error processing {data_item['video_id']}: {e}")
            result = {
                "video_id": data_item.get('video_id', ''),
                "grounding_objects": [],
                "keyframe_timestamps": [],
                "answer": "",
                "error": str(e)
            }
            
        data_item.update(result)
        results.append(data_item)
        if (idx + 1) % args.save_batch == 0 or (idx + 1) == len(dataset):
            # Save batch results to output_json
            with open(args.kfs_path, 'w', encoding='utf-8') as f_out:
                json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    print(f"Batch processing completed. Results saved to {args.kfs_path}")

if __name__ == "__main__":
    main()

# Example (step 2: YOLO-World search over grounded objects -> keyframe distribution):
#   python scripts/get_key_frames.py \
#       --obj_path ./output/lvb_obj_result.json \
#       --kfs_path ./output/lvb_keyframes.json
