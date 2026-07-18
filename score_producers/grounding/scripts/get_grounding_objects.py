'''
1st step of the pipeline: get grounding objects on each video & question pair.

Driver for the T*-derived grounding library under this producer
(see THIRD_PARTY_LICENSES at the repository root).
'''

import json
import sys
import argparse
import numpy as np
import os

sys.path.append('./')
sys.path.append('./YOLO-World/')
from lib.interface_llm import UniversalGrounder
from lib.interface_yolo import YoloInterface
from lib.framework import GroundingFramework, initialize_yolo
from utils.data_loader import VideoMME2qgate_json, LongVideoBench2qgate_json

np.random.seed(2025)

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Searcher: Video Frame Search and QA Tool")

    # Data meta processing arguments
    parser.add_argument('--dataset', type=str, default="longvideobench", choices=["longvideobench", "videomme"], help='The video dataset used for processing.')
    parser.add_argument('--video_root', type=str, default='./Datasets/LVBench/videos', help='Root directory where the input video files are stored.')
    parser.add_argument('--anno_path', type=str, default='./Datasets/LVBench/lvb_val.json', help='LongVideoBench only: path to the official validation JSON.')
    parser.add_argument('--obj_path', type=str, default='./runs/obj/obj_result.json', help='Path to save the object grounding results.')
    
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
    parser.add_argument('--prompt_type', type=str, default='cot', help='Prompt type used.')
    parser.add_argument('--upload_video', type=int, default=1, help='Upload video or not to OpenAI API.')
    return parser.parse_args()

def grounding_objects_onVideo(args, data_item,
        yolo_scorer: YoloInterface,
        grounder: UniversalGrounder) -> dict:
    """
    grounding objects on a single video.

    Args:
        args (argparse.Namespace): Parsed arguments.
        data_item (dict): Dictionary containing 'video_path', 'question', and 'options'.
        yolo_scorer : YOLO interface instance.
        grounder: Universal Grounder instance.

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
        update_method="spline"
    )

    # Use Grounder to get target and cue objects
    target_objects, cue_objects, relations = framework.get_grounded_objects(args.prompt_type, args.upload_video)

    result = {
        "video_path": data_item['video_path'],
        "grounding_objects": framework.results.get('Searching_Objects', []),
    }

    return result

def main():
    """
    Main function to execute FrameSearcher.
    """
    args = parse_arguments()
    
    if args.dataset == "longvideobench":
        dataset = LongVideoBench2qgate_json(video_root=args.video_root, anno_path=args.anno_path)

    elif args.dataset == "videomme":
        dataset = VideoMME2qgate_json(video_root=args.video_root)

    else:
        raise ValueError(f"Unsupported --dataset '{args.dataset}' (choose longvideobench or videomme).")

    print(f"Loaded {len(dataset)} records for {args.dataset}.")

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
    for idx, data_item in enumerate(dataset):
        
        print(f"Processing {idx+1}/{len(dataset)}: {data_item['video_id']}")
        try:
            result = grounding_objects_onVideo(args, data_item=data_item, grounder=grounder, yolo_scorer=yolo_interface)
            print(f"Completed: {data_item['video_id']}\n")

        except Exception as e:
            print(f"Error processing {data_item['video_id']}: {e}")
            result = {
                "video_id": data_item.get('video_id', ''),
                "grounding_objects": [],
                "answer": data_item.get('answer', ''),
                "error": str(e)
            }
                
        data_item.update(result)        
        results.append(data_item)
    
    if not os.path.exists(os.path.dirname(args.obj_path)):
        os.makedirs(os.path.dirname(args.obj_path))

    with open(args.obj_path, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, indent=4, ensure_ascii=False)
    
    print(f"Batch processing completed. Results saved to {args.obj_path}")        
        
if __name__ == "__main__":
    main()

# Example (step 1 of the grounding pipeline; requires the external YOLO-World
# repo and OPENAI_API_KEY in the environment):
#   python scripts/get_grounding_objects.py \
#       --dataset longvideobench \
#       --video_root <path/to/LVBench/videos> \
#       --obj_path ./output/lvb_obj_result.json