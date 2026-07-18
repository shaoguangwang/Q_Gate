"""
GroundingFramework: turn a question into search targets, run the frame search, and return grounded objects and keyframes.

It coordinates the universal grounder (target/cue extraction), the YOLO interface (object scoring), and the frame
searcher to identify the frames most relevant to a video question.

Derived from T* (https://github.com/LongVideoHaystack/TStar), MIT-licensed.
See THIRD_PARTY_LICENSES at the repository root.
"""

import os
import sys
import logging
import numpy as np
from typing import List, Optional, Tuple
# Import the local grounding interfaces
from lib.interface_llm import UniversalGrounder
from lib.interface_yolo import YoloWorldInterface, YoloInterface
from lib.interface_searcher import FrameSearcher
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GroundingFramework:
    """
    Main class for performing object-based frame search and question-answering in a video.
    """
    def __init__(
        self,
        video_path: str,
        yolo_scorer: YoloInterface,
        grounder: UniversalGrounder,
        question: str,
        options: str,
        search_nframes: int = 8,
        grid_rows: int = 4,
        grid_cols: int = 4,
        output_dir: str = './output',
        confidence_threshold: float = 0.6,
        search_budget: int = 1000,
        prefix: str = 'stitched_image',
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:0",
        update_method: str = "spline"
    ):
        """
        Initialize the grounding framework.

        Args:
            video_path (str): Path to the input video file.
            yolo_scorer: YOLO interface instance.
            grounder (UniversalGrounder): Universal Grounder instance.
            question (str): The question for question-answering.
            options (str): Multiple-choice options for the question.
            search_nframes (int, optional): Number of top frames to return. Default is 8.
            grid_rows (int, optional): Number of rows in the image grid. Default is 4.
            grid_cols (int, optional): Number of columns in the image grid. Default is 4.
            output_dir (str, optional): Directory to save outputs. Default is './output'.
            confidence_threshold (float, optional): YOLO detection confidence threshold. Default is 0.6.
            search_budget (float, optional): Fraction of frames that may be processed during search
                (the CLI scripts pass a ratio, e.g. 1.0). Default is 1000 (effectively uncapped).
            prefix (str, optional): Prefix for output filenames. Default is 'stitched_image'.
            config_path (str, optional): Path to the YOLO configuration file. Default is None.
            checkpoint_path (str, optional): Path to the YOLO model checkpoint. Default is None.
            device (str, optional): Device for model inference (e.g., "cuda:0" or "cpu"). Default is "cuda:0".
            update_method (str, optional): Score-distribution update strategy. Default is 'spline'.
        """
        self.video_path = video_path
        self.yolo_scorer = yolo_scorer
        self.grounder = grounder
        self.question = question
        self.options = options
        self.search_nframes = search_nframes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.search_budget = search_budget
        self.prefix = prefix
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.update_method = update_method

        self.video_id=self.video_path.split("/")[-1].split(".")[0]
        self.output_dir = os.path.join(self.output_dir, self.video_id)  # Output save path
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Grounding framework initialized successfully.")

        self.results = {}

    def get_grounded_objects(self, prompt_type : str, upload_video : int) -> Tuple[List[str], List[str], List[Tuple[str]]]:
        """
        Use Grounder to obtain target objects, cue objects, and their relations.

        Returns:
            Tuple[List[str], List[str], List[Tuple[str]]]: target objects, cue objects, and relations.
        """
        if prompt_type == "cot":
            target_objects, cue_objects, relations = self.grounder.inference_query_grounding_cot(
                video_path=self.video_path,
                question=self.question,
                options=self.options,
                upload_video=upload_video
            )
        else:
            target_objects, cue_objects = self.grounder.inference_query_grounding(
                video_path=self.video_path,
                question=self.question,
                options=self.options,
                upload_video=upload_video
            )
            relations = []

        logger.info(f"Target objects: {target_objects}")
        logger.info(f"Cue objects: {cue_objects}")
        self.results["Searching_Objects"] = {"target_objects": target_objects,
                                             "cue_objects": cue_objects,
                                             "relations": relations}

        return target_objects, cue_objects, relations

    def set_searching_targets(self, target_objects, cue_objects, relations):
        """
        Initialize and configure the FrameSearcher for video object search.

        Args:
            target_objects (List[str]): List of target objects to search for in the video.
            cue_objects (List[str]): List of cue objects to assist in locating target objects.

        Returns:
           FrameSearcher: Configured instance of the FrameSearcher class.

        Notes:
            - The `FrameSearcher` is responsible for performing the object search within the video
            using the specified targets and cues.
            - Key parameters such as the search frame limit (`search_nframes`), grid shape (`image_grid_shape`),
            confidence threshold (`confidence_threshold`), and search budget (`search_budget`) are passed
            to the `FrameSearcher`.
            - The `yolo_scorer` is used as the object detection model for evaluating the objects in the video.
        """
        video_searcher = FrameSearcher(
            video_path=self.video_path,
            target_objects=target_objects,
            cue_objects=cue_objects,
            relations=relations,
            search_nframes=self.search_nframes,
            image_grid_shape=(self.grid_rows, self.grid_cols),
            output_dir=self.output_dir,
            confidence_threshold=self.confidence_threshold,
            search_budget=self.search_budget,
            yolo_scorer=self.yolo_scorer,
            update_method=self.update_method
        )

        return video_searcher

    def perform_search(self, video_searcher: FrameSearcher) -> Tuple[List[np.ndarray], List[float]]:
        """
        Execute the frame search process and retrieve relevant frames and timestamps.

        Args:
            video_searcher (FrameSearcher): Instance of FrameSearcher.

        Returns:
            Tuple[List[np.ndarray], List[float]]: List of frames and their corresponding timestamps.
        """
        all_frames, time_stamps, num_iterations = video_searcher.search_with_visualization()
        logger.info(f"Found {len(all_frames)} frames, timestamps: {time_stamps}")

        self.results['timestamps'] = time_stamps
        self.results['num_iterations'] = num_iterations
        return all_frames, time_stamps

def initialize_yolo(
    config_path: str,
    checkpoint_path: str,
    device: str
) -> YoloInterface:
    """
    Initialize the YOLO object detection model.

    Args:
        config_path (str): Path to the YOLO configuration file.
        checkpoint_path (str): Path to the YOLO model checkpoint.
        device (str): Device for model inference (e.g., "cuda:0").

    Returns:
        YoloWorldInterface: Initialized YOLO interface instance.

    Raises:
        FileNotFoundError: If the configuration file or checkpoint file is not found.
    """

    yolo = YoloWorldInterface(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )
    logger.info("YoloWorldInterface initialized successfully.")
    return yolo
