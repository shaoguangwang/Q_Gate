"""YOLO-World open-vocabulary detection interface for object grounding, with bbox annotation and visualization helpers.

Derived from T* (https://github.com/LongVideoHaystack/TStar), MIT-licensed.
See THIRD_PARTY_LICENSES at the repository root.
"""

import os.path as osp
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.apis import init_detector
import torch
from typing import List

import supervision as sv
from supervision.draw.color import ColorPalette

class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h

class YoloInterface:
    def __init__(self):
        """
        Initialize the YOLO-World model with the given configuration and checkpoint.

        Args:
        """
        
        pass
    def set_BBoxAnnotator(self):
        self.BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
        self.LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                        text_scale=0.5,
                                        text_thickness=1,
                                        smart_position=True,
                                        color=ColorPalette.LEGACY)

class YoloWorldInterface(YoloInterface):
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
        """
        Initialize the YOLO-World model with the given configuration and checkpoint.

        Args:
            config_path (str): Path to the model configuration file.
            checkpoint_path (str): Path to the model checkpoint.
            device (str): Device to run the model on (e.g., 'cuda:0', 'cpu').
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device

        # Load configuration
        cfg = Config.fromfile(config_path)
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config_path))[0])
        cfg.load_from = checkpoint_path

        # Initialize the model
        self.model = init_detector(cfg, checkpoint=checkpoint_path, device=device)
        self.set_BBoxAnnotator()

        # Initialize the test pipeline
        # build test pipeline
        self.model.cfg.test_dataloader.dataset.pipeline[
            0].type = 'mmdet.LoadImageFromNDArray'
        self.test_pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)

    def reparameterize_object_list(self, target_objects: List[str], cue_objects: List[str]):
        """
        Reparameterize the detect object list to be used by the YOLO model.

        Args:
            target_objects (List[str]): List of target object names.
            cue_objects (List[str]): List of cue object names.
        """
        # Combine target objects and cue objects into the final text format
        combined_texts = target_objects + cue_objects

        # Format the text prompts for the YOLO model
        self.texts = [[obj.strip()] for obj in combined_texts] + [[' ']]

        # Reparameterize the YOLO model with the provided text prompts
        self.model.reparameterize(self.texts)

    def inference_detector(self, images, max_dets=50, score_threshold=0.2):
        data_info = dict(img_id=0, img=images[0], texts=self.texts)  # single-image detection path
        data_info = self.test_pipeline(data_info)
        data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                        data_samples=[data_info['data_samples']])
        detections_inbatch = []
        with torch.no_grad():
            outputs = self.model.test_step(data_batch)
            # convert to searcher interface format
            
            for output in outputs:
                pred_instances = output.pred_instances
                pred_instances = pred_instances[pred_instances.scores.float() >
                                                score_threshold]
                if len(pred_instances.scores) > max_dets:
                    indices = pred_instances.scores.float().topk(max_dets)[1]
                    pred_instances = pred_instances[indices]

                output.pred_instances = pred_instances

                if 'masks' in pred_instances:
                    masks = pred_instances['masks']
                else:
                    masks = None
                pred_instances = pred_instances.cpu().numpy()
                detections = sv.Detections(xyxy=pred_instances['bboxes'],
                    class_id=pred_instances['labels'],
                    confidence=pred_instances['scores'],
                    mask=masks)
                detections_inbatch.append(detections)
        self.detections_inbatch = detections_inbatch
        return detections_inbatch

    def bbox_visualization(self, images, detections_inbatch):
        anno_images = []
        for b, detections in enumerate(detections_inbatch):
            texts = self.texts
            labels = [
                f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
                zip(detections.class_id, detections.confidence)
            ]

            index = len(detections_inbatch) -1 
            image = images[index]
            anno_image = image.copy()
  
            anno_image = self.BOUNDING_BOX_ANNOTATOR.annotate(anno_image, detections)
            anno_image = self.LABEL_ANNOTATOR.annotate(anno_image, detections, labels=labels)
            anno_images.append(anno_image)
        
        return anno_images
