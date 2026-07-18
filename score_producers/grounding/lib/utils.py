"""Utility helpers for the grounding stream: image/base64 encoding and video frame loading.

Derived from T* (https://github.com/LongVideoHaystack/TStar), MIT-licensed.
See THIRD_PARTY_LICENSES at the repository root.
"""
from typing import List
from PIL import Image
import base64
import io
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV is not installed, video frame extraction will not work.")

def encode_image_to_base64(image) -> str:
    """
    Convert an image (PIL.Image or numpy.ndarray) to a Base64 encoded string.
    """
    try:
        # If the input is a numpy array, convert it to a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure it's a PIL Image before proceeding
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL.Image or numpy.ndarray")

        # Encode the image to Base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Error encoding image: {str(e)}")

def load_video_frames(video_path: str, num_frames: int = 32) -> List[Image.Image]:
    """
    Load frames from a video or image file.
    """
    try:
        if video_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # If it's an image file, load and duplicate it directly
            image = Image.open(video_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return [image] * num_frames  # Return multiple identical copies of the image

        elif video_path.lower().endswith(('.mp4', '.avi', '.mov')):
            # If it's a video file, extract frames
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                raise ValueError(f"No frames found in video: {video_path}")

            step = max(1, total_frames // num_frames)
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frames.append(frame)
                if len(frames) >= num_frames:
                    break
            cap.release()
            return frames[:num_frames]
        else:
            raise ValueError(f"Unsupported file format: {video_path}")

    except Exception as e:
        print(f"Error loading frames from {video_path}: {str(e)}")
        raise

