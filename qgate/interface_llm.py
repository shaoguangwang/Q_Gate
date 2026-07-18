"""LLM/MLLM backend layer for Q-Gate.

Two thin client interfaces — ``GPT4Interface`` (OpenAI-compatible chat-completions
endpoint) and ``LocalVLMInterface`` (a vLLM-style OpenAI-compatible server) — plus
``QGateGrounder``, the single entry point the pipeline talks to:

- Stage 2 (``fuse_scores.py``) calls ``grounder.VLM_model_interface.inference_text_only``
  to obtain the query-modulated gating weights.
- Stage 3 (``run_qa.py``) calls ``grounder.inference_qa`` for the downstream answer.

All backends share one QA prompt; the only per-backend difference is which endpoint
the frames/text are sent to.
"""
import os
from typing import List
import datetime
import json

import openai
import requests
from PIL import Image
from .utils import encode_image_to_base64
import warnings
warnings.filterwarnings("ignore")


class LocalVLMInterface:
    """Client for a local, OpenAI-compatible VLM server (e.g. vLLM)."""

    def __init__(self, model_name: str = "default", base_url: str = "http://localhost:8000/v1"):
        self.model_name = model_name
        self.client = openai.Client(api_key="None", base_url=base_url, timeout=90.0, max_retries=1)

    def inference_text_only(self, query: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Text-only inference (used by the Stage-2 gater)."""
        try:
            stream_request = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": [{"type": "text", "text": query}]}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            stream_response = ""
            for chunk in stream_request:
                if chunk.choices[0].delta.content is not None:
                    stream_response += chunk.choices[0].delta.content
            return stream_response
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_with_frames_all_in_one(self, query: str, frames: List[Image.Image], temperature: float = 0.2, max_tokens: int = 1000) -> str:
        """Multi-image + text inference (used by the Stage-3 QA call)."""
        try:
            inputs = [{"type": "text", "text": query}]
            for frame in frames:
                frame_base64 = encode_image_to_base64(frame)
                inputs.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                    "modalities": "multi-images",
                })
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": inputs}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in inference_with_frames_all_in_one: {str(e)}")
            return f"Error: {str(e)}"


class GPT4Interface:
    """Client for the OpenAI (or OpenAI-compatible) chat-completions endpoint."""

    def __init__(self, model="gpt-4o", api_key=None):
        # Full chat-completions endpoint; override via OPENAI_BASE_URL.
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY is not set.")
        self.model_name = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def inference_text_only(self, query: str, system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Text-only inference (used by the Stage-2 gater)."""
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data)).json()
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def inference_with_frames_all_in_one(self, query: str, frames: List[Image.Image], system_message: str = "You are a helpful assistant.", temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Multi-image + text inference (used by the Stage-3 QA call)."""
        inputs = [{"type": "text", "text": query}]
        for i, frame in enumerate(frames):
            try:
                frame_base64 = encode_image_to_base64(frame)
                inputs.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}", "detail": "low"},
                })
            except Exception as e:
                return f"Error encoding frame {i}: {str(e)}"
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": inputs},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(self.url, headers=self.headers, data=json.dumps(data)).json()
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {str(e)}"


class QGateGrounder:
    """Unified backend handle used by the fusion (Stage 2) and QA (Stage 3) stages.

    ``backend`` selects the downstream model; every backend shares the same QA prompt.
    - ``gpt4`` → OpenAI-compatible endpoint (GPT-4o by default).
    - ``qwen3_32b`` / ``internvl3_8b`` / ``internvl_26b`` → local OpenAI-compatible
      servers (e.g. vLLM); each reads its base URL from a dedicated env var so no
      host/port is hardcoded, and the served-model-name must match the serving command.
    """

    # backend name -> (served model name, env var for the local base URL, default base URL)
    _LOCAL_BACKENDS = {
        "qwen3_32b":    ("qwen3-32b",    "QWEN3_32B_BASE_URL",  "http://localhost:8001/v1"),
        "internvl_26b": ("internvl-26b", "INTERNVL26B_BASE_URL", "http://localhost:8002/v1"),
        "internvl3_8b": ("internvl3-8b", "INTERNVL3_BASE_URL",  "http://localhost:8003/v1"),
    }

    def __init__(self, backend: str = "gpt4", gpt4_model_name: str = "gpt-4o"):
        self.backend = backend.lower()

        if self.backend == "gpt4":
            self.VLM_model_interface = GPT4Interface(model=gpt4_model_name)
        elif self.backend in self._LOCAL_BACKENDS:
            served_name, env_var, default_url = self._LOCAL_BACKENDS[self.backend]
            self.VLM_model_interface = LocalVLMInterface(
                model_name=served_name,
                base_url=os.environ.get(env_var, default_url),
            )
        else:
            raise ValueError(
                "backend must be one of: 'gpt4', 'qwen3_32b', 'internvl3_8b', 'internvl_26b'."
            )

    def inference_qa(
        self,
        frames: List[Image.Image],
        frame_timestamps: List[float],
        question: str,
        options: str,
        temperature: float = 0.2,
        subtitle_info: str = "",
    ) -> str:
        """Answer a multiple-choice question from timestamped keyframes (and optional subtitles).

        The same prompt is used for every backend: each selected frame is prefixed with its
        ``[Image at H:MM:SS]`` timestamp so the model can align visual evidence with the
        timestamped subtitles.
        """
        def format_time(seconds: float) -> str:
            """Format seconds as an H:MM:SS timestamp string (e.g. 65 -> '0:01:05')."""
            return str(datetime.timedelta(seconds=int(seconds)))

        # Build the timestamped image placeholders: "[Image at H:MM:SS]\n<image>\n".
        if frames and frame_timestamps and len(frames) == len(frame_timestamps):
            image_placeholders = "".join(
                f"[Image at {format_time(ts)}]\n<image>\n" for ts, frame in zip(frame_timestamps, frames)
            )
        else:  # Fallback for safety
            image_placeholders = "\n".join(["<image>"] * len(frames))

        system_prompt = (
            "Select the best answer to the following multiple-choice question based on the video frames and subtitles.\n\n"
            + image_placeholders.strip()
            + subtitle_info
            + f"\nQuestion: {question}\n"
            + f"Options: {options}\n"
            + "Answer with the option's letter from the given choices directly.\n"
            + "Your response format should be strictly an upper case letter A,B,C,D or E.\n"
        )

        response = self.VLM_model_interface.inference_with_frames_all_in_one(
            query=system_prompt,
            frames=frames,
            temperature=temperature,
            max_tokens=30,
        )
        return (response or "").strip()


if __name__ == "__main__":
    # Minimal smoke test for the GPT-4o text-only backend.
    # Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) in the environment.
    grounder = QGateGrounder(backend="gpt4")
    answer = grounder.VLM_model_interface.inference_text_only(
        "What is the color of the sky?", max_tokens=10
    )
    print("GPT-4o answer:", answer)
