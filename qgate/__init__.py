"""Q-Gate: query-modulated multimodal keyframe selection for long video understanding.

This package provides the LLM/MLLM backend layer (``QGateGrounder``) used by the
Stage-2 fusion and Stage-3 QA scripts. See the repository README for the full pipeline.
"""

from .interface_llm import QGateGrounder

__version__ = "1.0.0"
__all__ = ["QGateGrounder", "__version__"]
