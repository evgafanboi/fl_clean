from abc import ABC, abstractmethod
from typing import Any, Dict

from ..config import FDConfig


class DistillationStrategy(ABC):
    """Base class for distillation-based federated learning strategies."""
    name: str = "Base"
    is_distillation_strategy: bool = True

    def __init__(self, config: FDConfig) -> None:
        self.config = config

    def extra_log_tokens(self) -> Dict[str, Any]:
        """Optional extra tokens for log filenames."""
        return {}

    def setup(self, context: "PipelineContext") -> None:
        """Called once after the pipeline context is prepared."""

    @abstractmethod
    def run_round(self, context: "PipelineContext", round_number: int) -> Dict[int, Dict[str, float]]:
        """Execute one communication round and return per-client metrics."""

    def finalize(self, context: "PipelineContext") -> None:
        """Optional cleanup after all rounds finish."""
