from abc import ABC, abstractmethod
from typing import Any, Dict

from ..config import FDConfig


class DistillationAlgorithm(ABC):
    name: str = "Base"

    def __init__(self, config: FDConfig) -> None:
        self.config = config

    def run(self) -> None:
        from ..pipeline import DistillationPipeline

        pipeline = DistillationPipeline(self.config, self)
        pipeline.run()

    def extra_log_tokens(self) -> Dict[str, Any]:
        """Optional extra tokens for log filenames."""
        return {}

    # Lifecycle hooks -------------------------------------------------
    def setup(self, context: "PipelineContext") -> None:  # type: ignore[name-defined]
        """Called once after the pipeline context is prepared."""

    @abstractmethod
    def run_round(self, context: "PipelineContext", round_number: int) -> Dict[int, Dict[str, float]]:  # type: ignore[name-defined]
        """Execute one communication round and return per-client metrics."""

    def finalize(self, context: "PipelineContext") -> None:  # type: ignore[name-defined]
        """Optional cleanup after all rounds finish."""
