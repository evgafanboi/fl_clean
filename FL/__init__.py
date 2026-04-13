import os

__all__ = ["FLConfig", "run_pipeline"]


def __getattr__(name):
    if name in ("FLConfig", "run_pipeline"):
        from .pipeline import FLConfig, run_pipeline
        return locals()[name]
    raise AttributeError(name)
