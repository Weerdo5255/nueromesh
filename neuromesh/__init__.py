"""NeuroMesh: an always-on, self-improving multi-agent system built on modular neural nodes."""
__version__ = "0.1.0"

__all__ = ["System", "__version__"]


def __getattr__(name):
    if name == "System":
        from .runtime import System
        return System
    raise AttributeError(name)
