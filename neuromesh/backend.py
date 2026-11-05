"""Compute backends for neural nodes.

Node math (forward pass, backprop, Adam) is written once in nodes.py against the
small set of operations below, so the same code runs on:

  numpy  - CPU, the default; no extra dependencies
  torch  - PyTorch on "cuda" (NVIDIA, or AMD via ROCm builds), "mps" (Apple Silicon) or "cpu"
  cupy   - CuPy on NVIDIA CUDA devices

Weights are always saved as numpy arrays, so a node trained on a GPU loads on a CPU
machine and vice versa. Text encoding stays on the CPU; batches are moved to the device
once per training slice.
"""
from __future__ import annotations

import contextlib
import logging

import numpy as np

log = logging.getLogger("neuromesh.backend")

GPU_HELP = ("Install a GPU build of PyTorch (https://pytorch.org/get-started/locally/; CUDA for NVIDIA, "
            "ROCm for AMD, the default build on Apple Silicon) or CuPy for NVIDIA (pip install "
            "cupy-cuda12x, matching your CUDA version). Or use device \"auto\" to fall back to the CPU.")


class BackendUnavailable(RuntimeError):
    """The requested device or library is not usable on this machine."""


class Backend:
    name = "numpy"
    device = "cpu"
    is_gpu = False

    # -- context / housekeeping
    def scope(self):
        return contextlib.nullcontext()

    def synchronize(self) -> None:
        pass

    def describe(self) -> str:
        return f"{self.name} on {self.device}"

    # -- transfer
    def asarray(self, a, integer: bool = False):
        raise NotImplementedError

    def to_numpy(self, a) -> np.ndarray:
        raise NotImplementedError

    # -- element-wise and reductions
    def copy(self, a):
        raise NotImplementedError

    def zeros_like(self, a):
        raise NotImplementedError

    def relu(self, a):
        raise NotImplementedError

    def gt0(self, a):
        """1.0 where a > 0 else 0.0, same dtype as a (the ReLU derivative)."""
        raise NotImplementedError

    def exp(self, a):
        raise NotImplementedError

    def log(self, a):
        raise NotImplementedError

    def sqrt(self, a):
        raise NotImplementedError

    def max_rows(self, a):
        """Row-wise max, keeping the reduced dimension."""
        raise NotImplementedError

    def sum_rows(self, a):
        """Row-wise sum, keeping the reduced dimension."""
        raise NotImplementedError

    def sum0(self, a):
        """Column-wise sum (over the batch)."""
        raise NotImplementedError

    def arange(self, n: int):
        raise NotImplementedError

    def argmax_rows(self, a) -> np.ndarray:
        """Row-wise argmax, returned as a numpy array."""
        raise NotImplementedError


class XPBackend(Backend):
    """Backends with a numpy-compatible array module (numpy itself, CuPy)."""

    def __init__(self, xp):
        self.xp = xp

    def asarray(self, a, integer=False):
        return self.xp.asarray(a, dtype=self.xp.int64 if integer else self.xp.float32)

    def to_numpy(self, a):
        return np.asarray(a)

    def copy(self, a):
        return a.copy()

    def zeros_like(self, a):
        return self.xp.zeros_like(a)

    def relu(self, a):
        return self.xp.maximum(a, 0)

    def gt0(self, a):
        return (a > 0).astype(a.dtype)

    def exp(self, a):
        return self.xp.exp(a)

    def log(self, a):
        return self.xp.log(a)

    def sqrt(self, a):
        return self.xp.sqrt(a)

    def max_rows(self, a):
        return a.max(axis=1, keepdims=True)

    def sum_rows(self, a):
        return a.sum(axis=1, keepdims=True)

    def sum0(self, a):
        return a.sum(axis=0)

    def arange(self, n):
        return self.xp.arange(n)

    def argmax_rows(self, a):
        return self.to_numpy(a.argmax(axis=1))


class NumpyBackend(XPBackend):
    name = "numpy"
    device = "cpu"

    def __init__(self):
        super().__init__(np)

    def describe(self) -> str:
        return f"numpy {np.__version__} on cpu"


class CupyBackend(XPBackend):
    name = "cupy"
    is_gpu = True

    def __init__(self, index: int = 0):
        import cupy  # noqa: PLC0415 - optional dependency
        count = cupy.cuda.runtime.getDeviceCount()
        if index >= count:
            raise BackendUnavailable(f"CuPy sees {count} CUDA device(s); cuda:{index} does not exist")
        super().__init__(cupy)
        self.cupy = cupy
        self.index = index
        self.device = f"cuda:{index}"

    def scope(self):
        # CuPy's current device is per thread, and nodes train in worker threads
        return self.cupy.cuda.Device(self.index)

    def to_numpy(self, a):
        return self.cupy.asnumpy(a)

    def synchronize(self):
        self.cupy.cuda.Device(self.index).synchronize()

    def describe(self) -> str:
        name = self.cupy.cuda.runtime.getDeviceProperties(self.index)["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        return f"cupy {self.cupy.__version__} on {self.device} ({name})"


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, device: str = "cuda"):
        import torch  # noqa: PLC0415 - optional dependency
        self.torch = torch
        self.device = device
        self.is_gpu = not device.startswith("cpu")

    def asarray(self, a, integer=False):
        t = self.torch
        dtype = t.int64 if integer else t.float32
        if isinstance(a, t.Tensor):
            return a.to(device=self.device, dtype=dtype)
        arr = np.ascontiguousarray(a, dtype=np.int64 if integer else np.float32)
        return t.from_numpy(arr).to(device=self.device, dtype=dtype)

    def to_numpy(self, a):
        return a.detach().cpu().numpy()

    def copy(self, a):
        return a.clone()

    def zeros_like(self, a):
        return self.torch.zeros_like(a)

    def relu(self, a):
        return self.torch.relu(a)

    def gt0(self, a):
        return (a > 0).to(a.dtype)

    def exp(self, a):
        return self.torch.exp(a)

    def log(self, a):
        return self.torch.log(a)

    def sqrt(self, a):
        return self.torch.sqrt(a)

    def max_rows(self, a):
        return a.amax(dim=1, keepdim=True)

    def sum_rows(self, a):
        return a.sum(dim=1, keepdim=True)

    def sum0(self, a):
        return a.sum(dim=0)

    def arange(self, n):
        return self.torch.arange(n, device=self.device)

    def argmax_rows(self, a):
        return self.to_numpy(a.argmax(dim=1))

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def describe(self) -> str:
        detail = ""
        if self.device.startswith("cuda"):
            index = int(self.device.split(":")[1]) if ":" in self.device else 0
            detail = f" ({self.torch.cuda.get_device_name(index)})"
        elif self.device == "mps":
            detail = " (Apple GPU)"
        return f"torch {self.torch.__version__} on {self.device}{detail}"


# ---------------------------------------------------------------------- selection
def _torch():
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError:
        return None


def _cupy_device_count() -> int:
    try:
        import cupy  # noqa: PLC0415
        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception:  # not installed, no driver, no device
        return 0


def _cuda_index(device: str) -> int:
    if device == "cuda":
        return 0
    try:
        return int(device.split(":", 1)[1])
    except (IndexError, ValueError):
        raise BackendUnavailable(f"cannot parse device '{device}'; use cuda, cuda:0, cuda:1, ...") from None


def _torch_cuda(torch, device: str) -> "TorchBackend | None":
    if not torch.cuda.is_available():
        return None
    index = _cuda_index(device)
    if index >= torch.cuda.device_count():
        raise BackendUnavailable(f"PyTorch sees {torch.cuda.device_count()} CUDA device(s); "
                                 f"cuda:{index} does not exist")
    return TorchBackend(f"cuda:{index}")


def _torch_mps(torch) -> "TorchBackend | None":
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return TorchBackend("mps")
    return None


def select_backend(device: str = "cpu", library: str = "auto") -> Backend:
    """Pick the compute backend.

    device:  "cpu" | "auto" | "cuda" | "cuda:N" | "mps"
    library: "auto" | "numpy" | "torch" | "cupy"

    "auto" uses the best GPU found (PyTorch CUDA/ROCm, then Apple MPS, then CuPy) and falls
    back to numpy on the CPU. An explicitly requested device that is unavailable raises
    BackendUnavailable instead of silently running somewhere else.
    """
    device = (device or "cpu").strip().lower()
    library = (library or "auto").strip().lower()
    if library not in ("auto", "numpy", "torch", "cupy"):
        raise BackendUnavailable(f"unknown GPU library '{library}' (use auto, numpy, torch or cupy)")

    if device in ("cpu", "numpy"):
        if library == "torch":
            if _torch() is None:
                raise BackendUnavailable("library 'torch' requested but PyTorch is not installed")
            return TorchBackend("cpu")
        if library == "cupy":
            raise BackendUnavailable("CuPy only runs on CUDA devices; use device 'cuda' with library 'cupy'")
        return NumpyBackend()

    if library == "numpy":
        raise BackendUnavailable(f"numpy runs on the CPU only; device '{device}' needs library torch or cupy")

    torch = _torch() if library in ("auto", "torch") else None
    if device == "auto":
        if torch is not None:
            backend = _torch_cuda(torch, "cuda") or _torch_mps(torch)
            if backend is not None:
                return backend
        if library in ("auto", "cupy") and _cupy_device_count() > 0:
            return CupyBackend(0)
        log.info("device 'auto': no usable GPU found, using numpy on the CPU")
        return NumpyBackend()

    if device.startswith("cuda"):
        if torch is not None:
            backend = _torch_cuda(torch, device)
            if backend is not None:
                return backend
        if library in ("auto", "cupy") and _cupy_device_count() > 0:
            return CupyBackend(_cuda_index(device))
        found = "PyTorch without CUDA" if torch is not None else "no PyTorch or CuPy with CUDA"
        raise BackendUnavailable(f"device '{device}' requested but no CUDA GPU is usable ({found}). {GPU_HELP}")

    if device == "mps":
        if torch is None:
            raise BackendUnavailable(f"device 'mps' needs PyTorch. {GPU_HELP}")
        backend = _torch_mps(torch)
        if backend is None:
            raise BackendUnavailable("device 'mps' requested but PyTorch reports no Apple Metal GPU")
        return backend

    raise BackendUnavailable(f"unknown device '{device}' (use cpu, auto, cuda, cuda:N or mps)")
