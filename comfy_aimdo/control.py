import os
import ctypes
import platform
from pathlib import Path
import logging
from enum import Enum

lib = None
_device_type = None


class DeviceType(Enum):
    CUDA = "cuda"
    XPU = "xpu"

    def __repr__(self):
        return f"<DeviceType.{self.name}>"


_BACKENDS = {
    DeviceType.CUDA: {
        "lib_suffix": "_cuda.so",
        "lib_prefix": "aimdo",
        "alloc_fn": "alloc_fn",
        "free_fn": "free_fn",
        "torch_module": "torch.cuda.memory",
        "allocator_class": "CUDAPluggableAllocator",
    },
    DeviceType.XPU: {
        "lib_suffix": "_xpu.so",
        "lib_prefix": "aimdo",
        "alloc_fn": "xpu_alloc_fn",
        "free_fn": "xpu_free_fn",
        "torch_module": "torch.xpu.memory",
        "allocator_class": "XPUPluggableAllocator",
    },
}


def _detect_device_type():
    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return DeviceType.XPU
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return DeviceType.CUDA
    except Exception:
        pass
    return None


def _get_backend_info(device_type):
    if device_type is None:
        return None
    return _BACKENDS.get(device_type)


def init():
    global lib, _device_type

    if lib is not None:
        return True

    _device_type = _detect_device_type()
    backend = _get_backend_info(_device_type)

    if backend is None:
        logging.info(f"comfy-aimdo: no supported device found")
        return False

    base_path = Path(__file__).parent.resolve()
    system = platform.system()

    lib_name = backend["lib_prefix"] + backend["lib_suffix"]

    try:
        if system == "Windows":
            lib = ctypes.CDLL(str(base_path / "aimdo.dll"))
        elif system == "Linux":
            lib = ctypes.CDLL(str(base_path / lib_name), mode=258)
        else:
            logging.info(f"comfy-aimdo os not supported {system}")
            logging.info(f"NOTE: comfy-aimdo is currently only support for Windows and Linux")
            return False
    except Exception as e:
        logging.info(f"comfy-aimdo failed to load {lib_name}: {e}")
        if _device_type:
            logging.info(f"NOTE: {backend['lib_prefix']} backend may not be built")
        return False

    lib.get_total_vram_usage.argtypes = []
    lib.get_total_vram_usage.restype = ctypes.c_uint64

    lib.init.argtypes = [ctypes.c_int]
    lib.init.restype = ctypes.c_bool

    return True

def init_device(device_id: int):
    if lib is None:
        return False

    return lib.init(device_id)

def deinit():
    global lib, _device_type
    if lib is not None:
        lib.cleanup()
    lib = None
    _device_type = None


def set_log_none(): lib.set_log_level_none()
def set_log_critical(): lib.set_log_level_critical()
def set_log_error(): lib.set_log_level_error()
def set_log_warning(): lib.set_log_level_warning()
def set_log_info(): lib.set_log_level_info()
def set_log_debug(): lib.set_log_level_debug()
def set_log_verbose(): lib.set_log_level_verbose()
def set_log_vverbose(): lib.set_log_level_vverbose()

def analyze():
    if lib is None:
        return
    lib.aimdo_analyze()

def get_total_vram_usage():
    return 0 if lib is None else lib.get_total_vram_usage()

def device_type():
    return _device_type

def get_backend_info():
    return _get_backend_info(_device_type)
