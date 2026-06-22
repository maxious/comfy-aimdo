import torch
import ctypes

import logging
from pathlib import Path
from importlib import import_module

from . import control

def get_tensor_from_raw_ptr(ptr, size, device):
    container = {
        "shape": (size,),
        "typestr": "|u1",
        "data": (ptr, False), #writable
        "version": 3,
    }

    class Holder:
        pass

    holder = Holder()
    holder.__cuda_array_interface__ = container

    return torch.as_tensor(holder, device=device)

def aimdo_to_tensor(alloc, device):
    _, ptr, size = alloc
    dt = control.device_type()
    if dt == control.DeviceType.CUDA:
        return get_tensor_from_raw_ptr(ptr, size, device)
    elif dt == control.DeviceType.XPU:
        numel = size
        tensor = torch.empty(numel, dtype=torch.uint8, device=device)
        return tensor
    else:
        raise NotImplementedError(f"aimdo_to_tensor not implemented for {dt}")

def hostbuf_to_tensor(hostbuf):
    byte_view = (ctypes.c_uint8 * hostbuf.size).from_address(hostbuf.get_raw_address())
    t = torch.frombuffer(byte_view, dtype=torch.uint8)
    return t

#pytorch doesnt have an API for a CUDAPluggableAllocator from an already loaded
#library. Rather than force a second load that pytorch owns, construct these
#pytorch internals outselves as sperate CDLL loads is far too risky.

def _get_allocator_for_device(device_type):
    backend = control.get_backend_info()
    if backend is None:
        return None

    alloc_fn_name = backend["alloc_fn"]
    free_fn_name = backend["free_fn"]
    torch_module_name = backend["torch_module"]
    allocator_class_name = backend["allocator_class"]

    torch_memory = import_module(torch_module_name)
    allocator_class = getattr(torch_memory, allocator_class_name)

    if device_type == control.DeviceType.CUDA:
        if control.lib is None:
            return None
        alloc_fn = ctypes.cast(
            getattr(control.lib, alloc_fn_name), ctypes.c_void_p
        ).value
        free_fn = ctypes.cast(getattr(control.lib, free_fn_name), ctypes.c_void_p).value
        if alloc_fn is None or free_fn is None:
            return None
        return allocator_class(alloc_fn, free_fn)

    elif device_type == control.DeviceType.XPU:
        base_path = Path(__file__).parent.resolve()
        lib_path = str(base_path / (backend["lib_prefix"] + backend["lib_suffix"]))
        return allocator_class(lib_path, alloc_fn_name, free_fn_name)

    return None


class PluggableAllocator:
    def __init__(self):
        self._allocator = _get_allocator_for_device(control.device_type())


def get_torch_allocator():
    #As of this writing (pytorch 2.10), pytorch MemPools + CUDAPluggableAllocator
    #considers the Mempool and pool usage context each as a hard reference to the
    #tensors completely preventing reasonable garbage collection. A read of the code
    #suggests that the assumptions of cudaGraphs completely prohibits pool cleanup
    #on VRAM pressure which ultimately makes this un-usable for our high pressure
    #allocator.
    logging.warning(f"WARNING: Aimdo+PluggableAllocator is experimental and unsupported.")
    return _get_allocator_for_device(control.device_type())
