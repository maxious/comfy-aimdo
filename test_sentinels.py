"""Sentinel-based correctness test for virtual memory management.

Verifies that VBAR pages preserve data across fault/unpin/eviction cycles
using per-layer sentinel values derived from the weight's VBAR offset.
"""
import gc
import math

import comfy_aimdo.control
comfy_aimdo.control.init()
comfy_aimdo.control.set_log_info()

import torch
from comfy_aimdo.model_vbar import ModelVBAR
from comfy_aimdo.torch import aimdo_to_tensor

dev_type = comfy_aimdo.control.device_type()
if dev_type == comfy_aimdo.control.DeviceType.XPU:
    device_str = f"xpu:{torch.xpu.current_device()}"
    gpu_size = torch.xpu.get_device_properties(torch.xpu.current_device()).total_memory
    def empty_cache():
        torch.xpu.empty_cache()
else:
    device_str = f"cuda:{torch.cuda.current_device()}"
    gpu_size = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    def empty_cache():
        torch.cuda.empty_cache()

comfy_aimdo.control.init_device(int(device_str.split(":")[1]))


def sentinel(offset):
    """Derive a sentinel from the offset — cheap, no CPU weight storage needed."""
    return offset ^ 0xDEADBEEF


def test_layer_fault_preserves_data():
    dtype = torch.float16
    M = 1024 ** 2
    vbar = ModelVBAR(gpu_size * 2, device=0)
    n_elements = M // dtype.itemsize

    alloc = vbar.alloc(M)
    raw = aimdo_to_tensor(alloc, torch.device(device_str))
    sig1 = vbar.fault(vbar.base_addr, M)
    assert sig1 is not None, "first fault should succeed"

    val = sentinel(vbar.base_addr)
    raw[0] = val & 0xFF
    raw[1] = (val >> 8) & 0xFF
    raw[2] = (val >> 16) & 0xFF
    raw[3] = (val >> 24) & 0xFF
    vbar.unpin(vbar.base_addr, M)

    sig2 = vbar.fault(vbar.base_addr, M)
    assert sig2 is not None, "re-fault should succeed"

    raw2 = aimdo_to_tensor(alloc, torch.device(device_str))
    readback = raw2[0].item() | (raw2[1].item() << 8) | (raw2[2].item() << 16) | (raw2[3].item() << 24)
    assert readback == val, f"sentinel mismatch: {readback:#x} != {val:#x}"

    assert vbar.loaded_size() > 0, "loaded size should be > 0"
    print(f"  PASS: sentinel preserved across re-fault ({vbar.loaded_size() / M:.0f} MB resident)")


def test_eviction_preserves_data_via_cpu():
    M = 1024 ** 2
    num_layers = 30
    layer_size = M * 4
    vbar = ModelVBAR(gpu_size * 5, device=0)

    offsets = []
    allocs = []
    for i in range(num_layers):
        a = vbar.alloc(layer_size)
        allocs.append(a)
        offsets.append(a[1])

    cpu = torch.zeros(4, dtype=torch.uint8)
    passed = 0
    failed = 0
    offloaded = 0

    for i, (a, off) in enumerate(zip(allocs, offsets)):
        val = sentinel(off)
        s = vbar.fault(off, layer_size)
        raw = aimdo_to_tensor(a, torch.device(device_str))

        if s is not None:
            cpu[0] = val & 0xFF
            cpu[1] = (val >> 8) & 0xFF
            cpu[2] = (val >> 16) & 0xFF
            cpu[3] = (val >> 24) & 0xFF
            raw[:4].copy_(cpu)
            vbar.unpin(off, layer_size)

            s2 = vbar.fault(off, layer_size)
            if s2 is None:
                offloaded += 1
                continue
            raw2 = aimdo_to_tensor(a, torch.device(device_str))
            readback = (raw2[0].item() | (raw2[1].item() << 8) |
                        raw2[2].item() << 16 | raw2[3].item() << 24)
            if readback == val:
                passed += 1
            else:
                failed += 1
            vbar.unpin(off, layer_size)
        else:
            offloaded += 1

    print(f"  PASS: {passed} eviction-proof, {offloaded} offloaded, {failed} failures")
    assert failed == 0, f"{failed} layers had sentinel corruption"
    assert offloaded > 0, "no layers were offloaded — test didn't exercise eviction"
    assert passed > 0, "no layers remained resident"


def test_watermark_is_monotonic():
    """Once watermark triggers, all higher-offset weights stay offloaded."""
    dtype = torch.float16
    M = 1024 ** 2
    n_layers = 40
    layer_sz = M * 4
    vbar = ModelVBAR(gpu_size * 5, device=0)

    allocs = []
    offsets = []
    for _ in range(n_layers):
        a = vbar.alloc(layer_sz)
        allocs.append(a)
        offsets.append(a[1])

    saw_offload = False
    after_offload_seen_load = False

    for i, (a, off) in enumerate(zip(allocs, offsets)):
        s = vbar.fault(off, layer_sz)
        if s is None:
            saw_offload = True
        else:
            vbar.unpin(off, layer_sz)
            if saw_offload:
                after_offload_seen_load = True
                break

    assert saw_offload, "watermark never triggered offloading"
    assert not after_offload_seen_load, (
        "watermark not monotonic: saw load after offload without reset"
    )

    vbar.deprioritize()

    for i, (a, off) in enumerate(zip(allocs, offsets)):
        s = vbar.fault(off, layer_sz)
        if s is not None:
            vbar.unpin(off, layer_sz)

    vbar.prioritize()
    reclaimed = 0
    for i, (a, off) in enumerate(zip(allocs, offsets)):
        s = vbar.fault(off, layer_sz)
        if s is not None:
            reclaimed += 1
            vbar.unpin(off, layer_sz)

    assert reclaimed > 0, "prioritize() did not reclaim any weights"
    print(f"  PASS: watermark monotonic, {reclaimed} reclaimed after prioritize()")


print("=== Test 1: Fault preserves data ===")
test_layer_fault_preserves_data()

print("\n=== Test 2: Eviction preserves data via CPU ===")
test_eviction_preserves_data_via_cpu()

print("\n=== Test 3: Watermark monotonicity ===")
test_watermark_is_monotonic()

print("\n=== ALL TESTS PASSED ===")
gc.collect()
empty_cache()
