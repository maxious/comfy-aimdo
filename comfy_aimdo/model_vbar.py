import ctypes

from . import control

_bound = False


def _bind_lib():
    global _bound
    if _bound or control.lib is None:
        return
    control.lib.vbar_allocate.argtypes = [ctypes.c_uint64, ctypes.c_int]
    control.lib.vbar_allocate.restype = ctypes.c_void_p
    control.lib.vbar_set_watermark_limit.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    control.lib.vbars_reset_watermark_limits.argtypes = []
    control.lib.vbar_prioritize.argtypes = [ctypes.c_void_p]
    control.lib.vbar_deprioritize.argtypes = [ctypes.c_void_p]
    control.lib.vbar_get.argtypes = [ctypes.c_void_p]
    control.lib.vbar_get.restype = ctypes.c_uint64
    control.lib.vbar_free.argtypes = [ctypes.c_void_p]
    control.lib.vbar_fault.argtypes = [
        ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    control.lib.vbar_fault.restype = ctypes.c_int
    control.lib.vbar_unpin.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    control.lib.vbar_loaded_size.argtypes = [ctypes.c_void_p]
    control.lib.vbar_loaded_size.restype = ctypes.c_size_t
    control.lib.vbar_free_memory.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    control.lib.vbar_free_memory.restype = ctypes.c_uint64
    control.lib.vbars_analyze.argtypes = [ctypes.c_bool]
    control.lib.vbars_analyze.restype = ctypes.c_uint64
    control.lib.vbar_get_nr_pages.argtypes = [ctypes.c_void_p]
    control.lib.vbar_get_nr_pages.restype = ctypes.c_size_t
    control.lib.vbar_get_watermark.argtypes = [ctypes.c_void_p]
    control.lib.vbar_get_watermark.restype = ctypes.c_size_t
    control.lib.vbar_get_residency.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
    ]
    _bound = True


class ModelVBAR:
    def __init__(self, size, device):
        _bind_lib()
        self._ptr = control.lib.vbar_allocate(int(size), device)
        if not self._ptr:
            raise MemoryError("VBAR allocation failed")
        self.device = device
        self.max_size = size
        self.offset = 0
        self.base_addr = control.lib.vbar_get(self._ptr)

    def prioritize(self):
        control.lib.vbar_prioritize(self._ptr)

    def deprioritize(self):
        control.lib.vbar_deprioritize(self._ptr)

    def alloc(self, num_bytes):
        self.offset = (self.offset + 511) & ~511

        if self.offset + num_bytes > self.max_size:
            raise MemoryError("VBAR OOM")

        alloc = self.base_addr + self.offset
        self.offset += num_bytes
        return (self, alloc, num_bytes)

    #define VBAR_PAGE_SIZE (32 << 20)

    #define VBAR_FAULT_SUCCESS      0
    #define VBAR_FAULT_OOM          1
    #define VBAR_FAULT_ERROR        2

    def fault(self, alloc, size):
        offset = alloc - self.base_addr
        # +2, one for misalignment and one for rounding
        signature = (ctypes.c_uint32 * (size // (32 * 1024 ** 2) + 2))()
        res = control.lib.vbar_fault(self._ptr, offset, size, signature)
        if res == 0:
            return signature
        elif res == 1:
            return None
        else:
            raise RuntimeError(f"Fault failed: {res}")

    def unpin(self, alloc, size):
        offset = alloc - self.base_addr
        control.lib.vbar_unpin(self._ptr, offset, size)

    def loaded_size(self):
        return control.lib.vbar_loaded_size(self._ptr)

    def set_watermark_limit(self, size_bytes):
        control.lib.vbar_set_watermark_limit(self._ptr, size_bytes)

    def free_memory(self, size_bytes):
        return control.lib.vbar_free_memory(self._ptr, int(size_bytes))

    def get_nr_pages(self):
        return control.lib.vbar_get_nr_pages(self._ptr)

    def get_watermark(self):
        return control.lib.vbar_get_watermark(self._ptr)

    def get_residency(self):
        """Returns a list of per-page status flags.
        Bit 0 (& 1): resident in VRAM
        Bit 1 (& 2): pinned
        """
        nr_pages = self.get_nr_pages()
        buf = (ctypes.c_uint8 * nr_pages)()
        control.lib.vbar_get_residency(self._ptr, buf, nr_pages)
        return list(buf)

    def __del__(self):
        ptr = getattr(self, "_ptr", None)
        if ptr is not None:
            try:
                control.lib.vbar_free(ptr)
            except (AttributeError, TypeError):
                pass
            self._ptr = None

def vbar_fault(alloc):
    vbar, offset, size = alloc
    return vbar.fault(offset, size)

def vbar_unpin(alloc):
    if alloc is not None:
        vbar, offset, size = alloc
        vbar.unpin(offset, size)

def vbar_signature_compare(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        raise ValueError(f"Signatures of mismatched length {len(a)} != {len(b)}")
    return memoryview(a) == memoryview(b)

def vbars_reset_watermark_limits():
    control.lib.vbars_reset_watermark_limits()

def vbars_analyze():
    if control.lib is None:
        return 0
    return control.lib.vbars_analyze(False)
