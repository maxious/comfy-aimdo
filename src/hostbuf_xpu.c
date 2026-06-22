#include "plat_xpu.h"

SHARED_EXPORT
void *hostbuf_allocate(uint64_t size) {
    void *ptr = NULL;
    size_t alloc_size = (size_t)size;

    if (!size) {
        return NULL;
    }

    ze_host_mem_alloc_desc_t host_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
    };

    if (!CHECK_ZE(zeMemAllocHost(aimdo_xpu_ctx, &host_desc, alloc_size, 1, &ptr))) {
        log(DEBUG, "%s: XPU host allocation failed (%zuk)\n", __func__, alloc_size / K);
        return NULL;
    }

    return ptr;
}

SHARED_EXPORT
void hostbuf_free(void *arg) {
    if (!arg) {
        return;
    }
    CHECK_ZE(zeMemFree(aimdo_xpu_ctx, arg));
}
