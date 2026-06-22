#include "vrambuf_xpu.h"

#define VRAM_CHUNK_SIZE      (16ULL * 1024 * 1024)

SHARED_EXPORT
void *vrambuf_create(int device, size_t max_size) {
    VramBuffer *buf;

    max_size = XPU_ALIGN_UP(max_size);

    buf = (VramBuffer *)calloc(1, sizeof(*buf) + sizeof(ze_physical_mem_handle_t) * max_size / VRAM_CHUNK_SIZE);
    if (!buf) {
        return NULL;
    }
    buf->device = device;
    buf->max_size = max_size;

    if (!CHECK_ZE(zeVirtualMemReserve(aimdo_xpu_ctx, NULL, max_size, &buf->base_ptr))) {
        log(ERROR, "%s: %d %zuk\n", __func__, device, max_size / K);
        free(buf);
        return NULL;
    }

    return (void *)buf;
}

SHARED_EXPORT
bool vrambuf_grow(void *arg, size_t required_size) {
    VramBuffer *buf = (VramBuffer *)arg;
    size_t grow_to;
    ze_physical_mem_handle_t handle;
    ze_result_t err;

    if (!buf) {
        return false;
    }
    if (required_size > buf->max_size) {
        return false;
    }
    if (required_size <= buf->allocated) {
        return true;
    }

    grow_to = ALIGN_UP(required_size, VRAM_CHUNK_SIZE);
    if (grow_to > buf->max_size) {
        grow_to = buf->max_size;
    }

    vbars_free(budget_deficit(grow_to - buf->allocated));
    while (buf->allocated < grow_to) {
        size_t to_allocate = grow_to - buf->allocated;
        if (to_allocate > VRAM_CHUNK_SIZE) {
            to_allocate = VRAM_CHUNK_SIZE;
        }
        err = three_stooges(aimdo_xpu_dev, aimdo_xpu_ctx,
                          (char *)buf->base_ptr + buf->allocated,
                          to_allocate, buf->device, &handle);
        if (err != ZE_RESULT_SUCCESS) {
            if (err != ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
                log(ERROR, "VRAM Allocation failed (non OOM)\n");
                return false;
            }
            log(DEBUG, "Pytorch allocator attempt exceeds available VRAM ...\n");
            vbars_free(VRAM_CHUNK_SIZE);
            err = three_stooges(aimdo_xpu_dev, aimdo_xpu_ctx,
                              (char *)buf->base_ptr + buf->allocated,
                              to_allocate, buf->device, &handle);
            if (err != ZE_RESULT_SUCCESS) {
                bool is_oom = err == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY;
                log(is_oom ? INFO : ERROR, "VRAM Allocation failed (%s)\n", is_oom ? "OOM" : "error");
                return false;
            }
        }

        buf->handles[buf->handle_count++] = handle;
        buf->allocated += to_allocate;
    }

    return true;
}

SHARED_EXPORT
void *vrambuf_get(void *arg) {
    VramBuffer *buf = (VramBuffer *)arg;

    if (!buf) {
        return NULL;
    }
    return buf->base_ptr;
}

SHARED_EXPORT
void vrambuf_destroy(void *arg) {
    VramBuffer *buf = (VramBuffer *)arg;
    size_t i;

    if (!buf) {
        return;
    }

    if (buf->allocated > 0) {
        CHECK_ZE(zeVirtualMemUnmap(aimdo_xpu_ctx, buf->base_ptr, buf->allocated));
    }

    for (i = 0; i < buf->handle_count; i++) {
        CHECK_ZE(zePhysicalMemDestroy(aimdo_xpu_ctx, buf->handles[i]));
    }

    CHECK_ZE(zeVirtualMemFree(aimdo_xpu_ctx, buf->base_ptr, buf->max_size));
    total_vram_usage -= buf->allocated;
    free(buf);
}
