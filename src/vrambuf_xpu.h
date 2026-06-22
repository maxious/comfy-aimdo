#pragma once

#include "plat_xpu.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct VramBuffer {
    void *base_ptr;
    size_t allocated;
    size_t max_size;
    int device;
    size_t handle_count;
    struct VramBuffer *next;
    ze_physical_mem_handle_t handles[];
} VramBuffer;

#define VRAM_CHUNK_SIZE      (16ULL * 1024 * 1024)

SHARED_EXPORT
void *vrambuf_create(int device, size_t max_size);

SHARED_EXPORT
bool vrambuf_grow(void *arg, size_t required_size);

SHARED_EXPORT
void *vrambuf_get(void *arg);

SHARED_EXPORT
void vrambuf_destroy(void *arg);
