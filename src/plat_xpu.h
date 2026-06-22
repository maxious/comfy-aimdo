#pragma once

#include <level_zero/ze_api.h>

#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>

/* control.c */
bool xpu_budget_deficit();

#define SHARED_EXPORT

#if defined(_WIN32) || defined(_WIN64)

#define SHARED_EXPORT __declspec(dllexport)

#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

#else

#define SHARED_EXPORT

#endif

static inline bool xpu_init() {
    return true;
}

static inline void xpu_cleanup() {
}

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/* NOTE: align_to must be power of 2 */
#define ALIGN_UP(x, align_to) (((x) + (align_to) - 1) & ~((align_to) - 1))

#define XPU_PAGE_SIZE   (2 << 20)
#define XPU_ALIGN_UP(s) ALIGN_UP(s, XPU_PAGE_SIZE)

typedef unsigned long long ull;
#define K 1024
#define M (K * K)
#define G (M * K)

enum DebugLevels {
    __NONE__ = -1,
    /* Default to everything so if python integration is hosed, we see prints. */
    ALL = 0,
    CRITICAL,
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    VERBOSE,
    VVERBOSE,
};

/* debug.c */
extern int log_level;
extern uint64_t log_shot_counter;
const char *get_level_str(int level);
void log_reset_shots();

#define do_log(do_shot_counter, level, ...) {                                                   \
    static uint64_t _sc_;                                                                       \
    if ((!log_level || log_level >= (level)) && _sc_ < log_shot_counter) {                      \
        _sc_ = (do_shot_counter) ? log_shot_counter : 0;                                        \
        fprintf(stderr, "aimdo-xpu: %s:%d:%s:", __FILE__, __LINE__, get_level_str(level));          \
        fprintf(stderr, __VA_ARGS__);                                                           \
        fflush(stderr);                                                                         \
    }                                                                                           \
}

#define log(level, ...) do_log(false, level, __VA_ARGS__)
#define log_shot(level, ...) do_log(true, level, __VA_ARGS__)

/* The default VRAM headroom. Different deficit methods with BYO headroom */
#define VRAM_HEADROOM (256 * 1024 * 1024)

/* control.c */
extern uint64_t vram_capacity;
extern uint64_t total_vram_usage;
extern uint64_t total_vram_last_check;
extern ssize_t deficit_sync;
extern const char *prevailing_deficit_method;

static inline size_t budget_deficit(size_t size) {
    ssize_t deficit_simple, deficit_delta;
    size_t deficit;

    xpu_budget_deficit();
    deficit_simple = (ssize_t)(total_vram_usage + VRAM_HEADROOM + size) - (ssize_t)vram_capacity;
    deficit_delta = deficit_sync + (ssize_t)total_vram_usage - (ssize_t)total_vram_last_check + size;
    deficit = (size_t)MAX(MAX(deficit_simple, deficit_delta), (ssize_t)0);
    if (deficit) {
        log(DEBUG, "%s: Prevailing Method: %s Deficit: %zu Alloc Size %zu\n", __func__,
            deficit_simple > deficit_delta ? "simple" : prevailing_deficit_method,
            deficit / M, size / M);
    }
    return deficit;
}

static inline int check_ze_impl(ze_result_t res, const char *label) {
    if (res != ZE_RESULT_SUCCESS && res != ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
        log(DEBUG, "Level Zero API FAILED : %s : result=%d\n", label, (int)res);
    }
    return (res == ZE_RESULT_SUCCESS);
}
#define CHECK_ZE(x) check_ze_impl((x), #x)

static inline ze_result_t three_stooges(ze_device_handle_t dev, ze_context_handle_t ctx,
                                        void *vaddr, size_t size, int device,
                                        ze_physical_mem_handle_t *out_handle) {
    ze_result_t err;
    ze_physical_mem_handle_t h = NULL;

    ze_physical_mem_desc_t physDesc = {
        .stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC,
        .pNext = NULL,
        .size = size,
    };

    err = zePhysicalMemCreate(ctx, dev, &physDesc, &h);
    if (!CHECK_ZE(err)) {
        return err;
    }

    ze_memory_access_attribute_t access = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
    err = zeVirtualMemMap(ctx, vaddr, size, h, 0, access);
    if (!CHECK_ZE(err)) {
        CHECK_ZE(zePhysicalMemDestroy(ctx, h));
        return err;
    }

    total_vram_usage += size;
    *out_handle = h;
    return ZE_RESULT_SUCCESS;
}

/* model-vbar.c */
size_t vbars_free(size_t size);
SHARED_EXPORT
uint64_t vbars_analyze(bool only_dirty);

/* pyt-xpu-alloc.c */
void *xpu_alloc_fn(size_t size, int device, void *stream);
void xpu_free_fn(void *ptr, size_t size, int device, void *stream);
void xpu_set_pending_alloc(void *ptr, size_t size);
void xpu_clear_pending_alloc(void);

void allocations_analyze();

/* control.c */
extern ze_context_handle_t aimdo_xpu_ctx;
extern ze_device_handle_t aimdo_xpu_dev;
SHARED_EXPORT
void aimdo_analyze();
