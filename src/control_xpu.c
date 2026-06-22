#include "plat_xpu.h"
#include "aimdo-time.h"

uint64_t vram_capacity;
uint64_t total_vram_usage;
uint64_t total_vram_last_check;
ssize_t deficit_sync;
const char *prevailing_deficit_method;
ze_context_handle_t aimdo_xpu_ctx;
ze_device_handle_t aimdo_xpu_dev;

bool xpu_budget_deficit() {
    uint64_t now = GET_TICK();
    static uint64_t last_check = 0;
    size_t free_vram = 0;
    size_t total_vram = 0;

    if (now - last_check < 2000) {
        return true;
    }
    last_check = now;
    total_vram_last_check = total_vram_usage;

    uint32_t mem_count = 0;
    if (CHECK_ZE(zeDeviceGetMemoryProperties(aimdo_xpu_dev, &mem_count, NULL)) && mem_count > 0) {
        ze_device_memory_properties_t mem_props = {};
        mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        if (CHECK_ZE(zeDeviceGetMemoryProperties(aimdo_xpu_dev, &mem_count, &mem_props))) {
            total_vram = mem_props.totalSize;
        }
    }

    free_vram = total_vram > total_vram_usage ? total_vram - total_vram_usage : 0;

    deficit_sync = (ssize_t)VRAM_HEADROOM - (ssize_t)free_vram;
    prevailing_deficit_method = "xpu_mem_props";
    return true;
}

SHARED_EXPORT
void aimdo_analyze() {
    log(DEBUG, "--- VRAM Stats ---\n");
    log(DEBUG, "  Aimdo Recorded Usage:  %7zu MB\n", total_vram_usage / M);

    vbars_analyze(true);
    allocations_analyze();
}

SHARED_EXPORT
uint64_t get_total_vram_usage() {
    return total_vram_usage;
}

SHARED_EXPORT
bool init(int xpu_device_id) {
    ze_result_t result;
    ze_driver_handle_t driver;
    ze_device_handle_t device;

    log_reset_shots();

    result = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (result != ZE_RESULT_SUCCESS) {
        log(ERROR, "zeInit failed: %d\n", result);
        return false;
    }

    uint32_t driver_count = 1;
    result = zeDriverGet(&driver_count, &driver);
    if (result != ZE_RESULT_SUCCESS) {
        log(ERROR, "zeDriverGet failed: %d\n", result);
        return false;
    }

    uint32_t device_count = 1;
    result = zeDeviceGet(driver, &device_count, &device);
    if (result != ZE_RESULT_SUCCESS) {
        log(ERROR, "zeDeviceGet failed: %d\n", result);
        return false;
    }

    if (xpu_device_id > 0) {
        device_count = 1;
        result = zeDeviceGet(driver, &device_count, &device);
        if (xpu_device_id >= (int)device_count) {
            log(ERROR, "XPU device %d not found\n", xpu_device_id);
            return false;
        }
    }

    ze_context_desc_t ctxDesc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
        .pNext = NULL,
        .flags = 0,
    };

    result = zeContextCreate(driver, &ctxDesc, &aimdo_xpu_ctx);
    if (result != ZE_RESULT_SUCCESS) {
        log(ERROR, "zeContextCreate failed: %d\n", result);
        return false;
    }

    aimdo_xpu_dev = device;

    uint32_t mem_count = 0;
    if (CHECK_ZE(zeDeviceGetMemoryProperties(device, &mem_count, NULL)) && mem_count > 0) {
        ze_device_memory_properties_t mem_props = {};
        mem_props.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
        if (CHECK_ZE(zeDeviceGetMemoryProperties(device, &mem_count, &mem_props))) {
            vram_capacity = mem_props.totalSize;
        }
    }

    char dev_name[256] = {0};
    ze_device_properties_t dev_props = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
        .pNext = NULL,
    };
    if (CHECK_ZE(zeDeviceGetProperties(device, &dev_props))) {
        snprintf(dev_name, sizeof(dev_name), "%s", dev_props.name);
    }

    log(INFO, "comfy-aimdo-xpu inited for GPU: %s (VRAM: %zu MB)\n",
        dev_name, (size_t)(vram_capacity / (1024 * 1024)));
    return true;
}

SHARED_EXPORT
void cleanup() {
    if (aimdo_xpu_ctx) {
        zeContextDestroy(aimdo_xpu_ctx);
        aimdo_xpu_ctx = NULL;
    }
}
