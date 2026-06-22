#include "plat_xpu.h"
#include "vrambuf_xpu.h"

#define VMM_HASH_SHIFT  21
#define VMM_HASH_SIZE   (1 << 12)

static VramBuffer* vmm_table[VMM_HASH_SIZE];

typedef struct VbarEntry {
    void *ptr;
    size_t size;
    struct VbarEntry *next;
} VbarEntry;

static VbarEntry *vbar_table[VMM_HASH_SIZE];

static void *pending_vbar_ptr;
static size_t pending_vbar_size;

static inline unsigned int vmm_hash(void *ptr) {
    return ((uintptr_t)ptr >> VMM_HASH_SHIFT) % VMM_HASH_SIZE;
}

SHARED_EXPORT
void xpu_set_pending_alloc(void *ptr, size_t size) {
    pending_vbar_ptr = ptr;
    pending_vbar_size = size;
}

SHARED_EXPORT
void xpu_clear_pending_alloc(void) {
    pending_vbar_ptr = NULL;
    pending_vbar_size = 0;
}

static void vbar_table_insert(void *ptr, size_t size) {
    unsigned int h = vmm_hash(ptr);
    VbarEntry *e = calloc(1, sizeof(*e));
    e->ptr = ptr;
    e->size = size;
    e->next = vbar_table[h];
    vbar_table[h] = e;
}

static bool vbar_table_remove(void *ptr) {
    unsigned int h = vmm_hash(ptr);
    for (VbarEntry **curr = &vbar_table[h]; *curr; curr = &(*curr)->next) {
        if ((*curr)->ptr == ptr) {
            VbarEntry *e = *curr;
            *curr = e->next;
            free(e);
            return true;
        }
    }
    return false;
}

void allocations_analyze() {
    size_t total_size = 0;
    int count = 0;

    log(DEBUG, "--- Allocation Analysis Start ---\n");

    for (int i = 0; i < VMM_HASH_SIZE; i++) {
        VramBuffer *entry = vmm_table[i];
        while (entry) {
            void *ptr = vrambuf_get(entry);
            size_t s = entry->allocated;

            log(DEBUG, "  [Bucket %4d] Ptr: %p | Size: %7zuk\n",
                i, ptr, s / K);

            total_size += s;
            count++;

            entry = entry->next;
        }
    }

    log(DEBUG, "%d Active Allocations for a total of %7zu MB\n", count, total_size / M);
}

SHARED_EXPORT
void *xpu_alloc_fn(size_t size, int device, void *stream) {
    log(VERBOSE, "%s (start): size=%zuk, device=%d\n", __func__, size / K, device);

    (void)stream;

    if (pending_vbar_ptr && pending_vbar_size == size) {
        vbar_table_insert(pending_vbar_ptr, size);
        log(VERBOSE, "%s (vbar): ptr=%p\n", __func__, pending_vbar_ptr);
        return pending_vbar_ptr;
    }

    VramBuffer *entry = vrambuf_create(device, size);
    if (!entry) {
        return NULL;
    }
    if (!vrambuf_grow(entry, size)) {
        vrambuf_destroy(entry);
        return NULL;
    }

    {
        unsigned int h = vmm_hash(vrambuf_get(entry));
        entry->next = vmm_table[h];
        vmm_table[h] = entry;
    }

    log(VERBOSE, "%s (return): ptr=%p\n", __func__, vrambuf_get(entry));
    return vrambuf_get(entry);
}

SHARED_EXPORT
void xpu_free_fn(void *ptr, size_t size, int device, void *stream) {
    log(VERBOSE, "%s (start) ptr=%p size=%zuk, device=%d\n", __func__, ptr, size / K, device);

    (void)size;
    (void)stream;

    if (ptr == NULL) {
        return;
    }

    if (vbar_table_remove(ptr)) {
        log(VERBOSE, "Released vbar-backed: ptr=%p\n", ptr);
        return;
    }

    for (VramBuffer **curr = &vmm_table[vmm_hash(ptr)]; *curr; curr = &(*curr)->next) {
        VramBuffer *entry = *curr;
        if (vrambuf_get(entry) != ptr || entry->device != device) {
            continue;
        }

        *curr = entry->next;
        vrambuf_destroy(entry);
        log(VERBOSE, "Freed: ptr=%p, size=%zuk, stream=%p\n", ptr, size / K, stream);
        return;
    }

    log(ERROR, "%s could not find VRAM@%p\n", __func__, ptr);
}
