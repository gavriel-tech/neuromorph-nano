#include "ringbuf.h"
#include <string.h>

static const char *const STATUS_NAMES[] = {"ok", "empty", "overflow"};

void rb_init(ringbuf_t *rb) {
    memset(rb, 0, sizeof(*rb));
}

rb_status rb_push(ringbuf_t *rb, float x) {
    if (RB_FULL(rb)) {
        /* Oldest sample is dropped so the newest always fits. */
        rb->tail++;
        rb->data[rb->head++ & RB_MASK] = x;
        return RB_OVERFLOW;
    }
    rb->data[rb->head++ & RB_MASK] = x;
    return RB_OK;
}

rb_status rb_pop(ringbuf_t *rb, float *out) {
    if (rb_is_empty(rb)) {
        return RB_EMPTY;
    }
    *out = rb->data[rb->tail++ & RB_MASK];
    return RB_OK;
}

size_t rb_len(const ringbuf_t *rb)
{
    return rb->head - rb->tail;
}

void rb_each(const ringbuf_t *rb, rb_visit_fn visit, void *user) {
    for (size_t i = rb->tail; i != rb->head; i++) {
        visit(rb->data[i & RB_MASK], user);
    }
}

const char *rb_status_name(rb_status s) {
    return (unsigned)s < 3 ? STATUS_NAMES[s] : "unknown";
}
