/* A fixed-size ring buffer of floats for streaming samples.
 * int not_a_function(void) { } stays inside this comment. */
#ifndef RINGBUF_H
#define RINGBUF_H

#include <stddef.h>

#define RB_CAPACITY 1024
#define RB_MASK (RB_CAPACITY - 1)
#define RB_FULL(rb) \
    (((rb)->head - (rb)->tail) == RB_CAPACITY)

typedef struct ringbuf {
    float data[RB_CAPACITY];
    size_t head;
    size_t tail;
} ringbuf_t;

typedef enum {
    RB_OK = 0,
    RB_EMPTY = 1,
    RB_OVERFLOW = 2
} rb_status;

typedef void (*rb_visit_fn)(float sample, void *user);

void rb_init(ringbuf_t *rb);
rb_status rb_push(ringbuf_t *rb, float x);
rb_status rb_pop(ringbuf_t *rb, float *out);
size_t rb_len(const ringbuf_t *rb);

static inline int rb_is_empty(const ringbuf_t *rb) {
    return rb->head == rb->tail;
}

#endif
