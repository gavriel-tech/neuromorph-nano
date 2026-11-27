#include <stdio.h>
#include "ringbuf.h"
#include "../kernels/scale.cuh"

/* Samples read from standard input before the buffer is scaled on the device. */
#define BATCH 256

static void print_sample(float s, void *user) {
    (void)user;
    printf("%.4f\n", s);
}

int main(int argc, char **argv) {
    ringbuf_t rb;
    float x;
    float gain = argc > 1 ? (float)atof(argv[1]) : 1.0f;
    rb_init(&rb);
    while (scanf("%f", &x) == 1) {
        if (rb_push(&rb, x) == RB_OVERFLOW) {
            fprintf(stderr, "dropped oldest sample\n");
        }
    }
    scale_on_device(rb.data, (int)rb_len(&rb), gain);
    rb_each(&rb, print_sample, NULL);
    return 0;
}
