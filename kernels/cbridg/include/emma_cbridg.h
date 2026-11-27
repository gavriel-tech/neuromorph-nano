/* Gavriel Technologies. Versioned, validated CUDA bridge. */
#ifndef EMMA_CBRIDG_H
#define EMMA_CBRIDG_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#define EMMA_NOEXCEPT noexcept
extern "C" {
#else
#define EMMA_NOEXCEPT
#endif
#ifdef EMMA_BUILD
#define EMMA_API __declspec(dllexport)
#else
#define EMMA_API __declspec(dllimport)
#endif
typedef int32_t emma_status;
typedef uint64_t emma_device;
#define EMMA_OK 0
#define EMMA_ERR_NULL -1
#define EMMA_ERR_RANK -2
#define EMMA_ERR_DTYPE -3
#define EMMA_ERR_OVERFLOW -4
#define EMMA_ERR_SHAPE -5
#define EMMA_ERR_CUDA -6
#define EMMA_ERR_STATE -7
#define EMMA_ERR_ABI -8
#define EMMA_ERR_INVALID_ARG -9
#define EMMA_ERR_ALIGNMENT -10
#define EMMA_ERR_PLUGIN -11
/* CUDA entry points in a CPU-only bridge return this status without touching outputs. */
#define EMMA_ERR_CUDA_NOT_BUILT -12
#define EMMA_OWNER_CALLER 1
#define EMMA_OWNER_MODULE 2
#define EMMA_F32 1
#define EMMA_F16 2
#define EMMA_BF16 3
#define EMMA_I8 4
#define EMMA_I32 5
#define EMMA_U8 6
#define EMMA_I4G 7
/* Non-owning contiguous view. data is a module buffer id, never a host pointer.
 * byte_offset is aligned to dtype. Module allocation is released by buffer_free.
 * rank 0 denotes a scalar; unused shape entries and reserved must be zero. */
typedef struct emma_tensor {
  uint64_t data;
  uint64_t byte_offset;
  int64_t shape[4];
  int32_t rank;
  int32_t dtype;
  int32_t owner;
  uint32_t reserved;
} emma_tensor;
/* Attributes interpreted by the selected op. All reserved values must be zero. */
typedef struct emma_attrs {
  uint64_t pos;
  float eps;
  float theta;
  uint32_t causal;
  uint32_t group_size;
  float scale;
  uint32_t reserved;
} emma_attrs;
/* Ordered bindings match opset.json and backnd v1. Unused tensors are zero.
 * Inputs/weights are read-only except kv-cache inputs 0 and 1 (session state).
 * Output memory is caller-selected module storage, committed asynchronously. */
typedef struct emma_op_args {
  emma_tensor inputs[4];
  emma_tensor weights[2];
  emma_tensor outputs[2];
  emma_attrs attrs;
  uint32_t input_count;
  uint32_t weight_count;
  uint32_t output_count;
  uint32_t reserved;
} emma_op_args;
/* Device profile, filled only on success; name is NUL-terminated UTF-8. */
typedef struct emma_device_info {
  uint64_t memory_bytes;
  uint32_t major;
  uint32_t minor;
  char name[128];
  uint64_t reserved;
} emma_device_info;
/* Returns major<<16|minor (2.0). No failure; no CUDA access. Thread-safe. */
EMMA_API uint32_t emma_abi_version(void) EMMA_NOEXCEPT;
/* Returns 0 for a production build, else the planted decoy compiled in (1 scale-twice,
 * 2 ignore-mask, 3 int4-high-first). No failure; no CUDA access. Thread-safe. */
EMMA_API uint32_t emma_build_mutant(void) EMMA_NOEXCEPT;
/* [out] available and total physical host memory in bytes. No CUDA access. Thread-safe. */
EMMA_API emma_status emma_host_memory(uint64_t* available, uint64_t* total) EMMA_NOEXCEPT;
/* Job admission. [in] UTF-8 path on the disk. [out] bytes free to the caller and total. */
EMMA_API emma_status emma_disk_free(const char* path, uint64_t* available,
                                    uint64_t* total) EMMA_NOEXCEPT;
/* [out] 1 when a process with this id is still running, else 0. */
EMMA_API emma_status emma_process_alive(uint32_t pid, uint8_t* alive) EMMA_NOEXCEPT;
/* Read count byte ranges (offsets, lengths) of the UTF-8 file at path into buffers, with the
 * Windows IoRing API when present (loaded at runtime) else positioned ReadFile calls.
 * [out] used_ioring 1 when IoRing served the reads. */
EMMA_API emma_status emma_read_ranges(const char* path, uint32_t count, const uint64_t* offsets,
                                      const uint32_t* lengths, uint8_t* const* buffers,
                                      uint8_t* used_ioring) EMMA_NOEXCEPT;
/* [out] free and total bytes on CUDA device `ordinal` (cudaMemGetInfo). */
EMMA_API emma_status emma_gpu_memory(int32_t ordinal, uint64_t* free_bytes,
                                     uint64_t* total_bytes) EMMA_NOEXCEPT;
/* Returns thread-local borrowed text, valid until the next call on this thread.
 * Does not clear the error. Never NULL. No CUDA access. */
EMMA_API const char* emma_last_error(void) EMMA_NOEXCEPT;
/* [in] required ABI, ordinal. [out] handle and profile, untouched on error.
 * Creates one explicit stream owned by the handle; serial calls are thread-safe.
 * Status: OK, NULL, ABI, INVALID_ARG, CUDA. Close with device_close. */
EMMA_API emma_status emma_device_open(uint32_t abi, int32_t ordinal, emma_device* out,
                                      emma_device_info* info) EMMA_NOEXCEPT;
/* [in] handle; zero is accepted. Frees module buffers and stream.
 * Status: OK, STATE, CUDA. Closed ids never become valid again. */
EMMA_API emma_status emma_device_close(emma_device device) EMMA_NOEXCEPT;
/* [in] handle, nonzero bytes. [out] module buffer id, untouched on error.
 * Status: OK, NULL, STATE, OVERFLOW, INVALID_ARG, CUDA. */
EMMA_API emma_status emma_buffer_alloc(emma_device device, uint64_t bytes,
                                       uint64_t* out) EMMA_NOEXCEPT;
/* [in] handle and buffer id; zero buffer accepted. Status: OK, STATE, CUDA. */
EMMA_API emma_status emma_buffer_free(emma_device device, uint64_t buffer) EMMA_NOEXCEPT;
/* [in] handle, module buffer id, caller-owned host bytes and exact byte length.
 * Copies finish before return so host slices need not outlive the call.
 * Status: OK, NULL, STATE, SHAPE, CUDA. No ownership transfer. */
EMMA_API emma_status emma_buffer_upload(emma_device device, uint64_t buffer, const void* src,
                                        uint64_t bytes) EMMA_NOEXCEPT;
/* [out] caller-owned host bytes. Otherwise same contract as upload. */
EMMA_API emma_status emma_buffer_download(emma_device device, uint64_t buffer, void* dst,
                                          uint64_t bytes) EMMA_NOEXCEPT;
/* [in] two buffers of one device with equal byte length. Copies device to device in
 * stream order without waiting. Status: OK, STATE, SHAPE, CUDA. */
EMMA_API emma_status emma_buffer_copy(emma_device device, uint64_t src, uint64_t dst,
                                      uint64_t bytes) EMMA_NOEXCEPT;
/* [in] copies all of src into dst at byte offset dst_offset, device to device in stream
 * order without waiting. Status: OK, STATE, SHAPE, CUDA. */
EMMA_API emma_status emma_buffer_copy_at(emma_device device, uint64_t src, uint64_t dst,
                                         uint64_t dst_offset, uint64_t bytes) EMMA_NOEXCEPT;
/* [in] explicit device stream owner. Status: OK, STATE, CUDA. */
EMMA_API emma_status emma_stream_sync(emma_device device) EMMA_NOEXCEPT;
/* All ops: [in] device/stream owner and borrowed args; enqueues on that stream.
 * Serialized against handle close. Status: OK, NULL, RANK, DTYPE, OVERFLOW,
 * SHAPE, CUDA, STATE, INVALID_ARG, ALIGNMENT. Validate before launch.
 * Runtime invalid ids, masks and nonfinite decode values return INVALID_ARG.
 * No out-parameter descriptor is modified. Sync before observing output bytes. */
#define EMMA_OP_DECL(name) \
  EMMA_API emma_status emma_op_##name(emma_device device, const emma_op_args* args) EMMA_NOEXCEPT
EMMA_OP_DECL(embed);
EMMA_OP_DECL(rmsnorm);
EMMA_OP_DECL(rope);
EMMA_OP_DECL(gqa_attention);
EMMA_OP_DECL(swiglu);
EMMA_OP_DECL(residual_add);
EMMA_OP_DECL(matmul_int8);
EMMA_OP_DECL(matmul_f16);
EMMA_OP_DECL(kv_cache);
EMMA_OP_DECL(decode_greedy);
EMMA_OP_DECL(matmul_int4);
EMMA_OP_DECL(adapter_add);
/* matmul-affine4 (op 12): x inputs[0] [.., K] f32, biases inputs[1] [N, K/G] f32,
 * codes weights[0] [N, K] I4G unsigned low nibble first, scales weights[1] [N, K/G] f32,
 * y [.., N] f32; w = code * scale + bias (MLX affine int4, group G). */
EMMA_OP_DECL(matmul_affine4);
/* sparsify (op 13): y = |x| >= attrs.scale ? x : 0, one float input, y like x. */
EMMA_OP_DECL(sparsify);
#undef EMMA_OP_DECL
/* Native plugins (emma_plugin.h). Ids are opaque, like device ids.
 * open: [in] UTF-8 absolute path of the library. [out] plugin id. ABI or size mismatch: ABI. */
EMMA_API emma_status emma_plugin_open(const char* path, uint64_t* out) EMMA_NOEXCEPT;
/* [out] the plugin's own NUL-terminated strings, valid until close. */
EMMA_API emma_status emma_plugin_info(uint64_t plugin, const char** name, const char** doc,
                                      const char** ports) EMMA_NOEXCEPT;
/* [in] input JSON bytes. [out] up to cap bytes of output JSON; *len written or needed.
 * OVERFLOW: cap too small, *len holds the size needed. PLUGIN: the plugin failed, and
 * emma_last_error holds its message. Safe to call from several threads. */
EMMA_API emma_status emma_plugin_run(uint64_t plugin, const uint8_t* input, uint64_t input_len,
                                     uint8_t* out, uint64_t cap, uint64_t* len) EMMA_NOEXCEPT;
/* Unloads once no run is in progress. A stale id returns STATE. */
EMMA_API emma_status emma_plugin_close(uint64_t plugin) EMMA_NOEXCEPT;
#ifdef __cplusplus
}
static_assert(sizeof(emma_tensor) == 64);
static_assert(alignof(emma_tensor) == 8);
static_assert(sizeof(emma_attrs) == 32);
static_assert(sizeof(emma_op_args) == 560);
static_assert(sizeof(emma_device_info) == 152);
static_assert(alignof(emma_attrs) == 8);
static_assert(alignof(emma_op_args) == 8);
static_assert(alignof(emma_device_info) == 8);
static_assert(offsetof(emma_tensor, data) == 0);
static_assert(offsetof(emma_tensor, byte_offset) == 8);
static_assert(offsetof(emma_tensor, shape) == 16);
static_assert(offsetof(emma_tensor, rank) == 48);
static_assert(offsetof(emma_tensor, dtype) == 52);
static_assert(offsetof(emma_tensor, owner) == 56);
static_assert(offsetof(emma_tensor, reserved) == 60);
static_assert(offsetof(emma_attrs, pos) == 0);
static_assert(offsetof(emma_attrs, eps) == 8);
static_assert(offsetof(emma_attrs, theta) == 12);
static_assert(offsetof(emma_attrs, causal) == 16);
static_assert(offsetof(emma_attrs, group_size) == 20);
static_assert(offsetof(emma_attrs, scale) == 24);
static_assert(offsetof(emma_attrs, reserved) == 28);
static_assert(offsetof(emma_op_args, inputs) == 0);
static_assert(offsetof(emma_op_args, weights) == 256);
static_assert(offsetof(emma_op_args, outputs) == 384);
static_assert(offsetof(emma_op_args, attrs) == 512);
static_assert(offsetof(emma_op_args, input_count) == 544);
static_assert(offsetof(emma_op_args, weight_count) == 548);
static_assert(offsetof(emma_op_args, output_count) == 552);
static_assert(offsetof(emma_op_args, reserved) == 556);
static_assert(offsetof(emma_device_info, memory_bytes) == 0);
static_assert(offsetof(emma_device_info, major) == 8);
static_assert(offsetof(emma_device_info, minor) == 12);
static_assert(offsetof(emma_device_info, name) == 16);
static_assert(offsetof(emma_device_info, reserved) == 144);
#endif
#endif
