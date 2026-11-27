// Gavriel Technologies. Lane S, card speed (2026-11-18): element kernels of element.cu laid out for
// throughput, with the same per-element arithmetic (built like element.cu, --fmad=false), so the
// outputs are bit-identical to the kernels they replace (checked by nmengn.gpu.fastel). Separate
// image (fastel.fatbin); element.fatbin is unchanged.
//
// binary8_kernel: binary_kernel (op 4 SwiGLU gate, op 5 add) on binary16 tensors, eight values per
// thread through 16-byte reads and writes.
// rope2_kernel: rope_kernel for the queries and the keys in one launch; one thread per token and
// rotary pair computes the angle, its cosine and its sine once and applies them to every head of
// both tensors (rope_kernel computed them again for each of the 24 heads, in two launches).
// norm_warp_kernel: norm_row_kernel with one warp per row in place of one block: each lane squares
// its values in registers, then the warp adds the squares in the same left-to-right order (each
// value broadcast from the lane holding it), so the sum, the factor and every output are those of
// norm_row_kernel; eight rows per block, no shared memory. SLOTS values per lane (width = 32 x SLOTS).
// segments_kernel: segment_kernel (segment.cu) for every packed sequence of a batch in one launch,
// the block-diagonal attention of a batched forward (one block mask per sequence group): a table
// gives each block its sequence (first packed row, length, first row of the block within it); the
// keys a sequence reads are the shared prefix (weights[0] and [1], P rows) and then its own rows
// of the packed keys, in the order and 32-key tiles of the concatenated copy segment_kernel read,
// so every output is the same bit for bit, without the per-sequence copies and launches.
// Package 5 (2026-11-18): the table has five numbers a block; the last two name a body (first
// packed row, rows) a tail sequence reads after the prefix and before its own rows (two questions
// over one group's pieces); a body of 0 rows reads as before, bit for bit.
// segments_wide_kernel<PER> (lane S package 3): segments_kernel with PER query rows per warp (8 * PER
// per block) instead of two, so a short sequence's rows share one pass over the prefix keys and
// values instead of one pass per 16 rows. The tiles hold the binary16 values as they are (half the
// shared memory; each read converts exactly to the float segments_kernel holds), and each row's
// arithmetic is that of segments_kernel in the same order (the same 32-key tiles), so outputs are
// bit-identical (nmengn.gpu.segments).
#include "../cbridg/src/internal.hpp"
#include "attention.cuh"
namespace emma {
__global__ __launch_bounds__(128) void binary8_kernel(int op, emma_op_args a) {
  const uint64_t n = elements(a.inputs[0]);
  const uint64_t base = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (base >= n) {
    return;
  }
  const auto* xl = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* xr = reinterpret_cast<const __half*>(a.inputs[1].data);
  auto* yo = reinterpret_cast<__half*>(a.outputs[0].data);
  auto one = [&](float left, float right) {
    if (op == 4) {
      left = rounded(left / (1.0F + expf(-left)), EMMA_F16) * right;
    } else {
      left += right;
    }
    return __float2half_rn(left);
  };
  if (base + 8 <= n) {
    const uint4 lv = *reinterpret_cast<const uint4*>(xl + base);
    const uint4 rv = *reinterpret_cast<const uint4*>(xr + base);
    const __half* lh = reinterpret_cast<const __half*>(&lv);
    const __half* rh = reinterpret_cast<const __half*>(&rv);
    uint4 ov;
    __half* oh = reinterpret_cast<__half*>(&ov);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      oh[e] = one(__half2float(lh[e]), __half2float(rh[e]));
    }
    *reinterpret_cast<uint4*>(yo + base) = ov;
  } else {
    for (uint64_t i = base; i < n; ++i) {
      yo[i] = one(__half2float(xl[i]), __half2float(xr[i]));
    }
  }
}
__global__ __launch_bounds__(128) void rope2_kernel(emma_op_args a) {
  const auto q = a.inputs[0];
  const auto k = a.inputs[1];
  const uint64_t width = static_cast<uint64_t>(q.shape[q.rank - 1]);
  const uint64_t half = width / 2;
  const uint64_t qheads = static_cast<uint64_t>(q.shape[q.rank - 2]);
  const uint64_t kheads = static_cast<uint64_t>(k.shape[k.rank - 2]);
  const uint64_t tokens = elements(q) / (qheads * width);
  const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= tokens * half) {
    return;
  }
  const uint64_t t = i / half;
  const uint64_t j = i % half;
  const int32_t pos = reinterpret_cast<const int32_t*>(a.inputs[2].data)[t];
  const float frequency =
      1.0F / powf(a.attrs.theta, static_cast<float>(2 * j) / static_cast<float>(width));
  const float angle = static_cast<float>(pos) * frequency;
  const float cs = rounded(cosf(angle), EMMA_F16);
  const float sn = rounded(sinf(angle), EMMA_F16);
  auto turn = [&](const emma_tensor& x, const emma_tensor& y, uint64_t heads) {
    const auto* xh = reinterpret_cast<const __half*>(x.data);
    auto* yh = reinterpret_cast<__half*>(y.data);
    for (uint64_t h = 0; h < heads; ++h) {
      const uint64_t row = t * heads + h;
      const float left = __half2float(xh[row * width + j]);
      const float right = __half2float(xh[row * width + j + half]);
      yh[row * width + j] =
          __float2half_rn(rounded(left * cs, EMMA_F16) + rounded(-right * sn, EMMA_F16));
      yh[row * width + j + half] =
          __float2half_rn(rounded(right * cs, EMMA_F16) + rounded(left * sn, EMMA_F16));
    }
  };
  turn(q, a.outputs[0], qheads);
  turn(k, a.outputs[1], kheads);
}
template <int SLOTS>
__global__ __launch_bounds__(256) void norm_warp_kernel(emma_op_args a) {
  constexpr uint64_t kWidth = 32 * SLOTS;
  const auto x = a.inputs[0];
  const auto y = a.outputs[0];
  const uint64_t rows = elements(x) / kWidth;
  const uint64_t row = static_cast<uint64_t>(blockIdx.x) * 8 + threadIdx.x / 32;
  const int lane = static_cast<int>(threadIdx.x % 32);
  if (row >= rows) {
    return;
  }
  const auto* xh = reinterpret_cast<const __half*>(x.data) + row * kWidth;
  float v[SLOTS];
  float sq[SLOTS];
#pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    v[s] = __half2float(xh[s * 32 + lane]);
    sq[s] = v[s] * v[s];
  }
  float sum = 0;
#pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
#pragma unroll
    for (int l = 0; l < 32; ++l) {
      sum += __shfl_sync(0xffffffffu, sq[s], l);
    }
  }
  const float factor = 1.0F / sqrtf(sum / static_cast<float>(kWidth) + a.attrs.eps);
  auto* yh = reinterpret_cast<__half*>(y.data) + row * kWidth;
#pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const uint64_t j = s * 32 + lane;
    yh[j] = __float2half_rn(rounded(v[s] * factor, EMMA_F16) * load(a.weights[0], j));
  }
}
template __global__ void norm_warp_kernel<4>(emma_op_args a);
template __global__ void norm_warp_kernel<32>(emma_op_args a);
namespace {
constexpr int kSegTile = 32;
constexpr int kSegWarps = 8;
constexpr int kSegPer = 2;
constexpr int kSegWidth = 128;
__device__ inline float segs_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_xor_sync(0xffffffff, value, offset);
  }
  return value;
}
__device__ inline float segs_warp_max(float value) {
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  }
  return value;
}
}  // namespace
// g.tokens: rows of the packed batch; g.cached: rows of the shared prefix (0 for none).
__global__ __launch_bounds__(256) void segments_kernel(emma_op_args a, Geometry g, const int* table) {
  __shared__ float key[kSegTile][kSegWidth + 1];
  __shared__ float value[kSegTile][kSegWidth];
  __shared__ float query[kSegWarps * kSegPer][kSegWidth];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const uint64_t q0 = static_cast<uint64_t>(table[5 * blockIdx.x]);
  const uint64_t n = static_cast<uint64_t>(table[5 * blockIdx.x + 1]);
  const uint64_t local0 = static_cast<uint64_t>(table[5 * blockIdx.x + 2]);
  // A tail sequence also reads a body: `bl` packed rows from `bs` after the prefix (package 5).
  const uint64_t bs = static_cast<uint64_t>(table[5 * blockIdx.x + 3]);
  const uint64_t bl = static_cast<uint64_t>(table[5 * blockIdx.x + 4]);
  const uint64_t prefix = g.cached;
  const uint64_t cached = prefix + bl + n;
  const uint64_t kh = blockIdx.y / (g.heads / g.kv_heads);
  const float scale = 1.0f / sqrtf(static_cast<float>(g.width));
  uint64_t token[kSegPer], row[kSegPer], visible[kSegPer];
  bool active[kSegPer];
  float maximum[kSegPer], denominator[kSegPer], out[kSegPer][4];
  for (int r = 0; r < kSegPer; ++r) {
    token[r] = local0 + r * kSegWarps + warp;
    row[r] = (q0 + token[r]) * g.heads + blockIdx.y;
    active[r] = token[r] < n && enabled(a, g, row[r]);
    visible[r] = a.attrs.causal != 0 ? cached - n + token[r] + 1 : cached;
    maximum[r] = -CUDART_INF_F;
    denominator[r] = 0.0f;
    for (int j = 0; j < 4; ++j) {
      out[r][j] = 0.0f;
    }
    for (int d = lane; d < kSegWidth; d += 32) {
      query[r * kSegWarps + warp][d] = active[r] && d < static_cast<int>(g.width) ? load(a.inputs[0], row[r] * g.width + d) : 0.0f;
    }
  }
  const uint64_t last = min(n, local0 + kSegWarps * kSegPer);
  const uint64_t limit = a.attrs.causal != 0 ? min(cached, cached - n + last) : cached;
  for (uint64_t first = 0; first < limit; first += kSegTile) {
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(kSegTile * kSegWidth); i += blockDim.x) {
      const uint64_t s = i >> 7, d = i & 127;
      const uint64_t at = first + s;
      const bool in = at < cached;
      float kv = 0.0f;
      float vv = 0.0f;
      if (in && at < prefix) {
        const uint64_t x = (at * g.kv_heads + kh) * g.width + d;
        kv = load(a.weights[0], x);
        vv = load(a.weights[1], x);
      } else if (in) {
        const uint64_t x = (((at < prefix + bl) ? bs + at - prefix : q0 + at - prefix - bl) * g.kv_heads + kh) * g.width + d;
        kv = load(a.inputs[1], x);
        vv = load(a.inputs[2], x);
      }
      key[s][d] = kv;
      value[s][d] = vv;
    }
    __syncthreads();
    float dot[kSegPer] = {0.0f, 0.0f};
#pragma unroll 8
    for (int d = 0; d < kSegWidth; ++d) {
      const float k = key[lane][d];
      for (int r = 0; r < kSegPer; ++r) {
        dot[r] += query[r * kSegWarps + warp][d] * k;
      }
    }
    float probability[kSegPer], previous[kSegPer];
    bool any[kSegPer];
    for (int r = 0; r < kSegPer; ++r) {
      const bool seen = active[r] && first + lane < visible[r];
      const float score = seen ? dot[r] * scale : -CUDART_INF_F;
      const float next = fmaxf(maximum[r], segs_warp_max(score));
      any[r] = isfinite(next);
      previous[r] = any[r] && isfinite(maximum[r]) ? expf(maximum[r] - next) : 0.0f;
      probability[r] = seen ? expf(score - next) : 0.0f;
      if (any[r]) {
        denominator[r] = denominator[r] * previous[r] + segs_warp_sum(probability[r]);
        maximum[r] = next;
        for (int j = 0; j < 4; ++j) {
          out[r][j] *= previous[r];
        }
      }
    }
    for (int s = 0; s < kSegTile; ++s) {
      const float p0 = __shfl_sync(0xffffffff, probability[0], s);
      const float p1 = __shfl_sync(0xffffffff, probability[1], s);
      for (int j = 0; j < 4; ++j) {
        const float v = value[s][lane + 32 * j];
        out[0][j] += p0 * v;
        out[1][j] += p1 * v;
      }
    }
  }
  for (int r = 0; r < kSegPer; ++r) {
    if (token[r] >= n) {
      continue;
    }
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = lane + 32 * j;
      if (d < g.width) {
        store(a.outputs[0], row[r] * g.width + d, active[r] && denominator[r] > 0.0f ? out[r][j] / denominator[r] : 0.0f);
      }
    }
  }
}
template <int PER>
__global__ __launch_bounds__(256) void segments_wide_kernel(emma_op_args a, Geometry g, const int* table) {
  constexpr int kRowsB = kSegWarps * PER;
  constexpr int kKp = kSegWidth + 2;  // 65 words a key row: lane-indexed reads hit distinct banks
  __shared__ __align__(16) __half key[kSegTile][kKp];
  __shared__ __align__(16) __half value[kSegTile][kSegWidth];
  __shared__ __align__(16) __half query[kRowsB][kSegWidth];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const uint64_t q0 = static_cast<uint64_t>(table[5 * blockIdx.x]);
  const uint64_t n = static_cast<uint64_t>(table[5 * blockIdx.x + 1]);
  const uint64_t local0 = static_cast<uint64_t>(table[5 * blockIdx.x + 2]);
  // A tail sequence also reads a body: `bl` packed rows from `bs` after the prefix (package 5).
  const uint64_t bs = static_cast<uint64_t>(table[5 * blockIdx.x + 3]);
  const uint64_t bl = static_cast<uint64_t>(table[5 * blockIdx.x + 4]);
  const uint64_t prefix = g.cached;
  const uint64_t cached = prefix + bl + n;
  const uint64_t kh = blockIdx.y / (g.heads / g.kv_heads);
  const float scale = 1.0f / sqrtf(static_cast<float>(g.width));
  const auto* qh = reinterpret_cast<const __half*>(a.inputs[0].data);
  uint64_t token[PER], row[PER], visible[PER];
  bool active[PER];
  float maximum[PER], denominator[PER], out[PER][4];
  for (int r = 0; r < PER; ++r) {
    token[r] = local0 + r * kSegWarps + warp;
    row[r] = (q0 + token[r]) * g.heads + blockIdx.y;
    active[r] = token[r] < n && enabled(a, g, row[r]);
    visible[r] = a.attrs.causal != 0 ? cached - n + token[r] + 1 : cached;
    maximum[r] = -CUDART_INF_F;
    denominator[r] = 0.0f;
    for (int j = 0; j < 4; ++j) {
      out[r][j] = 0.0f;
    }
    for (int d = lane; d < kSegWidth; d += 32) {
      query[r * kSegWarps + warp][d] = active[r] && d < static_cast<int>(g.width) ? qh[row[r] * g.width + d] : __float2half_rn(0.0f);
    }
  }
  const uint64_t last = min(n, local0 + kRowsB);
  const uint64_t limit = a.attrs.causal != 0 ? min(cached, cached - n + last) : cached;
  const auto* pkh = reinterpret_cast<const __half*>(a.weights[0].data);
  const auto* pvh = reinterpret_cast<const __half*>(a.weights[1].data);
  const auto* kh16 = reinterpret_cast<const __half*>(a.inputs[1].data);
  const auto* vh16 = reinterpret_cast<const __half*>(a.inputs[2].data);
  const __half hz = __float2half_rn(0.0f);
  for (uint64_t first = 0; first < limit; first += kSegTile) {
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < static_cast<uint32_t>(kSegTile * kSegWidth); i += blockDim.x) {
      const uint64_t s = i >> 7, d = i & 127;
      const uint64_t at = first + s;
      __half kv = hz;
      __half vv = hz;
      if (at < prefix) {
        const uint64_t x = (at * g.kv_heads + kh) * g.width + d;
        kv = pkh[x];
        vv = pvh[x];
      } else if (at < cached) {
        const uint64_t x = (((at < prefix + bl) ? bs + at - prefix : q0 + at - prefix - bl) * g.kv_heads + kh) * g.width + d;
        kv = kh16[x];
        vv = vh16[x];
      }
      key[s][d] = kv;
      value[s][d] = vv;
    }
    __syncthreads();
    float dot[PER];
    for (int r = 0; r < PER; ++r) {
      dot[r] = 0.0f;
    }
#pragma unroll 8
    for (int d = 0; d < kSegWidth; ++d) {
      const float k = __half2float(key[lane][d]);
#pragma unroll
      for (int r = 0; r < PER; ++r) {
        dot[r] += __half2float(query[r * kSegWarps + warp][d]) * k;
      }
    }
    float probability[PER], previous[PER];
    bool any[PER];
#pragma unroll
    for (int r = 0; r < PER; ++r) {
      const bool seen = active[r] && first + lane < visible[r];
      const float score = seen ? dot[r] * scale : -CUDART_INF_F;
      const float next = fmaxf(maximum[r], segs_warp_max(score));
      any[r] = isfinite(next);
      previous[r] = any[r] && isfinite(maximum[r]) ? expf(maximum[r] - next) : 0.0f;
      probability[r] = seen ? expf(score - next) : 0.0f;
      if (any[r]) {
        denominator[r] = denominator[r] * previous[r] + segs_warp_sum(probability[r]);
        maximum[r] = next;
        for (int j = 0; j < 4; ++j) {
          out[r][j] *= previous[r];
        }
      }
    }
    for (int s = 0; s < kSegTile; ++s) {
      float p[PER];
#pragma unroll
      for (int r = 0; r < PER; ++r) {
        p[r] = __shfl_sync(0xffffffff, probability[r], s);
      }
      for (int j = 0; j < 4; ++j) {
        const float v = __half2float(value[s][lane + 32 * j]);
#pragma unroll
        for (int r = 0; r < PER; ++r) {
          out[r][j] += p[r] * v;
        }
      }
    }
  }
  for (int r = 0; r < PER; ++r) {
    if (token[r] >= n) {
      continue;
    }
    for (int j = 0; j < 4; ++j) {
      const uint64_t d = lane + 32 * j;
      if (d < g.width) {
        store(a.outputs[0], row[r] * g.width + d, active[r] && denominator[r] > 0.0f ? out[r][j] / denominator[r] : 0.0f);
      }
    }
  }
}
template __global__ void segments_wide_kernel<4>(emma_op_args, Geometry, const int*);
template __global__ void segments_wide_kernel<8>(emma_op_args, Geometry, const int*);
namespace fseg {
constexpr int kRows = 64;
constexpr int kKeys = 64;
constexpr int kWidth = 128;
constexpr int kChunk = 32;
constexpr int kHalf = kKeys / 2;
constexpr int kPadC = kChunk + 4;
constexpr int kPadV = kWidth + 4;
constexpr int kPadP = kRows + 4;
constexpr int kScoreArea = 2 * kRows * kPadC;
constexpr int kValueArea = kHalf * kPadV;
constexpr int kFirst = kScoreArea > kValueArea ? kScoreArea : kValueArea;
constexpr int kShared = kFirst + kKeys * kPadP;
__device__ inline float max16(float v) {
  for (int offset = 8; offset > 0; offset /= 2) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
  }
  return v;
}
__device__ inline float sum16(float v) {
  for (int offset = 8; offset > 0; offset /= 2) {
    v += __shfl_xor_sync(0xffffffff, v, offset);
  }
  return v;
}
__device__ inline void load8(const __half* base, uint64_t at, bool in, float* dst) {
  float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 hi = lo;
  if (in) {
    const uint4 raw = *reinterpret_cast<const uint4*>(base + at);
    const __half2* h = reinterpret_cast<const __half2*>(&raw);
    const float2 a = __half22float2(h[0]);
    const float2 b = __half22float2(h[1]);
    const float2 c = __half22float2(h[2]);
    const float2 d = __half22float2(h[3]);
    lo = make_float4(a.x, a.y, b.x, b.y);
    hi = make_float4(c.x, c.y, d.x, d.y);
  }
  reinterpret_cast<float4*>(dst)[0] = lo;
  reinterpret_cast<float4*>(dst)[1] = hi;
}
}  // namespace fseg
// flash_segments_kernel (lane S package 4): flash_kernel (flash.cu) for every packed sequence of a
// batch in one launch, the long-group block-diagonal attention of the served path. A table gives
// each block its sequence (first packed row, length, first row of its 64-row tile); keys and values
// are read in place, the shared prefix (weights[0] and [1], P rows) then the sequence's own rows,
// in the 64-key tiles and order flash_kernel reads over the copied rows, with the same products,
// so outputs are bit-identical to one flash_kernel launch per sequence on copies
// (nmengn.gpu.flash_segments).
__global__ __launch_bounds__(256) void flash_segments_kernel(emma_op_args a, Geometry g, const int* table) {
  using namespace fseg;
  __shared__ __align__(16) float smem[kShared];
  float* qc = smem;
  float* kc = smem + kRows * kPadC;
  float* vc = smem;
  float* ps = smem + kFirst;
  const int tid = static_cast<int>(threadIdx.x);
  const int ty = tid / 16;
  const int tx = tid % 16;
  const uint64_t qs = static_cast<uint64_t>(table[5 * blockIdx.x]);
  const uint64_t n = static_cast<uint64_t>(table[5 * blockIdx.x + 1]);
  const uint64_t q0 = static_cast<uint64_t>(table[5 * blockIdx.x + 2]);
  // A tail sequence also reads a body: `bl` packed rows from `bs` after the prefix (package 5).
  const uint64_t bs = static_cast<uint64_t>(table[5 * blockIdx.x + 3]);
  const uint64_t bl = static_cast<uint64_t>(table[5 * blockIdx.x + 4]);
  const uint64_t prefix = g.cached;
  const uint64_t cached = prefix + bl + n;
  const uint64_t head = blockIdx.y;
  const uint64_t kh = head / (g.heads / g.kv_heads);
  const uint64_t past = prefix + bl;
  const bool causal = a.attrs.causal != 0;
  const auto* qh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* kp = reinterpret_cast<const __half*>(a.inputs[1].data);
  const auto* vp = reinterpret_cast<const __half*>(a.inputs[2].data);
  const auto* kpre = reinterpret_cast<const __half*>(a.weights[0].data);
  const auto* vpre = reinterpret_cast<const __half*>(a.weights[1].data);
  bool active[4];
  uint64_t visible[4];
  float maximum[4], denominator[4], out[4][8];
  for (int i = 0; i < 4; ++i) {
    const uint64_t token = q0 + ty * 4 + i;
    active[i] = token < n && enabled(a, g, (qs + token) * g.heads + head);
    visible[i] = causal ? past + token + 1 : cached;
    maximum[i] = -CUDART_INF_F;
    denominator[i] = 0.0f;
    for (int e = 0; e < 8; ++e) {
      out[i][e] = 0.0f;
    }
  }
  const uint64_t last_row = min(q0 + kRows, n);
  const uint64_t end = causal ? min(cached, past + last_row) : cached;
  const float root = sqrtf(static_cast<float>(g.width));
  for (uint64_t k0 = 0; k0 < end; k0 += kKeys) {
    float s[4][4];
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        s[i][j] = 0.0f;
      }
    }
    for (int dc = 0; dc < kWidth; dc += kChunk) {
      __syncthreads();
      {
        const int row = tid >> 2;
        const int part = tid & 3;
        const uint64_t token = q0 + row;
        load8(qh, ((qs + token) * g.heads + head) * g.width + dc + part * 8, token < n, qc + row * kPadC + part * 8);
        const uint64_t key = k0 + row;
        const bool pre = key < prefix;
        const uint64_t at = pre ? (key * g.kv_heads + kh) * g.width : (((key < prefix + bl) ? bs + key - prefix : qs + key - prefix - bl) * g.kv_heads + kh) * g.width;
        load8(pre ? kpre : kp, at + dc + part * 8, key < cached, kc + row * kPadC + part * 8);
      }
      __syncthreads();
#pragma unroll
      for (int d = 0; d < kChunk; d += 4) {
        float4 q[4], k[4];
        for (int i = 0; i < 4; ++i) {
          q[i] = *reinterpret_cast<const float4*>(qc + (ty * 4 + i) * kPadC + d);
        }
        for (int j = 0; j < 4; ++j) {
          k[j] = *reinterpret_cast<const float4*>(kc + (tx + 16 * j) * kPadC + d);
        }
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            float acc = s[i][j];
            acc = fmaf(q[i].x, k[j].x, acc);
            acc = fmaf(q[i].y, k[j].y, acc);
            acc = fmaf(q[i].z, k[j].z, acc);
            acc = fmaf(q[i].w, k[j].w, acc);
            s[i][j] = acc;
          }
        }
      }
    }
    float p[4][4];
    for (int i = 0; i < 4; ++i) {
      float tile = -CUDART_INF_F;
      for (int j = 0; j < 4; ++j) {
        const uint64_t key = k0 + tx + 16 * j;
        const bool seen = active[i] && key < visible[i] && key < cached;
        s[i][j] = seen ? s[i][j] / root : -CUDART_INF_F;
        tile = fmaxf(tile, s[i][j]);
      }
      const float next = fmaxf(maximum[i], max16(tile));
      const bool any = isfinite(next);
      const float previous = any ? (isfinite(maximum[i]) ? expf(maximum[i] - next) : 0.0f) : 1.0f;
      float sum = 0.0f;
      for (int j = 0; j < 4; ++j) {
        p[i][j] = isfinite(s[i][j]) ? expf(s[i][j] - next) : 0.0f;
        sum += p[i][j];
      }
      sum = sum16(sum);
      denominator[i] = denominator[i] * previous + sum;
      for (int e = 0; e < 8; ++e) {
        out[i][e] *= previous;
      }
      maximum[i] = next;
    }
    for (int j = 0; j < 4; ++j) {
      *reinterpret_cast<float4*>(ps + (tx + 16 * j) * kPadP + ty * 4) = make_float4(p[0][j], p[1][j], p[2][j], p[3][j]);
    }
    for (int half = 0; half < 2; ++half) {
      if (k0 + half * kHalf >= end) {
        break;
      }
      __syncthreads();
      {
        const int row = tid >> 3;
        const int part = tid & 7;
        const uint64_t key = k0 + half * kHalf + row;
        const bool pre = key < prefix;
        const uint64_t at = (pre ? (key * g.kv_heads + kh) * g.width : (((key < prefix + bl) ? bs + key - prefix : qs + key - prefix - bl) * g.kv_heads + kh) * g.width) + part * 16;
        const __half* base = pre ? vpre : vp;
        load8(base, at, key < cached, vc + row * kPadV + part * 16);
        load8(base, at + 8, key < cached, vc + row * kPadV + part * 16 + 8);
      }
      __syncthreads();
#pragma unroll 4
      for (int s2 = 0; s2 < kHalf; ++s2) {
        const float4 pr = *reinterpret_cast<const float4*>(ps + (half * kHalf + s2) * kPadP + ty * 4);
        const float4 v0 = *reinterpret_cast<const float4*>(vc + s2 * kPadV + tx * 4);
        const float4 v1 = *reinterpret_cast<const float4*>(vc + s2 * kPadV + 64 + tx * 4);
        const float pp[4] = {pr.x, pr.y, pr.z, pr.w};
        for (int i = 0; i < 4; ++i) {
          out[i][0] = fmaf(pp[i], v0.x, out[i][0]);
          out[i][1] = fmaf(pp[i], v0.y, out[i][1]);
          out[i][2] = fmaf(pp[i], v0.z, out[i][2]);
          out[i][3] = fmaf(pp[i], v0.w, out[i][3]);
          out[i][4] = fmaf(pp[i], v1.x, out[i][4]);
          out[i][5] = fmaf(pp[i], v1.y, out[i][5]);
          out[i][6] = fmaf(pp[i], v1.z, out[i][6]);
          out[i][7] = fmaf(pp[i], v1.w, out[i][7]);
        }
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    const uint64_t token = q0 + ty * 4 + i;
    if (token >= n) {
      continue;
    }
    const uint64_t base = ((qs + token) * g.heads + head) * g.width;
    const bool ok = active[i] && denominator[i] > 0.0f;
    for (int e = 0; e < 4; ++e) {
      store(a.outputs[0], base + tx * 4 + e, ok ? out[i][e] / denominator[i] : 0.0f);
      store(a.outputs[0], base + 64 + tx * 4 + e, ok ? out[i][4 + e] / denominator[i] : 0.0f);
    }
  }
}
// flash_segments_h_kernel (lane S package 5, 2026-11-18): flash_segments_kernel with binary16
// products. The query and key chunks and the value half tiles stay binary16 in shared memory;
// each 32-wide chunk of a score is summed as half2 fused multiply-adds and added to its binary32
// total, and each 32-key half tile of the weighted values is summed the same way (probabilities
// rounded to binary16) and added to the binary32 output; softmax, masks and the order of tiles as
// in flash_segments_kernel. Not bit-identical: outputs within the served path's rounding
// (checked against flash_segments_kernel by nmengn.gpu.flash_h).
namespace fsegh {
constexpr int kRows = 64;
constexpr int kKeys = 64;
constexpr int kWidth = 128;
constexpr int kChunk = 32;
constexpr int kHalf = kKeys / 2;
constexpr int kPadC = kChunk + 8;   // halves
constexpr int kPadV = kWidth + 8;   // halves
constexpr int kPadP = kRows + 4;    // floats
}  // namespace fsegh
__global__ __launch_bounds__(256) void flash_segments_h_kernel(emma_op_args a, Geometry g, const int* table) {
  using namespace fsegh;
  __shared__ __align__(16) __half qc[kRows][kPadC];
  __shared__ __align__(16) __half kc[kKeys][kPadC];
  __shared__ __align__(16) __half vc[kHalf][kPadV];
  __shared__ __align__(16) float ps[kKeys][kPadP];
  const int tid = static_cast<int>(threadIdx.x);
  const int ty = tid / 16;
  const int tx = tid % 16;
  const uint64_t qs = static_cast<uint64_t>(table[5 * blockIdx.x]);
  const uint64_t n = static_cast<uint64_t>(table[5 * blockIdx.x + 1]);
  const uint64_t q0 = static_cast<uint64_t>(table[5 * blockIdx.x + 2]);
  const uint64_t bs = static_cast<uint64_t>(table[5 * blockIdx.x + 3]);
  const uint64_t bl = static_cast<uint64_t>(table[5 * blockIdx.x + 4]);
  const uint64_t prefix = g.cached;
  const uint64_t cached = prefix + bl + n;
  const uint64_t head = blockIdx.y;
  const uint64_t kh = head / (g.heads / g.kv_heads);
  const uint64_t past = prefix + bl;
  const bool causal = a.attrs.causal != 0;
  const auto* qh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* kp = reinterpret_cast<const __half*>(a.inputs[1].data);
  const auto* vp = reinterpret_cast<const __half*>(a.inputs[2].data);
  const auto* kpre = reinterpret_cast<const __half*>(a.weights[0].data);
  const auto* vpre = reinterpret_cast<const __half*>(a.weights[1].data);
  const uint4 zero4 = make_uint4(0, 0, 0, 0);
  auto key_at = [&](uint64_t key) -> uint64_t {
    return key < prefix ? key * g.kv_heads + kh : (((key < prefix + bl) ? bs + key - prefix : qs + key - prefix - bl) * g.kv_heads + kh);
  };
  bool active[4];
  uint64_t visible[4];
  float maximum[4], denominator[4], out[4][8];
  for (int i = 0; i < 4; ++i) {
    const uint64_t token = q0 + ty * 4 + i;
    active[i] = token < n && enabled(a, g, (qs + token) * g.heads + head);
    visible[i] = causal ? past + token + 1 : cached;
    maximum[i] = -CUDART_INF_F;
    denominator[i] = 0.0f;
    for (int e = 0; e < 8; ++e) {
      out[i][e] = 0.0f;
    }
  }
  const uint64_t last_row = min(q0 + kRows, n);
  const uint64_t end = causal ? min(cached, past + last_row) : cached;
  const float root = sqrtf(static_cast<float>(g.width));
  const __half2 hz2 = __float2half2_rn(0.0f);
  for (uint64_t k0 = 0; k0 < end; k0 += kKeys) {
    float s[4][4];
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        s[i][j] = 0.0f;
      }
    }
    for (int dc = 0; dc < kWidth; dc += kChunk) {
      __syncthreads();
      {
        const int row = tid >> 2;
        const int part = tid & 3;
        const uint64_t token = q0 + row;
        *reinterpret_cast<uint4*>(&qc[row][part * 8]) =
            token < n ? *reinterpret_cast<const uint4*>(qh + ((qs + token) * g.heads + head) * g.width + dc + part * 8) : zero4;
        const uint64_t key = k0 + row;
        const __half* kb = key < prefix ? kpre : kp;
        *reinterpret_cast<uint4*>(&kc[row][part * 8]) =
            key < cached ? *reinterpret_cast<const uint4*>(kb + key_at(key) * g.width + dc + part * 8) : zero4;
      }
      __syncthreads();
      __half2 acc[4][4];
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = hz2;
        }
      }
#pragma unroll
      for (int d = 0; d < kChunk; d += 8) {
        uint4 qv[4], kv[4];
        for (int i = 0; i < 4; ++i) {
          qv[i] = *reinterpret_cast<const uint4*>(&qc[ty * 4 + i][d]);
        }
        for (int j = 0; j < 4; ++j) {
          kv[j] = *reinterpret_cast<const uint4*>(&kc[tx + 16 * j][d]);
        }
        for (int i = 0; i < 4; ++i) {
          const __half2* q2 = reinterpret_cast<const __half2*>(&qv[i]);
          for (int j = 0; j < 4; ++j) {
            const __half2* k2 = reinterpret_cast<const __half2*>(&kv[j]);
            __half2 t = acc[i][j];
            t = __hfma2(q2[0], k2[0], t);
            t = __hfma2(q2[1], k2[1], t);
            t = __hfma2(q2[2], k2[2], t);
            t = __hfma2(q2[3], k2[3], t);
            acc[i][j] = t;
          }
        }
      }
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          s[i][j] += __low2float(acc[i][j]) + __high2float(acc[i][j]);
        }
      }
    }
    float p[4][4];
    for (int i = 0; i < 4; ++i) {
      float tile = -CUDART_INF_F;
      for (int j = 0; j < 4; ++j) {
        const uint64_t key = k0 + tx + 16 * j;
        const bool seen = active[i] && key < visible[i] && key < cached;
        s[i][j] = seen ? s[i][j] / root : -CUDART_INF_F;
        tile = fmaxf(tile, s[i][j]);
      }
      const float next = fmaxf(maximum[i], fseg::max16(tile));
      const bool any = isfinite(next);
      const float previous = any ? (isfinite(maximum[i]) ? expf(maximum[i] - next) : 0.0f) : 1.0f;
      float sum = 0.0f;
      for (int j = 0; j < 4; ++j) {
        p[i][j] = isfinite(s[i][j]) ? expf(s[i][j] - next) : 0.0f;
        sum += p[i][j];
      }
      sum = fseg::sum16(sum);
      denominator[i] = denominator[i] * previous + sum;
      for (int e = 0; e < 8; ++e) {
        out[i][e] *= previous;
      }
      maximum[i] = next;
    }
    for (int j = 0; j < 4; ++j) {
      *reinterpret_cast<float4*>(&ps[tx + 16 * j][ty * 4]) = make_float4(p[0][j], p[1][j], p[2][j], p[3][j]);
    }
    for (int half = 0; half < 2; ++half) {
      if (k0 + half * kHalf >= end) {
        break;
      }
      __syncthreads();
      {
        const int row = tid >> 3;
        const int part = tid & 7;
        const uint64_t key = k0 + half * kHalf + row;
        const __half* vb = key < prefix ? vpre : vp;
        const uint64_t at = key_at(key) * g.width + part * 16;
        *reinterpret_cast<uint4*>(&vc[row][part * 16]) = key < cached ? *reinterpret_cast<const uint4*>(vb + at) : zero4;
        *reinterpret_cast<uint4*>(&vc[row][part * 16 + 8]) = key < cached ? *reinterpret_cast<const uint4*>(vb + at + 8) : zero4;
      }
      __syncthreads();
      __half2 o2[4][4];
      for (int i = 0; i < 4; ++i) {
        for (int e = 0; e < 4; ++e) {
          o2[i][e] = hz2;
        }
      }
#pragma unroll 4
      for (int s2 = 0; s2 < kHalf; ++s2) {
        const float4 pr = *reinterpret_cast<const float4*>(&ps[half * kHalf + s2][ty * 4]);
        const uint2 va = *reinterpret_cast<const uint2*>(&vc[s2][tx * 4]);
        const uint2 vb2 = *reinterpret_cast<const uint2*>(&vc[s2][64 + tx * 4]);
        const __half2* v0 = reinterpret_cast<const __half2*>(&va);
        const __half2* v1 = reinterpret_cast<const __half2*>(&vb2);
        const float pp[4] = {pr.x, pr.y, pr.z, pr.w};
        for (int i = 0; i < 4; ++i) {
          const __half2 p2 = __float2half2_rn(pp[i]);
          o2[i][0] = __hfma2(p2, v0[0], o2[i][0]);
          o2[i][1] = __hfma2(p2, v0[1], o2[i][1]);
          o2[i][2] = __hfma2(p2, v1[0], o2[i][2]);
          o2[i][3] = __hfma2(p2, v1[1], o2[i][3]);
        }
      }
      for (int i = 0; i < 4; ++i) {
        out[i][0] += __low2float(o2[i][0]);
        out[i][1] += __high2float(o2[i][0]);
        out[i][2] += __low2float(o2[i][1]);
        out[i][3] += __high2float(o2[i][1]);
        out[i][4] += __low2float(o2[i][2]);
        out[i][5] += __high2float(o2[i][2]);
        out[i][6] += __low2float(o2[i][3]);
        out[i][7] += __high2float(o2[i][3]);
      }
    }
  }
  for (int i = 0; i < 4; ++i) {
    const uint64_t token = q0 + ty * 4 + i;
    if (token >= n) {
      continue;
    }
    const uint64_t base = ((qs + token) * g.heads + head) * g.width;
    const bool ok = active[i] && denominator[i] > 0.0f;
    for (int e = 0; e < 4; ++e) {
      store(a.outputs[0], base + tx * 4 + e, ok ? out[i][e] / denominator[i] : 0.0f);
      store(a.outputs[0], base + 64 + tx * 4 + e, ok ? out[i][4 + e] / denominator[i] : 0.0f);
    }
  }
}
// part_lse_kernel (lane S package 5, 2026-11-18): the tentacles' read reduced on the card. One block
// per (work item, part): a work item is one question (packed rows `q_start` .. + `rows`) at one
// head (local), reading the shared prefix (weights[0], P rows, roped keys) and its group's body
// (packed rows `body_start` .. + `body_len`); a part is (first position, positions) in that span.
// For each question row the raw scores q.k / sqrt(width) over the part's positions are reduced to
// their log-sum-exp; the block writes the mean over the rows (`part_lse` of the processor read,
// on the card, so only part scores come back). items: 8 ints each (q_start, rows, body_start,
// body_len, head, part_first, part_count, out_first); parts: 2 ints each.
__global__ __launch_bounds__(128) void part_lse_kernel(emma_op_args a, Geometry g, const int* items, const int* parts) {
  __shared__ float qrow[128];
  __shared__ float red_m[128];
  __shared__ float red_s[128];
  const int* it = items + 8 * blockIdx.x;
  const int part_count = it[6];
  if (static_cast<int>(blockIdx.y) >= part_count) {
    return;
  }
  const uint64_t q_start = static_cast<uint64_t>(it[0]);
  const int rows = it[1];
  const uint64_t body_start = static_cast<uint64_t>(it[2]);
  const uint64_t head = static_cast<uint64_t>(it[4]);
  const int* pt = parts + 2 * (it[5] + static_cast<int>(blockIdx.y));
  const uint64_t p0 = static_cast<uint64_t>(pt[0]);
  const uint64_t plen = static_cast<uint64_t>(pt[1]);
  const uint64_t prefix = g.cached;
  const uint64_t kh = head / (g.heads / g.kv_heads);
  const auto* qh = reinterpret_cast<const __half*>(a.inputs[0].data);
  const auto* kp = reinterpret_cast<const __half*>(a.inputs[1].data);
  const auto* kpre = reinterpret_cast<const __half*>(a.weights[0].data);
  const float root = sqrtf(static_cast<float>(g.width));
  const int tid = static_cast<int>(threadIdx.x);
  float acc = 0.0f;
  for (int t = 0; t < rows; ++t) {
    __syncthreads();
    if (tid < static_cast<int>(g.width)) {
      qrow[tid] = __half2float(qh[((q_start + t) * g.heads + head) * g.width + tid]);
    }
    __syncthreads();
    float m = -CUDART_INF_F;
    float s = 0.0f;
    for (uint64_t p = p0 + tid; p < p0 + plen; p += blockDim.x) {
      const __half* kr = p < prefix ? kpre + (p * g.kv_heads + kh) * g.width : kp + ((body_start + p - prefix) * g.kv_heads + kh) * g.width;
      float dot = 0.0f;
      for (uint64_t d = 0; d < g.width; d += 2) {
        const float2 k2 = __half22float2(*reinterpret_cast<const __half2*>(kr + d));
        dot += qrow[d] * k2.x;
        dot += qrow[d + 1] * k2.y;
      }
      const float x = dot / root;
      if (x > m) {
        s = s * expf(m - x) + 1.0f;
        m = x;
      } else {
        s += expf(x - m);
      }
    }
    red_m[tid] = m;
    red_s[tid] = s;
    __syncthreads();
    for (int stride = 64; stride > 0; stride /= 2) {
      if (tid < stride) {
        const float m1 = red_m[tid];
        const float m2 = red_m[tid + stride];
        const float mx = fmaxf(m1, m2);
        float sum = 0.0f;
        if (isfinite(m1)) {
          sum += red_s[tid] * expf(m1 - mx);
        }
        if (isfinite(m2)) {
          sum += red_s[tid + stride] * expf(m2 - mx);
        }
        red_m[tid] = mx;
        red_s[tid] = sum;
      }
      __syncthreads();
    }
    if (tid == 0 && isfinite(red_m[0])) {
      acc += red_m[0] + logf(red_s[0]);
    }
  }
  if (tid == 0) {
    reinterpret_cast<float*>(a.outputs[0].data)[it[7] + static_cast<int>(blockIdx.y)] = acc / static_cast<float>(rows > 0 ? rows : 1);
  }
}
}  // namespace emma
