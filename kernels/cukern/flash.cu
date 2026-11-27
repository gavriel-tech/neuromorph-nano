// Gavriel Technologies. sm_75 tiled attention for one causal sequence over a cached prefix (the
// joint read's encoding and its label tails). A block holds 64 query rows of one head and walks
// the keys in tiles of 64: the scores as a 64 by 64 product with a 4 by 4 register tile per thread
// (head dimensions streamed through shared memory 32 at a time), an online softmax per row, then
// the weighted values with a 4 rows by 8 dimensions register tile per thread. Float accumulation
// with explicit fused multiply-adds, binary16 in and out, the causal and head-mask rules of
// prefill_kernel; a different summation order, so outputs match prefill2_kernel within rounding,
// not bit for bit. Launched only for binary16 inputs of head width 128 (the host checks).
#include "../cbridg/src/internal.hpp"
#include "attention.cuh"
namespace emma {
namespace {
constexpr int kRows = 64;
constexpr int kKeys = 64;
constexpr int kWidth = 128;
constexpr int kChunk = 32;
constexpr int kHalf = kKeys / 2;
constexpr int kPadC = kChunk + 4;
constexpr int kPadV = kWidth + 4;
constexpr int kPadP = kRows + 4;
// Shared floats: the query and key chunks (the value half tile reuses them), then the
// probabilities, keys by rows.
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
// Eight binary16 values from 16 aligned bytes at element `at` into two float4 slots (zeros when
// the row is outside the tensor).
__device__ inline void load8(const emma_tensor& t, uint64_t at, bool in, float* dst) {
  float4 lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float4 hi = lo;
  if (in) {
    const uint4 raw = *reinterpret_cast<const uint4*>(reinterpret_cast<const __half*>(t.data) + at);
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
}  // namespace
__global__ __launch_bounds__(256) void flash_kernel(emma_op_args a, Geometry g) {
  __shared__ __align__(16) float smem[kShared];
  float* qc = smem;
  float* kc = smem + kRows * kPadC;
  float* vc = smem;
  float* ps = smem + kFirst;
  const int tid = static_cast<int>(threadIdx.x);
  const int ty = tid / 16;
  const int tx = tid % 16;
  const uint64_t blocks = (g.tokens + kRows - 1) / kRows;
  // The last query rows read the most keys: their blocks start first.
  const uint64_t q0 = (blocks - 1 - blockIdx.x) * kRows;
  const uint64_t head = blockIdx.y;
  const uint64_t kh = head / (g.heads / g.kv_heads);
  const uint64_t past = g.cached - g.tokens;
  const bool causal = a.attrs.causal != 0;
  bool active[4];
  uint64_t visible[4];
  float maximum[4], denominator[4], out[4][8];
  for (int i = 0; i < 4; ++i) {
    const uint64_t token = q0 + ty * 4 + i;
    active[i] = token < g.tokens && enabled(a, g, token * g.heads + head);
    visible[i] = causal ? past + token + 1 : g.cached;
    maximum[i] = -CUDART_INF_F;
    denominator[i] = 0.0f;
    for (int e = 0; e < 8; ++e) {
      out[i][e] = 0.0f;
    }
  }
  const uint64_t last_row = min(q0 + kRows, g.tokens);
  const uint64_t end = causal ? min(g.cached, past + last_row) : g.cached;
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
        load8(a.inputs[0], (token * g.heads + head) * g.width + dc + part * 8, token < g.tokens, qc + row * kPadC + part * 8);
        const uint64_t key = k0 + row;
        load8(a.inputs[1], (key * g.kv_heads + kh) * g.width + dc + part * 8, key < g.cached, kc + row * kPadC + part * 8);
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
    // Online softmax per row: the 16 threads of a row hold its 64 scores, 4 each.
    float p[4][4];
    for (int i = 0; i < 4; ++i) {
      float tile = -CUDART_INF_F;
      for (int j = 0; j < 4; ++j) {
        const uint64_t key = k0 + tx + 16 * j;
        const bool seen = active[i] && key < visible[i] && key < g.cached;
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
        const uint64_t at = (key * g.kv_heads + kh) * g.width + part * 16;
        load8(a.inputs[2], at, key < g.cached, vc + row * kPadV + part * 16);
        load8(a.inputs[2], at + 8, key < g.cached, vc + row * kPadV + part * 16 + 8);
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
    if (token >= g.tokens) {
      continue;
    }
    const uint64_t base = (token * g.heads + head) * g.width;
    const bool ok = active[i] && denominator[i] > 0.0f;
    for (int e = 0; e < 4; ++e) {
      store(a.outputs[0], base + tx * 4 + e, ok ? out[i][e] / denominator[i] : 0.0f);
      store(a.outputs[0], base + 64 + tx * 4 + e, ok ? out[i][4 + e] / denominator[i] : 0.0f);
    }
  }
}
}  // namespace emma
