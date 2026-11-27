// Gavriel Technologies. Private host bridge implementation.
#pragma once
#ifndef EMMA_CPU_ONLY
#include <cuda_runtime.h>
#endif

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "emma_cbridg.h"
namespace emma {
struct Failure : std::runtime_error {
  emma_status status;
  Failure(emma_status value, const char* message) : std::runtime_error(message), status(value) {}
};
extern thread_local std::array<char, 1024> last_error;
inline void set_error(const char* message) noexcept {
  size_t i = 0;
  while (i + 1 < last_error.size() && message[i] != '\0') {
    last_error[i] = message[i];
    ++i;
  }
  last_error[i] = '\0';
}
inline void require(bool ok, emma_status status, const char* message) {
  if (!ok) {
    throw Failure(status, message);
  }
}
#ifndef EMMA_CPU_ONLY
inline void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) {
    // Clear the thread's last-error state so a failed call (for example opening a missing
    // device) is not reported again by the next kernel launch check.
    static_cast<void>(cudaGetLastError());
    throw Failure(EMMA_ERR_CUDA, cudaGetErrorString(status));
  }
}
struct Allocation {
  void* address = nullptr;
  uint64_t bytes = 0;
  ~Allocation() {
    if (address != nullptr) {
      (void)cudaFree(address);
    }
  }
};
struct Device {
  int ordinal = 0;
  cudaStream_t stream = nullptr;
  std::unordered_map<uint64_t, std::unique_ptr<Allocation>> buffers;
  // Buffers whose head-mask values were checked and not written since (by an upload or as an
  // operation output); their check is not repeated, which saved a device read-back and stream
  // wait per attention operation (2026-11-05).
  std::unordered_set<uint64_t> checked;
  ~Device() {
    (void)cudaSetDevice(ordinal);
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
    }
    buffers.clear();
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }
};
extern std::mutex mutex;
extern std::unordered_map<uint64_t, std::unique_ptr<Device>> devices;
extern uint64_t next_id;
Device& get_device(uint64_t id);
Allocation& get_buffer(Device& device, uint64_t id);
uint64_t tensor_bytes(const emma_tensor& tensor);
void validate_shape(int op, const emma_op_args& args);
void validate_bindings(Device& device, emma_op_args& args);
void launch(int op, const emma_op_args& args, cudaStream_t stream);
emma_status run(int op, emma_device device, const emma_op_args* args);
#endif
}  // namespace emma
namespace emma {
template <typename R>
R error_result(emma_status status) noexcept {
  if constexpr (std::is_same_v<R, emma_status>) {
    return status;
  } else if constexpr (std::is_same_v<R, uint32_t>) {
    return 0;
  } else {
    return "bridge error message unavailable";
  }
}
template <typename F>
auto guard(F&& function) noexcept -> decltype(function()) {
  using Result = decltype(function());
  try {
    if constexpr (!std::is_same_v<Result, const char*>) {
      last_error[0] = '\0';
    }
    return function();
  } catch (const Failure& e) {
    set_error(e.what());
    return error_result<Result>(e.status);
  } catch (const std::exception& e) {
    set_error(e.what());
    return error_result<Result>(EMMA_ERR_INVALID_ARG);
  } catch (...) {
    set_error("unknown bridge exception");
    return error_result<Result>(EMMA_ERR_INVALID_ARG);
  }
}
}  // namespace emma
#define EMMA_GUARD(...) return emma::guard([&]() { __VA_ARGS__ });
