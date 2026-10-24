// Single-producer single-consumer lock-free ring buffer (wait-free, no allocation after init).
#pragma once
#include <atomic>
#include <cstddef>
#include <vector>

namespace clarity {

template <typename T>
class SpscRing {
 public:
  explicit SpscRing(std::size_t capacityPow2) : buf_(capacityPow2), mask_(capacityPow2 - 1) {}
  std::size_t capacity() const { return buf_.size(); }
  std::size_t readAvailable() const {
    return head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_relaxed);
  }
  std::size_t writeAvailable() const {
    return capacity() - (head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_acquire));
  }
  bool write(const T* src, std::size_t n) {
    if (writeAvailable() < n) return false;
    std::size_t h = head_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < n; ++i) buf_[(h + i) & mask_] = src[i];
    head_.store(h + n, std::memory_order_release);
    return true;
  }
  bool read(T* dst, std::size_t n) {
    if (readAvailable() < n) return false;
    std::size_t t = tail_.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < n; ++i) dst[i] = buf_[(t + i) & mask_];
    tail_.store(t + n, std::memory_order_release);
    return true;
  }

 private:
  std::vector<T> buf_;
  std::size_t mask_;
  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace clarity
