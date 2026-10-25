#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace clarity {

// Fixed-capacity latency recorder (pre-allocated; push() is RT-safe).
class LatencyRecorder {
 public:
  explicit LatencyRecorder(size_t cap) { v_.reserve(cap); }
  void push(double ms) { if (v_.size() < v_.capacity()) v_.push_back(ms); else ++dropped_; }
  size_t size() const { return v_.size(); }
  std::string json(const char* name, double deadlineMs) const {
    std::vector<double> s(v_);
    std::sort(s.begin(), s.end());
    auto q = [&](double p) { return s.empty() ? NAN : s[std::min(s.size() - 1, (size_t)(p * (s.size() - 1)))]; };
    double mean = 0; for (double x : s) mean += x; mean = s.empty() ? NAN : mean / s.size();
    size_t miss = std::count_if(s.begin(), s.end(), [&](double x) { return x > deadlineMs; });
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "\"%s\": {\"n\": %zu, \"mean_ms\": %.5f, \"p50_ms\": %.5f, \"p90_ms\": %.5f, \"p99_ms\": %.5f, "
                  "\"p999_ms\": %.5f, \"max_ms\": %.5f, \"deadline_ms\": %.4f, \"deadline_misses\": %zu, \"dropped\": %zu}",
                  name, s.size(), mean, q(0.5), q(0.9), q(0.99), q(0.999), s.empty() ? NAN : s.back(), deadlineMs, miss,
                  dropped_);
    return buf;
  }

 private:
  std::vector<double> v_;
  size_t dropped_ = 0;
};

}  // namespace clarity
