// Hearing-safety output stage (mirrors clarity_tse/dsp/limiter.py).
// The ceiling is re-clamped on every block and a hard clip follows the smooth limiter, so no
// configuration or input (incl. NaN/Inf) can produce |y| > kAbsoluteMaxCeilingDbfs.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace clarity {

constexpr float kAbsoluteMaxCeilingDbfs = -1.0f;
constexpr float kDefaultCeilingDbfs = -12.0f;
constexpr float kMaxUserGainDb = 20.0f;

inline float dbToLin(float db) { return std::pow(10.0f, db / 20.0f); }

class SafetyLimiter {
 public:
  explicit SafetyLimiter(float sampleRate = 16000.f, float ceilingDbfs = kDefaultCeilingDbfs,
                         float userGainDb = 0.f)
      : sr_(sampleRate) {
    setCeilingDbfs(ceilingDbfs);
    setUserGainDb(userGainDb);
    attack_ = std::exp(-1.f / std::max(1.f, 0.5e-3f * sr_));
    release_ = std::exp(-1.f / std::max(1.f, 80e-3f * sr_));
  }
  void setCeilingDbfs(float db) { ceilingDbfs_ = std::isfinite(db) ? db : kDefaultCeilingDbfs; }
  void setUserGainDb(float db) {
    if (!std::isfinite(db)) db = std::isnan(db) ? 0.f : (db > 0 ? kMaxUserGainDb : -60.f);
    userGainDb_ = std::clamp(db, -60.f, kMaxUserGainDb);
  }
  float ceiling() const { return dbToLin(std::min(ceilingDbfs_, kAbsoluteMaxCeilingDbfs)); }

  // In-place, real-time safe (no allocation, no locks).
  void process(float* x, std::size_t n) {
    const float c = ceiling();
    const float g = dbToLin(std::clamp(userGainDb_, -60.f, kMaxUserGainDb));
    for (std::size_t i = 0; i < n; ++i) {
      float v = std::isfinite(x[i]) ? x[i] * g : 0.f;
      float peak = std::fabs(v);
      float target = peak <= c ? 1.f : c / peak;
      float coef = target < env_ ? attack_ : release_;
      env_ = coef * env_ + (1.f - coef) * target;
      v *= std::min(env_, target);
      x[i] = std::clamp(v, -c, c);
    }
  }

 private:
  float sr_;
  float ceilingDbfs_ = kDefaultCeilingDbfs;
  float userGainDb_ = 0.f;
  float attack_ = 0.f, release_ = 0.f, env_ = 1.f;
};

}  // namespace clarity
