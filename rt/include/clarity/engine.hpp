// Streaming inference engine: hop-in / hop-out ONNX model with explicit state, followed by the
// hearing-safety limiter. process() is real-time safe after construction: all tensors are
// pre-allocated and two IoBindings (state A->B and B->A) are alternated, so the hot path performs
// no heap allocation and takes no locks.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "clarity/limiter.hpp"

namespace clarity {

struct ModelConfig {
  int hop = 0, channels = 0, sr = 16000, lookahead = 0, nStates = 0;
  std::vector<int64_t> filmShape;
  std::vector<std::vector<int64_t>> stateShapes;
  static ModelConfig load(const std::string& cfgPath);
};

class Engine {
 public:
  Engine(const std::string& onnxPath, const std::string& cfgPath, int intraThreads = 1);
  const ModelConfig& config() const { return cfg_; }
  // Set FiLM conditioning (from the *_film.onnx model or zeros = no enrollment). Not RT-safe.
  void setFilm(const std::vector<float>& film);
  // Target azimuth relative to head in degrees; NaN = unknown. RT-safe.
  void setTheta(float deg) { theta_[0] = deg; }
  // in: channels*hop samples, channel-major; out: hop samples. RT-safe.
  void process(const float* in, float* out);
  SafetyLimiter& limiter() { return limiter_; }

 private:
  ModelConfig cfg_;
  Ort::Env env_;
  Ort::SessionOptions opts_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo mem_;
  std::vector<float> hopBuf_, outBuf_, film_;
  std::array<float, 1> theta_{0.f};
  std::array<std::vector<std::vector<float>>, 2> states_;
  std::vector<Ort::Value> keep_;  // tensors referenced by the bindings
  std::array<std::unique_ptr<Ort::IoBinding>, 2> bind_;
  int parity_ = 0;
  SafetyLimiter limiter_;
  Ort::RunOptions runOpts_;
  void buildBindings();
};

}  // namespace clarity
