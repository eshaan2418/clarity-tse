#include "clarity/engine.hpp"

#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace clarity {

static std::vector<int64_t> parseShape(const std::string& s) {
  std::vector<int64_t> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stoll(tok));
  return v;
}

ModelConfig ModelConfig::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  ModelConfig c;
  std::string line;
  while (std::getline(f, line)) {
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string k = line.substr(0, eq), v = line.substr(eq + 1);
    if (k == "hop") c.hop = std::stoi(v);
    else if (k == "channels") c.channels = std::stoi(v);
    else if (k == "sr") c.sr = std::stoi(v);
    else if (k == "lookahead") c.lookahead = std::stoi(v);
    else if (k == "n_states") c.nStates = std::stoi(v);
    else if (k == "film_shape") c.filmShape = parseShape(v);
    else if (k.rfind("state_", 0) == 0) c.stateShapes.push_back(parseShape(v));
  }
  if (c.hop <= 0 || (int)c.stateShapes.size() != c.nStates) throw std::runtime_error("bad cfg " + path);
  return c;
}

static size_t numel(const std::vector<int64_t>& s) {
  return std::accumulate(s.begin(), s.end(), (size_t)1, [](size_t a, int64_t b) { return a * (size_t)b; });
}

Engine::Engine(const std::string& onnxPath, const std::string& cfgPath, int intraThreads)
    : cfg_(ModelConfig::load(cfgPath)),
      env_(ORT_LOGGING_LEVEL_WARNING, "clarity"),
      mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      limiter_((float)cfg_.sr) {
  opts_.SetIntraOpNumThreads(intraThreads);
  opts_.SetInterOpNumThreads(1);
  opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  opts_.AddConfigEntry("session.intra_op.allow_spinning", "0");
  session_ = std::make_unique<Ort::Session>(env_, onnxPath.c_str(), opts_);
  hopBuf_.assign((size_t)cfg_.channels * cfg_.hop, 0.f);
  outBuf_.assign((size_t)cfg_.hop, 0.f);
  film_.assign(numel(cfg_.filmShape), 0.f);
  for (int p = 0; p < 2; ++p)
    for (auto& s : cfg_.stateShapes) states_[p].emplace_back(numel(s), 0.f);
  buildBindings();
}

void Engine::setFilm(const std::vector<float>& film) {
  if (film.size() != film_.size()) throw std::runtime_error("film size mismatch");
  std::copy(film.begin(), film.end(), film_.begin());
}

void Engine::buildBindings() {
  keep_.clear();
  auto mk = [&](float* data, size_t n, const std::vector<int64_t>& shape) {
    keep_.push_back(Ort::Value::CreateTensor<float>(mem_, data, n, shape.data(), shape.size()));
    return keep_.size() - 1;
  };
  std::vector<int64_t> hopShape{1, cfg_.channels, cfg_.hop}, thShape{1}, outShape{1, cfg_.hop};
  keep_.reserve(4 + 4 * cfg_.nStates);
  size_t iHop = mk(hopBuf_.data(), hopBuf_.size(), hopShape);
  size_t iTh = mk(theta_.data(), 1, thShape);
  size_t iFilm = mk(film_.data(), film_.size(), cfg_.filmShape);
  size_t iOut = mk(outBuf_.data(), outBuf_.size(), outShape);
  std::vector<size_t> sIdx[2];
  for (int p = 0; p < 2; ++p)
    for (int i = 0; i < cfg_.nStates; ++i)
      sIdx[p].push_back(mk(states_[p][i].data(), states_[p][i].size(), cfg_.stateShapes[i]));
  for (int p = 0; p < 2; ++p) {
    bind_[p] = std::make_unique<Ort::IoBinding>(*session_);
    bind_[p]->BindInput("hop", keep_[iHop]);
    bind_[p]->BindInput("theta", keep_[iTh]);
    bind_[p]->BindInput("film", keep_[iFilm]);
    bind_[p]->BindOutput("out", keep_[iOut]);
    for (int i = 0; i < cfg_.nStates; ++i) {
      std::string in = "state_" + std::to_string(i), out = "new_state_" + std::to_string(i);
      bind_[p]->BindInput(in.c_str(), keep_[sIdx[p][i]]);
      bind_[p]->BindOutput(out.c_str(), keep_[sIdx[1 - p][i]]);
    }
  }
}

void Engine::process(const float* in, float* out) {
  std::copy(in, in + hopBuf_.size(), hopBuf_.begin());
  session_->Run(runOpts_, *bind_[parity_]);
  parity_ = 1 - parity_;
  std::copy(outBuf_.begin(), outBuf_.end(), out);
  limiter_.process(out, (size_t)cfg_.hop);
}

}  // namespace clarity
