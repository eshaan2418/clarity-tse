// clarity_bench: per-hop compute latency distribution of the streaming engine (no audio device),
// optionally under CPU contention (N busy threads). Output: JSON.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "clarity/engine.hpp"
#include "stats.hpp"

using namespace clarity;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  std::string model, json;
  double seconds = 60;
  int contention = 0, threads = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model") model = argv[++i];
    else if (a == "--seconds") seconds = std::stod(argv[++i]);
    else if (a == "--minutes") seconds = 60 * std::stod(argv[++i]);
    else if (a == "--contention") contention = std::stoi(argv[++i]);
    else if (a == "--threads") threads = std::stoi(argv[++i]);
    else if (a == "--json") json = argv[++i];
  }
  if (model.empty()) { std::fprintf(stderr, "usage: clarity_bench --model m.onnx [--seconds S] [--contention N]\n"); return 2; }
  std::string cfg = model.substr(0, model.size() - 5) + ".cfg";
  Engine eng(model, cfg, threads);
  const auto& c = eng.config();
  const double hopMs = 1000.0 * c.hop / c.sr;
  const size_t nHops = (size_t)(seconds * c.sr / c.hop);
  std::atomic<bool> stop{false};
  std::vector<std::thread> burners;
  for (int i = 0; i < contention; ++i)
    burners.emplace_back([&] { volatile double x = 1; while (!stop.load(std::memory_order_relaxed)) x = x * 1.0000001 + 1e-9; });
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.f, 0.05f);
  std::vector<float> in((size_t)c.channels * c.hop), out(c.hop);
  for (int w = 0; w < 200; ++w) { for (auto& v : in) v = nd(rng); eng.process(in.data(), out.data()); }
  LatencyRecorder rec(nHops);
  for (size_t k = 0; k < nHops; ++k) {
    for (auto& v : in) v = nd(rng);
    auto t0 = Clock::now();
    eng.process(in.data(), out.data());
    rec.push(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  stop = true;
  for (auto& t : burners) t.join();
  char head[256];
  std::snprintf(head, sizeof head, "{\"model\": \"%s\", \"hop\": %d, \"hop_ms\": %.4f, \"audio_seconds\": %.1f, \"contention_threads\": %d, \"ort_threads\": %d, ",
                model.c_str(), c.hop, hopMs, seconds, contention, threads);
  std::string s = std::string(head) + rec.json("compute", hopMs) + "}\n";
  std::fputs(s.c_str(), stdout);
  if (!json.empty()) { FILE* f = std::fopen(json.c_str(), "w"); if (f) { std::fputs(s.c_str(), f); std::fclose(f); } }
  return 0;
}
