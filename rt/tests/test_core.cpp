#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include "clarity/limiter.hpp"
#include "clarity/ring_buffer.hpp"

using namespace clarity;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static void testLimiterCannotBeBypassed() {
  const float ceilings[] = {-12.f, 0.f, 20.f, std::numeric_limits<float>::quiet_NaN(),
                            std::numeric_limits<float>::infinity()};
  const float gains[] = {0.f, 100.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.f, 10.f);
  for (float c : ceilings)
    for (float g : gains) {
      SafetyLimiter lim(16000.f, c, g);
      lim.setCeilingDbfs(c);  // post-construction mutation must not bypass either
      std::vector<float> x(20000);
      for (auto& v : x) v = nd(rng);
      x[100] = 1e30f; x[200] = std::numeric_limits<float>::infinity(); x[300] = std::numeric_limits<float>::quiet_NaN();
      lim.process(x.data(), x.size());
      float mx = 0.f;
      for (float v : x) { CHECK(std::isfinite(v)); mx = std::max(mx, std::fabs(v)); }
      CHECK(mx <= dbToLin(kAbsoluteMaxCeilingDbfs) + 1e-6f);
    }
}

static void testRingSpsc() {
  SpscRing<float> r(1024);
  const int N = 200000;
  std::thread prod([&] { float v = 0; for (int i = 0; i < N;) { if (r.write(&v, 1)) { v += 1; ++i; } } });
  float expect = 0; int got = 0;
  while (got < N) { float v; if (r.read(&v, 1)) { CHECK(v == expect); expect += 1; ++got; } }
  prod.join();
}

int main() {
  testLimiterCannotBeBypassed();
  testRingSpsc();
  if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
  std::printf("rt core tests passed\n");
  return 0;
}
