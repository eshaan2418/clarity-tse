// Android audio pipeline: Oboe full-duplex (AAudio low-latency when available) -> hop FIFO ->
// clarity::Engine (ONNX Runtime) -> SafetyLimiter -> output. The callback never allocates or locks.
#include <android/log.h>
#include <jni.h>
#include <oboe/Oboe.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <vector>

#include "clarity/engine.hpp"
#include "clarity/limiter.hpp"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "clarity", __VA_ARGS__)

namespace {

class Pipeline : public oboe::FullDuplexStream, public oboe::AudioStreamErrorCallback {
 public:
  Pipeline(std::unique_ptr<clarity::Engine> eng) : eng_(std::move(eng)), bypassLimiter_(16000.f) {
    const auto& c = eng_->config();
    H_ = c.hop; C_ = c.channels;
    in_.assign((size_t)C_ * 4096, 0.f);
    out_.assign(8192, 0.f);
    hopIn_.assign((size_t)C_ * H_, 0.f);
    hopOut_.assign(H_, 0.f);
  }
  oboe::DataCallbackResult onBothStreamsReady(const void* inputData, int numIn, void* outputData, int numOut) override {
    const float* x = static_cast<const float*>(inputData);
    float* y = static_cast<float*>(outputData);
    const int inCh = getInputStream()->getChannelCount();
    const int outCh = getOutputStream()->getChannelCount();
    // de-interleave input into per-channel FIFO
    for (int i = 0; i < numIn && inLen_ < 4096; ++i, ++inLen_)
      for (int ch = 0; ch < C_; ++ch) in_[(size_t)ch * 4096 + inLen_] = x[i * inCh + std::min(ch, inCh - 1)];
    while (inLen_ >= H_ && outLen_ + H_ <= 8192) {
      for (int ch = 0; ch < C_; ++ch) std::copy_n(&in_[(size_t)ch * 4096], H_, &hopIn_[(size_t)ch * H_]);
      for (int ch = 0; ch < C_; ++ch) std::copy(&in_[(size_t)ch * 4096 + H_], &in_[(size_t)ch * 4096 + inLen_], &in_[(size_t)ch * 4096]);
      inLen_ -= H_;
      auto t0 = std::chrono::steady_clock::now();
      if (bypass_.load(std::memory_order_relaxed)) {
        std::copy_n(hopIn_.data(), H_, hopOut_.data());
        bypassLimiter_.process(hopOut_.data(), H_);  // A/B "off" is still safety-limited
      } else {
        eng_->setTheta(theta_.load(std::memory_order_relaxed));
        eng_->process(hopIn_.data(), hopOut_.data());  // includes limiter
      }
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      sumMs_ += ms; maxMs_ = std::max(maxMs_, ms); ++hops_;
      std::copy_n(hopOut_.data(), H_, &out_[outLen_]);
      outLen_ += H_;
    }
    int n = std::min(numOut, outLen_);
    if (n < numOut) xruns_.fetch_add(1, std::memory_order_relaxed);  // underflow: emit silence
    for (int i = 0; i < numOut; ++i) {
      float v = i < n ? out_[i] : 0.f;
      for (int ch = 0; ch < outCh; ++ch) y[i * outCh + ch] = v;
    }
    std::copy(out_.begin() + n, out_.begin() + outLen_, out_.begin());
    outLen_ -= n;
    callbacks_.fetch_add(1, std::memory_order_relaxed);
    return oboe::DataCallbackResult::Continue;
  }
  void onErrorAfterClose(oboe::AudioStream*, oboe::Result r) override { LOGI("stream closed: %s", oboe::convertToText(r)); }

  std::unique_ptr<clarity::Engine> eng_;
  clarity::SafetyLimiter bypassLimiter_;
  std::atomic<bool> bypass_{false};
  std::atomic<float> theta_{0.f};
  std::atomic<uint64_t> callbacks_{0}, xruns_{0};
  double sumMs_ = 0, maxMs_ = 0; uint64_t hops_ = 0;
  int H_ = 80, C_ = 2, inLen_ = 0, outLen_ = 0;
  std::vector<float> in_, out_, hopIn_, hopOut_;
};

std::mutex gMu;
std::shared_ptr<oboe::AudioStream> gIn, gOut;
std::unique_ptr<Pipeline> gPipe;

oboe::Result openStreams(bool lowLatency, int sr, int channels) {
  oboe::AudioStreamBuilder ob;
  ob.setDirection(oboe::Direction::Output)->setPerformanceMode(lowLatency ? oboe::PerformanceMode::LowLatency : oboe::PerformanceMode::None)
      ->setSharingMode(lowLatency ? oboe::SharingMode::Exclusive : oboe::SharingMode::Shared)->setFormat(oboe::AudioFormat::Float)
      ->setChannelCount(2)->setSampleRate(sr)->setSampleRateConversionQuality(oboe::SampleRateConversionQuality::Medium)
      ->setUsage(oboe::Usage::Media)->setContentType(oboe::ContentType::Speech)
      ->setDataCallback(gPipe.get())->setErrorCallback(gPipe.get());
  auto r = ob.openStream(gOut);
  if (r != oboe::Result::OK) return r;
  oboe::AudioStreamBuilder ib;
  ib.setDirection(oboe::Direction::Input)->setPerformanceMode(lowLatency ? oboe::PerformanceMode::LowLatency : oboe::PerformanceMode::None)
      ->setSharingMode(lowLatency ? oboe::SharingMode::Exclusive : oboe::SharingMode::Shared)->setFormat(oboe::AudioFormat::Float)
      ->setChannelCount(channels)->setSampleRate(sr)->setSampleRateConversionQuality(oboe::SampleRateConversionQuality::Medium)
      ->setInputPreset(oboe::InputPreset::Unprocessed)->setBufferCapacityInFrames(gOut->getBufferCapacityInFrames() * 2);
  r = ib.openStream(gIn);
  if (r != oboe::Result::OK && channels > 1) {  // no stereo capture on this route: mono, duplicated in the callback
    LOGI("stereo input unavailable (%s), falling back to mono", oboe::convertToText(r));
    ib.setChannelCount(1);
    r = ib.openStream(gIn);
  }
  if (r != oboe::Result::OK) return r;
  gPipe->setSharedInputStream(gIn);
  gPipe->setSharedOutputStream(gOut);
  return gPipe->start();
}

void closeAll() {
  if (gPipe) gPipe->stop();
  if (gIn) gIn->close();
  if (gOut) gOut->close();
  gIn.reset(); gOut.reset();
}

}  // namespace

extern "C" {
JNIEXPORT jint JNICALL Java_org_clarity_app_NativeEngine_start(JNIEnv* env, jobject, jstring model, jstring cfg, jboolean low) {
  std::lock_guard<std::mutex> lk(gMu);
  closeAll();
  const char* m = env->GetStringUTFChars(model, nullptr);
  const char* c = env->GetStringUTFChars(cfg, nullptr);
  try {
    gPipe = std::make_unique<Pipeline>(std::make_unique<clarity::Engine>(m, c, 1));
  } catch (const std::exception& e) {
    LOGI("engine: %s", e.what());
    env->ReleaseStringUTFChars(model, m); env->ReleaseStringUTFChars(cfg, c);
    return -1;
  }
  env->ReleaseStringUTFChars(model, m); env->ReleaseStringUTFChars(cfg, c);
  auto r = openStreams(low, gPipe->eng_->config().sr, gPipe->eng_->config().channels);
  if (r != oboe::Result::OK) { LOGI("open: %s", oboe::convertToText(r)); closeAll(); return -2; }
  return 0;
}
JNIEXPORT void JNICALL Java_org_clarity_app_NativeEngine_stop(JNIEnv*, jobject) { std::lock_guard<std::mutex> lk(gMu); closeAll(); }
JNIEXPORT void JNICALL Java_org_clarity_app_NativeEngine_setBypass(JNIEnv*, jobject, jboolean b) { if (gPipe) gPipe->bypass_ = b; }
JNIEXPORT void JNICALL Java_org_clarity_app_NativeEngine_setTheta(JNIEnv*, jobject, jfloat d) { if (gPipe) gPipe->theta_ = d; }
JNIEXPORT void JNICALL Java_org_clarity_app_NativeEngine_setGainDb(JNIEnv*, jobject, jfloat d) {
  if (gPipe) { gPipe->eng_->limiter().setUserGainDb(d); gPipe->bypassLimiter_.setUserGainDb(d); }
}
JNIEXPORT void JNICALL Java_org_clarity_app_NativeEngine_setCeilingDbfs(JNIEnv*, jobject, jfloat d) {
  if (gPipe) { gPipe->eng_->limiter().setCeilingDbfs(d); gPipe->bypassLimiter_.setCeilingDbfs(d); }
}
JNIEXPORT void JNICALL Java_org_clarity_app_NativeEngine_setFilm(JNIEnv* env, jobject, jfloatArray f) {
  std::lock_guard<std::mutex> lk(gMu);
  if (!gPipe) return;
  std::vector<float> v(env->GetArrayLength(f));
  env->GetFloatArrayRegion(f, 0, (jsize)v.size(), v.data());
  // brief stop/start keeps the RT callback lock-free (setFilm is not RT-safe by design)
  if (gPipe) gPipe->stop();
  try { gPipe->eng_->setFilm(v); } catch (...) {}
  gPipe->start();
}
JNIEXPORT jfloatArray JNICALL Java_org_clarity_app_NativeEngine_stats(JNIEnv* env, jobject) {
  float s[6] = {0, 0, 0, 0, 0, 0};
  if (gPipe && gOut) {
    s[0] = (float)gPipe->callbacks_.load(); s[1] = (float)gPipe->xruns_.load();
    s[2] = gPipe->hops_ ? (float)(gPipe->sumMs_ / gPipe->hops_) : 0.f; s[3] = (float)gPipe->maxMs_;
    s[4] = (float)gOut->getSampleRate(); s[5] = (float)gOut->getFramesPerBurst();
  }
  jfloatArray a = env->NewFloatArray(6);
  env->SetFloatArrayRegion(a, 0, 6, s);
  return a;
}

// Loopback latency: play a click every 500 ms for 4 s, record, take median click-to-detection delay.
JNIEXPORT jfloat JNICALL Java_org_clarity_app_NativeEngine_measureLoopbackMs(JNIEnv*, jobject, jboolean low) {
  const int sr = 48000;
  oboe::AudioStreamBuilder b;
  std::shared_ptr<oboe::AudioStream> out, in;
  b.setDirection(oboe::Direction::Output)->setPerformanceMode(low ? oboe::PerformanceMode::LowLatency : oboe::PerformanceMode::None)
      ->setFormat(oboe::AudioFormat::Float)->setChannelCount(1)->setSampleRate(sr);
  if (b.openStream(out) != oboe::Result::OK) return -1.f;
  oboe::AudioStreamBuilder ib;
  ib.setDirection(oboe::Direction::Input)->setPerformanceMode(low ? oboe::PerformanceMode::LowLatency : oboe::PerformanceMode::None)
      ->setFormat(oboe::AudioFormat::Float)->setChannelCount(1)->setSampleRate(sr)->setInputPreset(oboe::InputPreset::Unprocessed);
  if (ib.openStream(in) != oboe::Result::OK) { out->close(); return -1.f; }
  const int total = sr * 4, period = sr / 2;
  std::vector<float> play(total, 0.f), rec(total, 0.f);
  for (int k = period / 2; k < total; k += period) for (int i = 0; i < 48; ++i) play[k + i] = 0.25f * (i % 2 ? -1.f : 1.f);  // safe level
  clarity::SafetyLimiter lim((float)sr);
  lim.process(play.data(), play.size());
  in->requestStart(); out->requestStart();
  int w = 0, r = 0;
  while (w < total || r < total) {
    if (w < total) { auto res = out->write(play.data() + w, std::min(192, total - w), 100000000); if (res) w += res.value(); }
    if (r < total) { auto res = in->read(rec.data() + r, std::min(192, total - r), 0); if (res) r += res.value(); }
  }
  in->requestStop(); out->requestStop(); in->close(); out->close();
  std::vector<float> d;
  for (int k = period / 2; k + period <= total; k += period) {
    int best = -1; float bv = 0.f;
    for (int i = k; i < k + period && i < total; ++i) if (std::fabs(rec[i]) > bv) { bv = std::fabs(rec[i]); best = i; }
    if (best > 0 && bv > 0.01f) d.push_back(1000.f * (best - k) / sr);
  }
  if (d.empty()) return -1.f;
  std::sort(d.begin(), d.end());
  return d[d.size() / 2];
}
}
