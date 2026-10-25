// clarity_rt: run the engine on a real audio clock for N minutes and count glitches.
//   macOS: CoreAudio default output unit (client format 16 kHz mono float; HAL does SRC).
//   other: simulated clock thread (sleep_until per buffer period) — labelled "simclock".
// Input is file-injected (interleaved float32 16 kHz, channels per model cfg) and looped;
// output is written to the device, or zeros with --mute (processing is identical).
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "clarity/engine.hpp"
#include "stats.hpp"

#ifdef __APPLE__
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#endif

using namespace clarity;
using Clock = std::chrono::steady_clock;

struct Ctx {
  Engine* eng = nullptr;
  std::vector<float> input;  // interleaved C x T
  size_t inPos = 0;
  int C = 2, H = 80, sr = 16000;
  std::vector<float> hopIn, hopOut, fifo;  // fifo: processed samples not yet delivered
  size_t fifoLen = 0;
  bool mute = true;
  LatencyRecorder* hopRec = nullptr;
  LatencyRecorder* cbRec = nullptr;
  std::atomic<uint64_t> callbacks{0}, hops{0}, lateCallbacks{0}, overloads{0}, underflows{0};
  double sumCbBudgetMs = 0;
};

static void produceHop(Ctx& c) {
  const size_t T = c.input.size() / c.C;
  for (int h = 0; h < c.H; ++h) {
    for (int ch = 0; ch < c.C; ++ch) c.hopIn[(size_t)ch * c.H + h] = c.input[c.inPos * c.C + ch];
    c.inPos = (c.inPos + 1) % T;
  }
  auto t0 = Clock::now();
  c.eng->process(c.hopIn.data(), c.hopOut.data());
  c.hopRec->push(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  std::memcpy(c.fifo.data() + c.fifoLen, c.hopOut.data(), sizeof(float) * c.H);
  c.fifoLen += c.H;
  c.hops.fetch_add(1, std::memory_order_relaxed);
}

// The real-time callback: no allocation, no locks, no I/O.
static void renderBlock(Ctx& c, float* out, uint32_t n) {
  auto t0 = Clock::now();
  while (c.fifoLen < n) produceHop(c);
  if (c.mute) std::memset(out, 0, sizeof(float) * n);
  else std::memcpy(out, c.fifo.data(), sizeof(float) * n);
  std::memmove(c.fifo.data(), c.fifo.data() + n, sizeof(float) * (c.fifoLen - n));
  c.fifoLen -= n;
  double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  double budget = 1000.0 * n / c.sr;
  c.cbRec->push(ms);
  if (ms > budget) c.lateCallbacks.fetch_add(1, std::memory_order_relaxed);
  c.callbacks.fetch_add(1, std::memory_order_relaxed);
}

#ifdef __APPLE__
static OSStatus caRender(void* ref, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32 n,
                         AudioBufferList* io) {
  auto& c = *static_cast<Ctx*>(ref);
  renderBlock(c, static_cast<float*>(io->mBuffers[0].mData), n);
  return noErr;
}
static OSStatus overloadListener(AudioObjectID, UInt32, const AudioObjectPropertyAddress*, void* ref) {
  static_cast<Ctx*>(ref)->overloads.fetch_add(1);
  return noErr;
}
#endif

int main(int argc, char** argv) {
  std::string model, json, inputPath, clockName;
  double minutes = 1;
  bool mute = false, simclock = false;
  int bufferFrames = 256;  // device frames per callback request
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model") model = argv[++i];
    else if (a == "--minutes") minutes = std::stod(argv[++i]);
    else if (a == "--json") json = argv[++i];
    else if (a == "--input") inputPath = argv[++i];
    else if (a == "--mute") mute = true;
    else if (a == "--simclock") simclock = true;
    else if (a == "--buffer") bufferFrames = std::stoi(argv[++i]);
  }
  if (model.empty()) { std::fprintf(stderr, "usage: clarity_rt --model m.onnx [--minutes M] [--mute] [--input x.f32]\n"); return 2; }
  Engine eng(model, model.substr(0, model.size() - 5) + ".cfg", 1);
  Ctx c;
  c.eng = &eng;
  c.C = eng.config().channels;
  c.H = eng.config().hop;
  c.sr = eng.config().sr;
  c.mute = mute;
  if (!inputPath.empty()) {
    std::ifstream f(inputPath, std::ios::binary | std::ios::ate);
    size_t bytes = (size_t)f.tellg();
    f.seekg(0);
    c.input.resize(bytes / 4);
    f.read(reinterpret_cast<char*>(c.input.data()), (std::streamsize)bytes);
  } else {
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 0.05f);
    c.input.resize((size_t)c.C * c.sr * 10);
    for (auto& v : c.input) v = nd(rng);
  }
  c.hopIn.assign((size_t)c.C * c.H, 0.f);
  c.hopOut.assign(c.H, 0.f);
  c.fifo.assign(8192 + 4 * c.H, 0.f);
  const size_t expectHops = (size_t)(minutes * 60 * c.sr / c.H) + 1000;
  LatencyRecorder hopRec(expectHops), cbRec(expectHops);
  c.hopRec = &hopRec;
  c.cbRec = &cbRec;
  double deviceRate = 0;
  uint32_t deviceBuffer = 0;
  auto start = Clock::now();
#ifdef __APPLE__
  if (!simclock) {
    clockName = "coreaudio";
    AudioComponentDescription d{kAudioUnitType_Output, kAudioUnitSubType_DefaultOutput, kAudioUnitManufacturer_Apple, 0, 0};
    AudioComponent comp = AudioComponentFindNext(nullptr, &d);
    AudioUnit unit;
    if (AudioComponentInstanceNew(comp, &unit) != noErr) { std::fprintf(stderr, "no output unit\n"); return 1; }
    AudioStreamBasicDescription fmt{};
    fmt.mSampleRate = c.sr;
    fmt.mFormatID = kAudioFormatLinearPCM;
    fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
    fmt.mBitsPerChannel = 32;
    fmt.mChannelsPerFrame = 1;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerFrame = 4;
    fmt.mBytesPerPacket = 4;
    AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &fmt, sizeof fmt);
    AURenderCallbackStruct cb{caRender, &c};
    AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb, sizeof cb);
    AudioDeviceID dev = 0;
    UInt32 sz = sizeof dev;
    AudioObjectPropertyAddress addr{kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, nullptr, &sz, &dev);
    UInt32 bf = (UInt32)bufferFrames;
    AudioObjectPropertyAddress bfa{kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioObjectSetPropertyData(dev, &bfa, 0, nullptr, sizeof bf, &bf);
    sz = sizeof bf;
    AudioObjectGetPropertyData(dev, &bfa, 0, nullptr, &sz, &bf);
    deviceBuffer = bf;
    Float64 rate = 0;
    sz = sizeof rate;
    AudioObjectPropertyAddress ra{kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioObjectGetPropertyData(dev, &ra, 0, nullptr, &sz, &rate);
    deviceRate = rate;
    AudioObjectPropertyAddress ov{kAudioDeviceProcessorOverload, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioObjectAddPropertyListener(dev, &ov, overloadListener, &c);
    AudioUnitInitialize(unit);
    start = Clock::now();
    AudioOutputUnitStart(unit);
    std::this_thread::sleep_for(std::chrono::duration<double>(minutes * 60));
    AudioOutputUnitStop(unit);
    AudioObjectRemovePropertyListener(dev, &ov, overloadListener, &c);
    AudioUnitUninitialize(unit);
    AudioComponentInstanceDispose(unit);
  } else
#endif
  {
    clockName = "simclock";
    const uint32_t n = (uint32_t)std::lround(bufferFrames * (double)c.sr / 48000.0);
    std::vector<float> out(n);
    auto period = std::chrono::duration<double>((double)n / c.sr);
    auto next = Clock::now();
    start = next;
    auto end = start + std::chrono::duration<double>(minutes * 60);
    while (Clock::now() < end) {
      renderBlock(c, out.data(), n);
      next += std::chrono::duration_cast<Clock::duration>(period);
      if (Clock::now() > next) c.underflows.fetch_add(1);
      std::this_thread::sleep_until(next);
    }
  }
  double wall = std::chrono::duration<double>(Clock::now() - start).count();
  const double hopMs = 1000.0 * c.H / c.sr;
  char head[768];
  std::snprintf(head, sizeof head,
                "{\"model\": \"%s\", \"clock\": \"%s\", \"minutes_requested\": %.2f, \"wall_seconds\": %.1f, "
                "\"device_rate\": %.0f, \"device_buffer_frames\": %u, \"mute\": %s, \"hop\": %d, \"callbacks\": %llu, "
                "\"hops\": %llu, \"late_callbacks\": %llu, \"os_overloads\": %llu, \"simclock_underflows\": %llu, "
                "\"xruns_total\": %llu, ",
                model.c_str(), clockName.c_str(), minutes, wall, deviceRate, deviceBuffer, mute ? "true" : "false", c.H,
                (unsigned long long)c.callbacks.load(), (unsigned long long)c.hops.load(),
                (unsigned long long)c.lateCallbacks.load(), (unsigned long long)c.overloads.load(),
                (unsigned long long)c.underflows.load(),
                (unsigned long long)(c.lateCallbacks.load() + c.overloads.load() + c.underflows.load()));
  std::string s = std::string(head) + hopRec.json("hop_compute", hopMs) + ", " +
                  cbRec.json("callback", 1000.0 * deviceBuffer / (deviceRate > 0 ? deviceRate : 48000.0)) + "}\n";
  std::fputs(s.c_str(), stdout);
  if (!json.empty()) { FILE* f = std::fopen(json.c_str(), "w"); if (f) { std::fputs(s.c_str(), f); std::fclose(f); } }
  return 0;
}
