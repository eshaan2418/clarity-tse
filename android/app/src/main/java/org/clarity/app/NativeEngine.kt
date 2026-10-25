package org.clarity.app

/** JNI bridge to the C++ pipeline (Oboe full-duplex + ONNX Runtime engine + safety limiter). */
object NativeEngine {
    init { System.loadLibrary("clarity_jni") }

    /** Returns 0 on success. lowLatency=true requests the AAudio low-latency (exclusive) path. */
    external fun start(modelPath: String, cfgPath: String, lowLatency: Boolean): Int
    external fun stop()
    external fun setBypass(bypass: Boolean)          // A/B: true = unprocessed (still limited)
    external fun setTheta(deg: Float)                // NaN = unknown direction
    external fun setFilm(film: FloatArray)           // conditioning from enrollment (zeros = none)
    external fun setGainDb(db: Float)
    external fun setCeilingDbfs(db: Float)
    external fun stats(): FloatArray                 // [callbacks, xruns, meanHopMs, maxHopMs, sampleRate, framesPerBurst]
    /** Plays clicks and records them; returns the median round-trip delay in ms, or -1. */
    external fun measureLoopbackMs(lowLatency: Boolean): Float
}
