package org.clarity.app

/** Mirrors rt/include/clarity/limiter.hpp. The native limiter re-clamps anyway; this keeps the UI honest. */
object Safety {
    const val ABSOLUTE_MAX_CEILING_DBFS = -1.0f
    const val DEFAULT_CEILING_DBFS = -12.0f
    const val MAX_USER_GAIN_DB = 20.0f
    const val MIN_USER_GAIN_DB = -60.0f

    fun clampGainDb(db: Float): Float =
        if (db.isNaN()) 0f else db.coerceIn(MIN_USER_GAIN_DB, MAX_USER_GAIN_DB)

    fun clampCeilingDbfs(db: Float): Float =
        if (!db.isFinite()) DEFAULT_CEILING_DBFS else minOf(db, ABSOLUTE_MAX_CEILING_DBFS)

    /** Slider position 0..100 -> gain in dB, conservative default at 40 (= -6 dB). */
    fun sliderToGainDb(pos: Int): Float = clampGainDb(-30f + 0.5f * pos.coerceIn(0, 100))

    const val NOTICE = "Research prototype - NOT a medical device. It does not diagnose, treat, or replace " +
        "hearing aids. Keep the volume low; stop if anything is uncomfortable. Audio never leaves this phone."
}
