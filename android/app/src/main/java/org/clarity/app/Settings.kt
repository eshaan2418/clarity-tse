package org.clarity.app

import android.content.Context

/** Persisted settings (survive process death). Enrollment data never leaves app-private storage. */
class Settings(ctx: Context) {
    private val p = ctx.getSharedPreferences("clarity", Context.MODE_PRIVATE)
    var audiogram: Audiogram
        get() = Audiogram.parse(p.getString("audiogram", "") ?: "") ?: Audiogram.NORMAL
        set(v) = p.edit().putString("audiogram", v.serialize()).apply()
    var gainSlider: Int
        get() = p.getInt("gain", 40)
        set(v) = p.edit().putInt("gain", v.coerceIn(0, 100)).apply()
    var focusMode: String
        get() = p.getString("mode", "facing") ?: "facing"
        set(v) = p.edit().putString("mode", v).apply()
    var noticeAccepted: Boolean
        get() = p.getBoolean("notice", false)
        set(v) = p.edit().putBoolean("notice", v).apply()
}
