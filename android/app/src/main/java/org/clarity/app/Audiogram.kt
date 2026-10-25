package org.clarity.app

/** User-entered audiogram (dB HL) at 250, 500, 1k, 2k, 3k, 4k, 6k, 8k Hz. Not checked or diagnosed. */
data class Audiogram(val levels: FloatArray) {
    init { require(levels.size == FREQS.size) }

    fun serialize(): String = levels.joinToString(",")

    companion object {
        val FREQS = intArrayOf(250, 500, 1000, 2000, 3000, 4000, 6000, 8000)
        val NORMAL = Audiogram(FloatArray(8))

        /** Parses "10,10,15,..." ; returns null on wrong count / non-numeric. Values clipped to -10..120. */
        fun parse(text: String): Audiogram? {
            val parts = text.split(",", ";", " ").map { it.trim() }.filter { it.isNotEmpty() }
            if (parts.size != FREQS.size) return null
            val v = parts.map { it.toFloatOrNull() ?: return null }
            if (v.any { !it.isFinite() }) return null
            return Audiogram(v.map { it.coerceIn(-10f, 120f) }.toFloatArray())
        }
    }

    override fun equals(other: Any?) = other is Audiogram && other.levels.contentEquals(levels)
    override fun hashCode() = levels.contentHashCode()
}
