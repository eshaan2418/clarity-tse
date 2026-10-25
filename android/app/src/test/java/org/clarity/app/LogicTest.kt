package org.clarity.app

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LogicTest {
    @Test fun gainIsCapped() {
        assertEquals(Safety.MAX_USER_GAIN_DB, Safety.clampGainDb(1000f))
        assertEquals(0f, Safety.clampGainDb(Float.NaN))
        for (p in -50..150) assertTrue(Safety.sliderToGainDb(p) <= Safety.MAX_USER_GAIN_DB)
    }

    @Test fun ceilingCannotExceedAbsoluteMax() {
        assertEquals(Safety.ABSOLUTE_MAX_CEILING_DBFS, Safety.clampCeilingDbfs(30f))
        assertEquals(Safety.DEFAULT_CEILING_DBFS, Safety.clampCeilingDbfs(Float.POSITIVE_INFINITY))
    }

    @Test fun audiogramParsing() {
        val a = Audiogram.parse("10, 10, 15, 25, 35, 40, 50, 200")
        assertNotNull(a)
        assertEquals(120f, a!!.levels[7])
        assertNull(Audiogram.parse("10,20"))
        assertNull(Audiogram.parse("a,b,c,d,e,f,g,h"))
        assertEquals(a, Audiogram.parse(a.serialize()))
    }

    @Test fun noticeMentionsMedicalDevice() { assertTrue(Safety.NOTICE.contains("NOT a medical device")) }
}
