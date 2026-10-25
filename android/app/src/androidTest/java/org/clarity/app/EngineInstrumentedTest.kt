package org.clarity.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.junit.Rule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class EngineInstrumentedTest {
    @get:Rule val mic: GrantPermissionRule = GrantPermissionRule.grant(android.Manifest.permission.RECORD_AUDIO)

    @Test fun engineStartsProcessesAndStops() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        fun copy(n: String) = File(ctx.filesDir, n).also { f -> ctx.assets.open(n).use { i -> f.outputStream().use { i.copyTo(it) } } }
        val rc = NativeEngine.start(copy("clarity.onnx").absolutePath, copy("clarity.cfg").absolutePath, true)
        assertEquals(0, rc)
        NativeEngine.setGainDb(100f)  // must be clamped natively
        Thread.sleep(8000)  // Oboe FullDuplexStream drains+discards ~50 callbacks before onBothStreamsReady
        val s = NativeEngine.stats()
        NativeEngine.stop()
        assertTrue("callbacks ran", s[0] > 10)
        android.util.Log.i("ClarityTest", "callbacks=${s[0]} xruns=${s[1]} meanHopMs=${s[2]} maxHopMs=${s[3]} rate=${s[4]}")
    }

    @Test fun bypassPathIsStillLimited() {
        // The A/B "off" path must also be safety-limited: covered natively by bypassLimiter_; here we check the
        // public API cannot raise gain or ceiling beyond the caps.
        assertEquals(Safety.MAX_USER_GAIN_DB, Safety.clampGainDb(1e9f))
        assertEquals(Safety.ABSOLUTE_MAX_CEILING_DBFS, Safety.clampCeilingDbfs(1e9f))
    }
}
