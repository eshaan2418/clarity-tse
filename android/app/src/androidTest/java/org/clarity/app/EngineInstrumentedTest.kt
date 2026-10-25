package org.clarity.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class EngineInstrumentedTest {
    @Test fun engineStartsProcessesAndStops() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        fun copy(n: String) = File(ctx.filesDir, n).also { f -> ctx.assets.open(n).use { i -> f.outputStream().use { i.copyTo(it) } } }
        val rc = NativeEngine.start(copy("clarity.onnx").absolutePath, copy("clarity.cfg").absolutePath, true)
        assertEquals(0, rc)
        NativeEngine.setGainDb(100f)  // must be clamped natively
        Thread.sleep(2000)
        val s = NativeEngine.stats()
        NativeEngine.stop()
        assertTrue("callbacks ran", s[0] > 10)
    }
}
