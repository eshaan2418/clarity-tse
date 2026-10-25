package org.clarity.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.annotation.SuppressLint
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.nio.FloatBuffer
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {
    private lateinit var settings: Settings
    private lateinit var status: TextView
    private lateinit var audio: AudioManager
    private var running = false
    private var lowLatency = true
    private var film: FloatArray? = null
    private var focusRequest: AudioFocusRequest? = null
    private val ui = Handler(Looper.getMainLooper())

    private val routeCallback = object : AudioDeviceCallback() {
        override fun onAudioDevicesAdded(added: Array<out AudioDeviceInfo>) = onRouteChanged()
        override fun onAudioDevicesRemoved(removed: Array<out AudioDeviceInfo>) = onRouteChanged()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = Settings(this)
        audio = getSystemService(AUDIO_SERVICE) as AudioManager
        setContentView(buildUi())
        if (!settings.noticeAccepted) {
            AlertDialog.Builder(this).setTitle("Not a medical device").setMessage(Safety.NOTICE)
                .setCancelable(false).setPositiveButton("I understand") { _, _ -> settings.noticeAccepted = true }.show()
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }
        audio.registerAudioDeviceCallback(routeCallback, ui)
        if (intent?.getBooleanExtra("measure_latency", false) == true) measureLatency()  // `make android-latency`
    }

    private fun buildUi(): ScrollView {
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 32, 32, 32) }
        col.addView(TextView(this).apply { text = Safety.NOTICE; textSize = 13f })
        status = TextView(this).apply { text = "stopped"; textSize = 15f }
        col.addView(status)
        col.addView(Button(this).apply {
            text = "Start / Stop"
            setOnClickListener { if (running) stopEngine() else startEngine() }
        })
        col.addView(Switch(this).apply {
            text = "Clarity processing (off = unprocessed A/B)"
            isChecked = true
            setOnCheckedChangeListener { _, on -> NativeEngine.setBypass(!on) }
        })
        col.addView(Switch(this).apply {
            text = "Low-latency route (wired / AAudio exclusive)"
            isChecked = true
            setOnCheckedChangeListener { _, on -> lowLatency = on; if (running) { stopEngine(); startEngine() } }
        })
        val modes = RadioGroup(this)
        val facing = RadioButton(this).apply { text = "Focus on who I'm facing"; id = 1 }
        val voice = RadioButton(this).apply { text = "Focus on this voice (enrolled)"; id = 2 }
        modes.addView(facing); modes.addView(voice)
        modes.check(if (settings.focusMode == "voice") 2 else 1)
        modes.setOnCheckedChangeListener { _, id -> settings.focusMode = if (id == 2) "voice" else "facing"; applyMode() }
        col.addView(modes)
        col.addView(TextView(this).apply { text = "Volume (hard-capped; limiter always on)" })
        col.addView(SeekBar(this).apply {
            max = 100; progress = settings.gainSlider
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    settings.gainSlider = p; NativeEngine.setGainDb(Safety.sliderToGainDb(p))
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        })
        col.addView(Button(this).apply { text = "Enroll a voice (15 s, with consent)"; setOnClickListener { enroll() } })
        val ag = EditText(this).apply { hint = "Audiogram dB HL @250,500,1k,2k,3k,4k,6k,8k"; setText(settings.audiogram.serialize()) }
        col.addView(ag)
        col.addView(Button(this).apply {
            text = "Save audiogram"
            setOnClickListener {
                val a = Audiogram.parse(ag.text.toString())
                if (a == null) toastStatus("Audiogram needs 8 numbers") else { settings.audiogram = a; toastStatus("Audiogram saved (not validated clinically)") }
            }
        })
        col.addView(Button(this).apply { text = "Measure system latency (loopback)"; setOnClickListener { measureLatency() } })
        return ScrollView(this).apply { addView(col) }
    }

    private fun assetToFile(name: String): File? = try {
        val f = File(filesDir, name)
        if (!f.exists()) assets.open(name).use { i -> f.outputStream().use { o -> i.copyTo(o) } }
        f
    } catch (e: Exception) { null }

    private fun requestFocus(): Boolean {
        val attrs = AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()
        val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN).setAudioAttributes(attrs)
            .setOnAudioFocusChangeListener { change ->
                // calls / other apps: stop cleanly; user restarts explicitly (no surprise loud audio)
                if (change == AudioManager.AUDIOFOCUS_LOSS || change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT) stopEngine()
            }.build()
        focusRequest = req
        return audio.requestAudioFocus(req) == AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }

    private fun startEngine() {
        if (!requestFocus()) { toastStatus("audio focus denied"); return }
        val model = assetToFile("clarity.onnx"); val cfg = assetToFile("clarity.cfg")
        if (model == null || cfg == null) { toastStatus("model assets missing (run make android-assets)"); return }
        val rc = NativeEngine.start(model.absolutePath, cfg.absolutePath, lowLatency)
        if (rc != 0) { toastStatus("engine start failed: $rc"); return }
        running = true
        NativeEngine.setGainDb(Safety.sliderToGainDb(settings.gainSlider))
        NativeEngine.setCeilingDbfs(Safety.DEFAULT_CEILING_DBFS)
        applyMode()
        pollStats()
    }

    private fun stopEngine() {
        if (!running) return
        NativeEngine.stop(); running = false
        focusRequest?.let { audio.abandonAudioFocusRequest(it) }
        toastStatus("stopped")
    }

    private fun applyMode() {
        if (settings.focusMode == "voice") {
            NativeEngine.setTheta(Float.NaN)  // voice-only: no direction prior
            film?.let { NativeEngine.setFilm(it) } ?: toastStatus("no voice enrolled - using direction")
        } else {
            NativeEngine.setTheta(0f)  // straight ahead (head-relative); on-device IMU fusion: see DECISIONS
            film?.let { NativeEngine.setFilm(it) }
        }
    }

    private fun onRouteChanged() {
        if (running) { stopEngine(); startEngine() }  // re-open streams on plug/unplug/BT connect
    }

    private fun pollStats() {
        if (!running) return
        val s = NativeEngine.stats()
        status.text = "running @%.0f Hz burst %d | callbacks %d xruns %d | hop %.2f ms (max %.2f)".format(
            s[4], s[5].toInt(), s[0].toInt(), s[1].toInt(), s[2], s[3])
        ui.postDelayed({ pollStats() }, 1000)
    }

    @SuppressLint("MissingPermission")
    private fun enroll() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) return
        val ecapa = assetToFile("ecapa.onnx"); val filmModel = assetToFile("clarity_film.onnx")
        if (ecapa == null || filmModel == null) { toastStatus("enrollment model not bundled: direction-only mode"); return }
        toastStatus("recording 15 s - only record people who agreed")
        thread {
            val sr = 16000; val n = sr * 15
            val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, sr, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT, AudioRecord.getMinBufferSize(sr, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT) * 4)
            val buf = FloatArray(n); var got = 0
            rec.startRecording()
            while (got < n) { val r = rec.read(buf, got, n - got, AudioRecord.READ_BLOCKING); if (r <= 0) break; got += r }
            rec.stop(); rec.release()
            val env = OrtEnvironment.getEnvironment()
            val emb = env.createSession(ecapa.absolutePath, OrtSession.SessionOptions()).use { s ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(buf), longArrayOf(1, n.toLong())).use { t ->
                    (s.run(mapOf(s.inputNames.first() to t))[0].value as Array<FloatArray>)[0]
                }
            }
            val norm = kotlin.math.sqrt(emb.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-8f)
            val e = FloatArray(emb.size) { emb[it] / norm }
            val f = env.createSession(filmModel.absolutePath, OrtSession.SessionOptions()).use { s ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(e), longArrayOf(1, e.size.toLong())).use { t ->
                    val out = s.run(mapOf("emb" to t))[0] as OnnxTensor
                    val fb = out.floatBuffer; FloatArray(fb.remaining()).also { fb.get(it) }
                }
            }
            // embedding stays in app-private memory only; raw audio is discarded
            ui.post { film = f; applyMode(); toastStatus("voice enrolled (stored only on this phone)") }
        }
    }

    private fun measureLatency() {
        if (running) stopEngine()
        toastStatus("measuring... keep the phone near the earbud/headset")
        thread {
            val ms = NativeEngine.measureLoopbackMs(lowLatency)
            val route = audio.getDevices(AudioManager.GET_DEVICES_OUTPUTS).firstOrNull { it.isSink }?.productName ?: "?"
            val msg = if (ms < 0) "no click detected" else "round-trip %.1f ms on %s (label: on-device)".format(ms, route)
            android.util.Log.i("ClarityLatency", "route=$route lowLatency=$lowLatency roundtrip_ms=$ms")
            ui.post { toastStatus(msg) }
        }
    }

    private fun toastStatus(s: String) { status.text = s }

    override fun onDestroy() {
        audio.unregisterAudioDeviceCallback(routeCallback)
        stopEngine()
        super.onDestroy()
    }
}
