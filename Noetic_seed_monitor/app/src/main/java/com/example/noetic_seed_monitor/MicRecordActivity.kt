package com.example.noetic_seed_monitor

import android.Manifest
import android.app.PendingIntent
import android.app.PictureInPictureParams
import android.app.RemoteAction
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.drawable.Icon
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Rational
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * 同期マイク録音 Activity。
 *
 * - AudioRecord で PCM 16bit mono 16kHz を録音
 * - チャンク毎に RMS を計算 → 波形バーをローリング表示
 * - PIP モード対応（停止ボタン RemoteAction 付き）
 * - duration_sec 経過 or 停止要求で終了
 * - 完了時に MicRecordBridge.onResult(wavBytes, durationSec, error?) を呼ぶ
 */
class MicRecordActivity : ComponentActivity() {

    companion object {
        const val EXTRA_DURATION_SEC = "duration_sec"
        private const val TAG = "MicRecordActivity"
        private const val SAMPLE_RATE = 16000
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        private const val BYTES_PER_SAMPLE = 2
        private const val WAVEFORM_BARS = 60
        const val ACTION_STOP_MIC = "com.example.noetic_seed_monitor.STOP_MIC"
        // 絶対上限（プライバシー保護、Python 側でも 30s 上限を持つが念のため）
        private const val HARD_MAX_SEC = 30.0f
    }

    private var durationSec: Float = 5.0f
    private var startMs: Long = 0L
    private var recordJob: Job? = null
    private var completed = false

    // 波形表示用の振幅履歴（StateFlow より単純な mutableState で十分）
    private val amplitudes = androidx.compose.runtime.mutableStateListOf<Float>().apply {
        repeat(WAVEFORM_BARS) { add(0f) }
    }
    private var elapsedSecState by mutableStateOf(0f)
    private var statusText by mutableStateOf("準備中...")

    // PIP 停止ボタンからの broadcast 受信
    private val stopReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == ACTION_STOP_MIC) {
                Log.d(TAG, "PIP stop action received")
                MicRecordBridge.stopRequested = true
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        durationSec = intent.getFloatExtra(EXTRA_DURATION_SEC, 5.0f).coerceIn(1.0f, HARD_MAX_SEC)

        MicRecordBridge.reset()

        // PIP 停止ブロードキャスト登録
        val filter = IntentFilter(ACTION_STOP_MIC)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(stopReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(stopReceiver, filter)
        }

        // 権限確認
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            finishWithResult(null, "RECORD_AUDIO permission missing")
            return
        }

        setContent { MicRecordScreen() }

        startRecording()
    }

    @Composable
    private fun MicRecordScreen() {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF0A0A1A)),
            contentAlignment = Alignment.Center,
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center,
                modifier = Modifier.fillMaxSize().padding(24.dp),
            ) {
                Text(
                    "🎙 録音中",
                    color = Color(0xFFFFB74D),
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Bold,
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    "${"%.1f".format(elapsedSecState)} / ${"%.1f".format(durationSec)} 秒",
                    color = Color.White,
                    fontSize = 18.sp,
                    fontFamily = FontFamily.Monospace,
                )
                Spacer(modifier = Modifier.height(32.dp))

                // 波形表示
                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(140.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(Color(0xFF1A1A2E)),
                ) {
                    val w = size.width
                    val h = size.height
                    val n = amplitudes.size
                    val barWidth = w / n * 0.7f
                    val gap = w / n * 0.3f
                    for (i in 0 until n) {
                        val a = amplitudes[i].coerceIn(0f, 1f)
                        val barH = h * a
                        val x = i * (barWidth + gap) + gap / 2
                        drawRect(
                            color = Color(0xFFFFB74D).copy(alpha = 0.3f + 0.7f * a),
                            topLeft = Offset(x, h / 2 - barH / 2),
                            size = Size(barWidth, barH.coerceAtLeast(2f)),
                        )
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    statusText,
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 12.sp,
                    fontFamily = FontFamily.Monospace,
                )
                Spacer(modifier = Modifier.height(32.dp))

                // 停止ボタン
                Button(
                    onClick = { MicRecordBridge.stopRequested = true },
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFEF5350)),
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                ) {
                    Text("■ 停止", color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }

    private fun startRecording() {
        startMs = System.currentTimeMillis()
        recordJob = CoroutineScope(Dispatchers.Default).launch {
            // 短い遅延の後 PIP に入る（UI が描画される時間を確保）
            delay(300)
            withContext(Dispatchers.Main) { enterPipModeNow() }
            recordLoop()
        }
    }

    private suspend fun recordLoop() {
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuf <= 0) {
            finishWithResult(null, "AudioRecord.getMinBufferSize failed: $minBuf")
            return
        }
        val bufSize = minBuf * 2

        val recorder = try {
            AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL, ENCODING, bufSize)
        } catch (e: SecurityException) {
            finishWithResult(null, "AudioRecord 構築失敗: ${e.message}")
            return
        }

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            recorder.release()
            finishWithResult(null, "AudioRecord not initialized")
            return
        }

        val totalSamples = (SAMPLE_RATE * durationSec).toInt()
        val pcmStream = ByteArrayOutputStream()
        // ~50ms チャンク（AudioRecord の最小バッファより小さい場合もあるが、その場合は最小バッファを使う）
        val chunkBytes = maxOf(SAMPLE_RATE * BYTES_PER_SAMPLE / 20, minBuf / 2)
        val readBuf = ByteArray(chunkBytes)
        var samplesRead = 0

        try {
            recorder.startRecording()
            withContext(Dispatchers.Main) { statusText = "録音中..." }
            while (samplesRead < totalSamples) {
                if (MicRecordBridge.stopRequested) {
                    Log.d(TAG, "Stop requested at $samplesRead samples")
                    break
                }
                val n = recorder.read(readBuf, 0, readBuf.size)
                if (n <= 0) break
                pcmStream.write(readBuf, 0, n)
                samplesRead += n / BYTES_PER_SAMPLE

                // 振幅計算 → UI 更新
                val rms = computeRms(readBuf, n)
                val elapsed = (System.currentTimeMillis() - startMs) / 1000f
                withContext(Dispatchers.Main) {
                    pushAmplitude(rms)
                    elapsedSecState = elapsed
                }
            }
            recorder.stop()
        } catch (e: Exception) {
            Log.e(TAG, "録音中エラー", e)
            recorder.release()
            finishWithResult(null, "録音中エラー: ${e.message}")
            return
        }
        recorder.release()

        val pcm = pcmStream.toByteArray()
        val actualDuration = samplesRead.toFloat() / SAMPLE_RATE
        Log.d(TAG, "録音完了: ${pcm.size} bytes (${samplesRead} samples = ${actualDuration}s)")
        val wavBytes = wrapWav(pcm, SAMPLE_RATE, 1, BYTES_PER_SAMPLE * 8)
        finishWithResult(wavBytes, null, actualDuration)
    }

    /** 16-bit PCM little-endian の RMS を 0-1 で返す */
    private fun computeRms(buf: ByteArray, lenBytes: Int): Float {
        if (lenBytes < 2) return 0f
        var sumSq = 0.0
        var n = 0
        var i = 0
        while (i < lenBytes - 1) {
            val s = ((buf[i + 1].toInt() shl 8) or (buf[i].toInt() and 0xff)).toShort().toInt()
            sumSq += (s * s).toDouble()
            n++
            i += 2
        }
        if (n == 0) return 0f
        val rms = sqrt(sumSq / n) / 32768.0
        // 軽い圧縮（小さい音も見えるように、log スケールに近い）
        return (rms * 4).coerceIn(0.0, 1.0).toFloat()
    }

    private fun pushAmplitude(amp: Float) {
        amplitudes.removeAt(0)
        amplitudes.add(amp)
    }

    private fun wrapWav(pcm: ByteArray, sampleRate: Int, channels: Int, bitsPerSample: Int): ByteArray {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataSize = pcm.size
        val chunkSize = 36 + dataSize
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(chunkSize)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)
        header.putShort(1)
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort(blockAlign.toShort())
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(dataSize)
        val out = ByteArray(44 + dataSize)
        System.arraycopy(header.array(), 0, out, 0, 44)
        System.arraycopy(pcm, 0, out, 44, dataSize)
        return out
    }

    private fun buildPipParams(): PictureInPictureParams {
        val builder = PictureInPictureParams.Builder().setAspectRatio(Rational(4, 3))
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try {
                val stopIntent = Intent(ACTION_STOP_MIC).setPackage(packageName)
                val pending = PendingIntent.getBroadcast(
                    this, 0, stopIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
                val stopAction = RemoteAction(
                    Icon.createWithResource(this, android.R.drawable.ic_media_pause),
                    "停止",
                    "録音を停止",
                    pending,
                )
                builder.setActions(listOf(stopAction))
            } catch (e: Exception) {
                Log.w(TAG, "PIP stop action setup failed", e)
            }
        }
        return builder.build()
    }

    private fun enterPipModeNow() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && !completed && !isInPictureInPictureMode) {
            try {
                enterPictureInPictureMode(buildPipParams())
                Log.d(TAG, "Entered PIP mode")
            } catch (e: Exception) {
                Log.w(TAG, "enterPictureInPictureMode failed", e)
            }
        }
    }

    private fun finishWithResult(wavBytes: ByteArray?, error: String?, actualDurationSec: Float = durationSec) {
        if (completed) return
        completed = true
        recordJob?.cancel()
        try {
            MicRecordBridge.onResult?.invoke(wavBytes, actualDurationSec, error)
        } catch (e: Exception) {
            Log.e(TAG, "onResult callback failed", e)
        }
        MicRecordBridge.onResult = null
        MicRecordBridge.stopRequested = false
        finish()
    }

    override fun onUserLeaveHint() {
        super.onUserLeaveHint()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && !completed) {
            try { enterPictureInPictureMode(buildPipParams()) }
            catch (e: Exception) { Log.w(TAG, "enterPictureInPictureMode failed", e) }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try { unregisterReceiver(stopReceiver) } catch (_: Exception) {}
        if (!completed) {
            completed = true
            recordJob?.cancel()
            // onResult が未実行なら null で通知
            try { MicRecordBridge.onResult?.invoke(null, 0f, "Activity destroyed") } catch (_: Exception) {}
            MicRecordBridge.onResult = null
            MicRecordBridge.stopRequested = false
        }
    }

    override fun onBackPressed() {
        MicRecordBridge.stopRequested = true
    }
}
