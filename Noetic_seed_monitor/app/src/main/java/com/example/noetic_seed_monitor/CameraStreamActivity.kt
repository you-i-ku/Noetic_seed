package com.example.noetic_seed_monitor

import android.annotation.SuppressLint
import android.app.PendingIntent
import android.app.PictureInPictureParams
import android.app.RemoteAction
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.drawable.Icon
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.util.Rational
import android.view.WindowManager
import android.widget.Button
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.gson.JsonObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * CameraX を使った非同期連続撮影 Activity。
 * フレーム毎に stream_frame メッセージで WebSocket 送信する。
 * PIP モード対応 — ユーザーがホームボタンなどで離脱したら小さくなる。
 * 外部からの停止要求（CameraStreamBridge.stopRequested）を毎フレーム前にチェックする。
 */
class CameraStreamActivity : ComponentActivity() {

    companion object {
        const val EXTRA_FACING = "facing"
        const val EXTRA_FRAMES = "frames"
        const val EXTRA_INTERVAL_SEC = "interval_sec"
        private const val TAG = "CameraStream"
        // frames=0 の無制限モードでの絶対安全上限（プライバシー・電池保護）
        private const val HARD_MAX_DURATION_MS = 600_000L  // 10分
        private const val HARD_MAX_FRAMES_SAFETY = 1000
        // PIP 内の停止ボタン用 broadcast action
        const val ACTION_STOP_STREAM = "com.example.noetic_seed_monitor.STOP_STREAM"
    }

    private lateinit var previewView: PreviewView
    private lateinit var statusText: TextView
    private lateinit var stopButton: Button
    private var imageCapture: ImageCapture? = null
    private var captureJob: Job? = null
    private var framesSent = 0
    private var completed = false

    private var facing: String = "back"
    private var frames: Int = 5   // 0 = 無制限モード（stop or ハード上限で終了）
    private var intervalSec: Float = 1.0f
    private var startMs: Long = 0L

    // PIP 内の停止ボタン用ブロードキャスト受信
    private val stopReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            if (intent?.action == ACTION_STOP_STREAM) {
                Log.d(TAG, "PIP stop action received")
                CameraStreamBridge.stopRequested = true
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_camera_stream)

        previewView = findViewById(R.id.preview_view)
        statusText = findViewById(R.id.status_text)
        stopButton = findViewById(R.id.stop_button)

        facing = intent.getStringExtra(EXTRA_FACING) ?: "back"
        val rawFrames = intent.getIntExtra(EXTRA_FRAMES, 5)
        frames = if (rawFrames == 0) 0 else rawFrames.coerceIn(1, 30)
        intervalSec = intent.getFloatExtra(EXTRA_INTERVAL_SEC, 1.0f).coerceIn(0.3f, 5.0f)

        // 開始時にフラグリセット
        CameraStreamBridge.reset()

        // PIP 停止ボタンからのブロードキャスト受信を登録
        val filter = IntentFilter(ACTION_STOP_STREAM)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(stopReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(stopReceiver, filter)
        }

        stopButton.setOnClickListener {
            Log.d(TAG, "Stop button pressed")
            CameraStreamBridge.stopRequested = true
        }

        statusText.text = "準備中..."
        startCamera()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            try {
                val provider = providerFuture.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val selector = if (facing == "front") {
                    CameraSelector.DEFAULT_FRONT_CAMERA
                } else {
                    CameraSelector.DEFAULT_BACK_CAMERA
                }
                imageCapture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()

                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, imageCapture)

                startMs = System.currentTimeMillis()
                captureJob = CoroutineScope(Dispatchers.Main).launch {
                    captureLoop()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Camera init failed", e)
                finishWithEnd()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private suspend fun captureLoop() {
        // プレビュー描画が始まるのを待ってから即 PIP に入る
        // （承認後すぐにアプリ内で小窓化される）
        delay(300)
        enterPipModeNow()

        // Pattern A (frames > 0): 指定枚数で自然終了
        // Pattern B (frames == 0): camera_stream_stop または絶対上限まで継続
        val unlimited = (frames == 0)
        var i = 0
        while (true) {
            if (completed) return
            if (CameraStreamBridge.stopRequested) {
                Log.d(TAG, "Stop requested, exiting loop at frame ${i + 1}")
                break
            }
            // Pattern A: 指定枚数に達したら終了
            if (!unlimited && i >= frames) break
            // Pattern B の絶対上限（プライバシー・電池保護）
            if (unlimited) {
                val elapsed = System.currentTimeMillis() - startMs
                if (elapsed > HARD_MAX_DURATION_MS) {
                    Log.w(TAG, "Hard limit: duration > ${HARD_MAX_DURATION_MS / 1000}s, stopping")
                    break
                }
                if (i >= HARD_MAX_FRAMES_SAFETY) {
                    Log.w(TAG, "Hard limit: frame count > $HARD_MAX_FRAMES_SAFETY, stopping")
                    break
                }
            }

            updateStatus(i)
            val bytes = captureOneFrame()
            if (bytes != null) {
                sendFrame(bytes, i + 1)
                framesSent++
                Log.d(TAG, "Frame ${i + 1}${if (unlimited) "" else "/$frames"} sent: ${bytes.size} bytes")
            } else {
                Log.w(TAG, "Frame ${i + 1} capture failed")
            }
            i++

            // インターバル待機（最後のフレーム後はスキップ）
            val hasNext = unlimited || i < frames
            if (hasNext && !CameraStreamBridge.stopRequested) {
                delay((intervalSec * 1000).toLong())
            }
        }
        updateStatus(i)
        delay(200)
        finishWithEnd()
    }

    /** PIP パラメータ（停止ボタン action 付き）を構築する */
    private fun buildPipParams(aspect: Rational = Rational(3, 4)): PictureInPictureParams {
        val builder = PictureInPictureParams.Builder().setAspectRatio(aspect)
        // API 26+: PIP 停止ボタン（RemoteAction）
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            try {
                val stopIntent = Intent(ACTION_STOP_STREAM).setPackage(packageName)
                val pending = PendingIntent.getBroadcast(
                    this, 0, stopIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
                val stopAction = RemoteAction(
                    Icon.createWithResource(this, android.R.drawable.ic_media_pause),
                    "停止",
                    "カメラストリームを停止",
                    pending
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
                enterPictureInPictureMode(buildPipParams(Rational(3, 4)))
                Log.d(TAG, "Entered PIP mode immediately")
            } catch (e: Exception) {
                Log.w(TAG, "immediate enterPictureInPictureMode failed", e)
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private fun updateStatus(capturedCount: Int) {
        val elapsed = (System.currentTimeMillis() - startMs) / 1000.0
        statusText.text = if (frames == 0) {
            "撮影中: ${capturedCount}枚 (${"%.1f".format(elapsed)}s) [無制限・最大10分]"
        } else {
            "撮影中: ${capturedCount}/${frames} (${"%.1f".format(elapsed)}s)"
        }
    }

    private suspend fun captureOneFrame(): ByteArray? = withContext(Dispatchers.Main) {
        val capture = imageCapture ?: return@withContext null
        kotlinx.coroutines.suspendCancellableCoroutine<ByteArray?> { cont ->
            capture.takePicture(
                ContextCompat.getMainExecutor(this@CameraStreamActivity),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        try {
                            val jpegBytes = imageProxyToJpegBytes(image)
                            image.close()
                            if (cont.isActive) cont.resumeWith(Result.success(jpegBytes))
                        } catch (e: Exception) {
                            image.close()
                            if (cont.isActive) cont.resumeWith(Result.success(null))
                        }
                    }

                    override fun onError(exc: ImageCaptureException) {
                        Log.e(TAG, "takePicture error", exc)
                        if (cont.isActive) cont.resumeWith(Result.success(null))
                    }
                }
            )
        }
    }

    private fun imageProxyToJpegBytes(image: ImageProxy): ByteArray? {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return try {
            // Android 側で粗めにリサイズ（WS転送量削減）。さらに Python 側で最終解像度に揃える
            resizeJpegIfTooLarge(bytes, maxSide = 1920)
        } catch (e: Exception) {
            Log.w(TAG, "resize failed, using raw bytes", e)
            bytes
        }
    }

    private fun resizeJpegIfTooLarge(jpegBytes: ByteArray, maxSide: Int): ByteArray {
        val opts = android.graphics.BitmapFactory.Options().apply {
            inJustDecodeBounds = true
        }
        android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, opts)
        val longest = maxOf(opts.outWidth, opts.outHeight)
        if (longest <= maxSide) return jpegBytes

        val scale = maxSide.toFloat() / longest
        val newW = (opts.outWidth * scale).toInt()
        val newH = (opts.outHeight * scale).toInt()
        var sampleSize = 1
        while (opts.outWidth / (sampleSize * 2) >= newW) sampleSize *= 2
        val decodeOpts = android.graphics.BitmapFactory.Options().apply {
            inSampleSize = sampleSize
        }
        val decoded = android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, decodeOpts)
            ?: return jpegBytes
        val scaled = android.graphics.Bitmap.createScaledBitmap(decoded, newW, newH, true)
        decoded.recycle()
        val baos = ByteArrayOutputStream()
        scaled.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, baos)
        scaled.recycle()
        return baos.toByteArray()
    }

    private fun sendFrame(jpegBytes: ByteArray, frameIndex: Int) {
        val sender = CameraStreamBridge.sendMessage ?: run {
            Log.w(TAG, "sendMessage not set, dropping frame $frameIndex")
            return
        }
        val b64 = android.util.Base64.encodeToString(jpegBytes, android.util.Base64.NO_WRAP)
        val now = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())

        // 画像寸法取得（軽いデコード）
        var w = 0
        var h = 0
        try {
            val opts = android.graphics.BitmapFactory.Options().apply {
                inJustDecodeBounds = true
            }
            android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size, opts)
            w = opts.outWidth
            h = opts.outHeight
        } catch (_: Exception) {}

        val meta = JsonObject().apply {
            addProperty("captured_at", now)
            addProperty("facing", facing)
            addProperty("frame_index", frameIndex)
            addProperty("total_frames", frames)
            addProperty("interval_sec", intervalSec)
            addProperty("size_bytes", jpegBytes.size)
            if (w > 0) addProperty("width", w)
            if (h > 0) addProperty("height", h)
        }

        val msg = JsonObject().apply {
            addProperty("type", "stream_frame")
            addProperty("data", b64)
            add("meta", meta)
        }
        sender(msg)
    }

    private fun finishWithEnd() {
        if (completed) return
        completed = true
        captureJob?.cancel()

        try {
            ProcessCameraProvider.getInstance(this).get().unbindAll()
        } catch (_: Exception) {
        }

        // stream_end 通知を Python 側に送る
        CameraStreamBridge.sendMessage?.invoke(
            JsonObject().apply {
                addProperty("type", "stream_end")
                addProperty("frame_count", framesSent)
            }
        )
        CameraStreamBridge.sendMessage = null
        CameraStreamBridge.stopRequested = false

        // 旧バッチ版の onComplete は null を渡して終了通知（互換）
        val legacy = CameraStreamBridge.onComplete
        CameraStreamBridge.onComplete = null
        legacy?.invoke(null)

        finish()
    }

    override fun onUserLeaveHint() {
        super.onUserLeaveHint()
        // ユーザーがホームボタンを押したら PIP モードに入る
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O && !completed) {
            try {
                enterPictureInPictureMode(buildPipParams(Rational(4, 3)))
            } catch (e: Exception) {
                Log.w(TAG, "enterPictureInPictureMode failed", e)
            }
        }
    }

    override fun onPictureInPictureModeChanged(isInPictureInPictureMode: Boolean, newConfig: android.content.res.Configuration) {
        super.onPictureInPictureModeChanged(isInPictureInPictureMode, newConfig)
        // PIP 中は UI 要素を隠す
        stopButton.visibility = if (isInPictureInPictureMode) android.view.View.GONE else android.view.View.VISIBLE
        statusText.visibility = if (isInPictureInPictureMode) android.view.View.GONE else android.view.View.VISIBLE
    }

    override fun onDestroy() {
        super.onDestroy()
        try { unregisterReceiver(stopReceiver) } catch (_: Exception) {}
        if (!completed) {
            completed = true
            captureJob?.cancel()
            CameraStreamBridge.sendMessage = null
            CameraStreamBridge.stopRequested = false
        }
    }

    override fun onBackPressed() {
        // 戻るボタンで手動停止
        CameraStreamBridge.stopRequested = true
    }
}
