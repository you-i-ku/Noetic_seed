package com.example.noetic_seed_monitor

import android.app.Activity
import android.content.Context
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import com.google.gson.JsonObject

/**
 * MediaProjection の許可ダイアログだけを出す透過的な Activity。
 * 許可が取れたら IkuMonitorService.startScreenCapture に引き渡して即 finish。
 * 実際のキャプチャループは Service 側で非同期に走る。
 *
 * camera_stream の CameraStreamActivity と違い、プレビューを持たず、
 * 画面を遮らない（ユーザーが見てる画面をそのままキャプチャしたいため）。
 */
class ScreenCaptureActivity : ComponentActivity() {

    companion object {
        const val EXTRA_FRAMES = "frames"
        const val EXTRA_INTERVAL_SEC = "interval_sec"
        private const val TAG = "ScreenCapture"
    }

    private val launcher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val frames = intent.getIntExtra(EXTRA_FRAMES, 5)
        val interval = intent.getFloatExtra(EXTRA_INTERVAL_SEC, 1.0f)
        val data = result.data
        if (result.resultCode == Activity.RESULT_OK && data != null) {
            Log.d(TAG, "MediaProjection permission granted")
            val svc = IkuMonitorService.instance
            if (svc != null) {
                svc.startScreenCapture(result.resultCode, data, frames, interval)
            } else {
                Log.w(TAG, "IkuMonitorService.instance is null, cannot start capture")
                notifyFailure("service not available")
            }
        } else {
            Log.d(TAG, "MediaProjection permission denied or cancelled")
            notifyFailure("permission denied")
        }
        finish()
    }

    private fun notifyFailure(reason: String) {
        // Python 側に stream_end を送って待機状態を解除させる
        CameraStreamBridge.sendMessage?.invoke(
            JsonObject().apply {
                addProperty("type", "stream_end")
                addProperty("frame_count", 0)
                addProperty("error", reason)
            }
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        try {
            launcher.launch(mpm.createScreenCaptureIntent())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to launch MediaProjection intent", e)
            notifyFailure(e.message ?: "launch failed")
            finish()
        }
    }
}
