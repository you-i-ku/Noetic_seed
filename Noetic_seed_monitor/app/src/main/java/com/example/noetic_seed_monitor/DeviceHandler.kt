package com.example.noetic_seed_monitor

import android.content.Context
import android.util.Base64
import android.util.Log
import com.google.gson.JsonObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * device_request を受けて各種デバイス操作を実行し、device_response を返す。
 */
object DeviceHandler {

    /**
     * device_request を処理。非同期で実行し、完了時に sendResponse() を呼ぶ。
     *
     * @param context Androidコンテキスト
     * @param request device_requestのJSON
     * @param sendResponse(JsonObject) 応答を送信するコールバック
     * @param requestCamera(onResult: (String?, Map<String,Any>?) -> Unit) カメラ撮影をUI層でトリガー
     */
    fun handle(
        context: Context,
        request: JsonObject,
        sendResponse: (JsonObject) -> Unit,
        requestCamera: (String, (String?, Map<String, Any>?) -> Unit) -> Unit,
        requestCameraStream: (String, Int, Float, (List<String>?, Map<String, Any>?) -> Unit) -> Unit,
    ) {
        val id = request.get("id")?.asString ?: return
        val action = request.get("action")?.asString ?: return
        val params = request.getAsJsonObject("params") ?: JsonObject()

        val scope = CoroutineScope(Dispatchers.Main)

        when (action) {
            "camera_capture" -> {
                val facing = params.get("facing")?.asString ?: "back"
                // メインスレッドで requestCamera を呼ぶ（ActivityResultLauncher.launch() 制約）
                scope.launch {
                    Log.d("DeviceHandler", "camera_capture: requestCamera呼び出し facing=$facing")
                    requestCamera(facing) { base64Image, meta ->
                        Log.d("DeviceHandler", "camera_capture コールバック: base64=${base64Image != null}")
                        if (base64Image != null) {
                            val metaObj = JsonObject()
                            meta?.forEach { (k, v) ->
                                when (v) {
                                    is String -> metaObj.addProperty(k, v)
                                    is Number -> metaObj.addProperty(k, v)
                                    is Boolean -> metaObj.addProperty(k, v)
                                    else -> metaObj.addProperty(k, v.toString())
                                }
                            }
                            val resp = JsonObject().apply {
                                addProperty("type", "device_response")
                                addProperty("id", id)
                                addProperty("success", true)
                                addProperty("data", base64Image)
                                add("meta", metaObj)
                            }
                            sendResponse(resp)
                        } else {
                            sendResponse(buildResponse(id, false, null, null, "camera capture failed or cancelled"))
                        }
                    }
                }
            }
            "camera_stream" -> {
                // async モード: Activity が起動し、フレーム毎に stream_frame を直接 WebSocket 送信する
                // DeviceHandler はここでは応答を返さない（応答待ちしない fire-and-forget）
                val facing = params.get("facing")?.asString ?: "back"
                val frames = params.get("frames")?.asInt?.coerceIn(1, 30) ?: 5
                val intervalSec = params.get("interval_sec")?.asFloat?.coerceIn(0.3f, 5.0f) ?: 1.0f
                scope.launch {
                    Log.d("DeviceHandler", "camera_stream (async): facing=$facing frames=$frames interval=$intervalSec")
                    // requestCameraStream はダミーのコールバックを渡す（async 版は使われない）
                    requestCameraStream(facing, frames, intervalSec) { _, _ ->
                        // async モードではここは呼ばれない（stream_end 経由で終了）
                    }
                }
            }
            "camera_stream_stop" -> {
                // AI 側からの停止命令: Bridge のフラグを立てるだけ（camera/screen 共通）
                CameraStreamBridge.stopRequested = true
                Log.d("DeviceHandler", "camera_stream_stop: stopRequested flag set")
                sendResponse(buildResponse(id, true, null, null, null))
            }
            "mic_record" -> {
                // MicRecordActivity を launch して PIP + 波形 + 停止ボタン UI で録音
                // 完了 → MicRecordBridge.onResult 経由で結果を受け取り device_response 返却
                val durationSec = params.get("duration_sec")?.asFloat?.coerceIn(1.0f, 30.0f) ?: 5.0f
                CoroutineScope(Dispatchers.Main).launch {
                    Log.d("DeviceHandler", "mic_record (Activity): duration=${durationSec}s")
                    if (!MicRecorder.hasPermission(context)) {
                        sendResponse(buildResponse(id, false, null, null, "RECORD_AUDIO permission missing"))
                        return@launch
                    }
                    // Bridge にコールバック登録 → Activity 起動
                    MicRecordBridge.reset()
                    MicRecordBridge.onResult = { wavBytes, actualDurationSec, error ->
                        if (wavBytes == null) {
                            sendResponse(buildResponse(id, false, null, null, error ?: "録音失敗"))
                        } else {
                            val b64 = Base64.encodeToString(wavBytes, Base64.NO_WRAP)
                            val metaObj = JsonObject().apply {
                                addProperty("requested_duration_sec", durationSec)
                                addProperty("actual_duration_sec", actualDurationSec)
                                addProperty("sample_rate", 16000)
                                addProperty("channels", 1)
                                addProperty("format", "wav_pcm16")
                                addProperty("bytes", wavBytes.size)
                            }
                            val resp = JsonObject().apply {
                                addProperty("type", "device_response")
                                addProperty("id", id)
                                addProperty("success", true)
                                addProperty("data", b64)
                                add("meta", metaObj)
                            }
                            sendResponse(resp)
                        }
                    }
                    val activityIntent = android.content.Intent(context, MicRecordActivity::class.java).apply {
                        addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                        putExtra(MicRecordActivity.EXTRA_DURATION_SEC, durationSec)
                    }
                    try {
                        context.startActivity(activityIntent)
                    } catch (e: Exception) {
                        Log.e("DeviceHandler", "MicRecordActivity 起動失敗", e)
                        MicRecordBridge.onResult = null
                        sendResponse(buildResponse(id, false, null, null, "Activity 起動失敗: ${e.message}"))
                    }
                }
            }
            "screen_peek" -> {
                // async: ScreenCaptureActivity を launch → permission → Service で capture loop
                val frames = params.get("frames")?.asInt ?: 5
                val intervalSec = params.get("interval_sec")?.asFloat?.coerceIn(0.3f, 5.0f) ?: 1.0f
                // frames = 0 or 1-30 を許可
                val framesValidated = if (frames == 0) 0 else frames.coerceIn(1, 30)
                scope.launch {
                    Log.d("DeviceHandler", "screen_peek (async): frames=$framesValidated interval=$intervalSec")
                    val intent = android.content.Intent(context, ScreenCaptureActivity::class.java).apply {
                        addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
                        putExtra(ScreenCaptureActivity.EXTRA_FRAMES, framesValidated)
                        putExtra(ScreenCaptureActivity.EXTRA_INTERVAL_SEC, intervalSec)
                    }
                    try {
                        context.startActivity(intent)
                    } catch (e: Exception) {
                        Log.e("DeviceHandler", "ScreenCaptureActivity 起動失敗", e)
                    }
                }
            }
            else -> {
                sendResponse(buildResponse(id, false, null, null, "unknown action: $action"))
            }
        }
    }

    private fun buildResponse(id: String, success: Boolean,
                              dataObj: JsonObject?, dataStr: String?,
                              error: String?): JsonObject {
        return JsonObject().apply {
            addProperty("type", "device_response")
            addProperty("id", id)
            addProperty("success", success)
            if (dataObj != null) add("data", dataObj)
            if (dataStr != null) addProperty("data", dataStr)
            if (error != null) addProperty("error", error)
        }
    }

    /** JPEG bytes を base64 にエンコード */
    fun encodeJpegBase64(bytes: ByteArray): String {
        return Base64.encodeToString(bytes, Base64.NO_WRAP)
    }
}

