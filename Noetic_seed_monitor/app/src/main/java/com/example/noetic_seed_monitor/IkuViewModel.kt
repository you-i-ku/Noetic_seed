package com.example.noetic_seed_monitor

import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.lifecycle.ViewModel
import com.google.gson.JsonObject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class ApprovalRequest(
    val id: String,
    val tool: String,
    val preview: String,
    val timestamp: String,
)

data class ProfileInfo(
    val name: String,
    val cycleId: Int = 0,
    val entropy: Float = 0.65f,
    val energy: Float = 50f,
)

data class IkuState(
    val connected: Boolean = false,
    val entropy: Float = 0.65f,
    val energy: Float = 50f,
    val cycleId: Int = 0,
    val toolLevel: Int = 0,
    val pressure: Float = 0f,
    val pendingCount: Int = 0,
    val pendingItems: List<Map<String, String>> = emptyList(),
    val selfModel: Map<String, String> = emptyMap(),
    val disposition: Map<String, Float> = emptyMap(),
    val e1: Float = 0f,
    val e2: Float = 0f,
    val e3: Float = 0f,
    val e4: Float = 0f,
    val negentropy: Float = 0f,
    val eHistory: List<List<Float>> = emptyList(),  // E値推移: [[e1,e2,e3,e4], ...]
    val logLines: List<String> = emptyList(),
    val replies: List<String> = emptyList(),
    val approvalRequests: List<ApprovalRequest> = emptyList(),
    val profiles: List<ProfileInfo> = emptyList(),
    val profileSelected: Boolean = false,
    val profileStarted: Boolean = false,  // WS再接続後、profileのmain.pyから最初のメッセージが来たか
    val pendingCameraCapture: Boolean = false,
    val error: String? = null,
)

class IkuViewModel : ViewModel() {
    private val _state = MutableStateFlow(IkuState())
    val state: StateFlow<IkuState> = _state.asStateFlow()

    private var wsClient: IkuWebSocketClient? = null
    private val maxLogLines = 200
    private var appContext: Context? = null
    private var notifId = 100
    private var _cameraCallback: ((String?, Map<String, Any>?) -> Unit)? = null
    private var _cameraFacing: String = "back"
    private var _cameraTrigger: (() -> Unit)? = null

    // camera_stream 用
    private var _cameraStreamCallback: ((List<String>?, Map<String, Any>?) -> Unit)? = null
    private var _cameraStreamTrigger: ((String, Int, Float) -> Unit)? = null

    val pendingCameraFacing: String get() = _cameraFacing

    /** MainActivity側から呼ぶ: カメラ起動トリガーを登録 */
    fun setCameraTrigger(trigger: () -> Unit) {
        _cameraTrigger = trigger
    }

    /** MainActivity側から呼ぶ: camera_stream 起動トリガーを登録 */
    fun setCameraStreamTrigger(trigger: (String, Int, Float) -> Unit) {
        _cameraStreamTrigger = trigger
    }

    fun finishCameraCapture(base64Image: String?, meta: Map<String, Any>?) {
        _cameraCallback?.invoke(base64Image, meta)
        _cameraCallback = null
        _state.value = _state.value.copy(pendingCameraCapture = false)
    }

    fun finishCameraStream(base64Frames: List<String>?, meta: Map<String, Any>?) {
        _cameraStreamCallback?.invoke(base64Frames, meta)
        _cameraStreamCallback = null
        _state.value = _state.value.copy(pendingCameraCapture = false)
    }

    /** カメラ起動トリガーを実行（メインスレッドで） */
    private fun triggerCameraCapture() {
        val trigger = _cameraTrigger
        if (trigger != null) {
            android.util.Log.d("IkuViewModel", "triggerCameraCapture: trigger呼び出し")
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                try {
                    trigger()
                } catch (e: Exception) {
                    android.util.Log.e("IkuViewModel", "trigger失敗", e)
                    finishCameraCapture(null, null)
                }
            }
        } else {
            android.util.Log.w("IkuViewModel", "triggerCameraCapture: _cameraTrigger未設定")
            finishCameraCapture(null, null)
        }
    }

    /** camera_stream 起動トリガーを実行（メインスレッドで） */
    private fun triggerCameraStream(facing: String, frames: Int, intervalSec: Float) {
        val trigger = _cameraStreamTrigger
        if (trigger != null) {
            android.util.Log.d("IkuViewModel", "triggerCameraStream: facing=$facing frames=$frames")
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                try {
                    trigger(facing, frames, intervalSec)
                } catch (e: Exception) {
                    android.util.Log.e("IkuViewModel", "stream trigger失敗", e)
                    finishCameraStream(null, null)
                }
            }
        } else {
            android.util.Log.w("IkuViewModel", "triggerCameraStream: _cameraStreamTrigger未設定")
            finishCameraStream(null, null)
        }
    }

    fun setContext(context: Context) {
        appContext = context.applicationContext
        createNotificationChannel()
        // ApprovalBridge: 通知ボタンから呼ばれる
        ApprovalBridge.callback = { apId, approved ->
            sendApproval(apId, approved)
        }
    }

    private fun createNotificationChannel() {
        val ctx = appContext ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel("iku_channel", "iku通知",
                NotificationManager.IMPORTANCE_HIGH).apply {
                description = "AIからのメッセージと承認要求"
            }
            val mgr = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            mgr.createNotificationChannel(channel)
        }
    }

    private fun sendNotification(title: String, body: String) {
        val ctx = appContext ?: return
        try {
            val mgr = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val notif = NotificationCompat.Builder(ctx, "iku_channel")
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle(title)
                .setContentText(body.take(100))
                .setStyle(NotificationCompat.BigTextStyle().bigText(body.take(300)))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .build()
            mgr.notify(notifId++, notif)
        } catch (_: SecurityException) {
        }
    }

    /** 承認/却下ボタン付き通知（approval_request用） */
    private fun sendApprovalNotification(apId: String, tool: String, preview: String) {
        val ctx = appContext ?: return
        try {
            val mgr = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val currentNotifId = notifId++

            val approveIntent = android.content.Intent(ctx, ApprovalActionReceiver::class.java).apply {
                action = ApprovalActionReceiver.ACTION_APPROVE
                putExtra("ap_id", apId)
                putExtra("notif_id", currentNotifId)
            }
            val denyIntent = android.content.Intent(ctx, ApprovalActionReceiver::class.java).apply {
                action = ApprovalActionReceiver.ACTION_DENY
                putExtra("ap_id", apId)
                putExtra("notif_id", currentNotifId)
            }

            val pendingFlag = android.app.PendingIntent.FLAG_UPDATE_CURRENT or android.app.PendingIntent.FLAG_IMMUTABLE
            val approvePending = android.app.PendingIntent.getBroadcast(
                ctx, currentNotifId * 2, approveIntent, pendingFlag
            )
            val denyPending = android.app.PendingIntent.getBroadcast(
                ctx, currentNotifId * 2 + 1, denyIntent, pendingFlag
            )

            val notif = NotificationCompat.Builder(ctx, "iku_channel")
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("🔐 $tool 承認待ち")
                .setContentText(preview.take(100))
                .setStyle(NotificationCompat.BigTextStyle().bigText(preview.take(500)))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .addAction(android.R.drawable.ic_menu_close_clear_cancel, "却下", denyPending)
                .addAction(android.R.drawable.ic_menu_send, "承認", approvePending)
                .build()
            mgr.notify(currentNotifId, notif)
        } catch (_: SecurityException) {
        }
    }

    fun connect(url: String, token: String) {
        wsClient?.disconnect()
        wsClient = IkuWebSocketClient(
            onMessage = { handleMessage(it) },
            onConnected = { _state.value = _state.value.copy(connected = true, error = null) },
            onDisconnected = { _state.value = _state.value.copy(connected = false) },
            onError = { _state.value = _state.value.copy(error = it) },
        )
        wsClient?.connect(url, token)
    }

    fun disconnect() {
        wsClient?.disconnect()
        wsClient = null
        _state.value = _state.value.copy(connected = false)
    }

    private fun handleMessage(json: JsonObject) {
        val type = json.get("type")?.asString ?: return
        var current = _state.value

        // profile選択後の再接続を検出: state/self/log/sync を受けたらprofileStarted=true
        if (current.profileSelected && !current.profileStarted
            && type in setOf("state", "self", "log", "sync", "e_values")) {
            current = current.copy(profileStarted = true)
            _state.value = current
        }

        when (type) {
            "log" -> {
                val text = json.get("text")?.asString ?: return
                val newLines = (current.logLines + text).takeLast(maxLogLines)
                _state.value = current.copy(logLines = newLines)
            }
            "state" -> {
                _state.value = current.copy(
                    entropy = json.get("entropy")?.asFloat ?: current.entropy,
                    energy = json.get("energy")?.asFloat ?: current.energy,
                    cycleId = json.get("cycle_id")?.asInt ?: current.cycleId,
                    toolLevel = json.get("tool_level")?.asInt ?: current.toolLevel,
                    pressure = json.get("pressure")?.asFloat ?: current.pressure,
                    pendingCount = json.get("pending_count")?.asInt ?: current.pendingCount,
                    pendingItems = json.getAsJsonArray("pending_items")?.map { item ->
                        val obj = item.asJsonObject
                        mapOf(
                            "type" to (obj.get("type")?.asString ?: ""),
                            "content" to (obj.get("content")?.asString ?: ""),
                            "id" to (obj.get("id")?.asString ?: ""),
                        )
                    } ?: current.pendingItems,
                )
            }
            "self" -> {
                val data = json.getAsJsonObject("data")
                if (data != null) {
                    val map = mutableMapOf<String, String>()
                    val disp = mutableMapOf<String, Float>()
                    // 表示しないキー
                    val skipKeys = setOf("disposition", "last_e_values", "drives_state")
                    for (key in data.keySet()) {
                        if (key in skipKeys) continue
                        val elem = data.get(key)
                        map[key] = when {
                            elem == null || elem.isJsonNull -> ""
                            elem.isJsonPrimitive -> elem.asString
                            elem.isJsonObject -> {
                                // ネストしたオブジェクトはkey:valueで展開
                                val obj = elem.asJsonObject
                                obj.keySet().joinToString(", ") { k ->
                                    val v = obj.get(k)
                                    "$k=${v?.asString ?: v.toString()}"
                                }
                            }
                            else -> elem.toString()
                        }
                    }
                    // disposition抽出
                    val dispObj = data.getAsJsonObject("disposition")
                    if (dispObj != null) {
                        for (k in dispObj.keySet()) {
                            try { disp[k] = dispObj.get(k).asFloat } catch (_: Exception) {}
                        }
                    }
                    _state.value = current.copy(selfModel = map, disposition = disp)
                }
            }
            "e_values" -> {
                val newE1 = json.get("e1")?.asFloat ?: current.e1
                val newE2 = json.get("e2")?.asFloat ?: current.e2
                val newE3 = json.get("e3")?.asFloat ?: current.e3
                val newE4 = json.get("e4")?.asFloat ?: current.e4
                val newHistory = (current.eHistory + listOf(listOf(newE1, newE2, newE3, newE4))).takeLast(20)
                _state.value = current.copy(
                    e1 = newE1, e2 = newE2, e3 = newE3, e4 = newE4,
                    negentropy = json.get("negentropy")?.asFloat ?: current.negentropy,
                    eHistory = newHistory,
                )
            }
            "reply" -> {
                val content = json.get("content")?.asString ?: return
                val newReplies = (current.replies + content).takeLast(50)
                val newLines = (current.logLines + "[AI] $content").takeLast(maxLogLines)
                _state.value = current.copy(replies = newReplies, logLines = newLines)
                sendNotification("iku", content.take(200))
            }
            "approval_request" -> {
                val req = ApprovalRequest(
                    id = json.get("id")?.asString ?: return,
                    tool = json.get("tool")?.asString ?: "",
                    preview = json.get("preview")?.asString ?: "",
                    timestamp = json.get("timestamp")?.asString ?: "",
                )
                _state.value = current.copy(
                    approvalRequests = current.approvalRequests + req
                )
                sendApprovalNotification(req.id, req.tool, req.preview)
            }
            "approval_result" -> {
                val apId = json.get("id")?.asString ?: return
                _state.value = current.copy(
                    approvalRequests = current.approvalRequests.filter { it.id != apId }
                )
            }
            "profile_list" -> {
                val arr = json.getAsJsonArray("profiles")
                if (arr != null) {
                    val profiles = arr.map { item ->
                        val obj = item.asJsonObject
                        ProfileInfo(
                            name = obj.get("name")?.asString ?: "",
                            cycleId = obj.get("cycle_id")?.asInt ?: 0,
                            entropy = obj.get("entropy")?.asFloat ?: 0.65f,
                            energy = obj.get("energy")?.asFloat ?: 50f,
                        )
                    }
                    _state.value = current.copy(profiles = profiles)
                }
            }
            "device_request" -> {
                val ctx = appContext ?: return
                DeviceHandler.handle(
                    context = ctx,
                    request = json,
                    sendResponse = { resp -> wsClient?.send(resp) },
                    requestCamera = { facing, callback ->
                        _cameraCallback = callback
                        _cameraFacing = facing
                        _state.value = _state.value.copy(pendingCameraCapture = true)
                        // Activity-level launcherを直接起動
                        triggerCameraCapture()
                    },
                    requestCameraStream = { facing, frames, intervalSec, callback ->
                        _cameraStreamCallback = callback
                        _cameraFacing = facing
                        _state.value = _state.value.copy(pendingCameraCapture = true)
                        triggerCameraStream(facing, frames, intervalSec)
                    },
                )
            }
            "sync" -> {
                // 再接続時のログバッファ
                val logs = json.getAsJsonArray("recent_logs")
                if (logs != null) {
                    val lines = mutableListOf<String>()
                    for (item in logs) {
                        val text = item.asJsonObject?.get("text")?.asString
                        if (text != null) lines.add(text)
                    }
                    _state.value = current.copy(logLines = lines.takeLast(maxLogLines))
                }
            }
        }
    }

    fun sendChat(text: String) {
        val msg = com.google.gson.JsonObject().apply {
            addProperty("type", "chat")
            addProperty("text", text)
        }
        wsClient?.send(msg)
        // ローカルのログにも追加
        val current = _state.value
        val newLines = (current.logLines + "[external] user: $text").takeLast(maxLogLines)
        _state.value = current.copy(logLines = newLines)
    }

    /** 外部（CameraStreamActivity など）から WebSocket にメッセージを送るための汎用エントリ */
    fun sendWsMessage(msg: com.google.gson.JsonObject) {
        wsClient?.send(msg)
    }

    fun runTestTool(name: String, args: Map<String, String>) {
        val argsObj = com.google.gson.JsonObject()
        args.forEach { (k, v) -> argsObj.addProperty(k, v) }
        val msg = com.google.gson.JsonObject().apply {
            addProperty("type", "test_tool")
            addProperty("tool", name)
            add("args", argsObj)
        }
        wsClient?.send(msg)
        val current = _state.value
        val newLines = (current.logLines + "[test] → $name $args").takeLast(maxLogLines)
        _state.value = current.copy(logLines = newLines)
    }

    fun selectProfile(name: String) {
        val msg = com.google.gson.JsonObject().apply {
            addProperty("type", "select_profile")
            addProperty("name", name)
        }
        wsClient?.send(msg)
        _state.value = _state.value.copy(
            profileSelected = true,
            profileStarted = false,  // 再接続待ち
        )
    }

    fun sendApproval(id: String, approved: Boolean) {
        val msg = com.google.gson.JsonObject().apply {
            addProperty("type", "approve")
            addProperty("id", id)
            addProperty("decision", if (approved) "yes" else "no")
        }
        wsClient?.send(msg)
        // ローカルからも除去
        val current = _state.value
        _state.value = current.copy(
            approvalRequests = current.approvalRequests.filter { it.id != id }
        )
    }

    override fun onCleared() {
        wsClient?.disconnect()
        super.onCleared()
    }
}
