package com.example.noetic_seed_monitor

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.gson.JsonObject
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * iku monitor の中核サービス。Foreground Service として常駐し、
 * WebSocket 接続と IkuState の source of truth を担う。
 *
 * 設計要点:
 * - state は companion 内の MutableStateFlow（プロセス生存中は維持）
 * - instance は @Volatile で Service インスタンスを公開（ViewModel から呼び出し用）
 * - Foreground notification で Android に kill されないようにする
 * - Activity が destroy されても WebSocket は生き続ける
 * - Service recreation 時は onCreate で再接続（state は companion に残る）
 */
class IkuMonitorService : Service() {

    companion object {
        private const val TAG = "IkuMonitorService"
        const val CHANNEL_ID_PERSISTENT = "noetic_seed_service_persistent"
        const val CHANNEL_ID_EVENTS = "noetic_seed_channel"
        const val NOTIF_ID_PERSISTENT = 1
        const val ACTION_WAKEUP = "com.example.noetic_seed_monitor.action.WAKEUP"

        // プロセス生存中は維持される state（Service instance 再生成にも耐える）
        private val _state = MutableStateFlow(IkuState())
        val state: StateFlow<IkuState> = _state.asStateFlow()

        @Volatile
        var instance: IkuMonitorService? = null
            private set

        // ViewModel から appContext を受け取るための経路（Service 未起動時用フォールバック）
        @Volatile
        private var appContextRef: Context? = null

        fun setContext(context: Context) {
            appContextRef = context.applicationContext
        }

        /** Service を起動する（startForegroundService 経由）。Activity の onCreate で呼ぶ */
        fun startIfNeeded(context: Context) {
            val intent = Intent(context, IkuMonitorService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
    }

    // === Service state (instance-level) ===
    private var wsClient: IkuWebSocketClient? = null
    private val maxLogLines = 200
    private var notifId = 100
    private var _cameraCallback: ((String?, Map<String, Any>?) -> Unit)? = null
    private var _cameraFacing: String = "back"
    private var _cameraTrigger: (() -> Unit)? = null
    private var _cameraStreamCallback: ((List<String>?, Map<String, Any>?) -> Unit)? = null
    private var _cameraStreamTrigger: ((String, Int, Float) -> Unit)? = null

    // === Screen capture state ===
    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var screenCaptureJob: Job? = null

    val pendingCameraFacing: String get() = _cameraFacing

    private val binder = LocalBinder()

    inner class LocalBinder : Binder() {
        fun getService(): IkuMonitorService = this@IkuMonitorService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        instance = this
        appContextRef = applicationContext
        createNotificationChannels()

        // 常駐通知を作成して Foreground mode に入る
        val persistentNotif = buildPersistentNotification("待機中")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            // Android 14+ は foregroundServiceType を明示
            startForeground(
                NOTIF_ID_PERSISTENT,
                persistentNotif,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            )
        } else {
            startForeground(NOTIF_ID_PERSISTENT, persistentNotif)
        }

        // ApprovalBridge の callback を Service 側に設定
        ApprovalBridge.callback = { apId, approved ->
            sendApproval(apId, approved)
        }

        Log.d(TAG, "Service onCreate, instance set")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onDestroy() {
        Log.d(TAG, "Service onDestroy")
        stopScreenCapture()
        wsClient?.disconnect()
        wsClient = null
        instance = null
        super.onDestroy()
    }

    // === Notification channel 設定 ===
    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            // 常駐通知用（音なし、低優先度）
            val persistentCh = NotificationChannel(
                CHANNEL_ID_PERSISTENT, "Noetic_seed monitoring",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Noetic_seed の監視セッションを継続するための常駐通知"
                setSound(null, null)
                enableVibration(false)
            }
            mgr.createNotificationChannel(persistentCh)
            // イベント通知用（reply, approval）
            val eventsCh = NotificationChannel(
                CHANNEL_ID_EVENTS, "Noetic_seed 通知",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "AIからのメッセージと承認要求"
            }
            mgr.createNotificationChannel(eventsCh)
        }
    }

    private fun buildPersistentNotification(statusText: String): Notification {
        val openAppIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pending = PendingIntent.getActivity(
            this, 0, openAppIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID_PERSISTENT)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle("Noetic_seed monitoring")
            .setContentText(statusText)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .setSilent(true)
            .setContentIntent(pending)
            .build()
    }

    private fun updatePersistentNotification(statusText: String) {
        try {
            val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            mgr.notify(NOTIF_ID_PERSISTENT, buildPersistentNotification(statusText))
        } catch (_: SecurityException) {
        }
    }

    private fun buildOpenAppPendingIntent(): PendingIntent {
        val openAppIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        return PendingIntent.getActivity(
            this, 0, openAppIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    private fun sendEventNotification(title: String, body: String) {
        try {
            val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val notif = NotificationCompat.Builder(this, CHANNEL_ID_EVENTS)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle(title)
                .setContentText(body.take(100))
                .setStyle(NotificationCompat.BigTextStyle().bigText(body.take(300)))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .setContentIntent(buildOpenAppPendingIntent())
                .build()
            mgr.notify(notifId++, notif)
        } catch (_: SecurityException) {
        }
    }

    /** 承認/却下ボタン付き通知（approval_request用） */
    private fun sendApprovalNotification(apId: String, tool: String, preview: String) {
        try {
            val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            val currentNotifId = notifId++

            val approveIntent = Intent(this, ApprovalActionReceiver::class.java).apply {
                action = ApprovalActionReceiver.ACTION_APPROVE
                putExtra("ap_id", apId)
                putExtra("notif_id", currentNotifId)
            }
            val denyIntent = Intent(this, ApprovalActionReceiver::class.java).apply {
                action = ApprovalActionReceiver.ACTION_DENY
                putExtra("ap_id", apId)
                putExtra("notif_id", currentNotifId)
            }

            val pendingFlag =
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            val approvePending = PendingIntent.getBroadcast(
                this, currentNotifId * 2, approveIntent, pendingFlag
            )
            val denyPending = PendingIntent.getBroadcast(
                this, currentNotifId * 2 + 1, denyIntent, pendingFlag
            )

            val notif = NotificationCompat.Builder(this, CHANNEL_ID_EVENTS)
                .setSmallIcon(android.R.drawable.ic_dialog_info)
                .setContentTitle("🔐 $tool 承認待ち")
                .setContentText(preview.take(100))
                .setStyle(NotificationCompat.BigTextStyle().bigText(preview.take(500)))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setAutoCancel(true)
                .setContentIntent(buildOpenAppPendingIntent())
                .addAction(android.R.drawable.ic_menu_close_clear_cancel, "却下", denyPending)
                .addAction(android.R.drawable.ic_menu_send, "承認", approvePending)
                .build()
            mgr.notify(currentNotifId, notif)
        } catch (_: SecurityException) {
        }
    }

    // === Camera trigger 登録 (Activity から呼ぶ) ===
    fun setCameraTrigger(trigger: () -> Unit) {
        _cameraTrigger = trigger
    }

    fun setCameraStreamTrigger(trigger: (String, Int, Float) -> Unit) {
        _cameraStreamTrigger = trigger
    }

    fun finishCameraCapture(base64Image: String?, meta: Map<String, Any>?) {
        _cameraCallback?.invoke(base64Image, meta)
        _cameraCallback = null
        updateSession(setRunning = false) { it.copy(pendingCameraCapture = false) }
    }

    fun finishCameraStream(base64Frames: List<String>?, meta: Map<String, Any>?) {
        _cameraStreamCallback?.invoke(base64Frames, meta)
        _cameraStreamCallback = null
        updateSession(setRunning = false) { it.copy(pendingCameraCapture = false) }
    }

    private fun triggerCameraCapture() {
        val trigger = _cameraTrigger
        if (trigger != null) {
            android.os.Handler(android.os.Looper.getMainLooper()).post {
                try {
                    trigger()
                } catch (e: Exception) {
                    Log.e(TAG, "camera trigger失敗", e)
                    finishCameraCapture(null, null)
                }
            }
        } else {
            Log.w(TAG, "_cameraTrigger 未設定")
            finishCameraCapture(null, null)
        }
    }

    private fun triggerCameraStream(facing: String, frames: Int, intervalSec: Float) {
        // Activity trigger 経由ではなく、Service context から直接 CameraStreamActivity を起動
        // （Activity が destroy されていても動作する。mic_record と同じパターン）

        // カメラ権限チェック（未許可なら Python 側にフレームが届かず黙って死ぬのを防ぐ）
        if (androidx.core.content.ContextCompat.checkSelfPermission(
                applicationContext, android.Manifest.permission.CAMERA
            ) != android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "CAMERA permission not granted, cannot start camera_stream")
            finishCameraStream(null, null)
            return
        }

        CameraStreamBridge.reset()
        CameraStreamBridge.sendMessage = { json -> wsClient?.send(json) }
        CameraStreamBridge.onComplete = { _ -> finishCameraStream(null, null) }
        val intent = android.content.Intent(applicationContext, CameraStreamActivity::class.java).apply {
            addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)
            putExtra(CameraStreamActivity.EXTRA_FACING, facing)
            putExtra(CameraStreamActivity.EXTRA_FRAMES, frames)
            putExtra(CameraStreamActivity.EXTRA_INTERVAL_SEC, intervalSec)
        }
        try {
            applicationContext.startActivity(intent)
        } catch (e: Exception) {
            Log.e(TAG, "CameraStreamActivity 起動失敗", e)
            CameraStreamBridge.sendMessage = null
            CameraStreamBridge.onComplete = null
            finishCameraStream(null, null)
        }
    }

    // === WebSocket 接続 ===
    fun connect(url: String, token: String) {
        wsClient?.disconnect()
        _state.value = _state.value.copy(phase = AppPhase.Reconnecting, session = null)

        // URL/token を保存（ConnectScreen のプリフィル用）
        try {
            val prefs = applicationContext.getSharedPreferences("noetic_seed", Context.MODE_PRIVATE)
            prefs.edit().putString("ws_url", url).putString("ws_token", token).apply()
        } catch (_: Exception) {}

        wsClient = IkuWebSocketClient(
            onMessage = { handleMessage(it) },
            onConnected = {
                _state.value = _state.value.copy(phase = AppPhase.WaitingForServer, error = null)
                updatePersistentNotification("接続中")
            },
            onDisconnected = {
                val wasRunning = _state.value.phase is AppPhase.Running
                _state.value = _state.value.copy(
                    phase = if (wasRunning) AppPhase.Reconnecting else _state.value.phase,
                )
                updatePersistentNotification("切断 — 再接続中...")
            },
            onDormant = {
                // セッション TTL 超過 (2分) → セッション死亡、完全リセット
                wsClient?.disconnect()
                wsClient = null
                _state.value = IkuState(phase = AppPhase.Disconnected)
                updatePersistentNotification("切断")
            },
            onError = { _state.value = _state.value.copy(error = it) },
        )
        wsClient?.connect(url, token)
    }

    fun disconnect() {
        wsClient?.disconnect()
        wsClient = null
        _state.value = _state.value.copy(phase = AppPhase.Disconnected, session = null)
        updatePersistentNotification("切断")
    }

    // === セッションデータ更新ヘルパ ===
    private fun updateSession(setRunning: Boolean = true, transform: (SessionData) -> SessionData) {
        val current = _state.value
        val s = current.session ?: SessionData()
        _state.value = current.copy(
            phase = if (setRunning) AppPhase.Running else current.phase,
            session = transform(s),
        )
    }

    // === WebSocket メッセージ処理 ===
    private fun handleMessage(json: JsonObject) {
        val type = json.get("type")?.asString ?: return
        val current = _state.value

        when (type) {
            "log" -> {
                val text = json.get("text")?.asString ?: return
                updateSession { it.copy(logLines = (it.logLines + text).takeLast(maxLogLines)) }
            }
            "state" -> {
                updateSession { s ->
                    s.copy(
                        entropy = json.get("entropy")?.asFloat ?: s.entropy,
                        energy = json.get("energy")?.asFloat ?: s.energy,
                        cycleId = json.get("cycle_id")?.asInt ?: s.cycleId,
                        toolLevel = json.get("tool_level")?.asInt ?: s.toolLevel,
                        pressure = json.get("pressure")?.asFloat ?: s.pressure,
                        pendingCount = json.get("pending_count")?.asInt ?: s.pendingCount,
                        pendingItems = json.getAsJsonArray("pending_items")?.map { item ->
                            val obj = item.asJsonObject
                            mapOf(
                                "type" to (obj.get("type")?.asString ?: ""),
                                "content" to (obj.get("content")?.asString ?: ""),
                                "id" to (obj.get("id")?.asString ?: ""),
                            )
                        } ?: s.pendingItems,
                        paused = json.get("paused")?.asBoolean ?: s.paused,
                    )
                }
            }
            "self" -> {
                val data = json.getAsJsonObject("data") ?: return
                val map = mutableMapOf<String, String>()
                val disp = mutableMapOf<String, Float>()
                val skipKeys = setOf("disposition", "last_e_values", "drives_state")
                for (key in data.keySet()) {
                    if (key in skipKeys) continue
                    val elem = data.get(key)
                    map[key] = when {
                        elem == null || elem.isJsonNull -> ""
                        elem.isJsonPrimitive -> elem.asString
                        elem.isJsonObject -> {
                            val obj = elem.asJsonObject
                            obj.keySet().joinToString(", ") { k ->
                                val v = obj.get(k)
                                "$k=${v?.asString ?: v.toString()}"
                            }
                        }
                        else -> elem.toString()
                    }
                }
                val dispObj = data.getAsJsonObject("disposition")
                if (dispObj != null) {
                    for (k in dispObj.keySet()) {
                        try { disp[k] = dispObj.get(k).asFloat } catch (_: Exception) {}
                    }
                }
                updateSession { it.copy(selfModel = map, disposition = disp) }
            }
            "e_values" -> {
                updateSession { s ->
                    val newE1 = json.get("e1")?.asFloat ?: s.e1
                    val newE2 = json.get("e2")?.asFloat ?: s.e2
                    val newE3 = json.get("e3")?.asFloat ?: s.e3
                    val newE4 = json.get("e4")?.asFloat ?: s.e4
                    s.copy(
                        e1 = newE1, e2 = newE2, e3 = newE3, e4 = newE4,
                        negentropy = json.get("negentropy")?.asFloat ?: s.negentropy,
                        eHistory = (s.eHistory + listOf(listOf(newE1, newE2, newE3, newE4))).takeLast(20),
                    )
                }
            }
            "reply" -> {
                val content = json.get("content")?.asString ?: return
                updateSession { s ->
                    s.copy(
                        replies = (s.replies + content).takeLast(50),
                        logLines = (s.logLines + "[AI] $content").takeLast(maxLogLines),
                    )
                }
                val senderName = _state.value.session?.selfModel?.get("name")
                    ?.takeIf { it.isNotBlank() } ?: "Noetic_seed"
                sendEventNotification(senderName, content.take(200))
            }
            "approval_request" -> {
                val req = ApprovalRequest(
                    id = json.get("id")?.asString ?: return,
                    tool = json.get("tool")?.asString ?: "",
                    preview = json.get("preview")?.asString ?: "",
                    timestamp = json.get("timestamp")?.asString ?: "",
                )
                updateSession(setRunning = false) { it.copy(approvalRequests = it.approvalRequests + req) }
                sendApprovalNotification(req.id, req.tool, req.preview)
            }
            "approval_result" -> {
                val apId = json.get("id")?.asString ?: return
                updateSession(setRunning = false) { it.copy(approvalRequests = it.approvalRequests.filter { r -> r.id != apId }) }
            }
            "profile_list" -> {
                // 既にプロファイル選択済み or 稼働中なら無視
                // （server.py の 0.5s 定期 broadcast が selectProfile 後も到着する対策）
                if (current.phase is AppPhase.ProfileLoading || current.phase is AppPhase.Running) return
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
                    _state.value = current.copy(
                        phase = AppPhase.ProfileSelect(profiles),
                        session = null,
                        error = null,
                    )
                }
            }
            "device_request" -> {
                DeviceHandler.handle(
                    context = applicationContext,
                    request = json,
                    sendResponse = { resp -> wsClient?.send(resp) },
                    requestCamera = { facing, callback ->
                        _cameraCallback = callback
                        _cameraFacing = facing
                        updateSession(setRunning = false) { it.copy(pendingCameraCapture = true) }
                        triggerCameraCapture()
                    },
                    requestCameraStream = { facing, frames, intervalSec, callback ->
                        _cameraStreamCallback = callback
                        _cameraFacing = facing
                        triggerCameraStream(facing, frames, intervalSec)
                    },
                )
            }
            "sync" -> {
                // sync はログ復元のみ。Running への遷移はしない
                // （server.py のプロファイル選択モードでも空 sync が飛ぶため）
                // Running への遷移は state/self/e_values/log に任せる
                val logs = json.getAsJsonArray("recent_logs")
                if (logs != null) {
                    val lines = mutableListOf<String>()
                    for (item in logs) {
                        val text = item.asJsonObject?.get("text")?.asString
                        if (text != null) lines.add(text)
                    }
                    if (lines.isNotEmpty()) {
                        updateSession(setRunning = false) { it.copy(logLines = lines.takeLast(maxLogLines)) }
                    }
                }
            }
            "llm_providers_list" -> {
                val arr = json.getAsJsonArray("providers")
                val providers = arr?.map { item ->
                    val obj = item.asJsonObject
                    LlmProviderInfo(
                        provider = obj.get("provider")?.asString ?: "",
                        baseUrl = obj.get("base_url")?.asString ?: "",
                        lastModel = obj.get("last_model")?.asString ?: "",
                        hasKey = obj.get("has_key")?.asBoolean ?: false,
                    )
                } ?: emptyList()
                val activeObj = json.getAsJsonObject("active")
                val active = if (activeObj != null) {
                    LlmActiveConfig(
                        provider = activeObj.get("provider")?.asString ?: "",
                        model = activeObj.get("model")?.asString ?: "",
                    )
                } else {
                    current.llmActive
                }
                _state.value = current.copy(llmProviders = providers, llmActive = active)
            }
            "set_llm_result" -> {
                val ok = json.get("ok")?.asBoolean ?: false
                val msg = if (ok) {
                    val p = json.get("provider")?.asString ?: ""
                    val m = json.get("model")?.asString ?: ""
                    "設定完了: $p / $m（次サイクルから反映）"
                } else {
                    "エラー: ${json.get("error")?.asString ?: "unknown"}"
                }
                _state.value = current.copy(llmSetResult = msg)
                if (ok) requestLlmProviders()
            }
        }
    }

    // === 送信メソッド ===
    fun sendChat(text: String) {
        val msg = JsonObject().apply {
            addProperty("type", "chat")
            addProperty("text", text)
        }
        wsClient?.send(msg)
        updateSession(setRunning = false) { it.copy(logLines = (it.logLines + "[external] $text").takeLast(maxLogLines)) }
    }

    fun sendWsMessage(msg: JsonObject) {
        wsClient?.send(msg)
    }

    fun sendServerCommand(action: String) {
        val msg = JsonObject().apply {
            addProperty("type", "server_command")
            addProperty("action", action)
        }
        wsClient?.send(msg)
        when (action) {
            "pause" -> updateSession(setRunning = false) { it.copy(paused = true) }
            "resume" -> updateSession(setRunning = false) { it.copy(paused = false) }
        }
    }

    fun runTestTool(name: String, args: Map<String, String>) {
        val argsObj = JsonObject()
        args.forEach { (k, v) -> argsObj.addProperty(k, v) }
        val msg = JsonObject().apply {
            addProperty("type", "test_tool")
            addProperty("tool", name)
            add("args", argsObj)
        }
        wsClient?.send(msg)
        updateSession(setRunning = false) { it.copy(logLines = (it.logLines + "[test] → $name $args").takeLast(maxLogLines)) }
    }

    fun selectProfile(name: String) {
        val msg = JsonObject().apply {
            addProperty("type", "select_profile")
            addProperty("name", name)
        }
        wsClient?.send(msg)
        _state.value = _state.value.copy(phase = AppPhase.ProfileLoading(name), session = null)
        updatePersistentNotification("profile起動中: $name")
    }

    fun sendApproval(id: String, approved: Boolean) {
        val msg = JsonObject().apply {
            addProperty("type", "approve")
            addProperty("id", id)
            addProperty("decision", if (approved) "yes" else "no")
        }
        wsClient?.send(msg)
        val s = _state.value.session ?: return
        _state.value = _state.value.copy(session = s.copy(approvalRequests = s.approvalRequests.filter { it.id != id }))
    }

    fun requestLlmProviders() {
        val msg = JsonObject().apply {
            addProperty("type", "get_llm_providers")
        }
        wsClient?.send(msg)
    }

    fun setLlm(provider: String, model: String, apiKey: String = "", baseUrl: String = "") {
        val msg = JsonObject().apply {
            addProperty("type", "set_llm")
            addProperty("provider", provider)
            addProperty("model", model)
            addProperty("api_key", apiKey)
            addProperty("base_url", baseUrl)
        }
        wsClient?.send(msg)
    }

    fun clearLlmSetResult() {
        _state.value = _state.value.copy(llmSetResult = null)
    }

    // === Screen capture (MediaProjection) ===
    // ScreenCaptureActivity から呼ばれる。permission granted 後の処理。
    fun startScreenCapture(resultCode: Int, resultData: Intent, frames: Int, intervalSec: Float) {
        stopScreenCapture()

        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = try {
            mpm.getMediaProjection(resultCode, resultData)
        } catch (e: Exception) {
            Log.e(TAG, "getMediaProjection failed", e)
            return
        }

        // MediaProjection が予期せず停止した時のコールバック
        mediaProjection?.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                Log.d(TAG, "MediaProjection stopped (system callback)")
                stopScreenCapture()
            }
        }, Handler(Looper.getMainLooper()))

        // 画面解像度を取得して縮小（ネットワーク負荷軽減）
        val dm = resources.displayMetrics
        val targetLongEdge = 1280
        val scale = targetLongEdge.toFloat() / maxOf(dm.widthPixels, dm.heightPixels)
        val width = (dm.widthPixels * scale).toInt().coerceAtLeast(320)
        val height = (dm.heightPixels * scale).toInt().coerceAtLeast(320)
        val density = dm.densityDpi

        imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2)
        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "noetic_seed_screen",
            width, height, density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader?.surface, null, null
        )

        CameraStreamBridge.reset()

        screenCaptureJob = CoroutineScope(Dispatchers.IO).launch {
            runScreenCaptureLoop(frames, intervalSec, width, height)
        }
    }

    private suspend fun runScreenCaptureLoop(
        totalFrames: Int, intervalSec: Float,
        width: Int, height: Int
    ) {
        val unlimited = totalFrames == 0
        val startMs = System.currentTimeMillis()
        var sent = 0
        var i = 0
        try {
            // 初回は軽く待ってから開始（VirtualDisplay のフレーム準備）
            delay(500)
            while (true) {
                if (CameraStreamBridge.stopRequested) {
                    Log.d(TAG, "screen capture: stop requested")
                    break
                }
                if (!unlimited && i >= totalFrames) break
                if (unlimited) {
                    val elapsed = System.currentTimeMillis() - startMs
                    if (elapsed > 600_000L) {
                        Log.w(TAG, "screen capture: hard limit (10min)")
                        break
                    }
                    if (i >= 1000) {
                        Log.w(TAG, "screen capture: hard limit (1000 frames)")
                        break
                    }
                }

                val image = imageReader?.acquireLatestImage()
                if (image != null) {
                    try {
                        val jpegBytes = imageToJpeg(image, width, height)
                        if (jpegBytes != null) {
                            sendScreenFrame(jpegBytes, i + 1, totalFrames)
                            sent++
                        }
                    } finally {
                        try { image.close() } catch (_: Exception) {}
                    }
                }
                i++

                if (unlimited || i < totalFrames) {
                    delay((intervalSec * 1000).toLong())
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "screen capture loop error", e)
        } finally {
            withContext(Dispatchers.Main) {
                stopScreenCapture()
                CameraStreamBridge.sendMessage?.invoke(
                    JsonObject().apply {
                        addProperty("type", "stream_end")
                        addProperty("frame_count", sent)
                    }
                )
                CameraStreamBridge.sendMessage = null
                CameraStreamBridge.stopRequested = false
            }
        }
    }

    private fun imageToJpeg(image: Image, width: Int, height: Int): ByteArray? {
        return try {
            val planes = image.planes
            val buffer = planes[0].buffer
            val pixelStride = planes[0].pixelStride
            val rowStride = planes[0].rowStride
            val rowPadding = rowStride - pixelStride * width

            val bitmapWidth = width + rowPadding / pixelStride
            val bitmap = android.graphics.Bitmap.createBitmap(
                bitmapWidth, height,
                android.graphics.Bitmap.Config.ARGB_8888
            )
            bitmap.copyPixelsFromBuffer(buffer)

            // パディングが入ってたら actual width でクロップ
            val cropped = if (bitmapWidth > width) {
                val c = android.graphics.Bitmap.createBitmap(bitmap, 0, 0, width, height)
                bitmap.recycle()
                c
            } else {
                bitmap
            }

            val baos = java.io.ByteArrayOutputStream()
            cropped.compress(android.graphics.Bitmap.CompressFormat.JPEG, 85, baos)
            cropped.recycle()
            baos.toByteArray()
        } catch (e: Exception) {
            Log.e(TAG, "imageToJpeg failed", e)
            null
        }
    }

    private fun sendScreenFrame(jpegBytes: ByteArray, frameIndex: Int, totalFrames: Int) {
        val sender = CameraStreamBridge.sendMessage ?: return
        val b64 = android.util.Base64.encodeToString(jpegBytes, android.util.Base64.NO_WRAP)
        val now = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", java.util.Locale.US)
            .format(java.util.Date())

        // 寸法取得
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
            addProperty("source", "screen")
            addProperty("frame_index", frameIndex)
            addProperty("total_frames", totalFrames)
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

    fun stopScreenCapture() {
        screenCaptureJob?.cancel()
        screenCaptureJob = null
        try { virtualDisplay?.release() } catch (_: Exception) {}
        virtualDisplay = null
        try { imageReader?.close() } catch (_: Exception) {}
        imageReader = null
        try { mediaProjection?.stop() } catch (_: Exception) {}
        mediaProjection = null
    }
}
