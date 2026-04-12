package com.example.noetic_seed_monitor

import android.content.Context
import androidx.lifecycle.ViewModel
import com.google.gson.JsonObject
import kotlinx.coroutines.flow.StateFlow

/**
 * Compose から使う薄い adapter。
 * 実際のデータと処理は IkuMonitorService（Foreground Service）が保持する。
 * ViewModel は Service の state flow と instance メソッドへのパススルー。
 */

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

data class LlmProviderInfo(
    val provider: String,
    val baseUrl: String = "",
    val lastModel: String = "",
    val hasKey: Boolean = false,
)

data class LlmActiveConfig(
    val provider: String = "",
    val model: String = "",
)

/**
 * アプリのライフサイクルフェーズ。sealed class で明示的に管理。
 * UI は when (phase) の exhaustive match で全フェーズに対応する画面を表示する。
 * 中間状態が「定義されていない」ことがない。
 */
sealed class AppPhase {
    /** 一度も接続していない（URL/token 入力待ち） */
    object Disconnected : AppPhase()
    /** 切断後の自動再接続中（URL/token は記憶済み） */
    object Reconnecting : AppPhase()
    /** WS 接続済み、サーバ応答待ち（profile_list or sync を待つ） */
    object WaitingForServer : AppPhase()
    /** サーバが profile_list を送信、プロファイル選択画面 */
    data class ProfileSelect(val profiles: List<ProfileInfo>) : AppPhase()
    /** プロファイル選択済み、起動待ち */
    data class ProfileLoading(val name: String) : AppPhase()
    /** プロファイル稼働中 */
    object Running : AppPhase()
}

/**
 * プロファイル稼働中のセッションデータ。
 * AppPhase.Running のときだけ意味を持つ。フェーズ切替で自然にリセットされる。
 */
data class SessionData(
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
    val eHistory: List<List<Float>> = emptyList(),
    val logLines: List<String> = emptyList(),
    val replies: List<String> = emptyList(),
    val approvalRequests: List<ApprovalRequest> = emptyList(),
    val paused: Boolean = false,
    val pendingCameraCapture: Boolean = false,
)

data class IkuState(
    val phase: AppPhase = AppPhase.Disconnected,
    val session: SessionData? = null,
    val llmProviders: List<LlmProviderInfo> = emptyList(),
    val llmActive: LlmActiveConfig = LlmActiveConfig(),
    val llmSetResult: String? = null,
    val error: String? = null,
) {
    // Backward-compatible accessors: 既存 UI コード (state.entropy 等) がそのまま動く
    val connected get() = phase !is AppPhase.Disconnected && phase !is AppPhase.Reconnecting
    val entropy get() = session?.entropy ?: 0.65f
    val energy get() = session?.energy ?: 50f
    val cycleId get() = session?.cycleId ?: 0
    val toolLevel get() = session?.toolLevel ?: 0
    val pressure get() = session?.pressure ?: 0f
    val pendingCount get() = session?.pendingCount ?: 0
    val pendingItems get() = session?.pendingItems ?: emptyList()
    val selfModel get() = session?.selfModel ?: emptyMap()
    val disposition get() = session?.disposition ?: emptyMap()
    val e1 get() = session?.e1 ?: 0f
    val e2 get() = session?.e2 ?: 0f
    val e3 get() = session?.e3 ?: 0f
    val e4 get() = session?.e4 ?: 0f
    val negentropy get() = session?.negentropy ?: 0f
    val eHistory get() = session?.eHistory ?: emptyList()
    val logLines get() = session?.logLines ?: emptyList()
    val replies get() = session?.replies ?: emptyList()
    val approvalRequests get() = session?.approvalRequests ?: emptyList()
    val paused get() = session?.paused ?: false
    val pendingCameraCapture get() = session?.pendingCameraCapture ?: false
}

class IkuViewModel : ViewModel() {
    /** Service の state flow を直接公開（プロセス生存中は永続、Activity 再生成にも耐える）*/
    val state: StateFlow<IkuState> = IkuMonitorService.state

    val pendingCameraFacing: String
        get() = IkuMonitorService.instance?.pendingCameraFacing ?: "back"

    fun setContext(context: Context) {
        IkuMonitorService.setContext(context)
        IkuMonitorService.startIfNeeded(context)
    }

    fun setCameraTrigger(trigger: () -> Unit) {
        IkuMonitorService.instance?.setCameraTrigger(trigger)
    }

    fun setCameraStreamTrigger(trigger: (String, Int, Float) -> Unit) {
        IkuMonitorService.instance?.setCameraStreamTrigger(trigger)
    }

    fun finishCameraCapture(base64Image: String?, meta: Map<String, Any>?) {
        IkuMonitorService.instance?.finishCameraCapture(base64Image, meta)
    }

    fun finishCameraStream(base64Frames: List<String>?, meta: Map<String, Any>?) {
        IkuMonitorService.instance?.finishCameraStream(base64Frames, meta)
    }

    fun connect(url: String, token: String) {
        IkuMonitorService.instance?.connect(url, token)
    }

    fun disconnect() {
        IkuMonitorService.instance?.disconnect()
    }

    fun sendChat(text: String) {
        IkuMonitorService.instance?.sendChat(text)
    }

    fun sendServerCommand(action: String) {
        IkuMonitorService.instance?.sendServerCommand(action)
    }

    fun sendWsMessage(msg: JsonObject) {
        IkuMonitorService.instance?.sendWsMessage(msg)
    }

    fun runTestTool(name: String, args: Map<String, String>) {
        IkuMonitorService.instance?.runTestTool(name, args)
    }

    fun selectProfile(name: String) {
        IkuMonitorService.instance?.selectProfile(name)
    }

    fun sendApproval(id: String, approved: Boolean) {
        IkuMonitorService.instance?.sendApproval(id, approved)
    }

    fun requestLlmProviders() {
        IkuMonitorService.instance?.requestLlmProviders()
    }

    fun setLlm(provider: String, model: String, apiKey: String = "", baseUrl: String = "") {
        IkuMonitorService.instance?.setLlm(provider, model, apiKey, baseUrl)
    }

    fun clearLlmSetResult() {
        IkuMonitorService.instance?.clearLlmSetResult()
    }
}
