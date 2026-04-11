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
    val eHistory: List<List<Float>> = emptyList(),
    val logLines: List<String> = emptyList(),
    val replies: List<String> = emptyList(),
    val approvalRequests: List<ApprovalRequest> = emptyList(),
    val profiles: List<ProfileInfo> = emptyList(),
    val profileSelected: Boolean = false,
    val profileStarted: Boolean = false,
    val pendingCameraCapture: Boolean = false,
    val llmProviders: List<LlmProviderInfo> = emptyList(),
    val llmActive: LlmActiveConfig = LlmActiveConfig(),
    val llmSetResult: String? = null,
    val error: String? = null,
)

class IkuViewModel : ViewModel() {
    /** Service の state flow を直接公開（プロセス生存中は永続、Activity 再生成にも耐える）*/
    val state: StateFlow<IkuState> = IkuMonitorService.state

    val pendingCameraFacing: String
        get() = IkuMonitorService.instance?.pendingCameraFacing ?: "back"

    fun setContext(context: Context) {
        IkuMonitorService.setContext(context)
        // Service 未起動ならここで startForegroundService
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
