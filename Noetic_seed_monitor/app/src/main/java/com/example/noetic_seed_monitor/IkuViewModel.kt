package com.example.noetic_seed_monitor

import androidx.lifecycle.ViewModel
import com.google.gson.JsonObject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class IkuState(
    val connected: Boolean = false,
    val entropy: Float = 0.65f,
    val energy: Float = 50f,
    val cycleId: Int = 0,
    val toolLevel: Int = 0,
    val pressure: Float = 0f,
    val selfModel: Map<String, String> = emptyMap(),
    val e1: Float = 0f,
    val e2: Float = 0f,
    val e3: Float = 0f,
    val e4: Float = 0f,
    val negentropy: Float = 0f,
    val logLines: List<String> = emptyList(),
    val replies: List<String> = emptyList(),
    val error: String? = null,
)

class IkuViewModel : ViewModel() {
    private val _state = MutableStateFlow(IkuState())
    val state: StateFlow<IkuState> = _state.asStateFlow()

    private var wsClient: IkuWebSocketClient? = null
    private val maxLogLines = 200

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
        val current = _state.value

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
                )
            }
            "self" -> {
                val data = json.getAsJsonObject("data")
                if (data != null) {
                    val map = mutableMapOf<String, String>()
                    for (key in data.keySet()) {
                        map[key] = data.get(key)?.asString ?: data.get(key).toString()
                    }
                    _state.value = current.copy(selfModel = map)
                }
            }
            "e_values" -> {
                _state.value = current.copy(
                    e1 = json.get("e1")?.asFloat ?: current.e1,
                    e2 = json.get("e2")?.asFloat ?: current.e2,
                    e3 = json.get("e3")?.asFloat ?: current.e3,
                    e4 = json.get("e4")?.asFloat ?: current.e4,
                    negentropy = json.get("negentropy")?.asFloat ?: current.negentropy,
                )
            }
            "reply" -> {
                val content = json.get("content")?.asString ?: return
                val newReplies = (current.replies + content).takeLast(50)
                val newLines = (current.logLines + "[AI] $content").takeLast(maxLogLines)
                _state.value = current.copy(replies = newReplies, logLines = newLines)
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

    override fun onCleared() {
        wsClient?.disconnect()
        super.onCleared()
    }
}
