package com.example.noetic_seed_monitor

import com.google.gson.Gson
import com.google.gson.JsonObject
import okhttp3.*
import java.util.concurrent.TimeUnit

class IkuWebSocketClient(
    private val onMessage: (JsonObject) -> Unit,
    private val onConnected: () -> Unit,
    private val onDisconnected: () -> Unit,
    private val onError: (String) -> Unit,
    private val onDormant: (() -> Unit)? = null,
) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private val gson = Gson()
    private var ws: WebSocket? = null
    private var serverUrl: String = ""
    private var token: String = ""
    private var shouldReconnect = true
    private var dormant = false
    private var hasEverConnected = false  // 一度も接続成功してなければ初回失敗で即 dormant

    // 指数バックオフ: 200ms → 1s → 5s → 30s → 60s（以降 60s 維持）
    private val backoffSchedule = longArrayOf(200L, 1000L, 5000L, 30000L, 60000L)
    private var consecutiveFailures = 0
    private var firstFailureTime = 0L
    // 2分連続失敗でセッション死亡（Disconnected へ遷移）
    private val dormantThresholdMs = 2 * 60 * 1000L

    fun connect(url: String, authToken: String) {
        serverUrl = url
        token = authToken
        shouldReconnect = true
        dormant = false
        hasEverConnected = false
        resetBackoff()
        doConnect()
    }

    private fun doConnect() {
        val request = Request.Builder().url(serverUrl).build()
        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                val auth = JsonObject().apply {
                    addProperty("type", "auth")
                    addProperty("token", token)
                }
                webSocket.send(gson.toJson(auth))
                hasEverConnected = true
                resetBackoff()
                onConnected()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = gson.fromJson(text, JsonObject::class.java)
                    onMessage(json)
                } catch (e: Exception) {
                    // ignore malformed JSON
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                onDisconnected()
                scheduleReconnect()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                onError(t.message ?: "Unknown error")
                onDisconnected()
                scheduleReconnect()
            }
        })
    }

    private fun scheduleReconnect() {
        if (!shouldReconnect || dormant) return

        // 一度も接続成功してない（初回 Connect で即失敗）→ リトライせず即 dormant
        if (!hasEverConnected) {
            dormant = true
            onDormant?.invoke()
            return
        }

        if (firstFailureTime == 0L) {
            firstFailureTime = System.currentTimeMillis()
        }

        // dormant 判定: 連続失敗が閾値を超えたら能動再接続を停止
        val elapsed = System.currentTimeMillis() - firstFailureTime
        if (elapsed > dormantThresholdMs) {
            dormant = true
            onDormant?.invoke()
            return
        }

        val delay = backoffSchedule[minOf(consecutiveFailures, backoffSchedule.size - 1)]
        consecutiveFailures++

        Thread {
            Thread.sleep(delay)
            if (shouldReconnect && !dormant) doConnect()
        }.start()
    }

    private fun resetBackoff() {
        consecutiveFailures = 0
        firstFailureTime = 0L
    }

    /** dormant 状態から能動再接続を再開する（通知の「再接続」ボタン用） */
    fun wakeUp() {
        if (!shouldReconnect) return
        dormant = false
        resetBackoff()
        doConnect()
    }

    fun isDormant(): Boolean = dormant

    fun disconnect() {
        shouldReconnect = false
        dormant = false
        resetBackoff()
        ws?.close(1000, "bye")
        ws = null
    }

    fun send(msg: JsonObject) {
        ws?.send(gson.toJson(msg))
    }
}
