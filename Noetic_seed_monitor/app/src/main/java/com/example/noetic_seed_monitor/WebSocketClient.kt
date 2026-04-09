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
) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private val gson = Gson()
    private var ws: WebSocket? = null
    private var serverUrl: String = ""
    private var token: String = ""
    private var shouldReconnect = true
    private var reconnectDelay = 1000L

    fun connect(url: String, authToken: String) {
        serverUrl = url
        token = authToken
        shouldReconnect = true
        reconnectDelay = 1000L
        doConnect()
    }

    private fun doConnect() {
        val request = Request.Builder().url(serverUrl).build()
        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                // 認証トークンを送信
                val auth = JsonObject().apply {
                    addProperty("type", "auth")
                    addProperty("token", token)
                }
                webSocket.send(gson.toJson(auth))
                reconnectDelay = 1000L
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
        if (!shouldReconnect) return
        Thread {
            Thread.sleep(reconnectDelay)
            reconnectDelay = (reconnectDelay * 2).coerceAtMost(30000L)
            if (shouldReconnect) doConnect()
        }.start()
    }

    fun disconnect() {
        shouldReconnect = false
        ws?.close(1000, "bye")
        ws = null
    }

    fun send(msg: JsonObject) {
        ws?.send(gson.toJson(msg))
    }
}
