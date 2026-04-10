package com.example.noetic_seed_monitor

import com.google.gson.JsonObject

/**
 * CameraStreamActivity と外部（IkuViewModel / DeviceHandler）の間のブリッジ。
 * ApprovalBridge と同じ singleton パターン。
 *
 * async 版: フレームは到着するたびに sendMessage で WebSocket 送信される。
 * stopRequested フラグを外部が true にすると、Activity は次のキャプチャ前にチェックして停止する。
 */
data class CameraStreamResult(
    val base64Frames: List<String>,
    val meta: Map<String, Any>,
)

object CameraStreamBridge {
    /** 旧バッチ版用のコールバック（後方互換）。null なら失敗 or キャンセル */
    @Volatile
    var onComplete: ((CameraStreamResult?) -> Unit)? = null

    /** async 版: Activity からフレーム毎に呼ばれる WebSocket 送信関数 */
    @Volatile
    var sendMessage: ((JsonObject) -> Unit)? = null

    /** async 版: 外部（Python 側 stop コマンド経由 or UI 停止）から停止要求 */
    @Volatile
    var stopRequested: Boolean = false

    fun reset() {
        stopRequested = false
    }
}
