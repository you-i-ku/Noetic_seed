package com.example.noetic_seed_monitor

/**
 * MicRecordActivity と外部（DeviceHandler / Service）の間のブリッジ。
 * CameraStreamBridge と同じ singleton パターン。
 *
 * 同期型: Activity が録音完了 (or 停止) 時に onResult を呼ぶ。
 * stopRequested フラグを外部 or PIP 停止ボタンが立てると、Activity は次のチャンク取得後に終了する。
 */
object MicRecordBridge {
    /** Activity から呼ばれる結果コールバック。
     *  wavBytes != null → 成功（base64 化は呼び出し側で）
     *  wavBytes == null → 失敗 or キャンセル */
    @Volatile
    var onResult: ((wavBytes: ByteArray?, durationSec: Float, error: String?) -> Unit)? = null

    /** 外部からの停止要求（PIP 停止ボタン or 戻るボタン or 上限到達） */
    @Volatile
    var stopRequested: Boolean = false

    fun reset() {
        stopRequested = false
    }
}
