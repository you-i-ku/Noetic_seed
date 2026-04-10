package com.example.noetic_seed_monitor

import android.app.NotificationManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/**
 * 通知のActionボタン（承認/却下）をハンドルするReceiver。
 * ボタンタップ → 承認結果をViewModelにコールバック → WebSocketで送信。
 */
class ApprovalActionReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val apId = intent.getStringExtra("ap_id") ?: return
        val action = intent.action ?: return
        val notifId = intent.getIntExtra("notif_id", -1)

        val approved = when (action) {
            ACTION_APPROVE -> true
            ACTION_DENY -> false
            else -> return
        }

        // ViewModelに通知（グローバル参照経由）
        ApprovalBridge.callback?.invoke(apId, approved)

        // 通知をキャンセル
        if (notifId >= 0) {
            val mgr = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            mgr.cancel(notifId)
        }
    }

    companion object {
        const val ACTION_APPROVE = "com.example.noetic_seed_monitor.ACTION_APPROVE"
        const val ACTION_DENY = "com.example.noetic_seed_monitor.ACTION_DENY"
    }
}

/**
 * 通知からのActionをViewModelに橋渡しするシングルトン。
 * BroadcastReceiverはViewModelに直接アクセスできないため。
 */
object ApprovalBridge {
    var callback: ((String, Boolean) -> Unit)? = null
}
