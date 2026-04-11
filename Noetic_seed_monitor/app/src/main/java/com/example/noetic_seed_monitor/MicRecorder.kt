package com.example.noetic_seed_monitor

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 短時間マイク録音ユーティリティ。
 *
 * AudioRecord を使い、PCM 16-bit mono 16kHz で録音する。
 * Whisper / YAMNet の標準入力フォーマットなので、Python 側でそのまま使える。
 * 出力は WAV header 付きの ByteArray。
 */
object MicRecorder {
    private const val TAG = "MicRecorder"
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
    private const val BYTES_PER_SAMPLE = 2

    /** マイク権限が許可されているか */
    fun hasPermission(context: Context): Boolean {
        return ContextCompat.checkSelfPermission(
            context, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * duration_sec 秒間録音し、WAV bytes を返す。
     * パーミッションが無ければ null。
     * 録音失敗時も null。
     */
    fun record(context: Context, durationSec: Float): ByteArray? {
        if (!hasPermission(context)) {
            Log.w(TAG, "RECORD_AUDIO permission missing")
            return null
        }

        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuf <= 0) {
            Log.e(TAG, "AudioRecord.getMinBufferSize failed: $minBuf")
            return null
        }
        val bufSize = minBuf * 2  // 余裕

        val recorder = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE, CHANNEL, ENCODING, bufSize
            )
        } catch (e: SecurityException) {
            Log.e(TAG, "AudioRecord 構築失敗", e)
            return null
        }

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized")
            recorder.release()
            return null
        }

        val totalSamples = (SAMPLE_RATE * durationSec).toInt()
        val pcmStream = ByteArrayOutputStream()
        val readBuf = ByteArray(bufSize)
        var samplesRead = 0

        try {
            recorder.startRecording()
            while (samplesRead < totalSamples) {
                val n = recorder.read(readBuf, 0, readBuf.size)
                if (n <= 0) break
                pcmStream.write(readBuf, 0, n)
                samplesRead += n / BYTES_PER_SAMPLE
            }
            recorder.stop()
        } catch (e: Exception) {
            Log.e(TAG, "録音中エラー", e)
            return null
        } finally {
            recorder.release()
        }

        val pcm = pcmStream.toByteArray()
        Log.d(TAG, "録音完了: ${pcm.size} bytes (${samplesRead} samples = ${samplesRead.toFloat()/SAMPLE_RATE}s)")
        return wrapWav(pcm, SAMPLE_RATE, 1, BYTES_PER_SAMPLE * 8)
    }

    /**
     * PCM bytes に WAV ヘッダを付ける。RIFF/WAVE/fmt /data の標準フォーマット。
     * @param pcm PCM data
     * @param sampleRate サンプルレート（Hz）
     * @param channels チャンネル数
     * @param bitsPerSample サンプルあたりビット数（16）
     */
    private fun wrapWav(pcm: ByteArray, sampleRate: Int, channels: Int, bitsPerSample: Int): ByteArray {
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val dataSize = pcm.size
        val chunkSize = 36 + dataSize

        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(chunkSize)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16)  // PCM サブチャンクサイズ
        header.putShort(1)  // PCM フォーマット
        header.putShort(channels.toShort())
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort(blockAlign.toShort())
        header.putShort(bitsPerSample.toShort())
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(dataSize)

        val out = ByteArray(44 + dataSize)
        System.arraycopy(header.array(), 0, out, 0, 44)
        System.arraycopy(pcm, 0, out, 44, dataSize)
        return out
    }
}
