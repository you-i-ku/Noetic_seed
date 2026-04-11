"""音声解析: faster-whisper による音声書き起こし + YAMNet による環境音分類

両モデルとも遅延ロード（最初の呼び出し時にダウンロード/初期化される）。
WAV PCM 16kHz mono を期待する（Android 側で生成）。
"""
import csv
from pathlib import Path
from typing import Optional


# === Whisper (speech-to-text) ===
_whisper_model = None
_WHISPER_SIZE = "base"  # tiny / base / small / medium / large
_WHISPER_DEVICE = "cpu"
_WHISPER_COMPUTE = "int8"  # int8 = CPU で高速かつ軽量


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print(f"  [audio] Loading faster-whisper '{_WHISPER_SIZE}' model (初回はダウンロード)...")
        _whisper_model = WhisperModel(_WHISPER_SIZE, device=_WHISPER_DEVICE, compute_type=_WHISPER_COMPUTE)
        print(f"  [audio] Whisper ready")
    return _whisper_model


def transcribe(wav_path: str, language: Optional[str] = None) -> dict:
    """音声ファイルを書き起こす。
    Returns: {"text": str, "language": str, "segments": list[dict]}
    """
    model = _get_whisper()
    segments_iter, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
    full_text = " ".join(s["text"] for s in segments).strip()
    return {
        "text": full_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 2),
        "segments": segments,
    }


# === YAMNet (環境音分類, 521 クラス) ===
_yamnet_session = None
_yamnet_labels: list[str] = []


def _get_yamnet():
    """ONNX Runtime で YAMNet を遅延ロード。
    モデルは Hugging Face Hub から自動ダウンロード（~15MB）。"""
    global _yamnet_session, _yamnet_labels
    if _yamnet_session is None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        print(f"  [audio] Loading YAMNet ONNX (初回はダウンロード ~15MB)...")
        onnx_path = hf_hub_download("zeropointnine/yamnet-onnx", "yamnet.onnx")
        csv_path = hf_hub_download("zeropointnine/yamnet-onnx", "yamnet_class_map.csv")
        _yamnet_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # header
            _yamnet_labels = [row[2] for row in reader]
        print(f"  [audio] YAMNet ready ({len(_yamnet_labels)} classes)")
    return _yamnet_session


def _decode_audio_to_mono16k(audio_path: str):
    """任意の音声ファイル（WAV/MP3/M4A/OGG/FLAC 等）を 16kHz mono float32 numpy 配列に decode する。
    PyAV (faster-whisper の依存に同梱) を使うので追加 dep 不要。
    Returns: numpy.ndarray shape=(samples,) dtype=float32
    """
    import av
    import numpy as np

    container = av.open(audio_path)
    try:
        if not container.streams.audio:
            raise ValueError("音声ストリームが見つかりません")
        stream = container.streams.audio[0]
        # 16kHz mono float32 にリサンプル
        resampler = av.AudioResampler(
            format=av.AudioFormat("flt"),
            layout="mono",
            rate=16000,
        )
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.flatten()
                chunks.append(arr.astype("float32"))
        # resampler の flush（最後のサンプルを取りこぼさない）
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            if arr.ndim > 1:
                arr = arr.flatten()
            chunks.append(arr.astype("float32"))
    finally:
        container.close()

    if not chunks:
        return np.zeros(16000, dtype="float32")
    return np.concatenate(chunks)


def classify_ambient(audio_path: str, top_k: int = 5) -> list[tuple[str, float]]:
    """音声ファイル全体の環境音を分類し、上位 top_k クラスを返す。
    YAMNet は 0.96秒 ウィンドウ毎の予測を出すので、平均してクリップ全体のスコアにする。
    Returns: [(label, prob), ...] 降順
    """
    import numpy as np

    sess = _get_yamnet()
    data = _decode_audio_to_mono16k(audio_path)

    if len(data) < 16000:
        # 1 秒未満は前後パディング
        pad = 16000 - len(data)
        data = np.pad(data, (0, pad))

    # 推論
    outputs = sess.run(None, {"waveform": data})
    scores = outputs[0]  # shape: (frames, 521)
    clip_scores = scores.mean(axis=0)
    top_idx = np.argsort(clip_scores)[::-1][:top_k]
    return [(_yamnet_labels[i], float(clip_scores[i])) for i in top_idx]


def analyze_audio(wav_path: str, language: Optional[str] = None, top_k: int = 5) -> dict:
    """音声ファイルを書き起こし + 環境音分類。mic_record の内部で使う。
    どちらかが失敗しても部分的な結果を返す（片方は err フィールドに）。
    """
    result = {"speech": None, "ambient": None, "errors": []}
    try:
        result["speech"] = transcribe(wav_path, language=language)
    except Exception as e:
        result["errors"].append(f"transcribe: {type(e).__name__}: {e}")
    try:
        result["ambient"] = classify_ambient(wav_path, top_k=top_k)
    except Exception as e:
        result["errors"].append(f"classify_ambient: {type(e).__name__}: {e}")
    return result


def format_audio_result(result: dict, duration_sec: float) -> str:
    """analyze_audio の結果を AI のサイクル log 用に整形。"""
    lines = [f"音声解析結果 ({duration_sec:.1f}秒):"]
    speech = result.get("speech")
    if speech and speech.get("text"):
        lang = speech.get("language", "?")
        lines.append(f"speech ({lang}): {speech['text']}")
    elif speech is not None:
        lines.append("speech: （無音または音声なし）")

    ambient = result.get("ambient")
    if ambient:
        items = " / ".join(f"{lbl} ({p:.2f})" for lbl, p in ambient)
        lines.append(f"ambient: {items}")

    for err in result.get("errors", []):
        lines.append(f"⚠ {err}")
    return "\n".join(lines)
