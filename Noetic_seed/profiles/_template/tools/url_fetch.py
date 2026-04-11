"""URL 経由のメディア取得ヘルパ。view_image / listen_audio で共有。

サイズ上限・タイムアウト・拡張子検出を一箇所に集約する。
fetched ファイルは指定された保存先ディレクトリに書き出され、相対パスを返す。
"""
import hashlib
from pathlib import Path

import httpx


# サイズ上限（ダウンロード時に超えたら中断）
MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB
MAX_AUDIO_BYTES = 50 * 1024 * 1024   # 50 MB

# Content-Type → 拡張子のマップ
_IMAGE_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_AUDIO_CONTENT_TYPES = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/webm": ".webm",
}


def is_url(path: str) -> bool:
    """与えられた path が URL かどうか判定。"""
    p = path.strip().lower()
    return p.startswith("http://") or p.startswith("https://")


def _ext_from_url_or_ct(url: str, content_type: str, fallback: str) -> str:
    """URL の末尾拡張子か Content-Type から拡張子を決める。両方ダメなら fallback。"""
    # まず URL 末尾
    last = url.split("?")[0].split("#")[0].rsplit("/", 1)[-1].lower()
    if "." in last:
        ext = "." + last.rsplit(".", 1)[-1]
        if len(ext) <= 6:
            return ext
    # Content-Type
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _IMAGE_CONTENT_TYPES:
        return _IMAGE_CONTENT_TYPES[ct]
    if ct in _AUDIO_CONTENT_TYPES:
        return _AUDIO_CONTENT_TYPES[ct]
    return fallback


def _hashed_filename(url: str, ext: str, prefix: str) -> str:
    """URL から決定的なファイル名を生成（同じ URL なら同じ名前 → キャッシュとしても機能）。"""
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}{ext}"


def fetch_to_file(url: str, dest_dir: Path, kind: str, timeout_sec: float = 30.0) -> tuple[Path, dict]:
    """URL から取得して dest_dir に保存する。

    kind: "image" or "audio" — サイズ上限と拡張子のフォールバックを切り替える
    Returns: (保存先 Path, {"bytes": int, "content_type": str, "url": str})
    例外: httpx.HTTPError, ValueError, IOError
    """
    if kind == "image":
        max_bytes = MAX_IMAGE_BYTES
        fallback_ext = ".jpg"
        prefix = "img"
    elif kind == "audio":
        max_bytes = MAX_AUDIO_BYTES
        fallback_ext = ".bin"
        prefix = "aud"
    else:
        raise ValueError(f"unknown kind: {kind}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Noetic_seed/1.0)",
    }

    with httpx.stream("GET", url, timeout=timeout_sec, follow_redirects=True, headers=headers) as resp:
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        ext = _ext_from_url_or_ct(url, ct, fallback_ext)
        # 拡張子バリデーション
        if kind == "image" and ext not in (".jpg", ".jpeg", ".png", ".webp"):
            raise ValueError(f"非対応の画像形式: {ext} (Content-Type={ct})")
        if kind == "audio" and ext not in (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".webm"):
            raise ValueError(f"非対応の音声形式: {ext} (Content-Type={ct})")

        filename = _hashed_filename(url, ext, prefix)
        dest_path = dest_dir / filename

        total = 0
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    f.close()
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
                    raise ValueError(f"ファイルサイズが上限を超えました: {total} > {max_bytes}")
                f.write(chunk)

    return dest_path, {"bytes": total, "content_type": ct, "url": url}
