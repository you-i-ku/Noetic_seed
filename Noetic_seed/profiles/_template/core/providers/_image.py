"""自前の画像読込・base64 ヘルパ (provider 共用)。

claw-code 準拠。Noetic 既存 core/llm.py には依存しない。
Pillow が無ければリサイズなしで生バイトを base64 化する。
"""
import base64
import io
from pathlib import Path
from typing import Optional


def load_image_base64(image_path: str,
                      max_size: int = 2048) -> Optional[tuple]:
    """画像を読込→ (リサイズして) JPEG base64 + media_type を返す。

    戻り値: (base64_str, "image/jpeg") or None。
    Pillow があればリサイズ + JPEG 化。なければ生バイトを image/jpeg として送る
    (拡張子から型を判定するのは provider 側で)。
    """
    p = Path(image_path)
    if not p.exists():
        return None

    try:
        from PIL import Image
        img = Image.open(p)
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b = buf.getvalue()
    except ImportError:
        b = p.read_bytes()
    except Exception:
        return None

    return (base64.b64encode(b).decode("ascii"), "image/jpeg")
