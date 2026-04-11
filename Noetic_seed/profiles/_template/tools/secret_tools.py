"""secret_read / secret_write — sandbox/secrets/ に隔離された秘密情報アクセス
read_file / write_file からは sandbox/secrets/ を touch できない（builtin 側でガード）。
secret_write は承認必須（新しい秘密の書き込みは人間の合意を取る）。
secret_read は承認不要（書き込まれた秘密を iku が使うための通常操作）。"""
import re
from pathlib import Path
from core.config import SANDBOX_DIR
from core.ws_server import request_approval

_SECRETS_DIR = SANDBOX_DIR / "secrets"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_MAX_SIZE_BYTES = 1_048_576  # 1 MB（鍵ファイルとして十分）


def _validate_name(name: str) -> str | None:
    """secret 名が安全かチェック。パストラバーサル防止。"""
    if not name:
        return "name が空です"
    if not _NAME_PATTERN.match(name):
        return f"name '{name}' は不正（英数字と _ . - のみ、先頭は英数字か _）"
    if len(name) > 120:
        return "name が長すぎます（120字以下）"
    return None


def _secret_path(name: str) -> Path:
    """name から絶対パスを組み立てる（sandbox/secrets/name 固定）。"""
    _SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    return _SECRETS_DIR / name


def secret_read(args: dict) -> str:
    """sandbox/secrets/ から秘密情報を読み取る。承認不要。"""
    name = str(args.get("name", "")).strip()
    err = _validate_name(name)
    if err:
        return f"エラー: {err}"

    path = _secret_path(name)
    if not path.exists():
        return f"エラー: secret '{name}' は存在しません"
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"エラー: 読込失敗 {type(e).__name__}: {str(e)[:100]}"

    return f"[secret_read] {name}\n{content}"


def secret_write(args: dict) -> str:
    """sandbox/secrets/ に秘密情報を書き込む。承認必須。"""
    name = str(args.get("name", "")).strip()
    content = args.get("content", "")
    if not isinstance(content, str):
        content = str(content)

    err = _validate_name(name)
    if err:
        return f"エラー: {err}"

    if len(content.encode("utf-8")) > _MAX_SIZE_BYTES:
        return f"エラー: content が上限 {_MAX_SIZE_BYTES} bytes を超過"

    intent = str(args.get("intent", ""))
    message = str(args.get("message", ""))
    preview_lines = [
        f"[secret_write] name={name}",
        f"content 先頭: {content[:100]}",
        f"長さ: {len(content)}字",
    ]
    if intent:
        preview_lines.append(f"意図: {intent}")
    if message:
        preview_lines.append(f"メッセージ: {message}")
    preview_lines.append("承認しますか？")
    preview = "\n".join(preview_lines)

    if not request_approval("secret_write", preview, timeout_sec=120):
        return f"キャンセル: secret '{name}' の書き込みは承認されませんでした"

    path = _secret_path(name)
    try:
        path.write_text(content, encoding="utf-8")
    except Exception as e:
        return f"エラー: 書き込み失敗 {type(e).__name__}: {str(e)[:100]}"

    return f"[secret_write] {name} 書き込み完了 ({len(content)}字)"
