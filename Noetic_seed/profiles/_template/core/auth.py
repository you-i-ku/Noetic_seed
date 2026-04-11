"""認証プロバイダ（http_request から呼ばれる）
secrets.json の auth_profiles を参照してヘッダ/パラメータを組み立てる。
汎用的なパターン（bearer / api_key）のみ。サービス固有の認証フロー
（GitHub App JWT、AWS SigV4 等）は iku が sandbox で自作する。"""
import json
from core.config import BASE_DIR

_SECRETS_FILE = BASE_DIR / "secrets.json"
_secrets_cache: dict | None = None


def _load_secrets() -> dict:
    """secrets.json をキャッシュ付きで読む。"""
    global _secrets_cache
    if _secrets_cache is not None:
        return _secrets_cache
    if not _SECRETS_FILE.exists():
        _secrets_cache = {}
        return _secrets_cache
    try:
        _secrets_cache = json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
    except Exception:
        _secrets_cache = {}
    return _secrets_cache


def reload_secrets() -> None:
    """secrets.json の再読み込み（手動編集後に呼ぶ）。"""
    global _secrets_cache
    _secrets_cache = None


def get_auth_profile(name: str) -> dict | None:
    """auth_profiles から指定名のプロファイルを取得（機密情報を含む）。内部向け。"""
    secrets = _load_secrets()
    return secrets.get("auth_profiles", {}).get(name)


# 機密フィールド名（iku からは隠す）
_SENSITIVE_FIELDS = {"token", "key", "secret", "password", "private_key"}


def get_auth_profile_info(name: str) -> dict | None:
    """auth profile のメタ情報のみを返す（token/key 等の機密フィールドを除外）。
    iku が自分でサービス固有認証フロー（GitHub App JWT 等）を書く時に使う。"""
    profile = get_auth_profile(name)
    if not profile:
        return None
    return {k: v for k, v in profile.items() if k not in _SENSITIVE_FIELDS}


def list_auth_profile_names() -> list[str]:
    """利用可能な auth profile 名の一覧を返す。機密情報は含まない。"""
    secrets = _load_secrets()
    return sorted(secrets.get("auth_profiles", {}).keys())


# === LLM クレデンシャル（内部専用、tools/ からは import 禁止）===
# このファイルを介してのみ LLM キーにアクセスする。
# tools/auth_tools.py は llm_providers セクションを絶対に露出しない。
# iku が exec_code で直接 core.auth._load_secrets() を呼ぶことは原理的に可能だが、
# その場合 exec_code の承認ゲートで人間がコードを確認する設計になっている。

def get_llm_credentials(provider: str) -> dict | None:
    """LLM プロバイダのクレデンシャル取得（core/llm.py 専用）。
    tools/ からは使わない。"""
    secrets = _load_secrets()
    providers = secrets.get("llm_providers", {})
    return providers.get(provider)


def list_llm_providers() -> list[str]:
    """登録済み LLM プロバイダ名の一覧。api_key が空でないものだけを返す。
    （Android UI にプロバイダ候補として渡すため、名前のみ）"""
    secrets = _load_secrets()
    providers = secrets.get("llm_providers", {})
    return sorted(
        name for name, cfg in providers.items()
        if cfg.get("api_key", "") or name == "lmstudio"  # lmstudio は key 不要
    )


def get_llm_provider_metadata(provider: str) -> dict | None:
    """プロバイダのメタ情報（api_key を除く）を取得。Android UI 表示用。"""
    cfg = get_llm_credentials(provider)
    if cfg is None:
        return None
    return {
        "provider": provider,
        "base_url": cfg.get("base_url", ""),
        "last_model": cfg.get("last_model", ""),
        "has_key": bool(cfg.get("api_key", "")),
    }


def save_llm_provider(provider: str, base_url: str = "", api_key: str = "", last_model: str = "") -> str | None:
    """LLM プロバイダ情報を secrets.json に保存。Android からの set_llm で使う。
    api_key が空文字なら既存のキーを温存する（再入力不要）。
    戻り値: エラーメッセージ or None（成功）"""
    if not _SECRETS_FILE.exists():
        _SECRETS_FILE.write_text(
            json.dumps({"auth_profiles": {}, "llm_providers": {}}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    try:
        secrets = json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return f"secrets.json パース失敗: {e}"

    if "llm_providers" not in secrets:
        secrets["llm_providers"] = {}

    existing = secrets["llm_providers"].get(provider, {})
    merged = {
        "base_url": base_url or existing.get("base_url", ""),
        "api_key": api_key if api_key else existing.get("api_key", ""),
        "last_model": last_model or existing.get("last_model", ""),
    }
    secrets["llm_providers"][provider] = merged

    try:
        _SECRETS_FILE.write_text(
            json.dumps(secrets, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        return f"secrets.json 書き込み失敗: {e}"

    reload_secrets()
    return None


def apply_auth(headers: dict, params: dict, auth_name: str) -> tuple[dict, dict, str | None]:
    """認証プロファイルを適用してヘッダ/パラメータを組み立てる。
    戻り値: (headers, params, error_msg_or_None)"""
    if not auth_name:
        return headers, params, None

    profile = get_auth_profile(auth_name)
    if not profile:
        return headers, params, f"auth profile '{auth_name}' が secrets.json に存在しません"

    auth_type = profile.get("type", "")

    if auth_type == "bearer":
        token = profile.get("token", "")
        if not token:
            return headers, params, f"auth profile '{auth_name}' の token が空です"
        headers = dict(headers)
        headers["Authorization"] = f"Bearer {token}"
        return headers, params, None

    elif auth_type == "api_key":
        key = profile.get("key", "")
        if not key:
            return headers, params, f"auth profile '{auth_name}' の key が空です"
        header_name = profile.get("header", "")
        param_name = profile.get("param", "")
        if header_name:
            headers = dict(headers)
            headers[header_name] = key
        elif param_name:
            params = dict(params)
            params[param_name] = key
        else:
            return headers, params, f"auth profile '{auth_name}' に header/param 指定がありません"
        return headers, params, None

    else:
        return headers, params, f"未対応の auth type: '{auth_type}'（bearer/api_key のみ対応）"
