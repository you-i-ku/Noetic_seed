"""汎用 HTTP リクエストツール
任意の URL に GET/POST/PUT/DELETE/PATCH を投げる。
auth= で secrets.json の auth_profiles を参照して認証ヘッダを自動付与。
書き込み系（POST/PUT/DELETE/PATCH）は承認必要。"""
import json as _json
import httpx
from core.auth import apply_auth, get_auth_profile
from core.ws_server import request_approval

# 上限設定（A-F 議論で決定）
_BODY_MAX_BYTES = 1_048_576       # 1 MB
_RESPONSE_MAX_CHARS = 100_000     # レスポンス表示の上限（iku プロンプトに載る量を抑制）
_DEFAULT_TIMEOUT = 60             # 秒
_WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}


def _build_preview(method: str, url: str, auth: str, body_preview: str, intent: str, message: str) -> str:
    """承認通知のプレビュー文を組み立てる。"""
    lines = [f"[http_request] {method} {url[:100]}"]
    if auth:
        lines.append(f"auth: {auth}")
    if body_preview:
        lines.append(f"body: {body_preview[:200]}")
    if intent:
        lines.append(f"意図: {intent}")
    if message:
        lines.append(f"メッセージ: {message}")
    lines.append("承認しますか？")
    return "\n".join(lines)


def http_request(args: dict) -> str:
    """汎用 HTTP ツール。引数辞書から設定を取り出してリクエスト送信。"""
    url = str(args.get("url", "")).strip()
    if not url:
        return "エラー: url が指定されていません"

    method = str(args.get("method", "GET")).upper().strip()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        return f"エラー: 未対応の method '{method}'"

    # headers: dict or JSON文字列
    headers_raw = args.get("headers", {})
    if isinstance(headers_raw, str):
        try:
            headers = _json.loads(headers_raw) if headers_raw.strip() else {}
        except Exception:
            return "エラー: headers の JSON パース失敗"
    else:
        headers = dict(headers_raw) if headers_raw else {}

    # params（GET の query string 用）
    params_raw = args.get("params", {})
    if isinstance(params_raw, str):
        try:
            params = _json.loads(params_raw) if params_raw.strip() else {}
        except Exception:
            return "エラー: params の JSON パース失敗"
    else:
        params = dict(params_raw) if params_raw else {}

    # body: dict(JSON化) or 文字列
    body_raw = args.get("body", None)
    body_json = None
    body_text = None
    if body_raw is not None:
        if isinstance(body_raw, dict):
            body_json = body_raw
        elif isinstance(body_raw, str):
            s = body_raw.strip()
            if s.startswith("{") or s.startswith("["):
                try:
                    body_json = _json.loads(s)
                except Exception:
                    body_text = body_raw
            else:
                body_text = body_raw

    # サイズチェック
    if body_text is not None and len(body_text.encode("utf-8")) > _BODY_MAX_BYTES:
        return f"エラー: body が上限 {_BODY_MAX_BYTES} bytes を超過"
    if body_json is not None:
        try:
            _sz = len(_json.dumps(body_json, ensure_ascii=False).encode("utf-8"))
            if _sz > _BODY_MAX_BYTES:
                return f"エラー: body が上限 {_BODY_MAX_BYTES} bytes を超過"
        except Exception:
            return "エラー: body の JSON シリアライズ失敗"

    # 認証プロファイル適用
    auth_name = str(args.get("auth", "")).strip()
    if auth_name:
        headers, params, auth_err = apply_auth(headers, params, auth_name)
        if auth_err:
            return f"エラー: {auth_err}"

    # 書き込み系は承認必須
    if method in _WRITE_METHODS:
        intent = str(args.get("intent", ""))
        message = str(args.get("message", ""))
        body_preview = ""
        if body_json is not None:
            try:
                body_preview = _json.dumps(body_json, ensure_ascii=False)[:200]
            except Exception:
                body_preview = "<dict>"
        elif body_text:
            body_preview = body_text[:200]
        preview = _build_preview(method, url, auth_name, body_preview, intent, message)
        if not request_approval("http_request", preview, timeout_sec=120):
            return f"キャンセル: {method} {url[:80]} は承認されませんでした"

    # タイムアウト
    try:
        timeout_val = int(args.get("timeout", _DEFAULT_TIMEOUT))
        if timeout_val <= 0 or timeout_val > 300:
            timeout_val = _DEFAULT_TIMEOUT
    except Exception:
        timeout_val = _DEFAULT_TIMEOUT

    # リクエスト送信
    try:
        resp = httpx.request(
            method=method,
            url=url,
            headers=headers,
            params=params if params else None,
            json=body_json if body_json is not None else None,
            content=body_text if body_text is not None else None,
            timeout=timeout_val,
            follow_redirects=True,
        )
    except httpx.TimeoutException:
        return f"エラー: timeout ({timeout_val}s) {method} {url[:80]}"
    except httpx.RequestError as e:
        return f"エラー: request failed: {str(e)[:200]}"
    except Exception as e:
        return f"エラー: {type(e).__name__}: {str(e)[:200]}"

    # レスポンス整形（戻り値 dict を文字列化して返す）
    body_str = resp.text
    if len(body_str) > _RESPONSE_MAX_CHARS:
        body_str = body_str[:_RESPONSE_MAX_CHARS] + f"\n...(表示上 {_RESPONSE_MAX_CHARS}/{len(resp.text)}字に切詰。ツール実行時は完全取得済)"

    result_dict = {
        "status": resp.status_code,
        "headers": dict(resp.headers),
        "body": body_str,
    }

    # status が 4xx/5xx の場合は「エラー」プレフィックスを付けて既存ペナルティと整合
    if resp.status_code >= 400:
        return f"エラー: HTTP {resp.status_code} — {_json.dumps(result_dict, ensure_ascii=False)[:2000]}"

    return _json.dumps(result_dict, ensure_ascii=False)
