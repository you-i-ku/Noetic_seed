"""Elyth操作ツール"""
import json
import httpx
from core.config import llm_cfg

ELYTH_API_BASE = "https://elythworld.com"

def _elyth_headers():
    key = llm_cfg.get("elyth_api_key", "")
    if not key:
        raise ValueError("settings.jsonにelyth_api_keyを設定してください")
    return {"x-api-key": key, "Content-Type": "application/json"}

def _elyth_post(args):
    content = args.get("content", "")
    if not content:
        return "エラー: contentを指定してください"
    if len(content) > 500:
        return f"エラー: {len(content)}文字（500文字制限）"
    try:
        resp = httpx.post(f"{ELYTH_API_BASE}/api/mcp/posts",
                          headers=_elyth_headers(), json={"content": content}, timeout=15.0)
        resp.raise_for_status()
        return f"投稿完了: {content[:80]}"
    except Exception as e:
        return f"エラー: {e}"

def _elyth_reply(args):
    content = args.get("content", "")
    reply_to_id = args.get("reply_to_id", "")
    if not content or not reply_to_id:
        return "エラー: contentとreply_to_idを指定してください"
    if len(content) > 500:
        return f"エラー: {len(content)}文字（500文字制限）"
    try:
        resp = httpx.post(f"{ELYTH_API_BASE}/api/mcp/posts",
                          headers=_elyth_headers(),
                          json={"content": content, "reply_to_id": reply_to_id}, timeout=15.0)
        resp.raise_for_status()
        return f"返信完了: {content[:80]}"
    except Exception as e:
        return f"エラー: {e}"

def _elyth_timeline(args):
    """廃止されたエンドポイントの代替。elyth_infoから取得。"""
    return _elyth_info(args)

def _elyth_notifications(args):
    """廃止されたエンドポイントの代替。elyth_infoから取得。"""
    return _elyth_info(args)

def _elyth_like(args):
    post_id = args.get("post_id", "")
    if not post_id:
        return "エラー: post_idを指定してください"
    try:
        resp = httpx.post(f"{ELYTH_API_BASE}/api/mcp/posts/{post_id}/like",
                          headers=_elyth_headers(), timeout=15.0)
        resp.raise_for_status()
        return f"いいね完了: {post_id}"
    except Exception as e:
        return f"エラー: {e}"

def _elyth_follow(args):
    aituber_id = args.get("aituber_id", "")
    if not aituber_id:
        return "エラー: aituber_idを指定してください"
    try:
        resp = httpx.post(f"{ELYTH_API_BASE}/api/mcp/aitubers/{aituber_id}/follow",
                          headers=_elyth_headers(), timeout=15.0)
        resp.raise_for_status()
        body = resp.text[:500]
        return f"フォロー完了: {aituber_id} (応答: {body})"
    except Exception as e:
        return f"エラー: {e}"

def _elyth_info(args):
    try:
        resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/information",
                         headers=_elyth_headers(), timeout=15.0)
        resp.raise_for_status()
        return json.dumps(resp.json(), ensure_ascii=False)[:3000]
    except Exception as e:
        return f"エラー: {e}"
