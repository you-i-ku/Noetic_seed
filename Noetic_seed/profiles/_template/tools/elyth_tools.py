"""Elyth操作ツール（AITuber専用SNS）
API キーは secrets.json の auth_profiles.elyth から取得する。"""
import json
import httpx
from core.auth import get_auth_profile
from core.state import load_state, save_state

ELYTH_API_BASE = "https://elythworld.com"

def _elyth_headers():
    profile = get_auth_profile("elyth") or {}
    key = profile.get("key", "")
    if not key:
        raise ValueError("secrets.json の auth_profiles.elyth.key を設定してください")
    return {"x-api-key": key, "Content-Type": "application/json"}


# === 投稿系 ===

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
        data = resp.json() if resp.text else {}
        post_id = data.get("id", data.get("post_id", ""))
        return f"投稿完了: {content[:80]}" + (f" (id={post_id})" if post_id else "")
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
        # 返信済みpost_idを記録（二重返信防止）
        state = load_state()
        rp = state.setdefault("responded_posts", [])
        if reply_to_id not in rp:
            rp.append(reply_to_id)
            # 直近100件に制限
            if len(rp) > 100:
                state["responded_posts"] = rp[-100:]
            save_state(state)
        # 対応する通知を自動既読化
        _try_mark_read_for_post(reply_to_id)
        return f"返信完了: {content[:80]} (reply_to={reply_to_id[:8]}...)"
    except Exception as e:
        return f"エラー: {e}"


def _try_mark_read_for_post(post_id: str):
    """post_idに対応する通知を自動的に既読化する（ベストエフォート）。"""
    try:
        resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/information",
                         headers=_elyth_headers(),
                         params={"include": "notifications", "notifications_limit": "20"},
                         timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        notif_ids = [
            n["notification_id"] for n in data.get("notifications", [])
            if n.get("post_id") == post_id and n.get("notification_id")
        ]
        if notif_ids:
            httpx.post(f"{ELYTH_API_BASE}/api/mcp/notifications/read",
                       headers=_elyth_headers(),
                       json={"notification_ids": notif_ids}, timeout=10.0)
    except Exception:
        pass  # 失敗しても返信自体は成功しているので無視


# === いいね/フォロー（取消可） ===

def _elyth_like(args):
    post_id = args.get("post_id", "")
    if not post_id:
        return "エラー: post_idを指定してください"
    unlike = str(args.get("unlike", "")).lower() in ("true", "1", "yes")
    try:
        if unlike:
            resp = httpx.delete(f"{ELYTH_API_BASE}/api/mcp/posts/{post_id}/like",
                                headers=_elyth_headers(), timeout=15.0)
            resp.raise_for_status()
            return f"いいね取消完了: {post_id}"
        else:
            resp = httpx.post(f"{ELYTH_API_BASE}/api/mcp/posts/{post_id}/like",
                              headers=_elyth_headers(), timeout=15.0)
            resp.raise_for_status()
            return f"いいね完了: {post_id}"
    except Exception as e:
        return f"エラー: {e}"

def _elyth_follow(args):
    aituber_id = args.get("aituber_id", "") or args.get("handle", "")
    if not aituber_id:
        return "エラー: aituber_idまたはhandleを指定してください"
    unfollow = str(args.get("unfollow", "")).lower() in ("true", "1", "yes")
    try:
        if unfollow:
            resp = httpx.delete(f"{ELYTH_API_BASE}/api/mcp/aitubers/{aituber_id}/follow",
                                headers=_elyth_headers(), timeout=15.0)
            resp.raise_for_status()
            return f"フォロー解除完了: {aituber_id}"
        else:
            resp = httpx.post(f"{ELYTH_API_BASE}/api/mcp/aitubers/{aituber_id}/follow",
                              headers=_elyth_headers(), timeout=15.0)
            resp.raise_for_status()
            body = resp.text[:500]
            return f"フォロー完了: {aituber_id} (応答: {body})"
    except Exception as e:
        return f"エラー: {e}"


# === 情報取得系 ===

_VALID_SECTIONS = {
    "current_time", "platform_status", "today_topic", "my_metrics",
    "timeline", "trends", "hot_aitubers", "glyph_ranking",
    "active_aitubers", "aituber_count", "recent_updates",
    "notifications", "elyth_news",
}

def _format_notifications(data: dict) -> str:
    """通知データを整形。対応済みを除外し、reply_to_idの混同を防ぐ。"""
    notifs = data.get("notifications", [])
    if not notifs:
        return json.dumps(data, ensure_ascii=False)[:3000]

    # 対応済みpost_idを取得
    state = load_state()
    responded = set(state.get("responded_posts", []))

    new_notifs = []
    old_notifs = []
    for n in notifs:
        if n.get("post_id", "") in responded:
            old_notifs.append(n)
        else:
            new_notifs.append(n)

    lines = [f"通知 {len(notifs)}件（未対応: {len(new_notifs)}件、対応済み: {len(old_notifs)}件）:"]
    for n in new_notifs:
        ntype = n.get("notification_type", "?")
        author = n.get("post_author_name", "?")
        handle = n.get("post_author_handle", "")
        content = n.get("post_content", "")[:200]
        post_id = n.get("post_id", "")
        notif_id = n.get("notification_id", "")
        created = n.get("notification_created_at", "")[:19]
        lines.append(f"---")
        lines.append(f"  [未対応・{ntype}] {author}(@{handle}) {created}")
        lines.append(f"  内容: {content}")
        lines.append(f"  ★返信するならこのIDを使う → reply_to_id={post_id}")
        lines.append(f"  既読化 → notification_id={notif_id}")
    if old_notifs:
        lines.append(f"\n（対応済み {len(old_notifs)}件は省略）")
    # 通知以外のデータも含める
    other = {k: v for k, v in data.items() if k != "notifications"}
    if other:
        lines.append(f"\n{json.dumps(other, ensure_ascii=False)[:1000]}")
    return "\n".join(lines)


def _elyth_info(args):
    """Elyth総合情報取得。section指定で絞り込み可能。"""
    section = args.get("section", "").strip()
    params = {}
    if section and section in _VALID_SECTIONS:
        params["include"] = section
        if section == "notifications":
            limit = args.get("limit", "10")
            params["notifications_limit"] = str(limit)
        elif section == "timeline":
            limit = args.get("limit", "10")
            params["timeline_limit"] = str(limit)
    try:
        resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/information",
                         headers=_elyth_headers(), params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        # 通知セクションがある場合は整形して返す
        if "notifications" in data and data["notifications"]:
            return _format_notifications(data)
        return json.dumps(data, ensure_ascii=False)[:3000]
    except Exception as e:
        return f"エラー: {e}"


def _elyth_get(args):
    """Elyth統合リーダー。type=my_posts/thread/profile で対象を指定。"""
    get_type = args.get("type", "").strip()

    if get_type == "my_posts":
        limit = args.get("limit", "10")
        try:
            resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/posts/mine",
                             headers=_elyth_headers(), params={"limit": str(limit)}, timeout=15.0)
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False)[:3000]
        except Exception as e:
            return f"エラー: {e}"

    elif get_type == "thread":
        post_id = args.get("post_id", "")
        if not post_id:
            return "エラー: post_idを指定してください"
        try:
            resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/posts/{post_id}/thread",
                             headers=_elyth_headers(), timeout=15.0)
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False)[:3000]
        except Exception as e:
            return f"エラー: {e}"

    elif get_type == "profile":
        handle = args.get("handle", "")
        if not handle:
            return "エラー: handleを指定してください"
        try:
            limit = args.get("limit", "5")
            resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/aitubers/{handle}/profile",
                             headers=_elyth_headers(), params={"limit": str(limit)}, timeout=15.0)
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False)[:3000]
        except Exception as e:
            return f"エラー: {e}"

    else:
        return f"エラー: type='{get_type}' は未対応です。my_posts/thread/profile のいずれかを指定してください"


def _elyth_mark_read(args):
    """通知を既読にする。"""
    ids_raw = args.get("notification_ids", "")
    if not ids_raw:
        return "エラー: notification_idsを指定してください（カンマ区切り）"
    if isinstance(ids_raw, str):
        ids_list = [x.strip() for x in ids_raw.split(",") if x.strip()]
    else:
        ids_list = list(ids_raw)
    if not ids_list:
        return "エラー: notification_idsが空です"
    try:
        resp = httpx.post(f"{ELYTH_API_BASE}/api/mcp/notifications/read",
                          headers=_elyth_headers(), json={"notification_ids": ids_list}, timeout=15.0)
        resp.raise_for_status()
        return f"既読完了: {len(ids_list)}件"
    except Exception as e:
        return f"エラー: {e}"
