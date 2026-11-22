"""Web — WebFetch / WebSearch / RemoteTrigger.

claw-code 参照:
  - rust/crates/runtime/src/web_fetch.rs
  - rust/crates/runtime/src/web_search.rs
  - rust/crates/runtime/src/remote_trigger.rs

厳密 claw-code 準拠。Noetic 既存 tools/web.py への forward は**しない**。
WebSearch は Brave Search API (auth_profiles.brave 経由)。description 取得 +
count/offset 調整対応 (段階13 Phase 6 hotfix で DuckDuckGo HTML scrape から差替)。
"""
import re
from typing import Optional
from urllib.parse import quote_plus

import httpx

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry, ToolError, result_redactor
from core.runtime.tool_schema import ToolSpec


MAX_FETCH_BYTES = 2 * 1024 * 1024  # 2 MB
USER_AGENT = "Mozilla/5.0 (compatible; ClawCode/1.0)"


# ============================================================
# WebFetch
# ============================================================

def web_fetch(inp: dict) -> str:
    url = (inp.get("url") or "").strip()
    prompt = (inp.get("prompt") or "").strip()
    if not url:
        raise ToolError("Error: url is required")
    if not prompt:
        raise ToolError("Error: prompt is required")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ToolError("Error: url must start with http:// or https://")

    try:
        resp = httpx.get(
            url, timeout=30,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise ToolError(f"Error: HTTP {e.response.status_code}")
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error: {type(e).__name__}: {e}")

    body = result_redactor()(resp.text)
    if len(body.encode("utf-8", errors="replace")) > MAX_FETCH_BYTES:
        body = body[:MAX_FETCH_BYTES // 4]

    content_type = (resp.headers.get("content-type") or "").lower()
    if "html" in content_type:
        body = _html_to_text(body)

    # claw-code と同様、LLM に質問応答させる形式の文字列を返す
    # 本 Phase では LLM 呼出は含めず、prompt + 本文を返すだけ。
    # ConversationRuntime 側が hook で LLM に投げる設計。
    return (f"[WebFetch: {url}]\n"
            f"Prompt: {prompt}\n\n"
            f"--- Content ---\n{body[:20000]}")


def _html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        text = re.sub(r"<script[^>]*>.*?</script>", "", html,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)


# ============================================================
# WebSearch
# ============================================================

def web_search(inp: dict) -> str:
    query = (inp.get("query") or "").strip()
    if not query:
        raise ToolError("Error: query is required")

    # 副次改善 D: count / offset 調整可能 (Brave 仕様 count 1-20 / offset 0-9)
    raw_count = inp.get("count", 10)
    try:
        count = int(raw_count) if raw_count is not None else 10
    except (ValueError, TypeError):
        raise ToolError("Error: count must be an integer (1-20)")
    count = max(1, min(20, count))

    raw_offset = inp.get("offset", 0)
    try:
        offset = int(raw_offset) if raw_offset is not None else 0
    except (ValueError, TypeError):
        raise ToolError("Error: offset must be an integer (0-9)")
    offset = max(0, min(9, offset))

    allowed = inp.get("allowed_domains") or []
    blocked = inp.get("blocked_domains") or []

    # 認証 (auth_profiles.brave から API key 取得)
    from core.auth import apply_auth
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params = {"q": query, "count": count, "offset": offset}
    headers, params, err = apply_auth(headers, params, "brave")
    if err:
        raise ToolError((f"Error: {err}。secrets.json の auth_profiles.brave.key に "
                f"Brave Search API key を設定してください "
                f"(取得元: https://api-dashboard.search.brave.com)。"))

    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=30,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise ToolError(("Error: Brave Search rate limit (429)。"
                    "1 req/sec 制限を超過しました。数秒待ってから再試行してください。"))
        if e.response.status_code in (401, 403):
            raise ToolError((f"Error: Brave 認証失敗 (HTTP {e.response.status_code})。"
                    f"secrets.json の auth_profiles.brave.key を確認してください。"))
        raise ToolError(f"Error: Brave HTTP {e.response.status_code}")
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error: {type(e).__name__}: {e}")

    try:
        data = resp.json()
    except ValueError as e:
        raise ToolError(f"Error: Brave JSON parse 失敗: {e}")

    web_results = (data.get("web") or {}).get("results") or []

    # 副次改善 C: description (抜粋) を含めて返す
    results: list = []
    for r in web_results:
        url = r.get("url") or ""
        title = (r.get("title") or "").strip()
        description = (r.get("description") or "").strip()
        # description には <strong> 等のハイライトタグが混じる場合があるため除去
        description = re.sub(r"<[^>]+>", "", description).strip()

        if allowed and not any(d in url for d in allowed):
            continue
        if blocked and any(d in url for d in blocked):
            continue

        results.append((title, url, description))

    if not results:
        return f"No search results for: {query}"

    lines = [f"Search results for: {query} (count={count}, offset={offset})"]
    for title, url, description in results:
        lines.append(f"  - {title}")
        lines.append(f"    {url}")
        if description:
            lines.append(f"    {description}")
    return "\n".join(lines)


# ============================================================
# RemoteTrigger
# ============================================================

def remote_trigger(inp: dict) -> str:
    url = (inp.get("url") or "").strip()
    method = (inp.get("method") or "POST").upper()
    headers = inp.get("headers") or {}
    body = inp.get("body")

    if not url:
        raise ToolError("Error: url is required")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ToolError("Error: url must start with http:// or https://")
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        raise ToolError(f"Error: unsupported method '{method}'")

    kwargs: dict = {"headers": headers, "timeout": 60}
    if body is not None:
        if isinstance(body, (dict, list)):
            kwargs["json"] = body
        else:
            kwargs["content"] = str(body)

    try:
        resp = httpx.request(method, url, **kwargs)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Error: {type(e).__name__}: {e}")

    safe_text = result_redactor()(resp.text or "")
    snippet = safe_text[:2000]
    more = len(safe_text) - len(snippet)
    lines = [f"{method} {url} -> {resp.status_code}"]
    if snippet:
        lines.append(snippet)
    if more > 0:
        lines.append(f"[... {more} more chars truncated ...]")
    message = "\n".join(lines)
    if resp.status_code >= 400:
        raise ToolError(message, detail={"status_code": resp.status_code,
                                         "method": method, "url": url})
    return message


# ============================================================
# register
# ============================================================

def register(registry: ToolRegistry) -> None:
    specs = [
        ToolSpec(
            name="WebFetch",
            description="Fetch a URL and return the content to reason about with the given prompt.",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["url", "prompt"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=web_fetch,
        ),
        ToolSpec(
            name="WebSearch",
            description=(
                "Brave Search API でウェブ検索し、title / url / description を返す。"
                "count (1-20、default 10) / offset (0-9、default 0) で件数とページを"
                "調整可能。API key は secrets.json の auth_profiles.brave.key で管理。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "結果件数 (1-20、default 10)",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 9,
                        "description": "ページオフセット (0-9、default 0)",
                    },
                    "allowed_domains": {"type": "array",
                                        "items": {"type": "string"}},
                    "blocked_domains": {"type": "array",
                                        "items": {"type": "string"}},
                },
                "required": ["query"],
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=web_search,
        ),
        ToolSpec(
            name="RemoteTrigger",
            description="Send an HTTP request (webhook/remote action trigger).",
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "method": {"type": "string",
                               "enum": ["GET", "POST", "PUT",
                                        "DELETE", "PATCH"]},
                    "headers": {"type": "object"},
                    "body": {},
                },
                "required": ["url"],
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=remote_trigger,
        ),
    ]
    for s in specs:
        registry.register(s)
