"""Web — WebFetch / WebSearch / RemoteTrigger.

claw-code 参照:
  - rust/crates/runtime/src/web_fetch.rs
  - rust/crates/runtime/src/web_search.rs
  - rust/crates/runtime/src/remote_trigger.rs

厳密 claw-code 準拠。Noetic 既存 tools/web.py への forward は**しない**。
WebSearch は Phase 2 では DuckDuckGo HTML endpoint を使った簡易実装。
"""
import re
from typing import Optional
from urllib.parse import quote_plus

import httpx

from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
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
        return "Error: url is required"
    if not prompt:
        return "Error: prompt is required"
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: url must start with http:// or https://"

    try:
        resp = httpx.get(
            url, timeout=30,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    body = resp.text
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
        return "Error: query is required"

    allowed = inp.get("allowed_domains") or []
    blocked = inp.get("blocked_domains") or []

    try:
        resp = httpx.get(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            timeout=30,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    # 簡易パース: <a class="result__a" href="..." ... > title </a>
    results: list = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.+?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(resp.text):
        raw_url = m.group(1)
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        # DuckDuckGo は uddg= でリダイレクト URL を返すことがある
        redirect = re.search(r"uddg=([^&]+)", raw_url)
        if redirect:
            from urllib.parse import unquote
            actual_url = unquote(redirect.group(1))
        else:
            actual_url = raw_url

        if allowed and not any(d in actual_url for d in allowed):
            continue
        if blocked and any(d in actual_url for d in blocked):
            continue

        results.append((title, actual_url))
        if len(results) >= 10:
            break

    if not results:
        return f"No search results for: {query}"

    lines = [f"Search results for: {query}"]
    for title, url in results:
        lines.append(f"  - {title}")
        lines.append(f"    {url}")
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
        return "Error: url is required"
    if not (url.startswith("http://") or url.startswith("https://")):
        return "Error: url must start with http:// or https://"
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
        return f"Error: unsupported method '{method}'"

    kwargs: dict = {"headers": headers, "timeout": 60}
    if body is not None:
        if isinstance(body, (dict, list)):
            kwargs["json"] = body
        else:
            kwargs["content"] = str(body)

    try:
        resp = httpx.request(method, url, **kwargs)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"

    snippet = (resp.text or "")[:2000]
    more = len(resp.text or "") - len(snippet)
    lines = [f"{method} {url} -> {resp.status_code}"]
    if snippet:
        lines.append(snippet)
    if more > 0:
        lines.append(f"[... {more} more chars truncated ...]")
    return "\n".join(lines)


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
            description="Search the web and return ranked results with citations.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
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
