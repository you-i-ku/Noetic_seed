"""Web検索・URL取得ツール
API キーは secrets.json の auth_profiles.brave から取得する。"""
import httpx
from core.auth import get_auth_profile


def _web_search(args):
    query = args.get("query", "")
    if not query:
        return "エラー: queryを指定してください"
    n = min(int(args.get("max_results", "") or "5"), 10)
    brave_profile = get_auth_profile("brave") or {}
    brave_key = brave_profile.get("key", "")
    if not brave_key:
        return "エラー: secrets.json の auth_profiles.brave.key を設定してください"
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": n},
            headers={"X-Subscription-Token": brave_key, "Accept": "application/json"},
            timeout=15.0,
        )
        resp.raise_for_status()
        results = resp.json().get("web", {}).get("results", [])
        if not results:
            return "検索結果なし"
        lines = [f"「{query}」の検索結果（{len(results)}件）:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('title', '')}")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('description', '')}")
        return "\n".join(lines)
    except Exception as e:
        return f"エラー: {e}"


def _fetch_url(args):
    url = args.get("url", "")
    if not url:
        return "エラー: urlを指定してください"
    if not url.startswith("http"):
        url = "https://" + url
    try:
        resp = httpx.get(f"https://r.jina.ai/{url}", timeout=30.0,
                         headers={"Accept": "text/plain"})
        resp.raise_for_status()
        text = resp.text.strip()
        return text[:10000] + ("..." if len(text) > 10000 else "")
    except Exception as e:
        return f"エラー: {e}"
