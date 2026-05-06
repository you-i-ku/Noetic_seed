"""web tests (WebFetch / WebSearch / RemoteTrigger). httpx mock で検証。

WebSearch は Brave Search API 経路 (auth_profiles.brave + JSON response)。
段階13 Phase 6 hotfix で DuckDuckGo HTML scrape から差替済。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import httpx

from core import auth as _auth_mod
from core.runtime.registry import ToolRegistry
from core.runtime.tools import web


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _patch_httpx(get_resp=None, request_resp=None, capture=None):
    original_get = httpx.get
    original_request = httpx.request

    def mock_get(url, **kw):
        if capture is not None:
            capture["get_url"] = url
            capture["get_kwargs"] = kw
        r = get_resp or {"text": "mock", "status": 200,
                         "headers": {"content-type": "text/html"}}
        return httpx.Response(
            status_code=r["status"],
            text=r["text"],
            headers=r["headers"],
            request=httpx.Request("GET", url),
        )

    def mock_request(method, url, **kw):
        if capture is not None:
            capture["req_method"] = method
            capture["req_url"] = url
            capture["req_kwargs"] = kw
        r = request_resp or {"text": "mock", "status": 200,
                             "headers": {"content-type": "text/plain"}}
        return httpx.Response(
            status_code=r["status"],
            text=r["text"],
            headers=r["headers"],
            request=httpx.Request(method, url),
        )

    httpx.get = mock_get
    httpx.request = mock_request
    return (original_get, original_request)


def _restore(originals):
    httpx.get, httpx.request = originals


def _patch_auth(ok: bool = True):
    """apply_auth を mock。ok=True なら header 注入成功扱い、False なら error。"""
    original = _auth_mod.apply_auth

    def mock_apply_auth(headers, params, auth_name):
        if not ok:
            return headers, params, f"auth profile '{auth_name}' の key が空です"
        headers = dict(headers)
        headers["X-Subscription-Token"] = "dummy_brave_key"
        return headers, params, None

    _auth_mod.apply_auth = mock_apply_auth
    return original


def _restore_auth(original):
    _auth_mod.apply_auth = original


def _reg():
    r = ToolRegistry()
    web.register(r)
    return r


def _brave_resp(items):
    """Brave Search API の web.results 形式 JSON を作る helper。
    items: [{"title":..., "url":..., "description":...}, ...]
    """
    payload = {"web": {"results": items}}
    return {
        "text": json.dumps(payload),
        "status": 200,
        "headers": {"content-type": "application/json"},
    }


# ============================================================
# WebFetch
# ============================================================

def test_webfetch_html():
    print("== WebFetch: HTML strip ==")
    html = "<html><body><p>visible</p><script>bad</script></body></html>"
    orig = _patch_httpx(get_resp={"text": html, "status": 200,
                                   "headers": {"content-type": "text/html"}})
    try:
        out = _reg().execute("WebFetch", {
            "url": "https://example.com",
            "prompt": "describe",
        })
    finally:
        _restore(orig)
    return all([
        _assert("visible" in out, "本文残る"),
        _assert("bad" not in out, "script 除去"),
        _assert("https://example.com" in out, "URL 表示"),
    ])


def test_webfetch_non_http():
    print("== WebFetch: scheme check ==")
    out = _reg().execute("WebFetch", {"url": "file:///etc/passwd",
                                       "prompt": "x"})
    return _assert("http://" in out or "https://" in out, "拒否メッセージ")


def test_webfetch_empty():
    print("== WebFetch: required ==")
    out = _reg().execute("WebFetch", {"url": "", "prompt": "x"})
    return _assert("required" in out.lower(), "url required")


# ============================================================
# WebSearch — Brave Search API
# ============================================================

def test_websearch_parse():
    print("== WebSearch: Brave JSON parse + description ==")
    resp = _brave_resp([
        {"title": "Example Title",
         "url": "https://example.com/page",
         "description": "Sample <strong>description</strong> text."},
        {"title": "Direct Result",
         "url": "https://direct.com/",
         "description": "Direct sample text."},
    ])
    capture = {}
    auth_orig = _patch_auth(ok=True)
    orig = _patch_httpx(get_resp=resp, capture=capture)
    try:
        out = _reg().execute("WebSearch", {"query": "test"})
    finally:
        _restore(orig)
        _restore_auth(auth_orig)
    return all([
        _assert("Example Title" in out, "title parse"),
        _assert("https://example.com/page" in out, "url parse"),
        _assert("Direct Result" in out, "2 件目 title"),
        _assert("Sample description text." in out,
                "description 抜粋取得 (HTML タグ除去込み)"),
        _assert("brave.com" in capture["get_url"], "Brave endpoint 叩いてる"),
        _assert(capture["get_kwargs"]["params"]["q"] == "test", "q param"),
    ])


def test_websearch_filter():
    print("== WebSearch: allowed_domains filter ==")
    resp = _brave_resp([
        {"title": "Good", "url": "https://good.com/a", "description": "g"},
        {"title": "Bad", "url": "https://bad.com/b", "description": "b"},
    ])
    auth_orig = _patch_auth(ok=True)
    orig = _patch_httpx(get_resp=resp)
    try:
        out = _reg().execute("WebSearch", {
            "query": "x", "allowed_domains": ["good.com"],
        })
    finally:
        _restore(orig)
        _restore_auth(auth_orig)
    return all([
        _assert("Good" in out, "good.com 含む"),
        _assert("Bad" not in out, "bad.com 除外"),
    ])


def test_websearch_count_offset():
    print("== WebSearch: count / offset clamping + transit ==")
    resp = _brave_resp([
        {"title": "A", "url": "https://a/", "description": "ad"},
    ])
    capture = {}
    auth_orig = _patch_auth(ok=True)
    orig = _patch_httpx(get_resp=resp, capture=capture)
    try:
        # count=5, offset=2 をそのまま transit
        _reg().execute("WebSearch", {"query": "x", "count": 5, "offset": 2})
        params_a = dict(capture["get_kwargs"]["params"])
        # 上限超過は clamp (count=99 → 20、offset=99 → 9)
        _reg().execute("WebSearch", {"query": "x", "count": 99, "offset": 99})
        params_b = dict(capture["get_kwargs"]["params"])
        # 下限 (count=0 → 1、offset=-5 → 0)
        _reg().execute("WebSearch", {"query": "x", "count": 0, "offset": -5})
        params_c = dict(capture["get_kwargs"]["params"])
    finally:
        _restore(orig)
        _restore_auth(auth_orig)
    return all([
        _assert(params_a["count"] == 5 and params_a["offset"] == 2,
                "count=5/offset=2 が transit される"),
        _assert(params_b["count"] == 20 and params_b["offset"] == 9,
                "上限超過は 20/9 に clamp"),
        _assert(params_c["count"] == 1 and params_c["offset"] == 0,
                "下限超過は 1/0 に clamp"),
    ])


def test_websearch_no_api_key():
    print("== WebSearch: API key 未設定で error ==")
    auth_orig = _patch_auth(ok=False)
    orig = _patch_httpx(get_resp=_brave_resp([]))
    try:
        out = _reg().execute("WebSearch", {"query": "x"})
    finally:
        _restore(orig)
        _restore_auth(auth_orig)
    return all([
        _assert(out.startswith("Error:"), "Error: prefix"),
        _assert("brave" in out.lower(), "brave への誘導 message"),
        _assert("secrets.json" in out, "secrets.json 案内"),
    ])


def test_websearch_empty():
    print("== WebSearch: empty query ==")
    out = _reg().execute("WebSearch", {"query": ""})
    return _assert("required" in out.lower(), "required")


# ============================================================
# RemoteTrigger
# ============================================================

def test_remote_post():
    print("== RemoteTrigger: POST body ==")
    capture = {}
    orig = _patch_httpx(
        request_resp={"text": '{"ok":true}', "status": 201,
                      "headers": {"content-type": "application/json"}},
        capture=capture,
    )
    try:
        out = _reg().execute("RemoteTrigger", {
            "url": "https://hook.test/",
            "method": "POST",
            "headers": {"X-Test": "yes"},
            "body": {"key": "value"},
        })
    finally:
        _restore(orig)
    return all([
        _assert("201" in out, "status"),
        _assert("ok" in out, "body"),
        _assert(capture["req_method"] == "POST", "method"),
        _assert(capture["req_kwargs"].get("json") == {"key": "value"},
                "json body"),
        _assert(capture["req_kwargs"]["headers"]["X-Test"] == "yes",
                "headers"),
    ])


def test_remote_invalid_method():
    print("== RemoteTrigger: invalid method ==")
    out = _reg().execute("RemoteTrigger", {
        "url": "https://x/", "method": "OPTIONS",
    })
    return _assert("OPTIONS" in out, "拒否")


def test_remote_non_http():
    print("== RemoteTrigger: scheme check ==")
    out = _reg().execute("RemoteTrigger", {"url": "ftp://x/"})
    return _assert("http://" in out or "https://" in out, "拒否")


def test_register():
    print("== register: 3 tool ==")
    r = ToolRegistry()
    web.register(r)
    return _assert(
        {"WebFetch", "WebSearch", "RemoteTrigger"}.issubset(set(r.all_names())),
        "3 tool 全登録",
    )


def main():
    tests = [
        test_webfetch_html, test_webfetch_non_http, test_webfetch_empty,
        test_websearch_parse, test_websearch_filter,
        test_websearch_count_offset, test_websearch_no_api_key,
        test_websearch_empty,
        test_remote_post, test_remote_invalid_method, test_remote_non_http,
        test_register,
    ]
    print(f"Running {len(tests)} test groups...\n")
    passed = 0
    for t in tests:
        if t():
            passed += 1
        print()
    print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
