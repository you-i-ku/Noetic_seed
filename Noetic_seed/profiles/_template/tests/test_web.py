"""web tests (WebFetch / WebSearch / RemoteTrigger). httpx mock で検証。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

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


def _reg():
    r = ToolRegistry()
    web.register(r)
    return r


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
# WebSearch
# ============================================================

def test_websearch_parse():
    print("== WebSearch: DuckDuckGo HTML parse ==")
    html = """
    <html><body>
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fpage">
        Example Title
      </a>
      <a class="result__a" href="https://direct.com/">
        Direct Result
      </a>
    </body></html>
    """
    orig = _patch_httpx(get_resp={"text": html, "status": 200,
                                   "headers": {"content-type": "text/html"}})
    try:
        out = _reg().execute("WebSearch", {"query": "test"})
    finally:
        _restore(orig)
    return all([
        _assert("Example Title" in out, "title parse"),
        _assert("example.com/page" in out, "uddg デコード"),
        _assert("Direct Result" in out, "direct href"),
    ])


def test_websearch_filter():
    print("== WebSearch: allowed_domains filter ==")
    html = """
      <a class="result__a" href="https://good.com/a">Good</a>
      <a class="result__a" href="https://bad.com/b">Bad</a>
    """
    orig = _patch_httpx(get_resp={"text": html, "status": 200,
                                   "headers": {"content-type": "text/html"}})
    try:
        out = _reg().execute("WebSearch", {
            "query": "x", "allowed_domains": ["good.com"],
        })
    finally:
        _restore(orig)
    return all([
        _assert("Good" in out, "good.com 含む"),
        _assert("Bad" not in out, "bad.com 除外"),
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
        test_websearch_parse, test_websearch_filter, test_websearch_empty,
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
