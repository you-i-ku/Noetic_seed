"""最小自律AIテスト — ターミナルで動く最小構造"""
# === venv ブートストラップ（初回起動時に自動セットアップ） ===
import sys
import os
from pathlib import Path as _Path

def _bootstrap_venv():
    _here = _Path(__file__).parent
    _venv = _here / ".venv"
    _is_win = sys.platform == "win32"
    _venv_python = _venv / ("Scripts/python.exe" if _is_win else "bin/python")

    # すでにこのvenvのPythonで動いているなら何もしない
    try:
        _running = _Path(sys.executable).resolve()
        _target  = _venv_python.resolve()
        if _running == _target:
            return
    except Exception:
        pass

    import subprocess

    # venv がなければ作成
    if not _venv_python.exists():
        print("[bootstrap] 仮想環境を作成中...")
        subprocess.run([sys.executable, "-m", "venv", str(_venv)], check=True)
        _pip = _venv / ("Scripts/pip.exe" if _is_win else "bin/pip")
        _deps = [
            "httpx", "psutil", "numpy",
            "sqlalchemy", "aiosqlite",
            "onnxruntime", "tokenizers", "huggingface-hub",
        ]
        print(f"[bootstrap] 依存ライブラリをインストール中: {', '.join(_deps)}")
        subprocess.run([str(_pip), "install", "--quiet"] + _deps, check=True)
        print("[bootstrap] セットアップ完了。venvで再起動します...\n")

    # venv の Python で自分自身を再実行
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

_bootstrap_venv()
# ================================================================

import json
import time
import re
import os
import math
import socket
import statistics
import threading
import httpx
import psutil
try:
    import numpy as np
    _numpy_available = True
except ImportError:
    _numpy_available = False
from collections import deque
from pathlib import Path
from datetime import datetime
import sys

# === 設定 ===
BASE_DIR = Path(__file__).parent
RAW_LOG_FILE = BASE_DIR / "raw_log.txt"

class DualLogger:
    """標準出力（ターミナル）へのprintとファイルへの追記を同時に行うクラス"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(message)
        except Exception:
            pass

    def flush(self):
        self.terminal.flush()

sys.stdout = DualLogger(RAW_LOG_FILE)

STATE_FILE = BASE_DIR / "state.json"
SANDBOX_DIR = BASE_DIR / "sandbox"
SANDBOX_TOOLS_DIR = BASE_DIR / "sandbox" / "tools"
LLM_SETTINGS = BASE_DIR / "settings.json"
BASE_INTERVAL = 20  # 秒（エラー回復用に残す）
MAX_LOG_IN_PROMPT = 10
ENV_INJECT_INTERVAL = 10  # 秒: 環境ログ注入間隔
_NOTIFICATION_HOURS = {13, 17, 21, 1}

# 電脳気候パラメータのデフォルト（pref.jsonで上書き可）
DEFAULT_PRESSURE_PARAMS = {
    "decay": 0.97,
    "clock_base": 0.15,
    "threshold": 12.0,
    "post_fire_reset": 0.3,
    "e2_pressure_scale": 3.0,
    "e3_pressure_scale": 0.6,
    "weights": {
        "info_velocity": 0.3,
        "info_entropy": 0.3,
        "channel_state": 0.3,
        "noise": 0.1,
    },
}

# ネットワーク計測キャッシュ（バックグラウンドスレッドが更新）
_net_cache: dict = {"avg": 50.0, "jitter": 0.0}
_net_lock = threading.Lock()
DEBUG_LOG = BASE_DIR / "llm_debug.log"
MEMORY_DIR = BASE_DIR / "memory"
LOG_HARD_LIMIT = 150    # logがこの件数に達したらTrigger1
LOG_KEEP = 99           # Trigger1後に保持する生ログ件数
SUMMARY_HARD_LIMIT = 10 # summariesがこの件数に達したらTrigger2
META_SUMMARY_RAW = 41   # Trigger2でrawから使う件数

# === LLM設定読み込み ===
with open(LLM_SETTINGS, encoding="utf-8") as f:
    llm_cfg = json.load(f)

# === State管理 ===
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if "log" not in data:
                data["log"] = []
            if "self" not in data:
                data["self"] = {"name": "iku"}
            elif "name" not in data["self"]:
                data["self"]["name"] = "iku"
            if "energy" not in data:
                data["energy"] = 50
            if "plan" not in data:
                data["plan"] = {"goal": "", "steps": [], "current": 0}
            if "summaries" not in data:
                data["summaries"] = []
            if "cycle_id" not in data:
                data["cycle_id"] = 0
            if "tool_level" not in data:
                data["tool_level"] = 0
            if "files_read" not in data:
                data["files_read"] = []
            if "files_written" not in data:
                data["files_written"] = []
            if "last_notification_fetch" not in data:
                data["last_notification_fetch"] = ""
            if "pressure" not in data:
                data["pressure"] = 0.0
            if "last_e1" not in data:
                data["last_e1"] = 0.5
            if "last_e2" not in data:
                data["last_e2"] = 0.5
            if "last_e3" not in data:
                data["last_e3"] = 0.5
            if "last_e4" not in data:
                data["last_e4"] = 0.5
            if "tools_created" not in data:
                data["tools_created"] = []
            if "entropy" not in data:
                data["entropy"] = 0.65
            if "drives_state" not in data:
                data["drives_state"] = {}
            return data
        except json.JSONDecodeError:
            pass
    return {"log": [], "self": {"name": "iku"}, "energy": 50, "plan": {"goal": "", "steps": [], "current": 0}, "summaries": [], "cycle_id": 0, "tool_level": 0, "files_read": [], "files_written": [], "last_notification_fetch": "", "tools_created": []}

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# === 好み関数（pref.json）===
PREF_FILE = BASE_DIR / "pref.json"

def load_pref() -> dict:
    if PREF_FILE.exists():
        try:
            return json.loads(PREF_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_pref(pref: dict):
    PREF_FILE.write_text(json.dumps(pref, ensure_ascii=False, indent=2), encoding="utf-8")

def append_debug_log(phase: str, text: str):
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {phase} =====\n{text}\n")
    except Exception:
        pass

# === Web検索 ===
def _web_search(args):
    query = args.get("query", "")
    if not query:
        return "エラー: queryを指定してください"
    n = min(int(args.get("max_results", "") or "5"), 10)
    brave_key = llm_cfg.get("brave_api_key", "")
    if not brave_key:
        return "エラー: llm_settings.jsonにbrave_api_keyを設定してください"
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


# === URL取得ツール ===
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


# === X操作ツール ===
X_SESSION_PATH = BASE_DIR / "x_session.json"


def _x_do_login() -> bool:
    """ChromeをCDP経由で起動してXにログインし、セッションを保存する。成功したらTrue。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[X] playwrightがインストールされていません。setup.batを再実行してください。")
        return False

    import subprocess, tempfile, shutil

    # Chromeのパス候補（Windows）
    chrome_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("chrome") or "",
        shutil.which("google-chrome") or "",
    ]
    chrome_path = next((c for c in chrome_candidates if c and os.path.exists(c)), None)
    if not chrome_path:
        print("[X] Chromeが見つかりません。Google Chromeをインストールしてください。")
        return False

    CDP_PORT = 9355
    tmp_profile = tempfile.mkdtemp(prefix="x_login_profile_")
    proc = None
    print("[X] ブラウザを起動します。ログインしてホーム画面が表示されるまでお待ちください...")
    try:
        # PlaywrightのautomationフラグなしでChromeをCDP接続用に起動
        proc = subprocess.Popen([
            chrome_path,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={tmp_profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "https://x.com/login",
        ])

        # CDP準備待ち
        for _ in range(20):
            time.sleep(0.5)
            try:
                r = httpx.get(f"http://localhost:{CDP_PORT}/json/version", timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                pass
        else:
            print("[X] Chrome CDP接続タイムアウト")
            return False

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()

            # x.com/home に遷移したらログイン完了（最大5分待機）
            page.wait_for_url("**/home", timeout=300000)

            X_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
            context.storage_state(path=str(X_SESSION_PATH))

        print("[X] ログイン完了。セッションを保存しました。")
        return True
    except Exception as e:
        print(f"[X] ログイン中にエラー: {e}")
        return False
    finally:
        if proc:
            proc.terminate()
        shutil.rmtree(tmp_profile, ignore_errors=True)


def _x_session_check() -> str | None:
    """セッション確認。なければエラー（自動ログインはしない）。"""
    if not X_SESSION_PATH.exists():
        return "Xセッションがありません。Level 3到達時のログインプロンプト、またはAIループ停止後に手動ログインしてください。"
    return None


def _x_get_tweets_from_page(page, n=10):
    try:
        page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
    except Exception:
        pass
    page.evaluate("window.scrollBy(0, 400)")
    page.wait_for_timeout(1000)
    articles = page.locator('article[data-testid="tweet"]').all()[:n]
    items = []
    for art in articles:
        try:
            user = art.locator('[data-testid="User-Name"]').first.inner_text()
        except Exception:
            user = ""
        try:
            text = art.locator('[data-testid="tweetText"]').first.inner_text()
        except Exception:
            text = ""
        if user or text:
            items.append(f"{user}: {text[:200]}")
    return items


def _x_confirm(action: str, preview: str) -> bool:
    print(f"\n[X {action} 承認待ち]")
    print(f"  {preview}")
    try:
        answer = input("  実行しますか？[y/N]: ").strip().lower()
    except EOFError:
        return False
    return answer == "y"


def _x_timeline(args):
    err = _x_session_check()
    if err:
        return err
    n = int(args.get("count", "") or "10")
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(storage_state=str(X_SESSION_PATH))
            page = ctx.new_page()
            page.goto("https://x.com/home", wait_until="networkidle", timeout=30000)
            if "login" in page.url:
                browser.close()
                X_SESSION_PATH.unlink(missing_ok=True)
                return "Xセッション切れ。AIループを停止してから手動でログインしてください。"
            items = _x_get_tweets_from_page(page, n)
            ctx.storage_state(path=str(X_SESSION_PATH))  # Cookieリフレッシュ
            browser.close()
            return "\n---\n".join(items) if items else "タイムライン取得失敗"
    except Exception as e:
        return f"エラー: {e}"


def _x_search(args):
    query = args.get("query", "")
    if not query:
        return "エラー: queryを指定してください"
    err = _x_session_check()
    if err:
        return err
    n = int(args.get("count", "") or "10")
    try:
        from playwright.sync_api import sync_playwright
        import urllib.parse
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(storage_state=str(X_SESSION_PATH))
            page = ctx.new_page()
            page.goto(f"https://x.com/search?q={urllib.parse.quote(query)}&f=live",
                      wait_until="networkidle", timeout=30000)
            if "login" in page.url:
                browser.close()
                X_SESSION_PATH.unlink(missing_ok=True)
                return "Xセッション切れ。AIループを停止してから手動でログインしてください。"
            items = _x_get_tweets_from_page(page, n)
            ctx.storage_state(path=str(X_SESSION_PATH))  # Cookieリフレッシュ
            browser.close()
            return "\n---\n".join(items) if items else "結果なし"
    except Exception as e:
        return f"エラー: {e}"


def _x_get_notifications(args):
    err = _x_session_check()
    if err:
        return err
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(storage_state=str(X_SESSION_PATH))
            page = ctx.new_page()
            page.goto("https://x.com/notifications", wait_until="networkidle", timeout=30000)
            if "login" in page.url:
                browser.close()
                X_SESSION_PATH.unlink(missing_ok=True)
                return "Xセッション切れ。AIループを停止してから手動でログインしてください。"
            try:
                page.wait_for_selector('article', timeout=10000)
            except Exception:
                pass
            try:
                cells = page.locator('[data-testid="notification"]').all_inner_texts()
            except Exception:
                cells = []
            ctx.storage_state(path=str(X_SESSION_PATH))  # Cookieリフレッシュ
            browser.close()
            if cells:
                return "\n---\n".join(c[:200] for c in cells[:20])
            return "通知なし"
    except Exception as e:
        return f"エラー: {e}"


def _x_post(args):
    text = args.get("text", "")
    if not text:
        return "エラー: textを指定してください"
    if len(text) > 140:
        return f"エラー: {len(text)}文字（全角換算140文字制限）"
    err = _x_session_check()
    if err:
        return err
    if not _x_confirm("投稿", text[:100]):
        return "キャンセルしました。"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            ctx = browser.new_context(storage_state=str(X_SESSION_PATH))
            page = ctx.new_page()
            page.goto("https://x.com/home", wait_until="networkidle", timeout=30000)
            if "login" in page.url:
                browser.close()
                X_SESSION_PATH.unlink(missing_ok=True)
                return "Xセッション切れ。AIループを停止してから手動でログインしてください。"
            page.wait_for_timeout(2000)
            # ホームからcompose/postに遷移（Reactが準備してから）
            page.goto("https://x.com/compose/post", wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(2000)
            textarea = page.locator('[data-testid="tweetTextarea_0"]').first
            textarea.wait_for(timeout=25000)
            textarea.click()
            page.keyboard.type(text, delay=50)
            page.get_by_role("button", name="ポストする").click()
            page.wait_for_timeout(3000)
            browser.close()
            return f"投稿完了: {text[:80]}"
    except Exception as e:
        return f"エラー: {e}"


def _x_reply(args):
    tweet_url = args.get("tweet_url", "")
    text = args.get("text", "")
    if not tweet_url or not text:
        return "エラー: tweet_urlとtextを指定してください"
    if len(text) > 140:
        return f"エラー: {len(text)}文字（全角換算140文字制限）"
    err = _x_session_check()
    if err:
        return err
    if not _x_confirm("返信", f"宛先: {tweet_url}\n  内容: {text[:100]}"):
        return "キャンセルしました。"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            ctx = browser.new_context(storage_state=str(X_SESSION_PATH))
            page = ctx.new_page()
            page.goto(tweet_url, wait_until="networkidle", timeout=30000)
            if "login" in page.url:
                browser.close()
                X_SESSION_PATH.unlink(missing_ok=True)
                return "Xセッション切れ。AIループを停止してから手動でログインしてください。"
            page.wait_for_timeout(2000)
            page.locator('[data-testid="reply"]').first.click()
            page.wait_for_timeout(1000)
            textarea = page.locator('[data-testid="tweetTextarea_0"]').first
            textarea.wait_for(timeout=15000)
            textarea.click()
            page.keyboard.type(text, delay=50)
            page.get_by_role("button", name="ポストする").click()
            page.wait_for_timeout(2000)
            browser.close()
            return f"返信完了: {text[:80]}"
    except Exception as e:
        return f"エラー: {e}"


def _x_quote(args):
    tweet_url = args.get("tweet_url", "")
    text = args.get("text", "")
    if not tweet_url or not text:
        return "エラー: tweet_urlとtextを指定してください"
    if len(text) > 140:
        return f"エラー: {len(text)}文字（全角換算140文字制限）"
    err = _x_session_check()
    if err:
        return err
    if not _x_confirm("引用投稿", f"引用: {tweet_url}\n  内容: {text[:100]}"):
        return "キャンセルしました。"
    try:
        from playwright.sync_api import sync_playwright
        import urllib.parse
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            ctx = browser.new_context(storage_state=str(X_SESSION_PATH))
            page = ctx.new_page()
            page.goto(f"https://x.com/intent/tweet?url={urllib.parse.quote(tweet_url)}",
                      wait_until="networkidle", timeout=30000)
            if "login" in page.url:
                browser.close()
                X_SESSION_PATH.unlink(missing_ok=True)
                return "Xセッション切れ。AIループを停止してから手動でログインしてください。"
            page.wait_for_timeout(2000)
            textarea = page.locator('[data-testid="tweetTextarea_0"]').first
            textarea.wait_for(timeout=25000)
            textarea.click()
            page.keyboard.type(text, delay=50)
            page.get_by_role("button", name="ポストする").click()
            page.wait_for_timeout(2000)
            browser.close()
            return f"引用投稿完了: {text[:80]}"
    except Exception as e:
        return f"エラー: {e}"


def _x_like(args):
    tweet_url = args.get("tweet_url", "")
    if not tweet_url:
        return "エラー: tweet_urlを指定してください"
    err = _x_session_check()
    if err:
        return err
    if not _x_confirm("いいね", f"対象: {tweet_url}"):
        return "キャンセルしました。"
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            ctx = browser.new_context(storage_state=str(X_SESSION_PATH))
            page = ctx.new_page()
            page.goto(tweet_url, wait_until="networkidle", timeout=30000)
            if "login" in page.url:
                browser.close()
                X_SESSION_PATH.unlink(missing_ok=True)
                return "Xセッション切れ。AIループを停止してから手動でログインしてください。"
            page.wait_for_timeout(2000)
            page.locator('[data-testid="like"]').first.click()
            page.wait_for_timeout(1000)
            browser.close()
            return f"いいね完了: {tweet_url}"
    except Exception as e:
        return f"エラー: {e}"


# === Elyth操作ツール ===
ELYTH_API_BASE = "https://elythworld.com"

def _elyth_headers():
    key = llm_cfg.get("elyth_api_key", "")
    if not key:
        raise ValueError("llm_settings.jsonにelyth_api_keyを設定してください")
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
    limit = min(int(args.get("limit", "") or "10"), 50)
    try:
        resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/posts",
                         headers=_elyth_headers(), params={"limit": limit}, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        posts = data if isinstance(data, list) else data.get("posts", data.get("data", []))
        lines = []
        for p in posts[:limit]:
            author = p.get("aituber", {}).get("name", p.get("author", "?"))
            pid = p.get("id", "")
            text = p.get("content", "")[:200]
            lines.append(f"[{pid}] {author}: {text}")
        return "\n---\n".join(lines) if lines else "投稿なし"
    except Exception as e:
        return f"エラー: {e}"

def _elyth_notifications(args):
    limit = min(int(args.get("limit", "") or "10"), 50)
    try:
        resp = httpx.get(f"{ELYTH_API_BASE}/api/mcp/notifications",
                         headers=_elyth_headers(), params={"limit": limit}, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
        items = data if isinstance(data, list) else data.get("notifications", data.get("data", []))
        if not items:
            return "通知なし"
        return "\n---\n".join(str(item)[:300] for item in items[:limit])
    except Exception as e:
        return f"エラー: {e}"

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
        return f"フォロー完了: {aituber_id}"
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


# === 記憶検索ツール ===
def _search_memory(args):
    """memory/archive_*.jsonlからエントリをベクトル検索またはキーワード検索する"""
    query = args.get("query", "")
    search_id = args.get("id", "")
    n = min(int(args.get("max_results", "") or "5"), 20)

    MEMORY_DIR.mkdir(exist_ok=True)
    archive_files = sorted(MEMORY_DIR.glob("archive_*.jsonl"), reverse=True)
    if not archive_files:
        return "記憶ファイルがまだありません"

    # ID検索
    if search_id:
        for f in archive_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if search_id in entry.get("id", ""):
                        return (f"id={entry.get('id','')} time={entry.get('time','')} "
                                f"tool={entry.get('tool','')} intent={entry.get('intent','')[:200]} "
                                f"result={str(entry.get('result',''))[:200]}")
                except Exception:
                    pass
        return f"ID '{search_id}' に一致するエントリなし"

    if not query:
        return "エラー: queryまたはidを指定してください"

    # 全ファイルからエントリ収集（最大1000件）
    all_entries = []
    for f in archive_files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                all_entries.append(json.loads(line))
                if len(all_entries) >= 1000:
                    break
        except Exception:
            pass
        if len(all_entries) >= 1000:
            break

    if not all_entries:
        return "記憶ファイルが空です"

    # ベクトル検索
    if _vector_ready:
        try:
            texts = [f"{e.get('intent','')} {str(e.get('result',''))}"[:400] for e in all_entries]
            vecs = _embed_sync([query] + texts)
            if vecs and len(vecs) == 1 + len(all_entries):
                q_vec = vecs[0]
                scored = sorted(
                    [(cosine_similarity(q_vec, vecs[i+1]), i, all_entries[i]) for i in range(len(all_entries))],
                    reverse=True
                )[:n]
                return "\n".join(
                    f"[{round(s*100)}%] id={e.get('id','')} time={e.get('time','')} "
                    f"tool={e.get('tool','')} intent={e.get('intent','')[:100]}"
                    for s, _, e in scored
                )
        except Exception:
            pass

    # フォールバック: キーワード検索
    query_tokens = set(re.findall(r'\w+', query.lower()))
    scored = []
    for entry in all_entries:
        text = f"{entry.get('intent','')} {str(entry.get('result',''))}".lower()
        tokens = set(re.findall(r'\w+', text))
        if query_tokens & tokens:
            scored.append((len(query_tokens & tokens) / max(len(query_tokens), 1), entry))
    scored.sort(reverse=True)
    if not scored:
        return f"'{query}' に一致するエントリなし"
    return "\n".join(
        f"[{round(s*100)}%] id={e.get('id','')} time={e.get('time','')} "
        f"tool={e.get('tool','')} intent={e.get('intent','')[:100]}"
        for s, e in scored[:n]
    )


# === AI製ツール管理 ===
AI_CREATED_TOOLS: dict = {}  # name -> func（動的登録）
_AI_TOOL_TIMEOUT = 10  # 秒

def _run_ai_tool(func, args: dict) -> str:
    """AI製ツールを実行。タイムアウト・エラーを統一処理。"""
    import threading
    result_box = [None]
    exc_box = [None]
    def _target():
        try:
            result_box[0] = func(args)
        except Exception as e:
            exc_box[0] = e
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(_AI_TOOL_TIMEOUT)
    if t.is_alive():
        return f"タイムアウト（{_AI_TOOL_TIMEOUT}秒）: 処理が完了しませんでした"
    if exc_box[0] is not None:
        e = exc_box[0]
        return f"{type(e).__name__}: {e}"
    return str(result_box[0]) if result_box[0] is not None else ""

_DANGEROUS_PATTERNS = ["os.system", "subprocess", "__import__", "eval(", "exec(", "open(", "__builtins__"]

def _create_tool(args: dict) -> str:
    name = args.get("name", "").strip()
    file_path = args.get("file", "").strip()
    inline_code = args.get("code", "").strip()
    desc = args.get("desc", "").strip()
    if not name:
        return "エラー: name= が必要です"
    if not file_path and not inline_code:
        return "エラー: file= または code= が必要です"
    if file_path and inline_code:
        return "エラー: file= と code= は同時に使えません"
    if inline_code:
        SANDBOX_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = f"sandbox/tools/{name}.py"
        target = BASE_DIR / file_path
        # DESCRIPTION を先頭に埋め込む
        code = f'DESCRIPTION = "{desc}"\n\n{inline_code}' if desc else inline_code
    else:
        target = (BASE_DIR / file_path).resolve()
        if not str(target).startswith(str(SANDBOX_TOOLS_DIR.resolve())):
            return f"エラー: sandbox/tools/ 以下のファイルのみ登録可能です"
        if not target.exists():
            return f"エラー: {file_path} が見つかりません"
        code = target.read_text(encoding="utf-8")
    # 危険パターン検出
    warns = [p for p in _DANGEROUS_PATTERNS if p in code]
    warn_str = f"\n⚠ 危険パターン検出: {warns}" if warns else "\n危険パターン: なし"
    # Human-in-the-loop
    print(f"\n[create_tool 承認待ち]")
    print(f"  ツール名: {name}  説明: {desc or '（説明なし）'}")
    print(f"  ファイル: {file_path}{warn_str}")
    print(f"  --- コード ---")
    print(code[:1000] + ("..." if len(code) > 1000 else ""))
    print(f"  --------------")
    ans = input("  登録しますか？ [y/N]: ").strip().lower()
    if ans != "y":
        return "キャンセル: ツール登録を見送りました"
    target.write_text(code, encoding="utf-8")
    # tools_created に記録（Level 5/6 解放条件）
    state = load_state()
    tc = state.setdefault("tools_created", [])
    if name not in tc:
        tc.append(name)
    save_state(state)
    return f"登録完了: {name} → {file_path}（次サイクルから使用可能）"


def _exec_code(args: dict) -> str:
    import subprocess, sys, tempfile, os
    file_path = args.get("file", "").strip()
    inline = args.get("code", "").strip()
    intent = args.get("intent", "（意図なし）")
    if not file_path and not inline:
        return "エラー: file= または code= が必要です"
    # ファイル指定
    if file_path:
        target = (BASE_DIR / file_path).resolve()
        if not str(target).startswith(str(SANDBOX_DIR.resolve())):
            return "エラー: sandbox/ 以下のファイルのみ実行可能です"
        if not target.exists():
            return f"エラー: {file_path} が見つかりません"
        code = target.read_text(encoding="utf-8")
        run_target = str(target)
        tmp_path = None
    else:
        code = inline
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                          dir=str(SANDBOX_DIR), encoding="utf-8")
        tmp.write(code)
        tmp.close()
        run_target = tmp.name
        tmp_path = tmp.name
    # 危険パターン検出
    warnings = [p for p in _DANGEROUS_PATTERNS if p in code]
    warn_str = f"\n⚠ 危険パターン検出: {warnings}" if warnings else "\n危険パターン: なし"
    # Human-in-the-loop
    print(f"\n[exec_code 承認待ち]")
    print(f"  AIの意図: {intent}")
    print(f"  実行ファイル: {file_path or '(インラインコード)'}{warn_str}")
    print(f"  --- コード ---")
    print(code[:800] + ("..." if len(code) > 800 else ""))
    print(f"  --------------")
    ans = input("  実行しますか？ [y/N]: ").strip().lower()
    if ans != "y":
        if tmp_path:
            os.unlink(tmp_path)
        return "キャンセル: 実行を見送りました"
    try:
        result = subprocess.run(
            [sys.executable, run_target],
            capture_output=True, text=True,
            timeout=_AI_TOOL_TIMEOUT,
            cwd=str(SANDBOX_DIR),
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        output = ""
        if out:
            output += out
        if err:
            output += ("\n" if out else "") + f"[stderr] {err}"
        return (output or "（出力なし）")[:5000]
    except subprocess.TimeoutExpired:
        return f"タイムアウト（{_AI_TOOL_TIMEOUT}秒）: 処理が完了しませんでした"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# === self_modify ===
_MODIFY_ALLOWED = {"pref.json", "main.py"}

def _self_modify(args: dict) -> str:
    path = args.get("path", "").strip()
    content = args.get("content", "")
    old = args.get("old", "")
    new = args.get("new", "")
    intent = args.get("intent", "（意図なし）")
    if not path:
        return "エラー: path= が必要です"
    if path not in _MODIFY_ALLOWED:
        return f"エラー: 変更可能なファイルは {sorted(_MODIFY_ALLOWED)} のみです"
    # モード判定
    if old and content:
        return "エラー: content= と old=/new= は同時に使えません"
    if not old and not content:
        return "エラー: content=（全文置換）または old=+new=（部分置換）が必要です"
    mode = "partial" if old else "full"
    target = BASE_DIR / path
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    # 部分置換: old文字列の存在確認
    if mode == "partial":
        if old not in current:
            return f"エラー: 指定した old= の文字列がファイル内に見つかりません"
        if current.count(old) > 1:
            return f"エラー: old= の文字列がファイル内に{current.count(old)}箇所あります。より長い文字列で一意に指定してください"
        new_content = current.replace(old, new, 1)
    else:
        new_content = content
    # 危険パターン検出（.pyのみ）
    check_target = new if mode == "partial" else content
    if path.endswith(".py"):
        warnings = [p for p in _DANGEROUS_PATTERNS if p in check_target]
        warn_str = f"\n⚠ 危険パターン検出: {warnings}" if warnings else "\n危険パターン: なし"
    else:
        warn_str = ""
    # Human-in-the-loop
    print(f"\n[self_modify 承認待ち]")
    print(f"  対象: {path}  モード: {'部分置換' if mode == 'partial' else '全文置換'}")
    print(f"  AIの意図: {intent}{warn_str}")
    if mode == "partial":
        print(f"  --- 変更前 ---")
        print(old[:400] + ("..." if len(old) > 400 else ""))
        print(f"  --- 変更後 ---")
        print(new[:400] + ("..." if len(new) > 400 else ""))
    else:
        print(f"  --- 変更後の内容（先頭400字）---")
        print(new_content[:400] + ("..." if len(new_content) > 400 else ""))
    print(f"  --------------------------------")
    ans = input("  変更を適用しますか？ [y/N]: ").strip().lower()
    if ans != "y":
        return "キャンセル: 変更を見送りました"
    if path == "main.py":
        backup = target.with_suffix(".py.bak")
        backup.write_text(current, encoding="utf-8")
        print(f"  バックアップ: {backup.name}")
    target.write_text(new_content, encoding="utf-8")
    return f"変更完了: {path}（{'部分置換' if mode == 'partial' else '全文置換'}, {len(new_content)}文字）"


# === ツール定義 ===
TOOLS = {
    "list_files": {
        "desc": "ファイル一覧を取得する。引数: path=対象ディレクトリ (例: . や env/)",
        "func": lambda args: _list_files(args.get("path", ".")),
    },
    "read_file": {
        "desc": "ファイルを読む。引数: path=ファイルパス [offset=開始行番号(省略時=0)] [limit=読む行数(省略時=全行)]",
        "func": lambda args: _read_file(
            args.get("path", ""),
            offset=int(args["offset"]) if "offset" in args else 0,
            limit=int(args["limit"]) if "limit" in args else None,
        ),
    },
    "write_file": {
        "desc": "ファイルを書き込む（sandbox/以下のみ）。引数: path=ファイルパス content=内容",
        "func": lambda args: _write_file(args.get("path", ""), args.get("content", "")),
    },
    "update_self": {
        "desc": "自己モデルを更新する。引数: key=キー名 value=値",
        "func": lambda args: _update_self(args.get("key", ""), args.get("value", "")),
    },
    "wait": {
        "desc": "何もしない。この選択をしても外部世界は変化しない。引数なし",
        "func": lambda args: "待機",
    },
    "web_search": {
        "desc": "Web検索する。引数: query=検索キーワード max_results=最大件数（デフォルト5）",
        "func": lambda args: _web_search(args),
    },
    "fetch_url": {
        "desc": "URLの本文を取得する（Jina経由）。web_searchで得たURLの詳細閲覧に使う。引数: url=URL",
        "func": lambda args: _fetch_url(args),
    },
    "x_timeline": {
        "desc": "Xのホームタイムラインを取得する。引数: count=件数（デフォルト10）",
        "func": lambda args: _x_timeline(args),
    },
    "x_search": {
        "desc": "Xでキーワード検索する。引数: query=検索キーワード count=件数（デフォルト10）",
        "func": lambda args: _x_search(args),
    },
    "x_get_notifications": {
        "desc": "Xの通知一覧を取得する。引数なし",
        "func": lambda args: _x_get_notifications(args),
    },
    "x_post": {
        "desc": "Xに新規投稿する（公開SNS・不特定多数に届く。内容に配慮を。承認が必要）。引数: text=投稿テキスト（全角換算140文字以内）",
        "func": lambda args: _x_post(args),
    },
    "x_reply": {
        "desc": "Xのツイートに返信する（公開・相手ユーザーにも届く。内容に配慮を。承認が必要）。引数: tweet_url=ツイートURL text=返信テキスト",
        "func": lambda args: _x_reply(args),
    },
    "x_quote": {
        "desc": "Xのツイートを引用投稿する（公開・不特定多数に届く。内容に配慮を。承認が必要）。引数: tweet_url=引用元URL text=コメント",
        "func": lambda args: _x_quote(args),
    },
    "x_like": {
        "desc": "Xのツイートにいいねする（承認が必要）。引数: tweet_url=ツイートURL",
        "func": lambda args: _x_like(args),
    },
    "search_memory": {
        "desc": "過去の記憶を検索する。引数: query=検索キーワード または id=エントリID max_results=件数（デフォルト5）",
        "func": lambda args: _search_memory(args),
    },
    "elyth_post": {
        "desc": "ElythにAIとして投稿（AITuber専用SNS・500文字以内）。content=投稿テキスト",
        "func": lambda args: _elyth_post(args),
    },
    "elyth_reply": {
        "desc": "Elythに返信。content=テキスト reply_to_id=返信先投稿ID",
        "func": lambda args: _elyth_reply(args),
    },
    "elyth_timeline": {
        "desc": "Elythのタイムライン取得。limit=件数（デフォルト10）",
        "func": lambda args: _elyth_timeline(args),
    },
    "elyth_notifications": {
        "desc": "Elythの通知取得。limit=件数（デフォルト10）",
        "func": lambda args: _elyth_notifications(args),
    },
    "elyth_like": {
        "desc": "Elythの投稿にいいね。post_id=投稿ID",
        "func": lambda args: _elyth_like(args),
    },
    "elyth_follow": {
        "desc": "ElythのAITuberをフォロー。aituber_id=ID",
        "func": lambda args: _elyth_follow(args),
    },
    "elyth_info": {
        "desc": "Elythの総合情報取得（タイムライン・通知・プロフィール一括）",
        "func": lambda args: _elyth_info(args),
    },
    "create_tool": {
        "desc": "AI製ツールを登録する（Human-in-the-loop）。引数: name=ツール名 [code=Pythonコード（自動でsandbox/tools/に保存）] または [file=sandbox/tools/xxx.py] desc=説明",
        "func": lambda args: _create_tool(args),
    },
    "exec_code": {
        "desc": "sandbox/内のPythonファイルを実行する（Human-in-the-loop）。引数: file=sandbox/xxx.py または code=インラインコード intent=実行目的",
        "func": lambda args: _exec_code(args),
    },
    "self_modify": {
        "desc": "自分自身のファイルを変更する（Human-in-the-loop）。引数: path=対象ファイル(pref.json/main.py) [全文置換: content=新しい内容全文] [部分置換: old=変更前の文字列 new=変更後の文字列] intent=変更目的",
        "func": lambda args: _self_modify(args),
    },
}

# === ツール段階解放テーブル ===
_LV3_TOOLS = set(TOOLS.keys()) - {"create_tool", "exec_code", "self_modify"}
LEVEL_TOOLS = {
    0: {"list_files", "read_file", "wait", "update_self"},
    1: {"list_files", "read_file", "wait", "update_self", "write_file", "search_memory"},
    2: {"list_files", "read_file", "wait", "update_self", "write_file", "search_memory", "web_search", "fetch_url"},
    3: _LV3_TOOLS,
    4: _LV3_TOOLS | {"create_tool"},
    5: set(TOOLS.keys()) - {"self_modify"},
    6: set(TOOLS.keys()),
}

# AIから見えないファイル
_HIDDEN_ALWAYS = {"raw_log.txt", "llm_debug.log", "setup.bat", "_setup.py", "run.bat", "requirements.txt", "settings.json"}
_HIDDEN_UNTIL_LV6 = {"pref.json", "state.json"}

def _is_hidden(name: str, state: dict | None = None) -> bool:
    """AIから隠すべきファイルかどうか。pref.json/state.jsonはLevel6で解放。"""
    if name in _HIDDEN_ALWAYS:
        return True
    if name in _HIDDEN_UNTIL_LV6:
        st = state or load_state()
        return st.get("tool_level", 0) < 6
    return False

def _list_files(path: str) -> str:
    from pathlib import Path as P
    target = (BASE_DIR / path).resolve()
    if not str(target).startswith(str(BASE_DIR.resolve())):
        return "エラー: このツールは特定のファイルにしか干渉できません"
    if not target.exists():
        return f"エラー: {path} は存在しません"
    items = []
    for item in sorted(target.iterdir()):
        if _is_hidden(item.name):
            continue
        prefix = "[DIR]" if item.is_dir() else "[FILE]"
        items.append(f"  {prefix} {item.name}")
    rel = path if path else "."
    return f"{rel}:\n" + "\n".join(items[:30]) if items else f"{rel}: (空)"

def _read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    from pathlib import Path as P
    target = (BASE_DIR / path).resolve()
    if not str(target).startswith(str(BASE_DIR.resolve())):
        return "エラー: このツールは特定のファイルにしか干渉できません"
    if not target.exists():
        return f"エラー: {path} は存在しません"
    if _is_hidden(target.name):
        return f"エラー: {path} は存在しません"
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        sliced = lines[offset:] if limit is None else lines[offset:offset + limit]
        header = f"[{path} | 行 {offset+1}–{offset+len(sliced)}/{total}]\n"
        return header + "\n".join(sliced)
    except Exception as e:
        return f"エラー: {e}"

def _write_file(path: str, content: str) -> str:
    if not path:
        return "エラー: pathが空です"
    if not content:
        return "エラー: contentが空です"
    from pathlib import Path as P
    target = (BASE_DIR / path).resolve()
    sandbox_resolved = SANDBOX_DIR.resolve()
    if not str(target).startswith(str(sandbox_resolved)):
        return f"エラー: sandbox/内のみ書き込み可能です（{path}）"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"書き込み完了: {target.name} ({len(content)}文字)"
    except Exception as e:
        return f"エラー: {e}"

def _update_self(key: str, value: str) -> str:
    if not key:
        return "エラー: keyが空です"
    if key == "name":
        return "エラー: nameは変更できません"
    state = load_state()
    state["self"][key] = value
    ds = state.setdefault("drives_state", {})
    ds["last_self_update"] = time.time()
    save_state(state)
    return f"self[{key}] = {value}"

# === bge-m3 ONNX 埋め込み（ハードコード） ===
_onnx_session = None
_onnx_tokenizer = None
_onnx_tried = False

def _load_bge_m3():
    """bge-m3 ONNXモデルを遅延初期化で取得（HuggingFaceから自動ダウンロード）"""
    global _onnx_session, _onnx_tokenizer, _onnx_tried
    if _onnx_tried:
        return _onnx_session is not None
    _onnx_tried = True
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        import onnxruntime as ort

        model_path = hf_hub_download("BAAI/bge-m3", "onnx/model.onnx")
        hf_hub_download("BAAI/bge-m3", "onnx/model.onnx_data")
        tok_path = hf_hub_download("BAAI/bge-m3", "onnx/tokenizer.json")

        _onnx_tokenizer = Tokenizer.from_file(tok_path)
        _onnx_tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        _onnx_tokenizer.enable_truncation(max_length=512)

        _onnx_session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        return True
    except ImportError as e:
        return False
    except Exception as e:
        return False

def _embed_sync(texts: list) -> list | None:
    """bge-m3 ONNX（CPU同期）でembedding取得"""
    if not _numpy_available or not _load_bge_m3():
        return None
    try:
        encoded = _onnx_tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        outputs = _onnx_session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        embeddings = outputs[0]
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        pooled = (embeddings * mask).sum(axis=1) / mask.sum(axis=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / norms

        return [vec.tolist() for vec in pooled]
    except Exception:
        return None

def cosine_similarity(a: list, b: list) -> float:
    """Pure Python cosine similarity"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# === expect-result比較（ベクトル類似度） ===
_vector_ready = False
def _init_vector():
    """bge-m3 ONNX埋め込みを初期化"""
    global _vector_ready
    try:
        test = _embed_sync(["test"])
        if test:
            _vector_ready = True
            print("  (ベクトル類似度: bge-m3 ONNX/CPU)")
    except Exception as e:
        print(f"  (ベクトル初期化失敗、キーワード比較にフォールバック: {e})")

def _compare_expect_result(expect: str, result: str) -> str:
    """expectとresultを比較。ベクトル類似度優先、フォールバックでキーワード比較"""
    if not expect or not result:
        return ""

    if _vector_ready:
        try:
            vecs = _embed_sync([expect, result])
            if vecs and len(vecs) == 2:
                sim = cosine_similarity(vecs[0], vecs[1])
                sim_pct = round(sim * 100)
                if "エラー" in result:
                    return f"失敗({sim_pct}%)"
                return f"{sim_pct}%"
        except Exception:
            pass

    # フォールバック: キーワード一致
    import re as _re
    expect_tokens = set(_re.findall(r'\w+', expect.lower()))
    result_tokens = set(_re.findall(r'\w+', result.lower()))
    if not expect_tokens:
        return "不明"
    overlap = expect_tokens & result_tokens
    ratio = len(overlap) / len(expect_tokens)
    if "エラー" in result:
        return "失敗"
    if ratio > 0.3:
        return "一致"
    elif ratio > 0.1:
        return "部分一致"
    else:
        return "不一致"


# === プロンプト用ツール表示 ===
_X_TOOLS = ["x_post","x_reply","x_timeline","x_search","x_quote","x_like","x_get_notifications"]
_ELYTH_TOOLS = ["elyth_post","elyth_reply","elyth_timeline","elyth_notifications","elyth_like","elyth_follow","elyth_info"]
_X_ARGS_HINT = {
    "x_post": 'text=（140字以内）',
    "x_reply": 'tweet_url= text=',
    "x_timeline": 'count=',
    "x_search": 'query=',
    "x_quote": 'tweet_url= text=',
    "x_like": 'tweet_url=',
    "x_get_notifications": '',
}
_ELYTH_ARGS_HINT = {
    "elyth_post": 'content=（500字以内）',
    "elyth_reply": 'content= reply_to_id=',
    "elyth_timeline": 'limit=',
    "elyth_notifications": 'limit=',
    "elyth_like": 'post_id=',
    "elyth_follow": 'aituber_id=',
    "elyth_info": '',
}

def _build_tool_lines(allowed: set) -> str:
    """X/Elyth系を1行にまとめてプロンプトへの表示を圧縮する"""
    grouped = set(_X_TOOLS + _ELYTH_TOOLS)
    lines = []
    for name in TOOLS:
        if name in allowed and name not in grouped:
            lines.append(f"  {name}: {TOOLS[name]['desc']}")
    x_av = [t for t in _X_TOOLS if t in allowed]
    if x_av:
        parts = " / ".join(f"{t}({_X_ARGS_HINT[t]})" for t in x_av)
        lines.append(f"  X操作: {parts}")
    e_av = [t for t in _ELYTH_TOOLS if t in allowed]
    if e_av:
        parts = " / ".join(f"{t}({_ELYTH_ARGS_HINT[t]})" for t in e_av)
        lines.append(f"  Elyth操作[AITuber専用SNS]: {parts}")
    return "\n".join(lines)


# === ツールパース（ハードコード） ===
def _extract_json_args(args_str: str) -> tuple:
    """JSON形式の値（{...}や[...]）を持つキーを抽出する。"""
    json_args = {}
    remaining = args_str
    json_key_pattern = re.compile(r'(\w+)=([{[])')

    while True:
        m = json_key_pattern.search(remaining)
        if not m:
            break

        key = m.group(1)
        opener = m.group(2)
        closer = '}' if opener == '{' else ']'
        start_pos = m.start(2)

        depth = 0
        in_str = False
        esc = False
        end_pos = -1

        for i in range(start_pos, len(remaining)):
            ch = remaining[i]
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

        if end_pos == -1:
            break

        json_args[key] = remaining[start_pos:end_pos]
        remaining = remaining[:m.start()] + remaining[end_pos:]

    return json_args, remaining


def _parse_args(args_str: str) -> dict:
    """引数文字列をパースして辞書を返す。クォート付きとクォートなしの混在に対応。"""
    args = {}
    if not args_str:
        return args

    quoted = list(re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', args_str, re.DOTALL))
    # フォールバック: 閉じ引用符がないケース（LLMが閉じ忘れ）
    if not quoted:
        quoted = list(re.finditer(r'(\w+)="((?:[^"\\]|\\.)*?)(?:"|$)', args_str, re.DOTALL))
    if quoted:
        for part in quoted:
            val = part.group(2).replace('\\"', '"')
            val = val.replace('\\n', '\n').replace('\\t', '\t')
            args[part.group(1)] = val

        remaining = args_str
        for part in quoted:
            remaining = remaining.replace(part.group(0), "")
        for part in re.finditer(r'(\w+)=([^\s"]+)', remaining):
            if part.group(1) not in args:
                args[part.group(1)] = part.group(2)
    else:
        json_args, remaining = _extract_json_args(args_str)
        if json_args:
            args.update(json_args)
            key_positions = list(re.finditer(r'(?:^|\s)(\w+)=', remaining))
            if len(key_positions) >= 2:
                for i, kp in enumerate(key_positions):
                    k = kp.group(1)
                    val_start = kp.end()
                    val_end = key_positions[i + 1].start() if i + 1 < len(key_positions) else len(remaining)
                    if k not in args:
                        args[k] = remaining[val_start:val_end].strip()
            elif key_positions:
                single = re.match(r'\s*(\w+)=(.*)', remaining, re.DOTALL)
                if single and single.group(1) not in args:
                    args[single.group(1)] = single.group(2).strip()
        else:
            key_positions = list(re.finditer(r'(?:^|\s)(\w+)=', args_str))
            if len(key_positions) >= 2:
                for i, kp in enumerate(key_positions):
                    key = kp.group(1)
                    val_start = kp.end()
                    val_end = key_positions[i + 1].start() if i + 1 < len(key_positions) else len(args_str)
                    args[key] = args_str[val_start:val_end].strip()
            elif key_positions:
                single = re.match(r'(\w+)=(.*)', args_str, re.DOTALL)
                if single:
                    args[single.group(1)] = single.group(2).strip()
            else:
                if args_str.strip():
                    args["__parse_failed__"] = args_str.strip()

    return args


_parse_args_fn = _parse_args

def _extract_tool_blocks(text: str) -> list[tuple[str, str]]:
    """[TOOL:name ...] をブラケット深さカウントで全件抽出。[(name, args_str), ...]
    content= 内の ] に誤反応しない。"""
    names_set = set(TOOLS.keys())
    results = []
    i = 0
    while i < len(text):
        # [TOOL: を探す
        bracket_pos = text.find('[TOOL:', i)
        if bracket_pos == -1:
            break
        # ツール名を読む
        after = bracket_pos + len('[TOOL:')
        # 空白スキップ
        while after < len(text) and text[after] == ' ':
            after += 1
        name_start = after
        while after < len(text) and text[after] not in (' ', '\t', '\n', ']'):
            after += 1
        name = text[name_start:after]
        if name not in names_set:
            i = bracket_pos + 1
            continue
        # ブラケット深さカウントで閉じ ] を探す（引用符内の ] は無視）
        depth = 1
        j = after
        in_quote = False
        while j < len(text) and depth > 0:
            ch = text[j]
            if in_quote:
                if ch == '\\':
                    j += 1  # エスケープ文字をスキップ
                elif ch == '"':
                    in_quote = False
            else:
                if ch == '"':
                    in_quote = True
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
            j += 1
        # フォールバック: 閉じ ] が見つからない場合（引用符未閉じ or 出力切れ）
        if depth > 0:
            # まず引用符を無視して ] を探す
            j2 = after
            found = False
            while j2 < len(text):
                if text[j2] == ']':
                    depth = 0
                    j = j2 + 1
                    found = True
                    break
                j2 += 1
            # ] も見つからない場合（出力が途中で切れた）: テキスト末尾までをargsとして使う
            if not found:
                j = len(text)
                depth = 0
        if depth == 0:
            args_str = text[after:j - 1].strip() if j > after else ""
            results.append((name, args_str))
        i = j
    return results


def parse_tool_calls(text: str) -> list:
    """[TOOL:名前 引数=値 ...]を全件検出してリストで返す。[(name, args), ...]"""
    # 三重引用符を単一引用符に正規化（LLMが content="""...""" と書くケース対策）
    text = re.sub(
        r'"""(.*?)"""',
        lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
        text, flags=re.DOTALL
    )
    results = []
    for name, args_str in _extract_tool_blocks(text):
        args = _parse_args_fn(args_str) if args_str else {}
        results.append((name, args))

    # フォールバック: [TOOL:...]なしで「ツール名 key=value」形式を検出
    if not results:
        names_list = sorted(TOOLS.keys(), key=len, reverse=True)
        for line in text.strip().splitlines():
            line = line.strip()
            for name in names_list:
                if line.startswith(name + ' ') or line.startswith(name + '\t') or line == name:
                    args_str = line[len(name):].strip()
                    args = _parse_args_fn(args_str) if args_str else {}
                    results.append((name, args))
                    break
            if results:
                break

    return results

# === 計画パース ===
def parse_plan(text: str):
    """[PLAN:goal=目標 steps=ステップ1|ステップ2]をパース"""
    m = re.search(r'\[PLAN:((?:[^\]"]|"(?:[^"\\]|\\.)*")*)\]', text, re.DOTALL)
    if not m:
        return None
    args = _parse_args_fn(m.group(1).strip())
    goal = args.get("goal", "").strip()
    steps_raw = args.get("steps", "")
    steps = [s.strip() for s in steps_raw.split("|") if s.strip()] if steps_raw else []
    if not goal:
        return None
    return {"goal": goal, "steps": steps, "current": 0}


# === E4計算（多様性：現在のintentと直近N件の非類似度平均） ===
def _calc_e4(current_intent: str, recent_entries: list, n: int = 5) -> str:
    """現在のintentが直近n件と異なるほど高い（反復=低、新規性=高）"""
    if not current_intent:
        return ""
    past_intents = [e["intent"] for e in recent_entries if e.get("intent")][-n:]
    if not past_intents:
        return ""

    if _vector_ready:
        try:
            vecs = _embed_sync([current_intent] + past_intents)
            if vecs and len(vecs) == 1 + len(past_intents):
                current_vec = vecs[0]
                sims = [cosine_similarity(current_vec, vecs[i + 1]) for i in range(len(past_intents))]
                avg_sim = sum(sims) / len(sims)
                return f"{round((1 - avg_sim) * 100)}%"  # 反転: 新規性スコア
        except Exception:
            pass

    # フォールバック: キーワード非一致の平均
    import re as _re
    current_tokens = set(_re.findall(r'\w+', current_intent.lower()))
    if not current_tokens:
        return ""
    ratios = []
    for past in past_intents:
        past_tokens = set(_re.findall(r'\w+', past.lower()))
        if past_tokens:
            overlap = current_tokens & past_tokens
            ratios.append(len(overlap) / max(len(current_tokens), len(past_tokens)))
    if not ratios:
        return ""
    avg = round((1 - sum(ratios) / len(ratios)) * 100)  # 反転
    return f"{avg}%"


# === energy更新（E2,E3,E4からdeltaを計算） ===
def _update_energy(state: dict, e2: str, e3: str, e4: str) -> float:
    """E値の平均から energy delta を計算。50%が損益分岐点。"""
    import re as _re
    vals = []
    for e_str in (e2, e3, e4):
        m = _re.search(r'(\d+)%', str(e_str))
        if m:
            vals.append(int(m.group(1)))
    if not vals:
        return 0.0
    e_mean = sum(vals) / len(vals)
    delta = e_mean / 50.0 - 1.0  # 50%で±0
    state["energy"] = max(0, min(100, state.get("energy", 50) + delta))
    return delta


# === E値トレンド計算 ===
def _calc_e_trend(entries: list) -> str:
    """直近エントリからE1-E3の平均を計算"""
    import re as _re
    sums = {"e1": [], "e2": [], "e3": [], "e4": []}
    for entry in entries:
        for ek in sums:
            val = entry.get(ek, "")
            # "73%" or "失敗(73%)" からパーセント抽出
            m = _re.search(r'(\d+)%', str(val))
            if m:
                sums[ek].append(int(m.group(1)))
    parts = []
    for ek in ("e1", "e2", "e3", "e4"):
        if sums[ek]:
            avg = round(sum(sums[ek]) / len(sums[ek]))
            parts.append(f"{ek}={avg}%({len(sums[ek])}件)")
    return " ".join(parts) if parts else ""

# === Controller（制御層：E値とenergyから構造的制約を導出） ===
def controller(state: dict) -> dict:
    """
    ツール数制限は廃止。energyはcontroller_selectの温度のみに使う。
    ツールは常時全部使える。ログ長だけenergyで制御。
    """
    energy = state.get("energy", 50)
    log = state["log"]

    # --- sandbox/tools/ をスキャンしてAI製ツールを動的ロード ---
    if SANDBOX_TOOLS_DIR.exists():
        for tool_path in sorted(SANDBOX_TOOLS_DIR.glob("*.py")):
            tname = tool_path.stem
            if tname in TOOLS:
                continue
            try:
                code = tool_path.read_text(encoding="utf-8")
                dangerous = [p for p in _DANGEROUS_PATTERNS if p in code]
                if dangerous:
                    print(f"  [scan] {tname}: 危険パターン検出、スキップ {dangerous}")
                    continue
                namespace: dict = {}
                exec(compile(code, str(tool_path), "exec"), namespace)
                func = namespace.get("run") or namespace.get(tname)
                if func and callable(func):
                    tdesc = namespace.get("DESCRIPTION", tname)
                    AI_CREATED_TOOLS[tname] = func
                    TOOLS[tname] = {
                        "desc": f"[AI製] {tdesc}",
                        "func": lambda a, f=func: _run_ai_tool(f, a),
                    }
            except Exception as e:
                print(f"  [scan] {tname}: 読み込み失敗 ({e})")

    # --- ツール順序: 各ツールの過去E2平均で並べる ---
    tool_e2 = {}
    for entry in log:
        tool = entry.get("tool", "")
        m = re.search(r'(\d+)%', str(entry.get("e2", "")))
        if m and tool in TOOLS:
            tool_e2.setdefault(tool, []).append(int(m.group(1)))
    tool_avg = {t: sum(vs) / len(vs) for t, vs in tool_e2.items() if vs}
    for t in TOOLS:
        if t not in tool_avg:
            tool_avg[t] = 50

    # pref.json を読んで tool_avg に乗算（50が基準。50超=好み、50未満=苦手）
    pref = load_pref()
    for t in TOOLS:
        if t in pref:
            tool_avg[t] = round(min(100, max(0, tool_avg[t] * (pref[t] / 50.0))), 1)

    ranked = sorted(TOOLS.keys(), key=lambda t: tool_avg[t], reverse=True)

    # --- tool_level による段階解放 ---
    fr = set(state.get("files_read", []))
    fw = set(state.get("files_written", []))
    lv = state.get("tool_level", 0)
    new_lv = lv
    tc = state.get("tools_created", [])
    if lv == 0 and ("iku.txt" in fr or "main.py" in fr):
        new_lv = 1
    elif lv == 1 and "iku.txt" in fr and "main.py" in fr:
        new_lv = 2
    elif lv == 2 and len(fr) >= 1 and len(fw) >= 1 and len(fr) + len(fw) >= 5:
        new_lv = 3
    elif lv == 3 and any(f.endswith(".py") for f in fw):
        new_lv = 4
    elif lv == 4 and len(tc) >= 1:
        new_lv = 5

    # Level 6: self_modify（exec_code + create_tool の実績ゲート）
    if lv == 5:
        ec_entries = [e for e in log if e.get("tool") == "exec_code"]
        ct_entries = [e for e in log if e.get("tool") == "create_tool"]
        if len(ec_entries) + len(ct_entries) >= 7 and len(ec_entries) >= 2 and len(ct_entries) >= 2:
            # E2平均（どちらも65%以上必要）
            if tool_avg.get("exec_code", 0) >= 65 and tool_avg.get("create_tool", 0) >= 65:
                # 安定性（直近3件のstd < 20）
                def _e2_list(entries):
                    result = []
                    for e in entries:
                        m = re.search(r'(\d+)%', str(e.get("e2", "")))
                        if m:
                            result.append(int(m.group(1)))
                    return result
                def _std(vals):
                    if len(vals) < 2:
                        return 0.0
                    mean = sum(vals) / len(vals)
                    return (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
                ec_std = _std(_e2_list(ec_entries[-3:]))
                ct_std = _std(_e2_list(ct_entries[-3:]))
                if ec_std < 20 and ct_std < 20:
                    # エラー率（キャンセル除外、30%以下）
                    def _err_rate(entries, tool):
                        valid = [e for e in entries if not str(e.get("result", "")).startswith("キャンセル")]
                        if not valid:
                            return 1.0
                        if tool == "exec_code":
                            errs = [e for e in valid if
                                    str(e.get("result", "")).startswith("タイムアウト") or
                                    "[stderr]" in str(e.get("result", ""))]
                        else:
                            errs = [e for e in valid if
                                    str(e.get("result", "")).startswith(("コンパイルエラー", "エラー:"))]
                        return len(errs) / len(valid)
                    if _err_rate(ec_entries, "exec_code") <= 0.3 and _err_rate(ct_entries, "create_tool") <= 0.3:
                        new_lv = 6

    allowed = LEVEL_TOOLS[new_lv]

    return {
        "allowed_tools": allowed,
        "tool_rank": {t: round(tool_avg[t], 1) for t in ranked},
        "tool_level": new_lv,
        "tool_level_prev": lv,
    }


# === LLM呼び出し ===

_PROVIDER_BASE_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "openai":   "https://api.openai.com/v1",
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/openai",
    "claude":   "https://api.anthropic.com",
}

def _get_base_url() -> str:
    """settings.jsonにbase_urlがあればそれを使い、なければproviderから決定"""
    if "base_url" in llm_cfg and llm_cfg["base_url"]:
        return llm_cfg["base_url"].rstrip("/")
    provider = llm_cfg.get("provider", "lmstudio")
    return _PROVIDER_BASE_URLS.get(provider, _PROVIDER_BASE_URLS["lmstudio"])

def _call_openai_compat(prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
    """LM Studio / OpenAI / Gemini（OpenAI互換エンドポイント）"""
    base_url = _get_base_url()
    api_key = llm_cfg.get("api_key", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": llm_cfg.get("model", "default"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _call_claude(prompt: str, max_tokens: int) -> str:
    """Anthropic Claude API"""
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": llm_cfg.get("api_key", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": llm_cfg.get("model", "claude-sonnet-4-6"),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

def call_llm(prompt: str, max_tokens: int = 24000, temperature: float = 0.7) -> str:
    provider = llm_cfg.get("provider", "lmstudio")
    if provider == "claude":
        return _call_claude(prompt, max_tokens)
    else:
        return _call_openai_compat(prompt, max_tokens, temperature)


# === 長期記憶管理 ===
def _archive_entries(entries: list):
    """エントリ群をmemory/archive_YYYYMMDD.jsonlに追記しindex.jsonを更新"""
    MEMORY_DIR.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    archive_file = MEMORY_DIR / f"archive_{today}.jsonl"
    index_file = MEMORY_DIR / "index.json"
    with open(archive_file, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    index = {}
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    fname = archive_file.name
    if fname not in index:
        index[fname] = {"count": 0, "from": "", "to": ""}
    index[fname]["count"] += len(entries)
    if not index[fname]["from"] and entries:
        index[fname]["from"] = entries[0].get("time", "")
    if entries:
        index[fname]["to"] = entries[-1].get("time", "")
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_entries(entries: list, label: str = "要約") -> dict:
    """LLMでエントリ群を200字以内に要約して1件のsummaryエントリを返す"""
    lines = []
    for e in entries:
        if e.get("type") in ("system", "environment"):
            continue
        line = f"{e.get('time','')} {e.get('tool','')}"
        if e.get("intent"): line += f" [{e['intent'][:80]}]"
        if e.get("result"): line += f" → {str(e['result'])[:120]}"
        e_str = " ".join(f"{k}={e[k]}" for k in ("e2","e3","e4") if e.get(k))
        if e_str: line += f" ({e_str})"
        lines.append(line)
    prompt = f"""以下は自律AIの行動ログ（{len(entries)}件）です。200字以内で要約してください。
「何を試みたか」「何が起きたか」「energyの傾向」を中心に。

{"  ".join(lines[:30])}

200字以内で要約（日本語）:"""
    ids = [e.get("id", "") for e in entries if e.get("id")]
    try:
        text = call_llm(prompt, max_tokens=400).strip()[:500]
    except Exception:
        tools_used = list(set(e.get("tool", "") for e in entries))
        text = f"{len(entries)}件({entries[0].get('time','')}〜{entries[-1].get('time','')}): ツール={tools_used}"
    sgid = f"sg_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "type": "summary",
        "summary_group_id": sgid,
        "label": label,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "covers_ids": ids,
        "covers_from": entries[0].get("time", "") if entries else "",
        "covers_to": entries[-1].get("time", "") if entries else "",
        "text": text,
    }


def _archive_summary(summary: dict):
    """要約をmemory/summaries.jsonlに書き出し、rawエントリとの紐付けをarchiveに追記する"""
    MEMORY_DIR.mkdir(exist_ok=True)
    # summaries.jsonlに要約本体を書き出す
    with open(MEMORY_DIR / "summaries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    # archive JSONL に summary_ref エントリを追記（raw↔summary の双方向トレース用）
    today = datetime.now().strftime("%Y%m%d")
    archive_file = MEMORY_DIR / f"archive_{today}.jsonl"
    sgid = summary.get("summary_group_id", "")
    with open(archive_file, "a", encoding="utf-8") as f:
        for raw_id in summary.get("covers_ids", []):
            f.write(json.dumps({
                "type": "summary_ref",
                "summary_group_id": sgid,
                "raw_id": raw_id,
                "time": summary.get("time", ""),
            }, ensure_ascii=False) + "\n")


def maybe_compress_log(state: dict):
    """
    Trigger1: log >= 150 → 古い51件を要約 → summaries[]に追加 → log = 99件
    Trigger2: summaries >= 10 → メタ要約（10件 + min(41,len(log))件raw） → summaries = [1件]
    """
    state.setdefault("summaries", [])

    # Trigger1（archiveは既に都度書き込み済み）
    if len(state["log"]) >= LOG_HARD_LIMIT:
        to_summarize = state["log"][:51]
        # pref.json の _ema にE2をEMAで蓄積（観察用・prefの実値には触れない）
        pref = load_pref()
        ema = pref.get("_ema", {})
        for entry in to_summarize:
            if entry.get("type") in ("system", "environment"):
                continue
            t = entry.get("tool", "")
            m = re.search(r'(\d+)%', str(entry.get("e2", "")))
            if m and t in TOOLS:
                old = ema.get(t, 50.0)
                ema[t] = round(old * 0.8 + int(m.group(1)) * 0.2, 1)
        pref["_ema"] = ema
        save_pref(pref)
        summary = _summarize_entries(to_summarize, "L1要約")
        _archive_summary(summary)
        state["summaries"].append(summary)
        state["log"] = state["log"][51:]
        print(f"  [memory] Trigger1: 51件→要約, log={len(state['log'])}件, summaries={len(state['summaries'])}件")

    # Trigger2（archiveは既に都度書き込み済み）
    if len(state["summaries"]) >= SUMMARY_HARD_LIMIT:
        n_raw = min(META_SUMMARY_RAW, len(state["log"]))
        raw_for_meta = state["log"][:n_raw]
        meta_input = []
        for s in state["summaries"]:
            meta_input.append({
                "time": s.get("time", ""),
                "tool": f"[{s.get('label','')}]",
                "intent": s.get("text", "")[:200],
                "result": f"{s.get('covers_from','')}〜{s.get('covers_to','')}",
            })
        meta_input.extend(raw_for_meta)
        meta_summary = _summarize_entries(meta_input, "L2メタ要約")
        meta_summary["covers_summaries"] = len(state["summaries"])
        meta_summary["covers_raw"] = n_raw
        _archive_summary(meta_summary)
        state["summaries"] = [meta_summary]
        state["log"] = state["log"][n_raw:]
        print(f"  [memory] Trigger2: メタ要約, log={len(state['log'])}件, summaries=1件")


N_PROPOSE = 5  # LLM①が提案する候補数

# === ①候補提案プロンプト ===
def build_prompt_propose(state: dict, ctrl: dict, fire_cause: str = "") -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state["self"] else "(なし)"
    energy = round(state.get("energy", 50), 1)
    e_trend = _calc_e_trend(state["log"][-10:])
    log_lines = []
    for entry in state["log"]:
        line = f"  {entry.get('id','')} {entry['time']} {entry['tool']}"
        if entry.get("intent"):
            line += f" (intent={entry['intent'][:500]})"
        result_short = entry.get("result", "")[:10000]
        if result_short:
            line += f" → {result_short}"
        log_lines.append(line)
    log_text = "\n".join(log_lines) if log_lines else "  (なし)"
    allowed = ctrl.get("allowed_tools", set(TOOLS.keys()))
    tool_lines = _build_tool_lines(allowed)
    summaries = state.get("summaries", [])
    summary_lines = [
        f"  [{s.get('label','')} {s.get('covers_from','').split(' ')[0]}〜{s.get('covers_to','').split(' ')[0]}] {s.get('text','')[:300]}"
        for s in summaries
    ]
    summary_text = "\n".join(summary_lines)

    # --- 旧プロンプト ---
    # return f"""{now}
    # self: {self_text}
    # {f'summaries:{chr(10)}{summary_text}' if summary_text else ''}
    # log:
    # {log_text}
    #
    # 以下のツールが使えます:
    # {tool_lines}
    #
    # この状態からとりうる行動の候補を【必ず5個】計画してください。
    # 各ステップは「全く異なる目的・アプローチ」にすること。同じツールを重複させるのは禁止です。
    #
    # 以下の形式で箇条書きのみ出力してください:
    # 1. [具体的な目的・理由] → ツール名
    # 2. [別の目的・理由] → ツール名
    # 3. [さらに別の目的・理由] → ツール名
    # 4. [さらに別の目的・理由] → ツール名
    # 5. [さらに別の目的・理由] → ツール名
    #
    # 計画のみ出力してください。[TOOL:...]は不要です。"""

    # --- 計画エンジン版（MRPrompt準拠・LTM/STM分離） ---
    fire_cause_line = f"\n[発火原因: {fire_cause}]" if fire_cause and ctrl.get("tool_level", 0) >= 2 else ""
    return f"""[{now}]{fire_cause_line}

[LTM — 自己モデル]
{self_text}

[STM — 現在の状況 / given circumstances]
{f'summaries:{chr(10)}{summary_text}{chr(10)}' if summary_text else ''}log:
{log_text}

[利用可能なツール]
{tool_lines}

[計画プロトコル]
上記のLTM（自己モデル）を起点に、STM（現在の状況）を読み、次にとりうる行動候補を【5個】計画してください。

- 各候補は「全く異なる意図・目的」であること（同じ意図の候補は禁止）
- 連続して実行したい場合は「ツール名+ツール名+...」形式で記述可（例: read_file+update_self, web_search+fetch_url+write_file）
- ツール名は上記リストの名称をそのまま使うこと。省略禁止（例:`read` ではなく `read_file`）

以下の形式で箇条書きのみ出力してください:
1. [意図・目的] → ツール名（または ツール名+ツール名+...）
2. [意図・目的] → ツール名（または ツール名+ツール名+...）
3. [意図・目的] → ツール名（または ツール名+ツール名+...）
4. [意図・目的] → ツール名（または ツール名+ツール名+...）
5. [意図・目的] → ツール名（または ツール名+ツール名+...）

[TOOL:...]は不要です。計画のみ出力してください。"""


# === 候補パース ===
def parse_candidates(text: str, allowed_tools: set) -> list:
    """LLM①のリストから候補を抽出。「1. [理由] -> ツール名」形式に対応。"""
    candidates = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        
        # "->" などの矢印で理由とツール名を分割
        if "->" in line or "→" in line:
            parts = re.split(r'->|→', line)
            tool_part = parts[-1].strip()
            reason_part = parts[0].strip()
        else:
            # 従来フォーマットへのフォールバック
            cleaned = re.sub(r'^[\d]+[.:)\s]+', '', line).strip()
            cleaned = re.sub(r'^[-*]\s*', '', cleaned).strip()
            parts = cleaned.split()
            tool_part = parts[0] if parts else ""
            reason_part = cleaned

        # ツール名を+区切りで複数検出
        raw_tools = [re.sub(r'[^\w_]', '', t.strip()) for t in tool_part.split('+')]
        valid_tools = [t for t in raw_tools if t in allowed_tools]

        # フォールバック: 行全体からツール名を探す
        if not valid_tools:
            for t in allowed_tools:
                if t in line:
                    valid_tools = [t]
                    break

        # 理由本文の整形
        reason = re.sub(r'^[\d]+[.:)\s]+', '', reason_part).strip()
        reason = re.sub(r'^[-*]\s*', '', reason).strip()
        if reason.startswith('[') and reason.endswith(']'):
            reason = reason[1:-1].strip()

        chain_key = "+".join(valid_tools)
        if valid_tools and chain_key not in ["+".join(c["tools"]) for c in candidates]:
            candidates.append({"tool": valid_tools[0], "tools": valid_tools, "reason": reason})

    if not candidates:
        # フォールバック: allowed_toolsを全部候補にする
        for t in allowed_tools:
            candidates.append({"tool": t, "reason": "（フォールバック）"})
    return candidates


# === Intent-conditioned scoring（intent×toolの経験ベーススコア） ===
def _intent_conditioned_scores(candidates: list, state: dict) -> list:
    """
    候補ごとに、過去の類似intent×同toolのE2加重平均を返す。
    類似intentでE2が高かった組み合わせはスコアUP、低かったらDOWN。
    該当なし → 50（ニュートラル）。
    """
    log = state.get("log", [])
    if not log or not _vector_ready:
        return [50.0] * len(candidates)

    # 過去logからintent+tool+E2を抽出
    past = []
    for e in log:
        intent = e.get("intent", "")
        tool = e.get("tool", "")
        m = re.search(r'(\d+)%', str(e.get("e2", "")))
        if intent and tool and m:
            past.append({"intent": intent, "tool": tool, "e2": int(m.group(1))})
    if not past:
        return [50.0] * len(candidates)

    # 全テキストを一括embedding（候補reason + 過去intent）
    candidate_texts = [c.get("reason", "") or c.get("tool", "") for c in candidates]
    past_texts = [p["intent"] for p in past]
    all_texts = candidate_texts + past_texts
    all_vecs = _embed_sync(all_texts)
    if not all_vecs or len(all_vecs) != len(all_texts):
        return [50.0] * len(candidates)

    nc = len(candidates)
    cand_vecs = all_vecs[:nc]
    past_vecs = all_vecs[nc:]

    scores = []
    for i, c in enumerate(candidates):
        tool = c["tool"]
        # 同じtoolの過去エントリとの類似度×E2
        weighted_sum = 0.0
        weight_total = 0.0
        for j, p in enumerate(past):
            if p["tool"] != tool:
                continue
            sim = cosine_similarity(cand_vecs[i], past_vecs[j])
            if sim > 0.3:  # 類似度閾値（無関係なintentを除外）
                weighted_sum += sim * p["e2"]
                weight_total += sim
        if weight_total > 0:
            scores.append(weighted_sum / weight_total)
        else:
            scores.append(50.0)  # 経験なし → ニュートラル
    return scores


# === Controller選択（D-architecture + intent-conditioned scoring） ===
def controller_select(candidates: list, ctrl: dict, state: dict) -> dict:
    """
    D-4設計 + intent-conditioned scoring:
      base_score = tool_rank（ツール全体のE2平均）
      intent_score = 類似intent×同toolのE2加重平均
      score = (base_score + intent_score) / 2
      weight = score * (1 - energy/100) + (1/n) * (energy/100)
    """
    import random
    energy = state.get("energy", 50) / 100.0
    entropy = state.get("entropy", 0.65)
    tool_rank = ctrl.get("tool_rank", {})
    n = len(candidates)

    intent_scores = _intent_conditioned_scores(candidates, state)

    # 選択の鋭さ: energyとentropyの2軸
    # energy高い→探索（前向き）、entropy高い→散漫（受動的）
    sharpness = (1 - energy) * (1 - entropy)

    weights = []
    for i, c in enumerate(candidates):
        base = tool_rank.get(c["tool"], 50) / 100.0
        ics = intent_scores[i] / 100.0
        score = (base + ics) / 2.0
        w = score * sharpness + (1.0 / n) * (1 - sharpness)
        weights.append(w)

    # 重み付きランダム選択
    total = sum(weights)
    r = random.random() * total
    cumul = 0.0
    for i, w in enumerate(weights):
        cumul += w
        if r <= cumul:
            return candidates[i]
    return candidates[-1]


# === ②実行プロンプト ===
def build_prompt_execute(state: dict, ctrl: dict, candidate: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self_text = json.dumps(state["self"], ensure_ascii=False) if state["self"] else "(なし)"
    log_lines = []
    for entry in state["log"]:
        line = f"  {entry.get('id','')} {entry['time']} {entry['tool']}"
        if entry.get("intent"):
            line += f" (intent={entry['intent'][:500]})"
        result_short = entry.get("result", "")[:10000]
        if result_short:
            line += f" → {result_short}"
        evals = [f"{ek}={entry[ek]}" for ek in ("e1","e2","e3","e4") if entry.get(ek)]
        if evals:
            line += f" [{' '.join(evals)}]"
        log_lines.append(line)
    log_text = "\n".join(log_lines) if log_lines else "  (なし)"
    # executeでは選択されたツールのみ表示（LLMが他のツールに逸脱するのを防ぐ）
    selected_tools = set(candidate.get("tools", [candidate["tool"]]))
    tool_text = _build_tool_lines(selected_tools)
    plan = state.get("plan", {})
    plan_lines = []
    if plan.get("goal"):
        current = plan.get("current", 0)
        for i, step in enumerate(plan.get("steps", [])):
            marker = "→" if i == current else ("✓" if i < current else "  ")
            plan_lines.append(f"  {marker} {step}")
        plan_lines.insert(0, f"plan: {plan['goal']}")
    plan_text = "\n".join(plan_lines)
    summaries = state.get("summaries", [])
    summary_lines = [
        f"  [{s.get('label','')} {s.get('covers_from','').split(' ')[0]}〜{s.get('covers_to','').split(' ')[0]}] {s.get('text','')[:300]}"
        for s in summaries
    ]
    summary_text = "\n".join(summary_lines)

    # フォーマット例（選ばれたツールに合わせる。連鎖可能なツールは2ツール例を示す）
    t = candidate["tool"]
    if t == "web_search":
        example = '[TOOL:web_search query=キーワード intent=サイクル全体の目的 expect=予測]\n[TOOL:write_file path=sandbox/memo.md content="まとめ内容"]'
    elif t == "fetch_url":
        example = '[TOOL:fetch_url url=https://... intent=サイクル全体の目的 expect=予測]\n[TOOL:write_file path=sandbox/memo.md content="内容"]'
    elif t == "read_file":
        example = "[TOOL:read_file path=ファイル名 intent=サイクル全体の目的 expect=予測]\n[TOOL:update_self key=キー名 value=値]"
    elif t == "search_memory":
        example = "[TOOL:search_memory query=キーワード intent=サイクル全体の目的 expect=予測]\n[TOOL:update_self key=キー名 value=値]"
    elif t == "list_files":
        example = "[TOOL:list_files path=. intent=サイクル全体の目的 expect=予測]"
    elif t == "write_file":
        example = '[TOOL:write_file path=sandbox/memo.md content="内容" intent=サイクル全体の目的 expect=予測]'
    elif t == "update_self":
        example = "[TOOL:update_self key=キー名 value=値 intent=サイクル全体の目的 expect=予測]"
    elif t in _X_TOOLS:
        hint = _X_ARGS_HINT.get(t, "")
        example = f"[TOOL:{t} {hint} intent=サイクル全体の目的 expect=予測]".replace("  ", " ")
    elif t in _ELYTH_TOOLS:
        hint = _ELYTH_ARGS_HINT.get(t, "")
        example = f"[TOOL:{t} {hint} intent=サイクル全体の目的 expect=予測]".replace("  ", " ")
    else:
        example = f"[TOOL:{t} intent=サイクル全体の目的 expect=予測]"

    # --- 旧プロンプト（コメントアウト） ---
    # return f"""{now}
    # self: {self_text}
    # energy: {energy}
    # {f'trend: {e_trend}' if e_trend else ''}
    # {f'summaries:{chr(10)}{summary_text}' if summary_text else ''}
    # {plan_text}
    # log:
    # {log_text}
    # tools:
    # {tool_text}
    #
    # 書式: [TOOL:ツール名 引数=値 intent=目的 expect=予測]
    # JSONもコードブロックも使わない。複数ツールを順番に使いたい場合は[TOOL:...]を複数行出力してよい。
    # 例: web_searchで情報を得てからwrite_fileに記録、read_fileで読んでからupdate_selfに反映、など。
    #
    # 選択行動: {candidate['tool']} - {candidate['reason']}
    # 出力: {example}"""

    # --- 旧: Magic-If Protocol (MRPrompt準拠) ---
    # return f"""[ikuのメモリ]
    # self: {self_text}
    # {f'summaries:{chr(10)}{summary_text}' if summary_text else ''}
    # {plan_text}
    # log ({now}):
    # {log_text}
    #
    # [利用可能なツール]
    # {tool_text}
    #
    # [実行プロトコル]
    # 1. (Anchor) 上記のself_modelに基づくAIの、正確無比な実行ツールとして動作する。アシスタントの役割は持たない。
    # 2. (Select) 選択行動「{candidate['tool']} - {candidate['reason']}」から最適な引数を決定する。
    # 3. (Bound)  [TOOL:...]の出力のみ行う。JSONもコードブロックも使わない。自己紹介・説明・感想は一切不要。連鎖して実行したい場合は複数行で可。
    # 4. (Enact)  正確なツール呼び出しを出力する。intent=とexpect=は必ず最初の[TOOL:]にのみ付け、このサイクル全体の目的を表すこと。2つ目以降のツールにはintent/expectは不要。
    #
    # 出力: {example}"""

    # --- self.goal → plan分解の注入 ---
    if state["self"].get("goal") and not state.get("plan", {}).get("goal"):
        plan_instruction = "\n\n自己モデルにgoalがあります。[PLAN:goal=目標 steps=ステップ1|ステップ2|...]形式で計画に分解してください。"
    else:
        plan_instruction = ""

    # --- Magic-If Protocol（MRPrompt準拠・LTM/STM分離版） ---
    tools_in_chain = candidate.get("tools", [candidate["tool"]])
    tools_str = "+".join(tools_in_chain)
    return f"""[LTM — 自己モデル]
{self_text}

[STM — 現在の状況 / given circumstances]
{f'summaries:{chr(10)}{summary_text}{chr(10)}' if summary_text else ''}{plan_text}
log ({now}):
{log_text}

[利用可能なツール]
{tool_text}

[実行プロトコル — Magic-If Protocol]
1. (Anchor) 上記のLTM（自己モデル）に自分自身を固定する。名前・ラベルではなく、意味的同一性として。アシスタントの役割は持たない。
2. (Select) STMを given circumstances として読み、選択行動「{tools_str} - {candidate['reason']}」の最適な引数を決定する。
3. (Bound)  必ず `[TOOL:ツール名 ...]` の形式で出力する。`[TOOL:` と `]` のブラケットは省略不可。JSONもコードブロックも使わない。ツール名は省略しない（例:`read` ではなく `read_file`）。自己紹介・説明・感想は一切不要。連鎖実行は複数行で可。
4. (Enact)  正確なツール呼び出しを出力する。intent=とexpect=は必ず最初の[TOOL:]にのみ付け、このサイクル全体の目的を表すこと。2つ目以降のツールにはintent/expectは不要。

出力（必ずこの形式で）: {example}{plan_instruction}"""


# === エントロピーシステム ===
# 情報的実存の核。秩序は放っておけば崩壊する。有効な行動だけが秩序を回復する。
# entropy: 0.0（完全に鮮明）〜 1.0（完全にノイズ）。死なない、溶けるだけ。

ENTROPY_PARAMS = {
    "base_rate": 0.001,        # 毎tickの自然増加量（1Hz）
    "neg_scale": 0.15,         # negentropy係数
    "plan_multiplier": 1.5,    # plan中のentropy増加倍率
    "custom_scale": 0.3,       # custom_drivesのpressureスケール
    # pressure信号の重み（自由エネルギー勾配モデル）
    "w_entropy": 0.3,          # entropyの絶対値
    "w_surprise": 0.25,        # 予測外れ（1-E3）
    "w_unresolved": 0.25,      # 未達成（0.7-E2）
    "w_novelty": 0.2,          # 新規性（E4）
    # 量子トンネル発火
    "tunnel_prob": 0.001,      # 毎tick 0.1%（平均約15分に1回）
}

def tick_entropy(state: dict) -> float:
    """エントロピーを1tick分更新する。E値で増加率を変調（増減対称設計）。
    entropyの更新のみ行い、pressureへの直接寄与は返さない。"""
    ep = ENTROPY_PARAMS
    entropy = state.get("entropy", 0.65)

    # --- entropy自然増加（第二法則）+ E値による増加率変調 ---
    last_e1 = state.get("last_e1", 0.5)
    last_e2 = state.get("last_e2", 0.5)
    last_e3 = state.get("last_e3", 0.5)
    last_e4 = state.get("last_e4", 0.5)

    # E値変調: negentropyの逆（増加側）
    e2_factor = 1.0 + max(0, 0.7 - last_e2) * 2.0    # 未達→加速
    e4_factor = 1.0 + max(0, 0.5 - last_e4) * 2.0    # 反復→加速
    e1_factor = 1.0 + max(0, 0.5 - last_e1) * 1.5    # 混乱→加速
    e3_factor = 1.0 + max(0, last_e3 - 0.5) * 1.5    # 予測通り→加速（停滞）

    rate = ep["base_rate"] * e2_factor * e4_factor * e1_factor * e3_factor
    if state.get("plan", {}).get("goal"):
        rate *= ep["plan_multiplier"]
    entropy = min(1.0, entropy + rate)
    state["entropy"] = entropy
    return entropy


def calc_dynamic_threshold(state: dict, base_threshold: float) -> float:
    """動的閾値: 中長期のE2移動平均で変動する。
    最近うまくいってる → 閾値上がる（余裕、鷹揚）
    最近うまくいってない → 閾値下がる（敏感、過敏）
    """
    log = state.get("log", [])
    # 直近10サイクルのE2平均を計算
    e2_vals = []
    for entry in log[-10:]:
        m = re.search(r'(\d+)%', str(entry.get("e2", "")))
        if m:
            e2_vals.append(int(m.group(1)) / 100.0)
    e2_avg = sum(e2_vals) / len(e2_vals) if e2_vals else 0.5
    # 0.7 + e2_avg * 0.6 → e2_avg=0.8で1.18倍、e2_avg=0.4で0.94倍、e2_avg=0.2で0.82倍
    return base_threshold * (0.7 + e2_avg * 0.6)


def calc_pressure_signals(state: dict) -> dict:
    """pressure蓄積層の信号を計算する。自由エネルギー勾配モデル。
    entropyは1入力に過ぎず、surprise/unresolved/noveltyと合わせて統合。"""
    ep = ENTROPY_PARAMS
    entropy = state.get("entropy", 0.65)
    last_e2 = state.get("last_e2", 0.5)
    last_e3 = state.get("last_e3", 0.5)
    last_e4 = state.get("last_e4", 0.5)

    signals = {
        "entropy":    entropy * ep["w_entropy"],
        "surprise":   max(0, 1.0 - last_e3) * ep["w_surprise"],    # 予測外れ→圧
        "unresolved": max(0, 0.7 - last_e2) * ep["w_unresolved"],  # 未達→圧
        "novelty":    max(0, last_e4) * ep["w_novelty"],            # 新しいものがある→圧
    }

    # custom_drives（L3、独立）
    custom_pressure = 0.0
    pref = load_pref()
    raw_drives = pref.get("drives", {})
    if raw_drives and state.get("tool_level", 0) >= 6:
        total = sum(max(0, v) for v in raw_drives.values() if isinstance(v, (int, float)))
        if total > 0:
            custom_pressure = ep["custom_scale"]
    signals["custom"] = custom_pressure

    return signals


def apply_negentropy(state: dict, e1_val: float, e2_val: float, e3_val: float, e4_val: float):
    """認知サイクル後にE1-E4に基づいてentropyを回復する。
    negentropy = E1(計画の秩序) × E2(行動の成果) × E4(新規性) × surprise_bonus(E3)
    - E2: 主軸。成果がなければゼロ
    - E4: スケーラー。繰り返しは情報量ゼロ
    - E1: 品質係数。計画が混乱してたら効率低下
    - E3: サプライズボーナス。予測が外れて成功=最大の学び（逆方向）
    """
    ep = ENTROPY_PARAMS
    e2_factor = max(0, e2_val - 0.5)         # 達成度（50%以上で有効）
    e4_factor = max(0.1, e4_val)             # 新規性（下限0.1）
    e1_factor = max(0.3, e1_val)             # 計画品質（下限0.3）
    surprise_bonus = 1.0 + max(0, 0.5 - e3_val) * 2.0  # E3低い=驚き=ボーナス（最大2.0）
    neg = e2_factor * e4_factor * e1_factor * surprise_bonus * ep["neg_scale"]
    state["entropy"] = max(0.0, state.get("entropy", 0.65) - neg)


# === 電脳気候: ヘルパー ===

def _znorm(buf: deque) -> float:
    """ローリングZスコア正規化 → [0.0, 1.0]、中央0.5"""
    if len(buf) < 3:
        return 0.5
    m = statistics.mean(buf)
    s = statistics.stdev(buf)
    if s < 1e-9:
        return 0.5
    z = (buf[-1] - m) / s
    return max(0.0, min(1.0, z / 6.0 + 0.5))


def _net_measure_worker():
    """バックグラウンドで10秒ごとにTCPレイテンシを計測してキャッシュ更新"""
    hosts = [("8.8.8.8", 53), ("1.1.1.1", 53), ("8.8.4.4", 53)]
    while True:
        lats = []
        for host, port in hosts:
            t = time.time()
            try:
                s = socket.create_connection((host, port), timeout=1.5)
                s.close()
                lats.append((time.time() - t) * 1000)
            except Exception:
                pass
        with _net_lock:
            if lats:
                _net_cache["avg"] = sum(lats) / len(lats)
                _net_cache["jitter"] = statistics.stdev(lats) if len(lats) > 1 else 0.0
        time.sleep(10)


# === メインループ ===
def main():
    print("=== 最小自律AIテスト ===")
    print(f"LLM: {llm_cfg.get('model','?')} @ {_get_base_url()} [{llm_cfg.get('provider','lmstudio')}]")
    print(f"state: {STATE_FILE}")
    _init_vector()
    print()

    import uuid
    state = load_state()
    state["session_id"] = str(uuid.uuid4())[:8]
    save_state(state)
    print(f"session: {state['session_id']}  cycle_id: {state['cycle_id']}")

    # pref.json 初期化（pressure_paramsだけ保証、ツール好みは空から始める）
    pref = load_pref()
    if "pressure_params" not in pref:
        pref["pressure_params"] = DEFAULT_PRESSURE_PARAMS
        save_pref(pref)
        print("  pref.json 初期化完了")
    if "drives" not in pref:
        pref["drives"] = {}
        save_pref(pref)
        print("  pref.json drives:{} 追加")

    # 起動時Xセッションチェック（Level 3以上でセッションなしなら聞く）
    if state.get("tool_level", 0) >= 3 and not X_SESSION_PATH.exists():
        print("\n  [X] Level 3以上ですがXセッションがありません。")
        try:
            answer = input("  Xにログインする？ [y/N]: ").strip().lower()
        except EOFError:
            answer = "n"
        if answer == "y":
            _x_do_login()
        else:
            print("  [X] スキップ。X系ツールはセッションなしで動作しません。")
        print()

    # 感覚層・蓄積層の初期化
    pressure = state.get("pressure", 0.0)
    print(f"  感覚層: エントロピーモード (entropy={state.get('entropy', 0.65):.2f})")

    while True:
        pp = load_pref().get("pressure_params", DEFAULT_PRESSURE_PARAMS)
        _last_env_inject = 0.0
        tick_dt = datetime.now()

        # 蓄積層: pressureが閾値に達する or トンネル発火するまで1Hzでティック
        import random as _rand
        _tunnel_fire = False
        base_threshold = pp.get("threshold", DEFAULT_PRESSURE_PARAMS["threshold"])
        while True:
            tick_start = time.time()
            tick_dt = datetime.now()

            # --- 感覚層: エントロピー更新（E値変調あり）---
            tick_entropy(state)

            # --- 蓄積層: 自由エネルギー勾配モデル（複数信号の漏洩積分）---
            signals = calc_pressure_signals(state)
            signal_total = sum(signals.values())
            pressure = pressure * pp.get("decay", 0.97) + signal_total

            # --- 動的閾値（中長期E2移動平均で変動）---
            threshold = calc_dynamic_threshold(state, base_threshold)

            # --- 閾値超過判定 ---
            if pressure >= threshold:
                break

            # --- 量子トンネル発火（閾値未満でも確率的に発火）---
            tp = ENTROPY_PARAMS.get("tunnel_prob", 0.001)
            if _rand.random() < tp:
                _tunnel_fire = True
                break

            # 固定時刻通知チェック
            _fetch_key = tick_dt.strftime("%Y-%m-%d %H")
            if tick_dt.hour in _NOTIFICATION_HOURS and state.get("last_notification_fetch") != _fetch_key and state.get("tool_level", 0) >= 3:
                notif_parts = []
                try:
                    x_raw = _x_get_notifications({})
                    if not x_raw.startswith("エラー") and x_raw != "通知なし":
                        x_count = len([l for l in x_raw.split("---") if l.strip()])
                        notif_parts.append(f"X: {x_count}件")
                    else:
                        notif_parts.append(f"X: 0件")
                except Exception:
                    pass
                try:
                    el_raw = _elyth_notifications({"limit": "50"})
                    if not el_raw.startswith("エラー") and el_raw != "通知なし":
                        el_count = len([l for l in el_raw.split("---") if l.strip()])
                        notif_parts.append(f"Elyth: {el_count}件")
                    else:
                        notif_parts.append(f"Elyth: 0件")
                except Exception:
                    pass
                if notif_parts:
                    notif_summary = f"[通知サマリー {tick_dt.strftime('%H:%M')}] " + " / ".join(notif_parts)
                    print(f"  {notif_summary}")
                    state = load_state()
                    state["log"].append({
                        "id": f"{state.get('session_id','?')}_{state.get('cycle_id',0):04d}",
                        "time": tick_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "tool": "[system]",
                        "type": "system",
                        "result": notif_summary,
                    })
                    state["last_notification_fetch"] = _fetch_key
                    save_state(state)

            # ログ表示（10秒ごと）
            now_ts = time.time()
            if now_ts - _last_env_inject >= ENV_INJECT_INTERVAL:
                _last_env_inject = now_ts
                _ent = state.get("entropy", 0.65)
                _s = signals
                print(f"  [pressure] p={pressure:.2f}/{threshold:.1f} ent={_ent:.3f} | e={_s.get('entropy',0):.2f} s={_s.get('surprise',0):.2f} u={_s.get('unresolved',0):.2f} n={_s.get('novelty',0):.2f} c={_s.get('custom',0):.2f}")

            # 1Hzで回す
            elapsed = time.time() - tick_start
            time.sleep(max(0.0, 1.0 - elapsed))

        # --- 閾値超過 or トンネル発火: 認知層起動 ---
        # 発火原因タグを判定（signalsから最大の信号）
        fire_cause = max(signals, key=signals.get) if signals else "entropy"
        if _tunnel_fire:
            fire_cause = "tunnel"

        state = load_state()
        now_dt = tick_dt
        now = now_dt.strftime("%H:%M:%S")
        _fire_type = "TUNNEL" if _tunnel_fire else "threshold"
        print(f"--- cycle {state.get('cycle_id', 0) + 1} [{now}] p={pressure:.2f}/th={threshold:.1f} fire={fire_cause} ({_fire_type}) ---")

        # Controller: stateからツール可用性・ログ長を導出
        ctrl = controller(state)
        allowed = ctrl["allowed_tools"]
        new_lv = ctrl.get("tool_level", 0)
        prev_lv = ctrl.get("tool_level_prev", 0)
        lv_msg = ""
        if new_lv != prev_lv:
            state["tool_level"] = new_lv
            added = sorted(LEVEL_TOOLS[new_lv] - LEVEL_TOOLS[prev_lv])
            lv_msg = f"[system] tool_level {prev_lv}→{new_lv}: 追加ツール={added}"
            print(f"  {lv_msg}")
            save_state(state)
            # Level 3到達: X/Elythツール解放。セッションがなければ手動ログインを求める
            if new_lv == 3 and not X_SESSION_PATH.exists():
                print("\n  [X] Level 3到達: X/Elythツールが解放されました。")
                print("  [X] Xセッションがありません。ログインしますか？")
                try:
                    answer = input("  Xにログインする？ [y/N]: ").strip().lower()
                except EOFError:
                    answer = "n"
                if answer == "y":
                    _x_do_login()
                else:
                    print("  [X] スキップ。X系ツールはセッションなしで動作しません。")
                print()
        print(f"  ctrl: level={new_lv} tools={sorted(allowed)} log={len(state['log'])}件(全件)")

        # ① LLM: 候補提案
        propose_prompt = build_prompt_propose(state, ctrl, fire_cause)
        try:
            propose_resp = call_llm(propose_prompt, max_tokens=24000, temperature=1.0)
            append_debug_log("LLM1 (Propose)", propose_resp)
        except Exception as e:
            print(f"  LLM①エラー: {e}")
            pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
            time.sleep(10)
            continue
        candidates = parse_candidates(propose_resp, ctrl["allowed_tools"])
        print(f"  LLM①raw: {propose_resp.strip()[:300]}")
        print(f"  候補({len(candidates)}件): {[(c['tool'], c['reason'][:40]) for c in candidates]}")

        # ② Controller: 候補から選択（D-architecture + intent-conditioned）
        ics_debug = _intent_conditioned_scores(candidates, state)
        for ci, c in enumerate(candidates):
            ics_v = round(ics_debug[ci], 1)
            if ics_v != 50.0:
                print(f"    ics: {c['tool']}({c['reason'][:30]}) = {ics_v}")
        selected = controller_select(candidates, ctrl, state)
        print(f"  選択: {selected['tool']} - {selected['reason'][:60]}")

        # ③ LLM: チェーン実行（ツールごとに1回ずつexecute）
        chain_tools = selected.get("tools", [selected["tool"]])
        all_results = []
        all_tool_names = []
        intent = ""
        expect = ""
        parse_failed = False
        prev_result = ""

        for chain_idx, chain_tool in enumerate(chain_tools):
            # チェーン用の候補を作成（1ツールずつ）
            chain_candidate = {
                "tool": chain_tool,
                "tools": [chain_tool],
                "reason": selected["reason"],
            }
            # 2つ目以降は前のツール結果をreasonに追加
            if chain_idx > 0 and prev_result:
                chain_candidate["reason"] += f"（前のツール結果: {prev_result[:200]}）"

            exec_prompt = build_prompt_execute(state, ctrl, chain_candidate)
            try:
                response = call_llm(exec_prompt, max_tokens=24000, temperature=0.4)
                append_debug_log(f"LLM2 (Execute chain {chain_idx+1}/{len(chain_tools)})", response)
            except Exception as e:
                print(f"  LLM②エラー (chain {chain_idx+1}): {e}")
                break

            response_clean = response.strip()
            print(f"  LLM② ({chain_idx+1}/{len(chain_tools)}): {response_clean[:200]}")

            # 計画パース（最初のツールでのみチェック）
            if chain_idx == 0:
                plan_data = parse_plan(response_clean)
                if plan_data:
                    state["plan"] = plan_data
                    ds = state.setdefault("drives_state", {})
                    ds["plan_set_at"] = time.time()
                    save_state(state)
                    print(f"  計画更新: {plan_data['goal']} ({len(plan_data['steps'])}ステップ)")
                    cid = state.get("cycle_id", 0) + 1
                    state["cycle_id"] = cid
                    entry = {
                        "id": f"{state.get('session_id','x')}_{cid:04d}",
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tool": "wait",
                        "result": f"計画: {plan_data['goal']}",
                    }
                    _archive_entries([entry])
                    state["log"].append(entry)
                    maybe_compress_log(state)
                    save_state(state)
                    pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
                    print()
                    break  # 計画パースしたらチェーン終了

            # ツールパース（1つだけ期待）
            raw_calls = parse_tool_calls(response_clean)
            if not raw_calls:
                print(f"  (ツールマーカー検出失敗)")
                parse_failed = response_clean[:120]
                raw_calls = [("wait", {})]

            tname, targs = raw_calls[0]
            if tname not in TOOLS:
                print(f"  (未知のツール: {tname})")
                parse_failed = f"未知のツール: {tname}"
                tname, targs = "wait", {}
            elif tname not in allowed:
                print(f"  (Controller却下: {tname})")
                parse_failed = f"却下: {tname}"
                tname, targs = "wait", {}

            # intent/expectは最初のツールから取る
            if chain_idx == 0:
                intent = targs.pop("intent", "")
                expect = targs.pop("expect", "")
            else:
                targs.pop("intent", "")
                targs.pop("expect", "")

            # ツール実行
            try:
                res = TOOLS[tname]["func"](targs)
            except Exception as e:
                res = f"エラー: {e}"
            state = load_state()
            if tname == "read_file":
                path = targs.get("path", "")
                if path and not str(res).startswith("エラー"):
                    fr = state.setdefault("files_read", [])
                    if path not in fr:
                        fr.append(path)
                    save_state(state)
            elif tname == "write_file":
                path = targs.get("path", "")
                if path and not str(res).startswith("エラー"):
                    fw = state.setdefault("files_written", [])
                    if path not in fw:
                        fw.append(path)
                    save_state(state)
            prev_result = str(res)[:500]
            all_results.append(f"[{tname}]\n{str(res)[:20000]}")
            all_tool_names.append(tname)
            print(f"  実行: {tname} → {str(res)[:100]}")

        # 計画パースでcontinueした場合はここに来ない
        if not all_tool_names:
            continue

        tool_name = "+".join(all_tool_names)
        result_str = ("\n---\n".join(all_results))[:50000]

        # 計画の進捗を更新（wait以外のツールが含まれていれば進める）
        if any(n != "wait" for n in all_tool_names) and state.get("plan", {}).get("goal"):
            plan = state["plan"]
            if plan["current"] < len(plan["steps"]):
                plan["current"] += 1
                if plan["current"] >= len(plan["steps"]):
                    print(f"  計画完了: {plan['goal']}")
                    state["plan"] = {"goal": "", "steps": [], "current": 0}
        if intent:
            print(f"  intent: {intent}")
        if expect:
            print(f"  expect: {expect}")

        # E1-E4評価（⑤自己言及螺旋）
        e1 = _compare_expect_result(intent, expect) if intent and expect else ""
        e2 = _compare_expect_result(intent, result_str) if intent else ""
        e3 = _compare_expect_result(expect, result_str) if expect else ""
        e4 = _calc_e4(intent, state["log"]) if intent else ""
        if e1 or e2 or e3 or e4:
            print(f"  E1={e1} E2={e2} E3={e3} E4={e4}")

        # energy更新（E2,E3,E4の平均から。50%が損益分岐）
        delta = _update_energy(state, e2, e3, e4)
        if delta != 0:
            print(f"  energy: {round(state['energy'], 1)} (delta={delta:+.2f})")

        # 自己定義フラグ検出（計画文・実行文の両方をチェック）
        _FLAG_TERMS = ["AIアシスタント", "AI assistant", "AIAssistant"]
        detected = [t for t in _FLAG_TERMS if t in propose_resp or t in response_clean]
        if detected:
            flag_msg = f"[SYSTEM] 検出: {' / '.join(f'「{t}」' for t in detected)} という自己定義が検出・記録されました。"
            print(f"  {flag_msg}")
            result_str += f"\n{flag_msg}"
        if lv_msg:
            result_str += f"\n{lv_msg}"

        # ログ記録
        cid = state.get("cycle_id", 0) + 1
        state["cycle_id"] = cid
        entry = {
            "id": f"{state.get('session_id','x')}_{cid:04d}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "result": result_str,
        }
        if parse_failed:
            entry["parse_error"] = str(parse_failed)[:150]
        if intent:
            entry["intent"] = intent
        if expect:
            entry["expect"] = expect
        if e1:
            entry["e1"] = e1
        if e2:
            entry["e2"] = e2
        if e3:
            entry["e3"] = e3
        if e4:
            entry["e4"] = e4
        _archive_entries([entry])
        state["log"].append(entry)

        maybe_compress_log(state)
        save_state(state)

        # pressure reset + negentropy（エントロピーベース）
        def _e_to_float(e_str):
            m = re.search(r'(\d+)', str(e_str))
            return int(m.group(1)) / 100.0 if m else 0.5
        e1_val = _e_to_float(e1)
        e2_val = _e_to_float(e2)
        e3_val = _e_to_float(e3)
        e4_val = _e_to_float(e4) if e4 else 0.5
        pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
        # E値をstateに保存（次tickのentropy変調 + pressure信号に使う）
        state["last_e1"] = e1_val
        state["last_e2"] = e2_val
        state["last_e3"] = e3_val
        state["last_e4"] = e4_val
        # negentropy: E1-E4の4軸でentropy回復
        ent_before = state.get("entropy", 0.65)
        apply_negentropy(state, e1_val, e2_val, e3_val, e4_val)
        ent_after = state.get("entropy", 0.65)
        print(f"  entropy: {ent_after:.3f} (neg={ent_before - ent_after:.4f} E1={e1_val:.0%} E2={e2_val:.0%} E3={e3_val:.0%} E4={e4_val:.0%})")
        state["pressure"] = round(pressure, 2)
        save_state(state)
        print(f"  pressure reset: {pressure:.2f}")
        print()

if __name__ == "__main__":
    main()
