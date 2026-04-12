"""X(Twitter)操作ツール"""
import os
import time
import json
import httpx
from core.config import BASE_DIR

X_SESSION_PATH = BASE_DIR / "x_session.json"


def _find_chrome() -> str | None:
    import shutil
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        shutil.which("chrome") or "",
        shutil.which("google-chrome") or "",
    ]
    return next((c for c in candidates if c and os.path.exists(c)), None)


def _wait_for_cdp(port: int, timeout_s: int = 10):
    for _ in range(timeout_s * 2):
        time.sleep(0.5)
        try:
            r = httpx.get(f"http://localhost:{port}/json/version", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


def _x_do_login() -> bool:
    """ChromeをCDP経由で起動してXにログインし、セッションを保存する。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[X] playwrightがインストールされていません。setup.batを再実行してください。")
        return False

    import subprocess, tempfile, shutil

    chrome_path = _find_chrome()
    if not chrome_path:
        print("[X] Chromeが見つかりません。Google Chromeをインストールしてください。")
        return False

    CDP_PORT = 9355
    tmp_profile = tempfile.mkdtemp(prefix="x_login_profile_")
    proc = None
    print("[X] ブラウザを起動します。ログインしてホーム画面が表示されるまでお待ちください...")
    try:
        proc = subprocess.Popen([
            chrome_path,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={tmp_profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "https://x.com/login",
        ])

        if not _wait_for_cdp(CDP_PORT):
            print("[X] Chrome CDP接続タイムアウト")
            return False

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
            context = browser.contexts[0] if browser.contexts else browser.new_context()
            page = context.pages[0] if context.pages else context.new_page()
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
    """X操作は自動承認（Human-in-the-loopはexec_code/self_modifyのみ）"""
    print(f"  [X {action}] {preview[:80]}")
    return True


def _x_open_chrome(url: str, cdp_port: int, prefix: str):
    """リアルChromeをCDP経由で起動し、セッションcookieを注入して返す。"""
    import subprocess, tempfile, shutil
    chrome_path = _find_chrome()
    if not chrome_path:
        raise RuntimeError("Chromeが見つかりません。")
    tmp_profile = tempfile.mkdtemp(prefix=prefix)
    session_data = json.load(open(X_SESSION_PATH, encoding="utf-8"))
    cookies = session_data.get("cookies", [])
    proc = subprocess.Popen([
        chrome_path, f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={tmp_profile}", "--no-first-run",
        "--no-default-browser-check", "about:blank",
    ])
    if not _wait_for_cdp(cdp_port):
        proc.terminate()
        shutil.rmtree(tmp_profile, ignore_errors=True)
        raise RuntimeError("Chrome CDP接続タイムアウト")
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    context.add_cookies(cookies)
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    if "login" in page.url:
        pw.stop()
        proc.terminate()
        shutil.rmtree(tmp_profile, ignore_errors=True)
        raise RuntimeError("Xセッション切れ。AIループを停止してから手動でログインしてください。")
    return page, context, browser, pw, proc, tmp_profile


def _x_close(context, pw, proc, tmp_profile):
    """Chrome CDP セッションを閉じてクリーンアップする。"""
    import shutil
    try:
        context.storage_state(path=str(X_SESSION_PATH))
    except Exception:
        pass
    try:
        pw.stop()
    except Exception:
        pass
    if proc:
        proc.terminate()
    shutil.rmtree(tmp_profile, ignore_errors=True)


def _x_timeline(args):
    err = _x_session_check()
    if err:
        return err
    n = int(args.get("count", "") or "10")
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            "https://x.com/home", 9360, "x_tl_")
        items = _x_get_tweets_from_page(page, n)
        return "\n---\n".join(items) if items else "タイムライン取得失敗"
    except RuntimeError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: {e}"
    finally:
        if context:
            _x_close(context, pw, proc, tmp_profile)


def _x_search(args):
    query = args.get("query", "")
    if not query:
        return "エラー: queryを指定してください"
    err = _x_session_check()
    if err:
        return err
    n = int(args.get("count", "") or "10")
    import urllib.parse
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            f"https://x.com/search?q={urllib.parse.quote(query)}&f=live", 9361, "x_search_")
        items = _x_get_tweets_from_page(page, n)
        return "\n---\n".join(items) if items else "結果なし"
    except RuntimeError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: {e}"
    finally:
        if context:
            _x_close(context, pw, proc, tmp_profile)


def _x_get_notifications(args):
    err = _x_session_check()
    if err:
        return err
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            "https://x.com/notifications", 9362, "x_notif_")
        try:
            page.wait_for_selector('article', timeout=10000)
        except Exception:
            pass
        try:
            cells = page.locator('[data-testid="notification"]').all_inner_texts()
        except Exception:
            cells = []
        if cells:
            return "\n---\n".join(c[:200] for c in cells[:20])
        return "通知なし"
    except RuntimeError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: {e}"
    finally:
        if context:
            _x_close(context, pw, proc, tmp_profile)


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
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            "https://x.com/home", 9356, "x_post_")
        page.wait_for_timeout(2000)
        page.goto("https://x.com/compose/post", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2000)
        textarea = page.locator('[data-testid="tweetTextarea_0"]').first
        textarea.wait_for(timeout=25000)
        textarea.click()
        page.keyboard.type(text, delay=50)
        page.get_by_role("button", name="ポストする").click()
        page.wait_for_timeout(3000)
        return f"投稿完了: {text[:80]}"
    except RuntimeError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: {e}"
    finally:
        if context:
            _x_close(context, pw, proc, tmp_profile)


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
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            tweet_url, 9357, "x_reply_")
        page.wait_for_timeout(2000)
        page.locator('[data-testid="reply"]').first.click()
        page.wait_for_timeout(1000)
        textarea = page.locator('[data-testid="tweetTextarea_0"]').first
        textarea.wait_for(timeout=15000)
        textarea.click()
        page.keyboard.type(text, delay=50)
        page.get_by_role("button", name="ポストする").click()
        page.wait_for_timeout(2000)
        return f"返信完了: {text[:80]}"
    except RuntimeError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: {e}"
    finally:
        if context:
            _x_close(context, pw, proc, tmp_profile)


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
    import urllib.parse
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            f"https://x.com/intent/tweet?url={urllib.parse.quote(tweet_url)}", 9358, "x_quote_")
        page.wait_for_timeout(2000)
        textarea = page.locator('[data-testid="tweetTextarea_0"]').first
        textarea.wait_for(timeout=25000)
        textarea.click()
        page.keyboard.type(text, delay=50)
        page.get_by_role("button", name="ポストする").click()
        page.wait_for_timeout(2000)
        return f"引用投稿完了: {text[:80]}"
    except RuntimeError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: {e}"
    finally:
        if context:
            _x_close(context, pw, proc, tmp_profile)


def _x_like(args):
    tweet_url = args.get("tweet_url", "")
    if not tweet_url:
        return "エラー: tweet_urlを指定してください"
    err = _x_session_check()
    if err:
        return err
    if not _x_confirm("いいね", f"対象: {tweet_url}"):
        return "キャンセルしました。"
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            tweet_url, 9359, "x_like_")
        page.wait_for_timeout(2000)
        page.locator('[data-testid="like"]').first.click()
        page.wait_for_timeout(1000)
        return f"いいね完了: {tweet_url}"
    except RuntimeError as e:
        return f"エラー: {e}"
    except Exception as e:
        return f"エラー: {e}"
    finally:
        if context:
            _x_close(context, pw, proc, tmp_profile)
