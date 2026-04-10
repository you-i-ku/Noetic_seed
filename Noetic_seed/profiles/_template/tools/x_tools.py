"""X(Twitter)操作ツール"""
import os
import time
import json
import httpx
from core.config import BASE_DIR

X_SESSION_PATH = BASE_DIR / "x_session.json"


def _x_do_login() -> bool:
    """ChromeをCDP経由で起動してXにログインし、セッションを保存する。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[X] playwrightがインストールされていません。setup.batを再実行してください。")
        return False

    import subprocess, tempfile, shutil

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
        proc = subprocess.Popen([
            chrome_path,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={tmp_profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "https://x.com/login",
        ])

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
            ctx.storage_state(path=str(X_SESSION_PATH))
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
            ctx.storage_state(path=str(X_SESSION_PATH))
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
            ctx.storage_state(path=str(X_SESSION_PATH))
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
