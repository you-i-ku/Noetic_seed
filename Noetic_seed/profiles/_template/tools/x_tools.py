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

    # count に応じて動的スクロール（T-3）、上限4回
    collected = {}  # url -> item text （重複排除用）
    seen_ids = set()  # 処理済み article のスキップ用
    max_scrolls = min(4, max(1, (n // 3) + 1))
    for _ in range(max_scrolls):
        articles = page.locator('article[data-testid="tweet"]').all()
        for art in articles:
            if len(collected) >= n:
                break
            # 処理済みスキップ（DOM id やテキストで識別）
            try:
                art_id = art.locator('a:has(time)[href*="/status/"]').first.get_attribute("href", timeout=1000) or ""
            except Exception:
                art_id = ""
            if art_id in seen_ids:
                continue
            seen_ids.add(art_id)
            # tweet URL 抽出（T-1）
            tweet_url = ""
            if art_id:
                tweet_url = f"https://x.com{art_id}" if art_id.startswith("/") else art_id
            # ユーザー名・テキスト（短いタイムアウト）
            user = ""
            try:
                _un = art.locator('[data-testid="User-Name"]')
                if _un.count() > 0:
                    user = _un.first.inner_text(timeout=2000)
            except Exception:
                pass
            text = ""
            try:
                _tt = art.locator('[data-testid="tweetText"]')
                if _tt.count() > 0:
                    text = _tt.first.inner_text(timeout=2000)
            except Exception:
                pass
            if (user or text) and tweet_url not in collected:
                item = f"{user}: {text[:200]}"
                if tweet_url:
                    item += f"\nurl: {tweet_url}"
                collected[tweet_url or f"_no_url_{len(collected)}"] = item
        if len(collected) >= n:
            break
        page.evaluate("window.scrollBy(0, 600)")
        page.wait_for_timeout(1200)

    return list(collected.values())[:n]


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


def _grab_posted_url(page, text: str) -> str:
    """投稿直後にホームTLの最上位ツイートから URL を取得。テキスト一致確認付き。"""
    try:
        page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(1500)
        # フォロー中タブに切替（自分の投稿が確実にトップに出る）
        try:
            _ft = page.locator('div[role="tab"]', has_text="フォロー中")
            if _ft.count() == 0:
                _ft = page.locator('div[role="tab"]', has_text="Following")
            if _ft.count() > 0:
                _ft.first.click()
                page.wait_for_timeout(1500)
        except Exception:
            pass
        top = page.locator('article[data-testid="tweet"]').first
        top_text = ""
        try:
            top_text = top.locator('[data-testid="tweetText"]').first.inner_text()
        except Exception:
            pass
        if top_text and text[:20] in top_text:
            _tl = top.locator('a:has(time)[href*="/status/"]')
            if _tl.count() > 0:
                href = _tl.first.get_attribute("href", timeout=2000)
                if href:
                    return f"https://x.com{href}" if href.startswith("/") else href
    except Exception:
        pass
    return ""


def _resolve_x_feedback(parsed_notifications: list):
    """通知結果と pending_feedback を照合し、マッチしたら resolve + E2 遡及修正。"""
    import re as _re
    from core.state import load_state, save_state

    state = load_state()
    pf_list = state.get("pending_feedback", [])
    awaiting = [p for p in pf_list if p.get("status") == "awaiting" and p.get("tool", "").startswith("x_")]
    if not awaiting:
        return

    resolved_any = False
    for notif in parsed_notifications:
        ntype = notif.get("type", "")
        nfrom = notif.get("from", "").lower()
        ntext = notif.get("text", "").lower()

        for pf in awaiting:
            if pf["status"] != "awaiting":
                continue
            tool = pf.get("tool", "")
            snippet = pf.get("text_snippet", "")
            entity = pf.get("entity", "")
            matched = False

            if tool == "x_post" and ntype in ("like", "repost", "reply", "mention"):
                if snippet and len(snippet) >= 10:
                    matched = snippet[:20].lower() in ntext
                else:
                    matched = True

            elif tool in ("x_reply", "x_quote") and ntype in ("reply", "mention"):
                target = entity.split(":", 1)[1] if ":" in entity else ""
                if target:
                    matched = target.lower() in nfrom

            if matched:
                pf["status"] = "resolved"
                for le in state.get("log", []):
                    if le.get("id") == pf.get("log_entry_id"):
                        m = _re.search(r'(\d+)', str(le.get("e2", "")))
                        if m:
                            le["e2"] = f"{min(100, int(m.group(1)) + 40)}%"
                        break
                resolved_any = True
                break

    if resolved_any:
        save_state(state)


def _x_timeline(args):
    err = _x_session_check()
    if err:
        return err
    n = int(args.get("count", "") or "10")
    tab = args.get("tab", "following").strip().lower()
    page = context = pw = proc = tmp_profile = None
    try:
        page, context, browser, pw, proc, tmp_profile = _x_open_chrome(
            "https://x.com/home", 9360, "x_tl_")
        # デフォルトは「フォロー中」タブ（X のデフォルトは「おすすめ」）
        if tab != "recommend":
            try:
                # タブが描画されるまで待つ
                page.wait_for_selector('div[role="tab"]', timeout=8000)
                page.wait_for_timeout(500)
                following_tab = page.locator('div[role="tab"]', has_text="フォロー中")
                if following_tab.count() == 0:
                    following_tab = page.locator('div[role="tab"]', has_text="Following")
                if following_tab.count() > 0:
                    following_tab.first.click()
                    # タブ切替後のコンテンツ読み込みを待つ
                    page.wait_for_timeout(2000)
            except Exception:
                pass
        items = _x_get_tweets_from_page(page, n)
        if items:
            return "[X/Twitter タイムライン]\n" + "\n---\n".join(items)
        return "タイムライン取得失敗"
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
        if items:
            return f"[X/Twitter 検索: {query}]\n" + "\n---\n".join(items)
        return "結果なし"
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
            page.wait_for_selector('article, [data-testid="notification"]', timeout=10000)
        except Exception:
            pass
        page.wait_for_timeout(1500)

        # article 単位で構造化抽出（dict リストを作ってから文字列化）
        articles = page.locator('article').all()[:20]
        parsed = []
        for art in articles:
            user = ""
            try:
                _un = art.locator('[data-testid="User-Name"]')
                if _un.count() > 0:
                    user = _un.first.inner_text(timeout=2000)
            except Exception:
                pass
            body = ""
            try:
                body = art.inner_text(timeout=3000)[:300]
            except Exception:
                pass
            tweet_url = ""
            try:
                _tl = art.locator('a:has(time)[href*="/status/"]')
                if _tl.count() > 0:
                    href = _tl.first.get_attribute("href", timeout=2000)
                    if href:
                        tweet_url = f"https://x.com{href}" if href.startswith("/") else href
            except Exception:
                pass

            ntype = "other"
            body_lower = body.lower()
            if "liked" in body_lower or "いいね" in body:
                ntype = "like"
            elif "reposted" in body_lower or "リポスト" in body:
                ntype = "repost"
            elif "replied" in body_lower or "返信" in body:
                ntype = "reply"
            elif "followed" in body_lower or "フォロー" in body:
                ntype = "follow"
            elif "mentioned" in body_lower or "メンション" in body:
                ntype = "mention"

            parsed.append({
                "type": ntype, "from": user, "url": tweet_url,
                "text": " ".join(body.split())[:150],
            })

        if not parsed:
            try:
                cells = page.locator('[data-testid="notification"]').all_inner_texts()
            except Exception:
                cells = []
            if cells:
                return "[X/Twitter 通知]\n" + "\n---\n".join(c[:200] for c in cells[:20])
            return "通知なし"

        # 遅延フィードバック照合（pending_feedback の awaiting を resolve）
        try:
            _resolve_x_feedback(parsed)
        except Exception:
            pass

        # 文字列出力
        items = []
        for p in parsed:
            entry = f"type: {p['type']}"
            if p["from"]:
                entry += f"\nfrom: {p['from']}"
            if p["url"]:
                entry += f"\nurl: {p['url']}"
            entry += f"\ntext: {p['text']}"
            items.append(entry)
        return "[X/Twitter 通知]\n" + "\n---\n".join(items)
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
        posted_url = _grab_posted_url(page, text)
        result = f"投稿完了: {text[:80]}"
        if posted_url:
            result += f"\ntweet_url: {posted_url}"
        return result
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
        # 段階10 Step 4 付帯 D: Fix 5 精神で reply text truncation 撤去
        return f"返信完了: {text}\nreply_to: {tweet_url}"
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
        posted_url = _grab_posted_url(page, text)
        result = f"引用投稿完了: {text[:80]}\nquote_of: {tweet_url}"
        if posted_url:
            result += f"\ntweet_url: {posted_url}"
        return result
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
