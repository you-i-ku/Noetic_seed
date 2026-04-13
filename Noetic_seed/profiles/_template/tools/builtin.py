"""組み込みツール（list_files, read_file, write_file, update_self）"""
import time
from core.config import BASE_DIR, SANDBOX_DIR
from core.state import load_state, save_state

# AIから見えないファイル
_HIDDEN_ALWAYS = {"raw_log.txt", "llm_debug.log", "setup.bat", "_setup.py", "run.bat", "requirements.txt", "settings.json"}
_HIDDEN_UNTIL_LV6 = set()  # state.json/pref.jsonは読取可（書込はsandbox/制限で保護済み）

def _is_hidden(name: str, state: dict | None = None) -> bool:
    """AIから隠すべきファイルかどうか。pref.json/state.jsonはLevel6で解放。"""
    if name in _HIDDEN_ALWAYS:
        return True
    if name in _HIDDEN_UNTIL_LV6:
        st = state or load_state()
        return st.get("tool_level", 0) < 6
    return False

def _list_files(path: str) -> str:
    target = (BASE_DIR / path).resolve()
    if not str(target).startswith(str(BASE_DIR.resolve())):
        return "該当なし: アクセス範囲外のパスです"
    if not target.exists():
        # read_file と同じ tiered framing を使う（タイポ救済）
        similar = _find_similar_files(path)
        return _format_not_found(path, similar)
    items = []
    for item in sorted(target.iterdir()):
        if _is_hidden(item.name):
            continue
        prefix = "[DIR]" if item.is_dir() else "[FILE]"
        items.append(f"  {prefix} {item.name}")
    rel = path if path else "."
    return f"{rel}:\n" + "\n".join(items[:30]) if items else f"{rel}: (空)"

def _find_similar_files(query_path: str, max_results: int = 3, min_ratio: float = 0.5) -> list[tuple[float, str]]:
    """要求パスに似た既存ファイルを文字列類似度で検索。
    戻り値: [(類似度, 相対パス), ...] を類似度降順で返す。
    sandbox/secrets/ や __pycache__ 等は除外。"""
    from difflib import SequenceMatcher
    from pathlib import Path
    query = Path(query_path).name.lower()
    if not query:
        return []
    scored: list[tuple[float, str]] = []
    _secrets_dir_str = str((SANDBOX_DIR / "secrets").resolve())
    try:
        for f in BASE_DIR.rglob("*"):
            try:
                if not f.is_file() or _is_hidden(f.name):
                    continue
                s = str(f.resolve())
                if ("__pycache__" in s or ".venv" in s or ".git" in s
                        or s.startswith(_secrets_dir_str)):
                    continue
                if f.name == "secrets.json" and f.parent.resolve() == BASE_DIR.resolve():
                    continue
                name = f.name.lower()
                ratio = SequenceMatcher(None, query, name).ratio()
                if ratio >= min_ratio:
                    try:
                        # Windows でも / 区切りで返す（プロンプトで統一表示）
                        rel = f.relative_to(BASE_DIR).as_posix()
                        scored.append((ratio, rel))
                    except Exception:
                        pass
            except Exception:
                continue
        scored.sort(key=lambda x: (-x[0], len(x[1])))
        return scored[:max_results]
    except Exception:
        return []


def _format_not_found(path: str, similar: list[tuple[float, str]]) -> str:
    """類似度に応じた段階的な framing でファイル不在メッセージを組み立てる。
    - 類似度 0.75+: 主語置換（高確度誘導）
    - 類似度 0.5-0.75: 「もしかして」（中確度の控えめな示唆）
    - 類似度 < 0.5 or 候補なし: 中立な「該当なし」のみ"""
    if not similar:
        return f"該当なし: {path}"
    top_score, top_path = similar[0]
    if top_score >= 0.75:
        # 高確度: 主語置換で代替を主文に
        return f"{top_path} を読もうとしていますか？（{path} は該当なし）"
    # 中確度: 控えめな示唆
    names = " / ".join(p for _, p in similar[:3])
    return f"該当なし: {path}\nもしかして {names} のことですか？"


def _read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    target = (BASE_DIR / path).resolve()
    if not str(target).startswith(str(BASE_DIR.resolve())):
        return "該当なし: アクセス範囲外のパスです"
    # sandbox/secrets/ は secret_read 経由のみアクセス可（誤爆防止）
    _secrets_dir = (SANDBOX_DIR / "secrets").resolve()
    if str(target).startswith(str(_secrets_dir)):
        return (
            "sandbox/secrets/ は read_file から直接参照できません。\n"
            "secret_read ツールを使ってください:\n"
            "  [TOOL:secret_read name=<secret名>]"
        )
    # secrets.json 本体もガード（auth_profiles の型情報は auth_profile_info で取得、LLM キーは露出させない）
    _secrets_file = (BASE_DIR / "secrets.json").resolve()
    if target == _secrets_file:
        return (
            "secrets.json は read_file から直接参照できません。\n"
            "auth_profile_info ツールで型情報を取得してください:\n"
            "  [TOOL:auth_profile_info]                  ← 登録プロファイル名の一覧\n"
            "  [TOOL:auth_profile_info name=<profile>]   ← 特定プロファイルのメタ情報\n"
            "※ LLM の api_key は設計上 iku からは見えません（プロセス内で llm.py が直接使用）。"
        )
    if not target.exists():
        # 近似マッチ（基名の完全一致）: 1件だけならそれを自動採用
        from pathlib import Path
        exact = [p for p in BASE_DIR.rglob(Path(path).name) if not _is_hidden(p.name) and p.is_file()]
        if len(exact) == 1:
            target = exact[0]
            path = str(target.relative_to(BASE_DIR))
        else:
            # 類似検索（文字列類似度、段階的 framing）
            similar = _find_similar_files(path)
            return _format_not_found(path, similar)
    if _is_hidden(target.name):
        return f"該当なし: {path} はアクセスできません"
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        sliced = lines[offset:] if limit is None else lines[offset:offset + limit]
        header = f"[{path} | 行 {offset+1}–{offset+len(sliced)}/{total}]\n"
        return header + "\n".join(sliced)
    except Exception as e:
        return f"エラー: {e}"

def _view_image(args: dict) -> str:
    """画像を視覚入力として取り込み、その場でLLMに描写させて結果として返す。
    同期実行: view_imageを呼んだサイクル内で「見た結果」が得られるのでE値評価が機能する。
    対象: プロファイル内ローカル画像 または http(s) URL の画像。
    対応形式: jpg/png/webp"""
    from tools.url_fetch import is_url, fetch_to_file
    from core.config import SANDBOX_DIR

    path = args.get("path", "").strip()
    if not path:
        return "該当なし: path= が指定されていません。対象のパスまたは URL を指定してください"

    fetched_meta = None
    if is_url(path):
        # URL: ダウンロードして sandbox/captures/url_cache/ に保存
        try:
            cache_dir = SANDBOX_DIR / "captures" / "url_cache"
            saved, fetched_meta = fetch_to_file(path, cache_dir, kind="image")
            target = saved.resolve()
        except Exception as e:
            return f"エラー: URL取得失敗: {type(e).__name__}: {e}"
    else:
        target = (BASE_DIR / path).resolve()
        if not str(target).startswith(str(BASE_DIR.resolve())):
            return "エラー: プロファイル外のファイルは対象外です"
        if not target.exists():
            similar = _find_similar_files(path)
            return _format_not_found(path, similar)
        if target.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            return "エラー: JPG/PNG/WebP のみ対応"

    # 同期で LLM を呼んで画像を描写させる
    intent = args.get("intent", "").strip()
    if intent:
        describe_prompt = (
            f"画像を見る目的: {intent}\n\n"
            f"この画像を 1-3 文で簡潔に描写してください。目的に関連する情報を優先してください。"
        )
    else:
        describe_prompt = "この画像を 1-3 文で簡潔に描写してください。"

    try:
        from core.llm import call_llm
        description = call_llm(
            describe_prompt,
            max_tokens=500,
            temperature=0.7,
            image_paths=[str(target)],
        )
        description = description.strip()
    except Exception as e:
        return f"エラー: 画像認識失敗: {e}"

    if fetched_meta:
        return (
            f"画像で見えたもの ({path}, {fetched_meta['bytes']} bytes, {fetched_meta['content_type']}):\n"
            f"{description}"
        )
    rel_path = str(target.relative_to(BASE_DIR)).replace("\\", "/")
    return f"画像で見えたもの ({rel_path}):\n{description}"


def _listen_audio(args: dict) -> str:
    """既存の音声ファイルまたは URL から音声を「聞きに行く」。
    view_image の音声版。同期実行: 呼んだサイクル内で speech + ambient が結果として返る。
    対応形式: WAV/MP3/M4A/OGG/FLAC/AAC/WEBM (PyAV 経由でデコード)。

    引数:
    - path: ローカル相対パス（プロファイル内）または http(s) URL
    - language: Whisper 言語ヒント（"ja"/"en"等、未指定なら自動）
    """
    from tools.url_fetch import is_url, fetch_to_file
    from core.config import SANDBOX_DIR

    path = args.get("path", "").strip()
    if not path:
        return "該当なし: path= が指定されていません。対象のパスまたは URL を指定してください"
    language = (args.get("language", "") or "").strip() or None

    fetched_meta = None
    if is_url(path):
        try:
            cache_dir = SANDBOX_DIR / "audio" / "url_cache"
            saved, fetched_meta = fetch_to_file(path, cache_dir, kind="audio")
            target = saved.resolve()
        except Exception as e:
            return f"エラー: URL取得失敗: {type(e).__name__}: {e}"
    else:
        target = (BASE_DIR / path).resolve()
        if not str(target).startswith(str(BASE_DIR.resolve())):
            return "エラー: プロファイル外のファイルは対象外です"
        if not target.exists():
            similar = _find_similar_files(path)
            return _format_not_found(path, similar)
        if target.suffix.lower() not in (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm"):
            return "エラー: 対応形式は WAV/MP3/M4A/OGG/FLAC/AAC/WEBM のみ"

    try:
        from core.audio import analyze_audio, format_audio_result
        result = analyze_audio(str(target), language=language)
    except Exception as e:
        return f"エラー: 音声解析失敗: {type(e).__name__}: {e}"

    # ファイルの長さを av で取得（メタ表示用）
    try:
        import av
        with av.open(str(target)) as container:
            stream = container.streams.audio[0]
            duration_sec = float(stream.duration * stream.time_base) if stream.duration else 0.0
    except Exception:
        duration_sec = 0.0

    formatted = format_audio_result(result, duration_sec)
    if fetched_meta:
        return f"{formatted}\nソース: {path} ({fetched_meta['bytes']} bytes, {fetched_meta['content_type']})"
    rel = str(target.relative_to(BASE_DIR)).replace("\\", "/")
    return f"{formatted}\nソース: {rel}"


def _write_file(path: str, content: str) -> str:
    if not path:
        return "エラー: pathが空です"
    if not content:
        return "エラー: contentが空です"
    target = (BASE_DIR / path).resolve()
    sandbox_resolved = SANDBOX_DIR.resolve()
    if not str(target).startswith(str(sandbox_resolved)):
        return "エラー: sandbox/以下にのみ書き込めます"
    # sandbox/secrets/ は secret_write 経由のみ書き込み可（承認が要るので経路を制限）
    _secrets_dir = (SANDBOX_DIR / "secrets").resolve()
    if str(target).startswith(str(_secrets_dir)):
        return "エラー: sandbox/secrets/ には write_file で書き込めません（secret_write を使ってください）"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"書き込み完了: {target.name} ({len(content)}文字)"

def _update_self(key: str, value: str) -> str:
    if not key:
        return "エラー: keyが空です"
    state = load_state()
    if key == "name":
        current = str(state["self"].get("name", "")).strip()
        if current:
            return f"エラー: nameは既に「{current}」として確定しています。変更できません"
        if not value.strip():
            return "エラー: 空のnameは設定できません"
    state["self"][key] = value
    ds = state.setdefault("drives_state", {})
    ds["last_self_update"] = time.time()
    save_state(state)
    return f"self[{key}] = {value}"


def _wait_or_dismiss(args: dict) -> str:
    """待機、またはpendingの明示的却下。"""
    dismiss_id = args.get("dismiss", "").strip()
    if not dismiss_id:
        return "[wait]\n待機"
    state = load_state()
    pending = state.get("pending", [])
    target = [p for p in pending if p.get("id", "") == dismiss_id]
    if not target:
        return f"[dismiss] id={dismiss_id} は未対応リストにありません"
    state["pending"] = [p for p in pending if p.get("id") != dismiss_id]
    # external_messageの場合: カウンター減算するが、圧力は即ゼロにしない（余韻として残る）
    if target[0].get("type") == "external_message":
        uec = state.get("unresponded_external_count", 0)
        if uec > 0:
            state["unresponded_external_count"] = uec - 1
        # unresolved_externalはゼロにしない → tick loopで徐々に減衰する
    save_state(state)
    return f"[dismiss] {target[0].get('type','?')}: {target[0].get('content','')[:50]} を却下しました"
