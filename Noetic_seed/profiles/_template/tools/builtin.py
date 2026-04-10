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
    target = (BASE_DIR / path).resolve()
    if not str(target).startswith(str(BASE_DIR.resolve())):
        return "エラー: このツールは特定のファイルにしか干渉できません"
    if not target.exists():
        # 近似マッチ: ファイル名だけで検索
        from pathlib import Path
        candidates = [p for p in BASE_DIR.rglob(Path(path).name) if not _is_hidden(p.name) and p.is_file()]
        if len(candidates) == 1:
            target = candidates[0]
            path = str(target.relative_to(BASE_DIR))
        else:
            hint = ""
            if candidates:
                hint = f" (候補: {', '.join(str(c.relative_to(BASE_DIR)) for c in candidates[:3])})"
            return f"エラー: {path} は存在しません{hint}"
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

def _view_image(args: dict) -> str:
    """画像ファイルを視覚入力として取り込み、その場でLLMに描写させて結果として返す。
    同期実行: view_imageを呼んだサイクル内で「見た結果」が得られるのでE値評価が機能する。
    プロファイルディレクトリ内のjpg/png/webpが対象。"""
    path = args.get("path", "").strip()
    if not path:
        return "エラー: path= が必要です"
    target = (BASE_DIR / path).resolve()
    if not str(target).startswith(str(BASE_DIR.resolve())):
        return "エラー: プロファイル外のファイルは対象外です"
    if not target.exists():
        return f"エラー: {path} が見つかりません"
    if target.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        return "エラー: JPG/PNG/WebP のみ対応"

    # 同期で LLM を呼んで画像を描写させる（結果がサイクルのログに残る）
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

    rel_path = str(target.relative_to(BASE_DIR)).replace("\\", "/")
    return f"画像で見えたもの ({rel_path}):\n{description}"


def _write_file(path: str, content: str) -> str:
    if not path:
        return "エラー: pathが空です"
    if not content:
        return "エラー: contentが空です"
    target = (BASE_DIR / path).resolve()
    sandbox_resolved = SANDBOX_DIR.resolve()
    if not str(target).startswith(str(sandbox_resolved)):
        return "エラー: sandbox/以下にのみ書き込めます"
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
    # user_messageの場合: カウンター減算するが、圧力は即ゼロにしない（余韻として残る）
    if target[0].get("type") == "user_message":
        uec = state.get("unresponded_external_count", 0)
        if uec > 0:
            state["unresponded_external_count"] = uec - 1
        # unresolved_externalはゼロにしない → tick loopで徐々に減衰する
    save_state(state)
    return f"[dismiss] {target[0].get('type','?')}: {target[0].get('content','')[:50]} を却下しました"
