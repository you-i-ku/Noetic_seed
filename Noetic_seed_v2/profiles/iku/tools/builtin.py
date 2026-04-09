"""組み込みツール（list_files, read_file, write_file, update_self）"""
import time
from core.config import BASE_DIR, SANDBOX_DIR
from core.state import load_state, save_state

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
    if key == "name":
        return "エラー: nameは変更できません"
    state = load_state()
    state["self"][key] = value
    ds = state.setdefault("drives_state", {})
    ds["last_self_update"] = time.time()
    save_state(state)
    return f"self[{key}] = {value}"
