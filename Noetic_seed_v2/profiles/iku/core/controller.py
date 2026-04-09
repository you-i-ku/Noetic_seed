"""Controller v2 — ツール段階解放 + 自己モデル参照（propose-selectなし）"""
import os
from core.config import BASE_DIR, SANDBOX_TOOLS_DIR
from core.state import load_state
from tools.x_tools import X_SESSION_PATH


def controller(state: dict, tools_dict: dict, level_tools: dict,
               ai_created_tools: dict, dangerous_patterns: list, run_ai_tool_fn) -> dict:
    """ツールレベル計算 + 許可ツール決定。"""

    # AI製ツールのスキャン（sandbox/tools/*.py）
    if SANDBOX_TOOLS_DIR.exists():
        for f in SANDBOX_TOOLS_DIR.glob("*.py"):
            name = f.stem
            if name.startswith("_") or name in tools_dict:
                continue
            try:
                code = f.read_text(encoding="utf-8")
                is_dangerous = any(p in code for p in dangerous_patterns)
                if is_dangerous:
                    continue
                ns = {}
                exec(code, ns)
                func = ns.get("run") or ns.get("main")
                if func:
                    tools_dict[name] = {
                        "desc": ns.get("DESCRIPTION", f"AI製ツール: {name}"),
                        "func": lambda args, _f=func: _f(args),
                    }
                    ai_created_tools[name] = str(f)
            except Exception:
                pass

    # ツールレベル計算
    fr = state.get("files_read", [])
    fw = state.get("files_written", [])
    tc = state.get("tools_created", [])
    prev_lv = state.get("tool_level", 0)

    if prev_lv < 1 and len(fr) >= 1:
        lv = 1
    elif prev_lv < 2 and len(fr) >= 2:
        lv = 2
    elif prev_lv < 3 and len(fr) >= 1 and len(fw) >= 1 and (len(fr) + len(fw)) >= 5:
        lv = 3
    elif prev_lv < 4 and any(f.endswith(".py") for f in fw):
        lv = 4
    elif prev_lv < 5 and len(tc) >= 1:
        lv = 5
    elif prev_lv < 6:
        lv = prev_lv  # Level 6は明示的なトリガーが必要
    else:
        lv = prev_lv

    lv = max(lv, prev_lv)  # レベルは下がらない

    # 許可ツール
    allowed = set(level_tools.get(lv, level_tools.get(0, set())))

    # AI製ツール追加
    for name in ai_created_tools:
        if name in tools_dict:
            allowed.add(name)

    # Xセッションなし→X系ツール除外
    _x_tools = {"x_timeline", "x_search", "x_get_notifications", "x_post", "x_reply", "x_quote", "x_like"}
    if not X_SESSION_PATH.exists():
        allowed -= _x_tools

    return {
        "allowed_tools": allowed,
        "tool_level": lv,
        "tool_level_prev": prev_lv,
    }
