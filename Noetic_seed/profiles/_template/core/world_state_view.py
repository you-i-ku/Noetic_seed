"""world_state_view.py — V07 Phase 1 commit 1 (Homologous Structure: world ring)

<world_state> XML block を構築する helper。file_snapshot + recent_events を thin format で
auto inject する。P-Y (Homologous Structure) 整合、subjective_state と相同並置。

V07_PROMPT_PARADIGM_SHIFT_PLAN §3-2-2 / §4 Phase 1 commit 1 literal。

設計指針:
- pure helper (Codex audit 2026-05-14 P2-02 fix): _recent_events は raw_events list を
  引数で直接受ける (load_state 経由しない、fixture 注入可能、test 決定論)
- file_snapshot: state.json + raw_events.jsonl の size+mtime + orphan 検出
  (99 cycle smoke 真因 cover、V7-A3 確定 = 案 2)
- recent_events: 最新 5 件 (V7-A4 確定)、tool/args/result 3 field のみ (世界視点)
- bootstrap engine: 空 list でも explicit empty literal を表示 (`feedback_intent_over_form_homologous_structure` §How to apply、Codex audit P3-03 fix)
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.config import MEMORY_DIR, STATE_FILE


# profile root = state.json の親 dir (core.config の BASE_DIR と同義)
PROFILE_ROOT: Path = STATE_FILE.parent
STATE_PATH: Path = STATE_FILE
RAW_EVENTS_PATH: Path = MEMORY_DIR / "raw_events.jsonl"


def _to_iso(epoch: float) -> str:
    """epoch float → ISO 8601 UTC 文字列 (Z 終端)."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_meta(path: Path) -> dict:
    """size (bytes) + mtime (ISO 8601 UTC)。file 不在なら size=0 / mtime=None."""
    if not path.exists():
        return {"size": 0, "mtime": None}
    st = path.stat()
    return {"size": st.st_size, "mtime": _to_iso(st.st_mtime)}


def _detect_orphan_state_json(profile_root: Optional[Path] = None,
                               canonical: Optional[Path] = None) -> list:
    """profile root 以下を rglob で再帰探索、canonical 以外の state.json を全列挙。

    99 cycle smoke 真因 cover: iku が write_file の相対 path で誤生成した
    profile/state.json (orphan) を毎 cycle prompt に出して気づかせる。

    Args:
        profile_root: 探索開始 dir (None で module default PROFILE_ROOT)
        canonical: 正規 state.json path (None で module default STATE_PATH)

    Returns:
        list of str (profile_root 相対 path、canonical 自身は除外)
    """
    root = profile_root if profile_root is not None else PROFILE_ROOT
    canon = (canonical if canonical is not None else STATE_PATH).resolve()
    orphans: list = []
    try:
        for path in root.rglob("state.json"):
            try:
                if path.resolve() != canon:
                    orphans.append(str(path.relative_to(root)))
            except (ValueError, OSError):
                continue
    except OSError:
        pass
    return orphans


def _file_snapshot() -> dict:
    """state.json + raw_events.jsonl size/mtime + orphan 検出 dict を返す。

    V7-A3 確定 (案 2): 自己永続 (state.json) + 行動永続 (raw_events.jsonl) の
    二輪表現、P-Y (Homologous Structure) の世界輪側。
    """
    return {
        "state_json": _file_meta(STATE_PATH),
        "raw_events_jsonl": _file_meta(RAW_EVENTS_PATH),
        "orphan": _detect_orphan_state_json(),
    }


def _summarize_event(entry: dict, result_cap: int = 200) -> dict:
    """raw event entry を tool/args/result 3 field 形式に圧縮。

    V7-A4 確定 + 5 論点 ① 確定: world_state は世界視点、intent / e1-e4 評価は
    含まない (subjective layer の recent_history に集約)。

    result_cap: result string 上限 (default 200)、超えたら "...[N字]" marker。
    """
    result_raw = entry.get("result", "")
    result_str = str(result_raw)
    if len(result_str) > result_cap:
        result_str = result_str[:result_cap] + f"...[{len(result_str)}字]"
    return {
        "tool": entry.get("tool", ""),
        "args": entry.get("args"),
        "result": result_str,
    }


def _recent_events(raw_events: list, limit: int = 5,
                   result_cap: int = 200) -> list:
    """raw_events list の最新 N 件を tool/args/result 3 field 形式で返す。

    pure helper (Codex audit 2026-05-14 P2-02 fix): state を load_state 経由で取らず、
    raw_events list を引数で直接受ける。fixture 注入可能、test 決定論。

    Args:
        raw_events: raw_events.jsonl 由来 list (古い→新しい順を仮定、tail が最新)
        limit: 返す件数 (default 5、V7-A4 確定)
        result_cap: result string 上限 (default 200 字)

    Returns:
        list of {tool, args, result} dict (時系列降順、新しい→古い、LLM が直近を最初に見やすい)
    """
    if not raw_events:
        return []
    selected = raw_events[-limit:]
    return [_summarize_event(e, result_cap) for e in reversed(selected)]


def _format_args_repr(args, cap: int = 200) -> str:
    """args dict を JSON 文字列化して cap 字に切る (truncation marker 付与)。"""
    if args is None:
        return "(none)"
    try:
        rendered = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(args)
    if len(rendered) > cap:
        rendered = rendered[:cap] + "…"
    return rendered


def build_world_state(state: dict) -> str:
    """<world_state> XML block を構築する (V07 Phase 1 commit 1 entry point).

    P-Y (Homologous Structure) 整合: subjective_state と対称配置、auto inject、
    thin view、同等 affordance (world_fact_view tool は維持)。

    Bootstrap engine: 空 list でも "(no recent events)" 等の explicit literal を
    表示し、空白こそ affordance を駆動する (Codex audit P3-03 fix)。

    Args:
        state: Noetic state dict (`raw_events` key を含む、空 list 許容)

    Returns:
        "<world_state>\n  ...\n</world_state>" XML block 文字列
    """
    snapshot = _file_snapshot()
    raw_events = state.get("raw_events", []) or []
    events = _recent_events(raw_events, limit=5)

    sj = snapshot["state_json"]
    rj = snapshot["raw_events_jsonl"]
    snap_lines = [
        "  file_snapshot:",
        f"    state.json:       size={sj['size']}B, mtime={sj['mtime'] or '(missing)'}",
        f"    raw_events.jsonl: size={rj['size']}B, mtime={rj['mtime'] or '(missing)'}",
    ]
    if snapshot["orphan"]:
        orphan_repr = ", ".join(snapshot["orphan"])
        snap_lines.append(f"    orphan:           [{orphan_repr}]")
    else:
        snap_lines.append("    orphan:           (none detected)")

    event_lines = ["  recent_events:"]
    if events:
        for e in events:
            args_repr = _format_args_repr(e["args"])
            event_lines.append(
                f"    [{e['tool']}] args={args_repr} → {e['result']}"
            )
    else:
        event_lines.append("    (no recent events)")

    body = "\n".join(snap_lines + event_lines)
    return f"<world_state>\n{body}\n</world_state>"
