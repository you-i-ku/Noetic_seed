"""Noetic_seed v2 — 情報的実存としての自律AI
感覚層（圧力蓄積）→ 認知層（LLM 1回）→ 帰結（プログラム評価）→ 内省（定期）
"""
# === venv ブートストラップ ===
import sys
import os
from pathlib import Path as _Path

def _bootstrap_venv():
    _here = _Path(__file__).parent
    _venv = _here.parent.parent / ".venv"
    _is_win = sys.platform == "win32"
    _venv_python = _venv / ("Scripts/python.exe" if _is_win else "bin/python")
    try:
        if _Path(sys.executable).resolve() == _venv_python.resolve():
            return
    except Exception:
        pass
    import subprocess
    if not _venv_python.exists():
        print("[bootstrap] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(_venv)], check=True)
        _pip = _venv / ("Scripts/pip.exe" if _is_win else "bin/pip")
        _req = _here.parent.parent / "requirements.txt"
        if _req.exists():
            subprocess.run([str(_pip), "install", "-q", "-r", str(_req)], check=True)
        print("[bootstrap] Done. Restarting in venv...\n")
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

_bootstrap_venv()
# ================================================================

import re
import time
import copy
import math
import random
import uuid
from datetime import datetime
from collections import Counter

from core.config import (
    DualLogger, RAW_LOG_FILE, STATE_FILE, DEFAULT_PRESSURE_PARAMS,
    ENV_INJECT_INTERVAL, _NOTIFICATION_HOURS, llm_cfg, LOG_HARD_LIMIT
)
sys.stdout = DualLogger(RAW_LOG_FILE)

from core.state import load_state, save_state, append_debug_log
from core.llm import call_llm, _use_fc
from core.embedding import _init_vector
from core.eval import evaluate_cycle, update_energy, EXTERNAL_ACTION_TOOLS
from core.entropy import (
    ENTROPY_PARAMS, tick_entropy, calc_dynamic_threshold,
    calc_pressure_signals, apply_negentropy
)
from core.memory import archive_action, prune_log, get_relevant_memories
from core.prompt import build_propose_prompt, build_execute_prompt
from core.controller import controller
from core.reflection import should_reflect, reflect

from tools import TOOLS, LEVEL_TOOLS
from tools.sandbox import AI_CREATED_TOOLS, _DANGEROUS_PATTERNS, _run_ai_tool
from tools.x_tools import X_SESSION_PATH, _x_do_login
from core.ws_server import (
    start_ws_server, broadcast_log, broadcast_state,
    broadcast_self, broadcast_e_values, get_pending_chats
)


def _parse_candidates(text: str, allowed_tools: set) -> list:
    """LLM①の提案テキストから候補を抽出。"""
    import re
    candidates = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if "->" in line or "→" in line:
            parts = re.split(r'->|→', line)
            tool_part = parts[-1].strip()
            reason_part = parts[0].strip()
        else:
            cleaned = re.sub(r'^[\d]+[.:)\s]+', '', line).strip()
            parts = cleaned.split()
            tool_part = parts[0] if parts else ""
            reason_part = cleaned

        tool_clean = re.sub(r'[^a-zA-Z0-9_+]', '', tool_part.split('+')[0].strip())
        if tool_clean in allowed_tools:
            reason = re.sub(r'^[\d]+[.:)\s]+', '', reason_part).strip()
            if reason.startswith('[') and reason.endswith(']'):
                reason = reason[1:-1].strip()
            if tool_clean not in [c["tool"] for c in candidates]:
                candidates.append({"tool": tool_clean, "tools": [tool_clean], "reason": reason})

    if not candidates:
        for t in sorted(allowed_tools):
            candidates.append({"tool": t, "tools": [t], "reason": "(フォールバック)"})
    return candidates


def _select_candidate(candidates: list, state: dict) -> dict:
    """候補からランダム加重選択。均等に近いがエネルギーで微調整。"""
    if len(candidates) <= 1:
        return candidates[0] if candidates else {"tool": "wait", "tools": ["wait"], "reason": ""}
    n = len(candidates)
    weights = [1.0 / n] * n
    # ランダム性を確保しつつ、微小な差をつける
    for i, c in enumerate(candidates):
        # output_displayは直近で使われてたら少し下げる（ツールクールダウン的）
        recent_tools = [e.get("tool", "") for e in state.get("log", [])[-5:]]
        recent_count = recent_tools.count(c["tool"])
        weights[i] *= max(0.1, 1.0 - recent_count * 0.25)  # 連続使用で確率低下
    total = sum(weights)
    weights = [w / total for w in weights]
    idx = random.choices(range(n), weights=weights, k=1)[0]
    return candidates[idx]


def main():
    print("=== Noetic_seed v2 ===")
    print(f"LLM: {llm_cfg.get('model', '?')} [{llm_cfg.get('provider', 'lmstudio')}]")
    print(f"FC: {'ON' if _use_fc() else 'OFF (text-marker)'}")
    print(f"state: {STATE_FILE}")
    _init_vector()
    ws_token = start_ws_server()
    print()

    state = load_state()
    state["session_id"] = str(uuid.uuid4())[:8]
    save_state(state)
    print(f"session: {state['session_id']}  cycle_id: {state['cycle_id']}")
    broadcast_state(state)

    # 起動時Xセッションチェック
    if state.get("tool_level", 0) >= 3 and not X_SESSION_PATH.exists():
        print("\n  [X] Level 3+: Xセッションなし。")
        try:
            if input("  Xにログインする？ [y/N]: ").strip().lower() == "y":
                _x_do_login()
        except EOFError:
            pass
        print()

    pressure = state.get("pressure", 0.0)
    pp = DEFAULT_PRESSURE_PARAMS
    reflection_interval = llm_cfg.get("reflection_interval", 10)
    tick_interval = llm_cfg.get("tick_interval_sec", 1.0)

    print(f"  entropy={state.get('entropy', 0.65):.3f} energy={state.get('energy', 50):.1f}")

    while True:
        _tunnel_fire = False
        base_threshold = pp.get("threshold", 12.0)
        _last_log_time = 0.0

        # === 感覚層（毎tick） ===
        while True:
            tick_start = time.time()

            # behavioral_entropy計算（10tickごと）
            _tc = getattr(main, '_tc', 0) + 1
            main._tc = _tc
            if _tc % 10 == 0:
                _recent_tools = [e.get("tool", "?") for e in state.get("log", [])[-20:]]
                if len(_recent_tools) >= 2:
                    counts = Counter(_recent_tools)
                    total = sum(counts.values())
                    H = -sum((c / total) * math.log2(c / total) for c in counts.values())
                    maxH = math.log2(len(counts)) if len(counts) > 1 else 1.0
                    main._beh_ent = H / maxH if maxH > 0 else 0.0
                else:
                    main._beh_ent = 1.0

            beh = getattr(main, '_beh_ent', None)
            pred_err = state.get("last_prediction_error", 0.0)
            coh_drop = max(0, 1.0 - state.get("last_coherence", 1.0))

            tick_entropy(state, behavioral_entropy=beh,
                         prediction_error=pred_err, coherence_drop=coh_drop)
            signals = calc_pressure_signals(state)
            pressure = pressure * pp.get("decay", 0.97) + sum(signals.values())
            threshold = calc_dynamic_threshold(state, base_threshold)

            # 外部入力チェック
            for chat_text in get_pending_chats():
                state = load_state()
                _ext_id = f"{state['session_id']}_ext{int(time.time()*1000)%100000}"
                _ext_entry = {
                    "id": _ext_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tool": "[external]",
                    "type": "external",
                    "result": f"user: {chat_text}",
                }
                archive_action(_ext_entry)
                state["log"].append(_ext_entry)
                # pending追加
                state.setdefault("pending", []).append({
                    "type": "user_message",
                    "id": _ext_id,
                    "content": chat_text,
                    "timestamp": _ext_entry["time"],
                    "priority": 3.0,
                })
                state["unresponded_external_count"] = state.get("unresponded_external_count", 0) + 1
                save_state(state)
                pressure += 3.0
                _line = f"  [external] user: {chat_text[:80]}"
                print(_line)
                broadcast_log(_line)

            # 未応答leak
            if state.get("unresponded_external_count", 0) > 0:
                state["unresolved_external"] = min(0.5, state.get("unresolved_external", 0.0) + 0.15)

            if pressure >= threshold:
                break
            if random.random() < ENTROPY_PARAMS.get("tunnel_prob", 0.001):
                _tunnel_fire = True
                break

            # ログ表示
            now_ts = time.time()
            if now_ts - _last_log_time >= ENV_INJECT_INTERVAL:
                _last_log_time = now_ts
                _s = signals
                _ent = state.get("entropy", 0.65)
                _log_line = f"  [pressure] p={pressure:.2f}/{threshold:.1f} ent={_ent:.3f} | e={_s.get('entropy',0):.2f} s={_s.get('surprise',0):.2f} pn={_s.get('pending',0):.2f} st={_s.get('stagnation',0):.2f} d={_s.get('drives',0):.2f}"
                print(_log_line)
                broadcast_log(_log_line)
                broadcast_state(state)

            elapsed = time.time() - tick_start
            time.sleep(max(0.0, tick_interval - elapsed))

        # === 認知層 ===
        fire_cause = max(signals, key=signals.get) if signals else "entropy"
        if _tunnel_fire:
            fire_cause = "tunnel"

        state = load_state()
        now = datetime.now().strftime("%H:%M:%S")
        _fire_type = "TUNNEL" if _tunnel_fire else "threshold"
        _cycle_line = f"--- cycle {state['cycle_id'] + 1} [{now}] p={pressure:.2f}/th={threshold:.1f} fire={fire_cause} ({_fire_type}) ---"
        print(_cycle_line)
        broadcast_log(_cycle_line)

        # Controller
        ctrl = controller(state, TOOLS, LEVEL_TOOLS, AI_CREATED_TOOLS, _DANGEROUS_PATTERNS, _run_ai_tool)
        allowed = ctrl["allowed_tools"]
        new_lv = ctrl["tool_level"]
        prev_lv = ctrl["tool_level_prev"]
        if new_lv != prev_lv:
            state["tool_level"] = new_lv
            added = sorted(LEVEL_TOOLS.get(new_lv, set()) - LEVEL_TOOLS.get(prev_lv, set()))
            print(f"  [system] tool_level {prev_lv}→{new_lv}: +{added}")
            save_state(state)
        print(f"  ctrl: level={new_lv} tools={len(allowed)}個")

        # state_before スナップショット
        state_before = {
            "self": copy.deepcopy(state.get("self", {})),
            "files_written": list(state.get("files_written", [])),
            "files_read": list(state.get("files_read", [])),
            "plan": copy.deepcopy(state.get("plan", {})),
            "pending": list(state.get("pending", [])),
        }

        # === LLM① 候補提案 ===
        propose_msgs = build_propose_prompt(state, ctrl, TOOLS, fire_cause)
        try:
            propose_resp = call_llm(propose_msgs, max_tokens=2000, temperature=1.0)
            propose_text = propose_resp.get("text", "")
            append_debug_log("LLM1 (Propose)", propose_text[:500])
        except Exception as e:
            print(f"  LLM①エラー: {e}")
            pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
            time.sleep(10)
            continue

        candidates = _parse_candidates(propose_text, allowed)
        print(f"  LLM①: {propose_text.strip()[:200]}")
        print(f"  候補({len(candidates)}件): {[(c['tool'], c['reason'][:30]) for c in candidates]}")

        # 候補選択（ランダム加重）
        selected = _select_candidate(candidates, state)
        print(f"  選択: {selected['tool']} - {selected['reason'][:60]}")
        broadcast_log(f"  選択: {selected['tool']} - {selected['reason'][:40]}")

        # === LLM② 実行 ===
        use_fc = _use_fc()
        exec_msgs, tool_schemas = build_execute_prompt(state, ctrl, selected, TOOLS, use_fc)
        try:
            response = call_llm(exec_msgs, tools=tool_schemas if use_fc else None,
                                tool_names=allowed, max_tokens=4096,
                                temperature=0.4)
            append_debug_log("LLM2 (Execute)", response.get("text", "")[:500])
        except Exception as e:
            print(f"  LLM②エラー: {e}")
            pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
            time.sleep(10)
            continue

        if response.get("text"):
            print(f"  LLM②: {response['text'][:200]}")

        # ツール実行
        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            print("  (ツール呼び出しなし)")
            pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
            continue

        tc = tool_calls[0]
        tool_name = tc.get("name", "wait")
        tool_args = tc.get("arguments", {})
        intent = tool_args.pop("intent", "")
        expect = tool_args.pop("expect", "")

        if tool_name not in TOOLS:
            print(f"  (未知のツール: {tool_name})")
            tool_name = "wait"
            tool_args = {}
        elif tool_name not in allowed:
            print(f"  (Controller却下: {tool_name})")
            tool_name = "wait"
            tool_args = {}

        try:
            result = TOOLS[tool_name]["func"](tool_args)
        except Exception as e:
            result = f"エラー: {e}"

        result_str = str(result)[:20000]
        _exec_line = f"  実行: {tool_name} → {result_str[:100]}"
        print(_exec_line)
        broadcast_log(_exec_line)

        # ファイル追跡
        state = load_state()
        if tool_name == "read_file" and not result_str.startswith("エラー"):
            path = tool_args.get("path", "")
            if path:
                fr = state.setdefault("files_read", [])
                if path not in fr:
                    fr.append(path)
        elif tool_name == "write_file" and not result_str.startswith("エラー"):
            path = tool_args.get("path", "")
            if path:
                fw = state.setdefault("files_written", [])
                if path not in fw:
                    fw.append(path)

        # 外界作用ツール → pending消化 + unresolved_external
        if tool_name in EXTERNAL_ACTION_TOOLS:
            uec = state.get("unresponded_external_count", 0)
            if uec > 0:
                state["unresponded_external_count"] = uec - 1
            if state.get("unresponded_external_count", 0) <= 0:
                state["unresolved_external"] = 0.0
                state["unresponded_external_count"] = 0
            # user_message系pendingを1つ消化
            pending = state.get("pending", [])
            user_msgs = [p for p in pending if p.get("type") == "user_message"]
            if user_msgs:
                state["pending"] = [p for p in pending if p != user_msgs[0]]

        if intent:
            print(f"  intent: {intent}")
        if expect:
            print(f"  expect: {expect}")

        # === 評価（完全プログラム計測）===
        state_after = load_state()
        eval_result = evaluate_cycle(state_before, state_after, [tool_name], result_str, intent, expect)
        state = state_after

        # negentropy適用
        ent_before = state.get("entropy", 0.65)
        apply_negentropy(state, eval_result)
        ent_after = state.get("entropy", 0.65)
        eval_result["negentropy"] = round(ent_before - ent_after, 4)

        # energy更新
        delta_e = update_energy(state, eval_result)

        # state更新
        state["last_e_values"] = eval_result
        state["last_prediction_error"] = 1.0 - eval_result.get("prediction", 0.5)
        state["last_coherence"] = eval_result.get("coherence", 0.5)

        _ev = eval_result
        print(f"  eval: ach={_ev['achievement']:.2f} pred={_ev['prediction']:.2f} div={_ev['diversity']:.2f} coh={_ev['coherence']:.2f}")
        print(f"  energy: {state['energy']:.1f} (delta={delta_e:+.2f})  entropy: {ent_after:.3f} (neg={_ev['negentropy']:.4f})")

        # ログ記録
        cid = state.get("cycle_id", 0) + 1
        state["cycle_id"] = cid
        entry = {
            "id": f"{state['session_id']}_{cid:04d}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "result": result_str[:5000],
            "intent": intent,
            "expect": expect,
            "eval": eval_result,
        }
        archive_action(entry)
        state["log"].append(entry)
        prune_log(state)
        save_state(state)

        # broadcast
        broadcast_e_values(cid, _ev.get("achievement", 0), _ev.get("prediction", 0),
                           _ev.get("diversity", 0), _ev.get("coherence", 0), _ev.get("negentropy", 0))
        broadcast_state(state)
        broadcast_self(state)

        # pressure reset
        pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
        state["pressure"] = round(pressure, 2)
        save_state(state)
        print(f"  pressure reset: {pressure:.2f}")

        # === 内省（定期）===
        state["reflection_cycle"] = state.get("reflection_cycle", 0) + 1
        if should_reflect(state, reflection_interval):
            print("  [reflection] 内省開始...")
            reflect(state, call_llm)
            state["reflection_cycle"] = 0
            save_state(state)

        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] 終了します。")
        sys.exit(0)
