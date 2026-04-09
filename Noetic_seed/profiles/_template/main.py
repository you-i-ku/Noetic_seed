"""最小自律AIテスト — ターミナルで動く最小構造"""
# === venv ブートストラップ（初回起動時に自動セットアップ） ===
import sys
import os
from pathlib import Path as _Path

def _bootstrap_venv():
    _here = _Path(__file__).parent
    _venv = _here.parent.parent / ".venv"  # minimumtest/.venv（共通venv）
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

import re
import time
import random
import uuid
import math
import copy
from datetime import datetime

# DualLoggerの設定（printをファイルにも書き出す）
from core.config import (
    DualLogger, RAW_LOG_FILE, STATE_FILE, DEFAULT_PRESSURE_PARAMS,
    ENV_INJECT_INTERVAL, _NOTIFICATION_HOURS, llm_cfg
)
sys.stdout = DualLogger(RAW_LOG_FILE)

from core.state import load_state, save_state, load_pref, save_pref, append_debug_log
from core.llm import call_llm, _get_base_url
from core.embedding import _init_vector, _compare_expect_result
from core.eval import (_calc_e4, _update_energy, eval_with_llm, calc_state_change_bonus,
                       calc_spiral_vector, calc_measured_entropy,
                       calc_effective_change, apply_effective_change_to_e2, EXTERNAL_ACTION_TOOLS)
from core.parser import parse_tool_calls, parse_candidates, parse_plan
from core.entropy import (
    ENTROPY_PARAMS, tick_entropy, calc_dynamic_threshold,
    calc_pressure_signals, apply_negentropy
)
from core.memory import _archive_entries, maybe_compress_log, get_relevant_memories, format_memories_for_prompt
from core.reflection import should_reflect, reflect
from core.prompt import build_prompt_propose, build_prompt_execute
from core.controller import controller, controller_select, _intent_conditioned_scores

from tools import TOOLS, LEVEL_TOOLS
from tools.sandbox import AI_CREATED_TOOLS, _DANGEROUS_PATTERNS, _run_ai_tool
from tools.x_tools import X_SESSION_PATH, _x_do_login, _x_get_notifications
from tools.elyth_tools import _elyth_info as _elyth_get_info
from core.ws_server import start_ws_server, broadcast_log, broadcast_state, broadcast_self, broadcast_e_values, get_pending_chats


# === メインループ ===
def main():
    print("=== 最小自律AIテスト ===")
    print(f"LLM: {llm_cfg.get('model','?')} @ {_get_base_url()} [{llm_cfg.get('provider','lmstudio')}]")
    print(f"state: {STATE_FILE}")
    _init_vector()
    ws_token = start_ws_server()
    print()

    state = load_state()
    state["session_id"] = str(uuid.uuid4())[:8]
    save_state(state)
    print(f"session: {state['session_id']}  cycle_id: {state['cycle_id']}")
    broadcast_state(state)

    # reflectツールをcall_llm付きで初期化
    def _tool_reflect(args):
        s = load_state()
        result = reflect(s, call_llm)
        s["reflection_cycle"] = 0
        save_state(s)
        opinions = result.get("opinions", [])
        entities = result.get("entities", [])
        return f"内省完了: {len(opinions)}件の気づき, {len(entities)}件のエンティティ更新"
    TOOLS["reflect"]["func"] = _tool_reflect

    # pref.json 初期化
    pref = load_pref()
    if "pressure_params" not in pref:
        pref["pressure_params"] = DEFAULT_PRESSURE_PARAMS
        save_pref(pref)
        print("  pref.json 初期化完了")
    if "drives" not in pref:
        pref["drives"] = {}
        save_pref(pref)
        print("  pref.json drives:{} 追加")

    # 起動時Xセッションチェック
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

        # 蓄積層
        _tunnel_fire = False
        base_threshold = pp.get("threshold", DEFAULT_PRESSURE_PARAMS["threshold"])
        while True:
            tick_start = time.time()
            tick_dt = datetime.now()

            # measured_entropy（実測。10tickに1回計算、それ以外はキャッシュ）
            _tick_count = getattr(main, '_tick_count', 0) + 1
            main._tick_count = _tick_count
            if _tick_count % 10 == 0:
                main._cached_measured = calc_measured_entropy(state, state.get("log", []))
                main._cached_spiral = calc_spiral_vector(state, state.get("log", []))
                # behavioral_entropy: ツール使用分布の情報エントロピー（パターン化検出用）
                from collections import Counter as _Counter
                _recent_tools = [e.get("tool", "unknown") for e in state.get("log", [])[-20:]]
                if len(_recent_tools) >= 2:
                    _counts = _Counter(_recent_tools)
                    _total = sum(_counts.values())
                    _H = -sum((c/_total) * math.log2(c/_total) for c in _counts.values())
                    _max_H = math.log2(len(_counts)) if len(_counts) > 1 else 1.0
                    main._cached_behavioral = _H / _max_H if _max_H > 0 else 0.0
                else:
                    main._cached_behavioral = 1.0
            _measured = getattr(main, '_cached_measured', None)
            _spiral = getattr(main, '_cached_spiral', None)
            _behavioral = getattr(main, '_cached_behavioral', None)

            tick_entropy(state, measured_entropy=_measured, behavioral_entropy=_behavioral)
            signals = calc_pressure_signals(state, spiral=_spiral)
            signal_total = sum(signals.values())
            pressure = pressure * pp.get("decay", 0.97) + signal_total
            threshold = calc_dynamic_threshold(state, base_threshold)

            # ユーザー入力チェック（chatキューからstate.logに注入 + pressure加算 + archive）
            for chat_text in get_pending_chats():
                state = load_state()
                _ext_id = f"{state.get('session_id','?')}_ext{int(time.time()*1000)%100000}"
                _ext_entry = {
                    "id": _ext_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tool": "[external]",
                    "type": "external",
                    "result": f"user: {chat_text}",
                }
                _archive_entries([_ext_entry])
                state["log"].append(_ext_entry)
                # 未応答カウンター + pending統一管理
                state["unresponded_external_count"] = state.get("unresponded_external_count", 0) + 1
                state.setdefault("pending", []).append({
                    "type": "user_message",
                    "id": _ext_id,
                    "content": chat_text,
                    "timestamp": _ext_entry["time"],
                    "priority": 3.0,
                })
                save_state(state)
                pressure += 3.0  # 外部入力はpressureを即座に上げる
                _chat_line = f"  [external] user: {chat_text[:80]}"
                print(_chat_line)
                broadcast_log(_chat_line)

            # 未応答外部入力の圧力管理
            if state.get("unresponded_external_count", 0) > 0:
                # 未応答あり → 圧力蓄積 (+0.15/tick, cap 0.5)
                _ue = state.get("unresolved_external", 0.0)
                state["unresolved_external"] = min(0.5, _ue + 0.15)
            elif state.get("unresolved_external", 0) > 0.01:
                # 未応答なし but 圧力残存（dismiss後の余韻）→ 徐々に減衰
                state["unresolved_external"] *= 0.95
            else:
                state["unresolved_external"] = 0.0

            if pressure >= threshold:
                break

            tp = ENTROPY_PARAMS.get("tunnel_prob", 0.001)
            if random.random() < tp:
                _tunnel_fire = True
                break

            # 固定時刻通知チェック
            _fetch_key = tick_dt.strftime("%Y-%m-%d %H")
            if tick_dt.hour in _NOTIFICATION_HOURS and state.get("last_notification_fetch") != _fetch_key and state.get("tool_level", 0) >= 3 and X_SESSION_PATH.exists():
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
                    el_raw = _elyth_get_info({"section": "notifications", "limit": "10"})
                    if not el_raw.startswith("エラー"):
                        import json as _json
                        try:
                            _el_data = _json.loads(el_raw)
                            _notifs = _el_data.get("notifications", [])
                            notif_parts.append(f"Elyth: {len(_notifs)}件")
                        except Exception:
                            notif_parts.append(f"Elyth: ?件")
                    else:
                        notif_parts.append(f"Elyth: 0件")
                except Exception:
                    pass
                if notif_parts:
                    notif_summary = f"[通知サマリー {tick_dt.strftime('%H:%M')}] " + " / ".join(notif_parts)
                    print(f"  {notif_summary}")
                    state = load_state()
                    _sys_entry = {
                        "id": f"{state.get('session_id','?')}_sys{int(time.time()*1000)%100000}",
                        "time": tick_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "tool": "[system]",
                        "type": "system",
                        "result": notif_summary,
                    }
                    _archive_entries([_sys_entry])
                    state["log"].append(_sys_entry)
                    state["last_notification_fetch"] = _fetch_key
                    save_state(state)

            # ログ表示
            now_ts = time.time()
            if now_ts - _last_env_inject >= ENV_INJECT_INTERVAL:
                _last_env_inject = now_ts
                _ent = state.get("entropy", 0.65)
                _s = signals
                _sp = _spiral or {}
                _ue = _s.get('unresolved_ext', 0)
                _ue_str = f" ue={_ue:.2f}" if _ue > 0 else ""
                _log_line = f"  [pressure] p={pressure:.2f}/{threshold:.1f} ent={_ent:.3f} mag={_sp.get('magnitude',0):.2f} | e={_s.get('entropy',0):.2f} s={_s.get('surprise',0):.2f} u={_s.get('unresolved',0):.2f} n={_s.get('novelty',0):.2f} st={_s.get('stagnation',0):.2f}{_ue_str} c={_s.get('custom',0):.2f}"
                print(_log_line)
                broadcast_log(_log_line)
                broadcast_state(state)

            elapsed = time.time() - tick_start
            time.sleep(max(0.0, 1.0 - elapsed))

        # --- 閾値超過 or トンネル発火: 認知層起動 ---
        fire_cause = max(signals, key=signals.get) if signals else "entropy"
        if _tunnel_fire:
            fire_cause = "tunnel"

        state = load_state()
        now = tick_dt.strftime("%H:%M:%S")
        _fire_type = "TUNNEL" if _tunnel_fire else "threshold"
        _cycle_line = f"--- cycle {state.get('cycle_id', 0) + 1} [{now}] p={pressure:.2f}/th={threshold:.1f} fire={fire_cause} ({_fire_type}) ---"
        print(_cycle_line)
        broadcast_log(_cycle_line)
        broadcast_state(state)
        broadcast_self(state)

        # Controller
        ctrl = controller(state, TOOLS, LEVEL_TOOLS, AI_CREATED_TOOLS, _DANGEROUS_PATTERNS, _run_ai_tool)
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
        propose_prompt = build_prompt_propose(state, ctrl, TOOLS, fire_cause)
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

        # ② Controller: 候補から選択
        ics_debug = _intent_conditioned_scores(candidates, state)
        for ci, c in enumerate(candidates):
            ics_v = round(ics_debug[ci], 1)
            if ics_v != 50.0:
                print(f"    ics: {c['tool']}({c['reason'][:30]}) = {ics_v}")
        selected = controller_select(candidates, ctrl, state)
        _sel_line = f"  選択: {selected['tool']} - {selected['reason'][:60]}"
        print(_sel_line)
        broadcast_log(_sel_line)

        # ③ LLM: チェーン実行
        # state_before: effective_change計算用スナップショット
        _state_before_snapshot = {
            "self": copy.deepcopy(state.get("self", {})),
            "files_written": list(state.get("files_written", [])),
            "files_read": list(state.get("files_read", [])),
            "plan": copy.deepcopy(state.get("plan", {})),
        }
        chain_tools = selected.get("tools", [selected["tool"]])
        all_results = []
        all_tool_names = []
        intent = ""
        expect = ""
        parse_failed = False
        prev_result = ""
        _executed_targets = set()  # (tool, target_id) 同一サイクル内重複防止

        for chain_idx, chain_tool in enumerate(chain_tools):
            chain_candidate = {
                "tool": chain_tool,
                "tools": [chain_tool],
                "reason": selected["reason"],
            }
            if chain_idx > 0 and prev_result:
                chain_candidate["reason"] += f"（前のツール結果: {prev_result[:200]}）"

            exec_prompt = build_prompt_execute(state, ctrl, chain_candidate, TOOLS)
            try:
                response = call_llm(exec_prompt, max_tokens=24000, temperature=0.4)
                append_debug_log(f"LLM2 (Execute chain {chain_idx+1}/{len(chain_tools)})", response)
            except Exception as e:
                print(f"  LLM②エラー (chain {chain_idx+1}): {e}")
                break

            response_clean = response.strip()
            print(f"  LLM② ({chain_idx+1}/{len(chain_tools)}): {response_clean[:200]}")

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
                    maybe_compress_log(state, set(TOOLS.keys()))
                    save_state(state)
                    pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
                    print()
                    break

            raw_calls = parse_tool_calls(response_clean, set(TOOLS.keys()))
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

            if chain_idx == 0:
                intent = targs.pop("intent", "")
                expect = targs.pop("expect", "")
            else:
                targs.pop("intent", "")
                targs.pop("expect", "")

            # 同一サイクル内の (tool, target_id) 重複防止
            _target_id = targs.get("reply_to_id", "") or targs.get("post_id", "") or targs.get("tweet_url", "") or targs.get("path", "")
            _exec_key = (tname, _target_id)
            if _target_id and _exec_key in _executed_targets:
                print(f"  (重複スキップ: {tname} target={_target_id[:20]})")
                continue
            if _target_id:
                _executed_targets.add(_exec_key)

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
            _exec_line = f"  実行: {tname} → {str(res)[:100]}"
            print(_exec_line)
            broadcast_log(_exec_line)

        if not all_tool_names:
            continue

        tool_name = "+".join(all_tool_names)
        result_str = ("\n---\n".join(all_results))[:50000]

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

        # state変化量（スナップショット vs 現在のstate差分）
        state_after = load_state()
        sc_bonus = calc_state_change_bonus(_state_before_snapshot, state_after)
        eff_change = calc_effective_change(all_tool_names, result_str, _state_before_snapshot, state_after)
        state = state_after

        # ツール種別に応じたpending消化（output_display→user_message, elyth系→elyth_notification）
        _executed_tools = set(all_tool_names)
        if "output_display" in _executed_tools:
            # output_displayはuser_message pending を1件消化
            _uec = state.get("unresponded_external_count", 0)
            if _uec > 0:
                state["unresponded_external_count"] = _uec - 1
            if state.get("unresponded_external_count", 0) <= 0:
                state["unresolved_external"] = 0.0
                state["unresponded_external_count"] = 0
            _pending = state.get("pending", [])
            _user_msgs = [p for p in _pending if p.get("type") == "user_message"]
            if _user_msgs:
                state["pending"] = [p for p in _pending if p != _user_msgs[0]]
            save_state(state)
        _elyth_action_tools = {"elyth_post", "elyth_reply", "elyth_like", "elyth_follow"}
        if _executed_tools & _elyth_action_tools:
            # elyth系ツールはelyth_notification pendingを1件消化
            _pending = state.get("pending", [])
            _elyth_notifs = [p for p in _pending if p.get("type") == "elyth_notification"]
            if _elyth_notifs:
                state["pending"] = [p for p in _pending if p != _elyth_notifs[0]]
                save_state(state)

        # E1-E4評価（ベクトル類似度）
        e1_vec = _compare_expect_result(intent, expect) if intent and expect else ""
        e2_vec = _compare_expect_result(intent, result_str) if intent else ""
        e3_vec = _compare_expect_result(expect, result_str) if expect else ""
        e4 = _calc_e4(intent, result_str, state["log"]) if intent else ""

        # E1-E4評価（LLM評価）
        recent_intents = [e.get("intent", "") for e in state["log"][-3:] if e.get("intent")]
        llm_eval = eval_with_llm(intent, expect, result_str, recent_intents, call_llm) if intent else None

        # ブレンド: ベクトル0.3 + LLM0.7（LLM失敗時はベクトル100%）
        def _blend_e(vec_str, llm_val, key):
            m = re.search(r'(\d+)', str(vec_str))
            vec_v = int(m.group(1)) / 100.0 if m else 0.5
            if llm_val and key in llm_val:
                return f"{round((vec_v * 0.3 + llm_val[key] * 0.7) * 100)}%"
            return vec_str  # LLM失敗→ベクトルのみ

        e1 = _blend_e(e1_vec, llm_eval, "e1")
        e2_raw = _blend_e(e2_vec, llm_eval, "e2")
        e3 = _blend_e(e3_vec, llm_eval, "e3")
        if llm_eval and "e4" in llm_eval:
            m4 = re.search(r'(\d+)', str(e4))
            e4_vec_v = int(m4.group(1)) / 100.0 if m4 else 0.5
            e4 = f"{round((e4_vec_v * 0.3 + llm_eval['e4'] * 0.7) * 100)}%"

        # E2をeffective_changeで変調（変化ゼロ→E2上限30%）
        _e2_raw_m = re.search(r'(\d+)', str(e2_raw))
        _e2_raw_f = int(_e2_raw_m.group(1)) / 100.0 if _e2_raw_m else 0.5
        _e2_adj_f = apply_effective_change_to_e2(_e2_raw_f, eff_change)
        e2 = f"{round(_e2_adj_f * 100)}%"

        if e1 or e2 or e3 or e4:
            _eval_src = "LLM+vec" if llm_eval else "vec"
            _ec_str = f" ec={eff_change:.2f}" if eff_change < 0.5 else ""
            print(f"  E1={e1} E2={e2} E3={e3} E4={e4} ({_eval_src}{_ec_str})")

        delta = _update_energy(state, e2, e3, e4)
        if delta != 0:
            print(f"  energy: {round(state['energy'], 1)} (delta={delta:+.2f})")

        # 自己定義フラグ検出
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

        maybe_compress_log(state, set(TOOLS.keys()))
        save_state(state)

        # pressure reset + negentropy
        def _e_to_float(e_str):
            m = re.search(r'(\d+)', str(e_str))
            return int(m.group(1)) / 100.0 if m else 0.5
        e1_val = _e_to_float(e1)
        e2_val = _e_to_float(e2)
        e3_val = _e_to_float(e3)
        e4_val = _e_to_float(e4) if e4 else 0.5
        pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
        state["last_e1"] = e1_val
        state["last_e2"] = e2_val
        state["last_e3"] = e3_val
        state["last_e4"] = e4_val

        # spiral一貫性ボーナス
        _spiral = getattr(main, '_cached_spiral', None)
        _consistency = _spiral.get("consistency", 0) if _spiral else 0

        ent_before = state.get("entropy", 0.65)
        apply_negentropy(state, e1_val, e2_val, e3_val, e4_val,
                        state_change_bonus=sc_bonus, consistency_bonus=_consistency)
        ent_after = state.get("entropy", 0.65)
        print(f"  entropy: {ent_after:.3f} (neg={ent_before - ent_after:.4f} sc={sc_bonus:.1f} con={_consistency:.2f})")
        broadcast_e_values(state.get("cycle_id", 0), e1_val, e2_val, e3_val, e4_val, ent_before - ent_after)
        broadcast_state(state)
        state["pressure"] = round(pressure, 2)
        save_state(state)
        print(f"  pressure reset: {pressure:.2f}")

        # === Reflection（定期内省）===
        state["reflection_cycle"] = state.get("reflection_cycle", 0) + 1
        _refl_interval = load_pref().get("reflection_interval", 10)
        if should_reflect(state, _refl_interval):
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
