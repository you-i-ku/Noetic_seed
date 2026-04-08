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
from core.eval import _calc_e4, _update_energy
from core.parser import parse_tool_calls, parse_candidates, parse_plan
from core.entropy import (
    ENTROPY_PARAMS, tick_entropy, calc_dynamic_threshold,
    calc_pressure_signals, apply_negentropy
)
from core.memory import _archive_entries, maybe_compress_log
from core.prompt import build_prompt_propose, build_prompt_execute
from core.controller import controller, controller_select, _intent_conditioned_scores

from tools import TOOLS, LEVEL_TOOLS
from tools.sandbox import AI_CREATED_TOOLS, _DANGEROUS_PATTERNS, _run_ai_tool
from tools.x_tools import X_SESSION_PATH, _x_do_login, _x_get_notifications
from tools.elyth_tools import _elyth_notifications


# === メインループ ===
def main():
    print("=== 最小自律AIテスト ===")
    print(f"LLM: {llm_cfg.get('model','?')} @ {_get_base_url()} [{llm_cfg.get('provider','lmstudio')}]")
    print(f"state: {STATE_FILE}")
    _init_vector()
    print()

    state = load_state()
    state["session_id"] = str(uuid.uuid4())[:8]
    save_state(state)
    print(f"session: {state['session_id']}  cycle_id: {state['cycle_id']}")

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

            tick_entropy(state)
            signals = calc_pressure_signals(state)
            signal_total = sum(signals.values())
            pressure = pressure * pp.get("decay", 0.97) + signal_total
            threshold = calc_dynamic_threshold(state, base_threshold)

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
                    el_raw = _elyth_notifications({"limit": "50"})
                    if not el_raw.startswith("エラー") and el_raw != "通知なし":
                        el_count = len([l for l in el_raw.split("---") if l.strip()])
                        notif_parts.append(f"Elyth: {el_count}件")
                    else:
                        notif_parts.append(f"Elyth: 0件")
                except Exception:
                    pass
                if notif_parts:
                    notif_summary = f"[通知サマリー {tick_dt.strftime('%H:%M')}] " + " / ".join(notif_parts)
                    print(f"  {notif_summary}")
                    state = load_state()
                    state["log"].append({
                        "id": f"{state.get('session_id','?')}_{state.get('cycle_id',0):04d}",
                        "time": tick_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "tool": "[system]",
                        "type": "system",
                        "result": notif_summary,
                    })
                    state["last_notification_fetch"] = _fetch_key
                    save_state(state)

            # ログ表示
            now_ts = time.time()
            if now_ts - _last_env_inject >= ENV_INJECT_INTERVAL:
                _last_env_inject = now_ts
                _ent = state.get("entropy", 0.65)
                _s = signals
                print(f"  [pressure] p={pressure:.2f}/{threshold:.1f} ent={_ent:.3f} | e={_s.get('entropy',0):.2f} s={_s.get('surprise',0):.2f} u={_s.get('unresolved',0):.2f} n={_s.get('novelty',0):.2f} c={_s.get('custom',0):.2f}")

            elapsed = time.time() - tick_start
            time.sleep(max(0.0, 1.0 - elapsed))

        # --- 閾値超過 or トンネル発火: 認知層起動 ---
        fire_cause = max(signals, key=signals.get) if signals else "entropy"
        if _tunnel_fire:
            fire_cause = "tunnel"

        state = load_state()
        now = tick_dt.strftime("%H:%M:%S")
        _fire_type = "TUNNEL" if _tunnel_fire else "threshold"
        print(f"--- cycle {state.get('cycle_id', 0) + 1} [{now}] p={pressure:.2f}/th={threshold:.1f} fire={fire_cause} ({_fire_type}) ---")

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
        print(f"  選択: {selected['tool']} - {selected['reason'][:60]}")

        # ③ LLM: チェーン実行
        chain_tools = selected.get("tools", [selected["tool"]])
        all_results = []
        all_tool_names = []
        intent = ""
        expect = ""
        parse_failed = False
        prev_result = ""

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
            print(f"  実行: {tname} → {str(res)[:100]}")

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

        # E1-E4評価
        e1 = _compare_expect_result(intent, expect) if intent and expect else ""
        e2 = _compare_expect_result(intent, result_str) if intent else ""
        e3 = _compare_expect_result(expect, result_str) if expect else ""
        e4 = _calc_e4(intent, state["log"]) if intent else ""
        if e1 or e2 or e3 or e4:
            print(f"  E1={e1} E2={e2} E3={e3} E4={e4}")

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
        ent_before = state.get("entropy", 0.65)
        apply_negentropy(state, e1_val, e2_val, e3_val, e4_val)
        ent_after = state.get("entropy", 0.65)
        print(f"  entropy: {ent_after:.3f} (neg={ent_before - ent_after:.4f} E1={e1_val:.0%} E2={e2_val:.0%} E3={e3_val:.0%} E4={e4_val:.0%})")
        state["pressure"] = round(pressure, 2)
        save_state(state)
        print(f"  pressure reset: {pressure:.2f}")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] 終了します。")
        sys.exit(0)
