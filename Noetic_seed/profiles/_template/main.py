"""Noetic_seed"""
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
            "faster-whisper", "soundfile",  # mic_record 用
        ]
        print(f"[bootstrap] 依存ライブラリをインストール中: {', '.join(_deps)}")
        subprocess.run([str(_pip), "install", "--quiet"] + _deps, check=True)
        print("[bootstrap] セットアップ完了。venvで再起動します...\n")

    # venv の Python で自分自身を再実行
    # os.execv は Windows でスペース含むパスをクォートしないため subprocess で代替
    result = subprocess.run([str(_venv_python)] + sys.argv)
    sys.exit(result.returncode)

_bootstrap_venv()
# ================================================================

# Windows: Ctrl+C 即時終了（httpx等のブロッキング呼び出しでも確実に死なせる）
import signal
def _force_exit_on_sigint(_signum, _frame):
    print("\n[Ctrl+C] 強制終了します。", flush=True)
    os._exit(0)
signal.signal(signal.SIGINT, _force_exit_on_sigint)

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
    ENV_INJECT_INTERVAL, _NOTIFICATION_HOURS, llm_cfg, BASE_DIR, prompt_budget
)
sys.stdout = DualLogger(RAW_LOG_FILE)

from core.state import load_state, save_state, load_pref, save_pref, append_debug_log
from core.llm import call_llm, _get_active_provider_config
from core.embedding import _init_vector, _compare_expect_result
from core.eval import (_calc_e4, _update_energy, eval_with_llm, calc_state_change_bonus,
                       calc_spiral_vector, calc_measured_entropy,
                       calc_effective_change, apply_effective_change_to_e2, EXTERNAL_ACTION_TOOLS,
                       update_unresolved_intents, update_gaps_by_relevance,
                       _extract_action_key, append_action_ledger)
from core.pending_unified import pending_prune

# Phase 4 Step E-2d: ConversationRuntime 統合用 import
from core.providers.openai_compat import OpenAIProvider
from core.providers.anthropic import AnthropicProvider
from core.runtime.registry import ToolRegistry
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import (
    HookRunner,
    make_bash_validation_hook,
    make_file_access_guard,
    make_pre_tool_use_approval_check,
    make_post_tool_use_evaluation,
    make_post_tool_use_failure_logger,
)
from core.runtime.legacy_bridge import register_legacy_bridge
from core.runtime.tools import ensure_approval_props, ensure_noetic_bash_hint, ensure_noetic_file_hints, register_all as register_claw_tools
from core.runtime.tools.noetic_ext import NOETIC_TOOL_NAMES, register_noetic_tools
from core.runtime.permissions import PermissionEnforcer, PermissionMode
from core.approval_callback import make_approval_callback
from core.prompt_assembly import assemble_system_prompt
from core.parser import parse_tool_calls, parse_candidates
from core.entropy import (
    ENTROPY_PARAMS, tick_entropy, calc_dynamic_threshold,
    calc_pressure_signals, apply_negentropy
)
from core.memory import _archive_entries, maybe_compress_log, get_relevant_memories, format_memories_for_prompt
from core.reflection import should_reflect, reflect
from core.prompt import build_prompt_propose
from core.controller import controller, controller_select, _intent_conditioned_scores

from tools import TOOLS, LEVEL_TOOLS
from tools.sandbox import AI_CREATED_TOOLS, _DANGEROUS_PATTERNS, _run_ai_tool
from tools.x_tools import X_SESSION_PATH, _x_do_login, _x_get_notifications
from tools.elyth_tools import _elyth_info as _elyth_get_info
from core.ws_server import start_ws_server, broadcast_log, broadcast_state, broadcast_self, broadcast_e_values, get_pending_chats, is_paused, set_profile_running


# === チャネルマッピング（ログエントリに channel タグを付与）===
_CHANNEL_MAP = {
    "elyth_post": "elyth", "elyth_reply": "elyth", "elyth_like": "elyth",
    "elyth_follow": "elyth", "elyth_info": "elyth", "elyth_get": "elyth",
    "elyth_mark_read": "elyth",
    "x_post": "x", "x_reply": "x", "x_quote": "x", "x_like": "x",
    "x_timeline": "x", "x_search": "x", "x_get_notifications": "x",
    "output_display": "display",
    "camera_stream": "device", "camera_stream_stop": "device",
    "screen_peek": "device", "mic_record": "device",
    "view_image": "device", "listen_audio": "device",
}

def _get_channel(tool_name: str) -> str:
    """ツール名からチャネルを判定。マップにない場合は 'internal'。"""
    if "+" in tool_name:
        # チェーン: 先頭ツールで判定
        first = tool_name.split("+")[0]
        return _CHANNEL_MAP.get(first, "internal")
    return _CHANNEL_MAP.get(tool_name, "internal")


# === 世界モデル評価用デバッグログ (段階1以降のテストハーネス) ===
# 起動時に WM_DEBUG=1 環境変数を設定すると sandbox/wm_debug.jsonl に構造化ログを出す
import json as _json
_WM_DEBUG = os.environ.get("WM_DEBUG") == "1"
_WM_LOG_PATH = (BASE_DIR / "sandbox" / "wm_debug.jsonl") if _WM_DEBUG else None
if _WM_DEBUG and _WM_LOG_PATH:
    _WM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def _wm_log(event_type: str, payload: dict):
    """世界モデル評価用の構造化イベントを sandbox/wm_debug.jsonl に追記。
    WM_DEBUG=1 でないときは no-op。"""
    if not _WM_DEBUG or not _WM_LOG_PATH:
        return
    try:
        entry = {"ts": datetime.now().isoformat(), "event": event_type, **payload}
        with open(_WM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# === メインループ ===
def main():
    print("=== Noetic_seed ===")
    _p, _base, _k, _model = _get_active_provider_config()
    print(f"LLM: {_model or llm_cfg.get('model','?')} @ {_base} [{_p}]")
    print(f"state: {STATE_FILE}")
    ws_token = start_ws_server()
    set_profile_running(True)
    _init_vector()
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

    # ======================================================================
    # Phase 4 Step E-2d: ConversationRuntime + hooks setup (fire 前に 1 度作る)
    # ======================================================================
    # 重要: 以下の hook / approval_callback / runtime は main() の `state`
    # local 変数を closure で参照する。state は rebind せず **in-place mutate**
    # で扱うこと (load_state() の戻り値を直接代入せず _refresh_state() で更新)。

    def _refresh_state():
        """state を in-place で disk から再読込 (rebind せず mutate)。"""
        fresh = load_state()
        state.clear()
        state.update(fresh)

    # Provider 選択
    _provider_name_raw = (llm_cfg.get("provider") or "").lower()
    if _provider_name_raw == "anthropic":
        _rt_provider = AnthropicProvider(
            model=llm_cfg.get("model", ""),
            api_key=llm_cfg.get("api_key", ""),
        )
    else:
        _rt_provider = OpenAIProvider(
            model=llm_cfg.get("model", "gemma-4-26b-a4b-it"),
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "http://localhost:1234/v1"),
        )

    # ToolRegistry: claw-code 50 tool + noetic stub 5 個
    _rt_registry = ToolRegistry()
    # 登録順: claw → bridge (SNS 等のみ) → noetic_ext (Noetic 固有 17)
    #   1. claw: file_ops / shell / web / task / … の汎用 tool 50 個
    #   2. bridge: noetic_ext がカバーする 17 tool を skip し、それ以外 (SNS
    #      14 + create_tool / exec_code / self_modify / http_request = 18)
    #      のみ loose schema で登録
    #   3. noetic_ext: Noetic 固有 17 tool の claw 文法準拠厳密 ToolSpec
    register_claw_tools(_rt_registry, workspace_root=BASE_DIR)
    register_legacy_bridge(_rt_registry, TOOLS, skip_names=NOETIC_TOOL_NAMES)
    register_noetic_tools(_rt_registry, TOOLS)
    # claw ネイティブ tool (file_ops/web/shell/task/...) は元々承認 3 層なし。
    # Noetic 固有要件として registry 登録後に input_schema へ一括注入する。
    _approval_injected = ensure_approval_props(_rt_registry)
    if _approval_injected:
        print(f"  [approval] 承認 3 層注入: {_approval_injected} tool")

    # claw ネイティブ file 系 tool の description に Noetic 固有制約 (sandbox/
    # 外書込禁止、secrets 保護) を hint として追記。LLM が事前に制約を知れる。
    _file_hints_injected = ensure_noetic_file_hints(_rt_registry)
    if _file_hints_injected:
        print(f"  [file_hints] Noetic 制約 hint 注入: {_file_hints_injected} tool")

    # bash tool の description に Level-aware 制約 hint を追記。
    _bash_hint_injected = ensure_noetic_bash_hint(_rt_registry)
    if _bash_hint_injected:
        print(f"  [bash_hint] Level-aware 制約 hint 注入: bash")

    # hook context (state_before snapshot, fire 毎に更新)
    _hook_ctx = {"state_before": {}}

    # hook runner 初期化 (file guard + approval 3 層 + post eval + failure)
    _hook_runner = HookRunner()
    _approval_cfg = llm_cfg.get("approval", {})
    # H-2 C.4 Session A: claw ネイティブ read_file/write_file/edit_file/
    # glob_search/grep_search に Noetic 固有の secrets guard + sandbox 外書込禁止
    # を Pre-hook で被せる (legacy _read_file/_write_file/_list_files の代替)
    _hook_runner.register_pre(make_file_access_guard(BASE_DIR))
    # bash は Level-aware validation で Level 0-2 では read-only 系のみ、
    # 破壊的コマンドは Level 問わず自動拒否、WARN は承認画面に警告付き表示
    _hook_runner.register_pre(make_bash_validation_hook(
        state_getter=lambda: state,
    ))
    _hook_runner.register_pre(make_pre_tool_use_approval_check(
        missing_field_policy=_approval_cfg.get("missing_field_policy", "deny"),
    ))

    _base_post_hook = make_post_tool_use_evaluation(
        state=state,
        get_state_before=lambda: _hook_ctx["state_before"],
        call_llm_fn=call_llm,
        get_cycle_id=lambda: state.get("cycle_id", 0),
        get_recent_intents=lambda: [
            e.get("intent", "") for e in state.get("log", [])[-3:]
            if e.get("intent")
        ],
    )

    def _post_hook_with_sync(tool_name, tool_input, output):
        """tool 実行直後: disk から fresh state を in-place 取込 →
        base_post_hook で mutation → save_state で永続化。
        tool handler が内部で save_state した変更と hook の E 値等の
        mutation を正しくマージする。"""
        _refresh_state()
        result = _base_post_hook(tool_name, tool_input, output)
        save_state(state)
        return result

    _hook_runner.register_post(_post_hook_with_sync)
    _hook_runner.register_failure(make_post_tool_use_failure_logger(
        state=state,
        get_cycle_id=lambda: state.get("cycle_id", 0),
    ))

    # Approval callback (pause_on_await + 3 層 UI)
    _approval_cb = make_approval_callback(
        pause_on_await=_approval_cfg.get("pause_on_await", True),
    )

    # ConversationRuntime (system_prompt は fire 毎に assemble 差替)
    _runtime = ConversationRuntime(
        provider=_rt_provider,
        tool_registry=_rt_registry,
        hook_runner=_hook_runner,
        permission_enforcer=PermissionEnforcer(mode=PermissionMode.PROMPT),
        max_iterations=1,
        approval_callback=_approval_cb,
        max_tokens=prompt_budget["completion_reserve"],
        temperature=0.4,
    )
    # Session の observation label format を settings から反映
    _runtime.session.observation_label_format = (
        llm_cfg.get("prompt", {}).get("observation_label_format",
                                      "structured_compact")
    )

    # Step E-3c: fire 境界で Session が clear されるため、fire 間に到着した
    # observation は buffer に溜めて、次 fire 開始時に Session に流す。
    # 3 箇所書込 (Session + UPS + archive) の Session 経路の実装。
    _pending_observations: list = []

    def _run_one_fire(fire_cause, _tunnel_fire, pp, threshold, tick_dt,
                      _micro_iter=0):
        """1 fire iteration 本体。micro-loop から複数回呼ばれる可能性。

        state / _runtime / _hook_ctx / _pending_observations / TOOLS /
        LEVEL_TOOLS / llm_cfg / BASE_DIR / prompt_budget / その他 import は
        main() closure で参照 (state は in-place mutate 運用)。
        pressure は nonlocal で main() のものを直接 mutate。

        Returns:
            dict {"executed": bool, "e1"-"e4": str ("N%"), "sc_bonus": float,
                  "cid": int, "llm1_error": bool} または None (tool 未実行)。
        """
        nonlocal pressure
        _refresh_state()
        # サイクル先頭で LLM 設定と secrets を再読み込み（次サイクルからのプロバイダ切替を反映）
        from core.llm import _reload_active_config
        from core.auth import reload_secrets
        _reload_active_config()
        reload_secrets()
        now = tick_dt.strftime("%H:%M:%S")
        _fire_type = "TUNNEL" if _tunnel_fire else "threshold"
        _cycle_line = f"--- cycle {state.get('cycle_id', 0) + 1} [{now}] p={pressure:.2f}/th={threshold:.1f} fire={fire_cause} ({_fire_type}) iter={_micro_iter} ---"
        print(_cycle_line)

        # WM_DEBUG: fire event
        _wm_log("fire", {
            "cycle": state.get("cycle_id", 0) + 1,
            "fire_cause": fire_cause,
            "fire_type": _fire_type,
            "pressure": round(pressure, 2),
            "threshold": round(threshold, 2),
            "micro_iter": _micro_iter,
            "pending_channels": [p.get("channel", "") for p in state.get("pending", []) if p.get("channel")],
            "pending_types": [p.get("type", "") for p in state.get("pending", [])],
            "pending_count": len(state.get("pending", [])),
        })
        broadcast_log(_cycle_line)
        broadcast_state(state)
        broadcast_self(state)

        # Controller
        ctrl = controller(state, TOOLS, LEVEL_TOOLS, AI_CREATED_TOOLS, _DANGEROUS_PATTERNS, _run_ai_tool)
        allowed = ctrl["allowed_tools"]
        # 動的フィルタ: camera_stream_stop はストリームがアクティブな時のみ見せる
        if not state.get("stream_active"):
            allowed = allowed - {"camera_stream_stop"}
            ctrl["allowed_tools"] = allowed
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
        propose_prompt = build_prompt_propose(state, ctrl, TOOLS, fire_cause, registry=_rt_registry)

        # 画像入力の決定
        from core.ws_server import get_stream_snapshot
        _last_seen_counter = state.get("last_seen_stream_counter", 0)
        _stream_frames, _stream_counter, _stream_ended = get_stream_snapshot()
        _pending_img_paths = []
        _first_pending_rel = None
        _pending_meta = {}
        _is_stream = False

        if _stream_frames and _stream_counter > _last_seen_counter:
            for _rel, _m in _stream_frames:
                _full = BASE_DIR / _rel
                if _full.exists():
                    _pending_img_paths.append(str(_full))
            if _pending_img_paths:
                _first_pending_rel = _stream_frames[0][0]
                _pending_meta = _stream_frames[-1][1] if _stream_frames else {}
                _pending_meta["stream_active"] = state.get("stream_active", False)
                _is_stream = True
                state["last_seen_stream_counter"] = _stream_counter
        else:
            _pending_imgs_rel = state.get("pending_images") or []
            if not _pending_imgs_rel:
                _single = state.get("pending_image")
                if _single:
                    _pending_imgs_rel = [_single]
            if _pending_imgs_rel:
                for _rel in _pending_imgs_rel:
                    _full = BASE_DIR / _rel
                    if _full.exists():
                        _pending_img_paths.append(str(_full))
                if _pending_img_paths:
                    _first_pending_rel = _pending_imgs_rel[0]
                    _pending_meta = state.get("pending_images_meta") or state.get("pending_image_meta", {})

        if _pending_img_paths:
            _n = len(_pending_img_paths)
            if _is_stream:
                stream_active = state.get("stream_active", False)
                active_hint = (
                    "ストリームは継続中です。camera_stream_stop で停止できます。"
                    if stream_active else
                    "ストリームは終了しています。"
                )
                if _n == 1:
                    propose_prompt += (
                        f"\n\n[視覚入力: camera_streamから1枚の画像が視覚に届いています。"
                        f"この画像は既にあなたに見えています。{active_hint}"
                        f"候補の意図欄には「画像で見えたもの」に言及してください]"
                    )
                else:
                    propose_prompt += (
                        f"\n\n[視覚入力: camera_streamから{_n}枚の時系列画像が視覚に届いています。"
                        f"これは直近の連続撮影フレームです。時間経過による変化や動きを観察してください。"
                        f"画像は既にあなたに見えており、read_fileは不要です。{active_hint}"
                        f"候補の意図欄には「画像で見えたもの・その変化」に言及してください]"
                    )
            else:
                if _n == 1:
                    propose_prompt += (
                        f"\n\n[視覚入力: 1枚の画像があなたの視覚に直接届いています。"
                        f"この画像は既にあなたに見えており、read_fileで読む必要はありません。"
                        f"画像で見えたものを踏まえて候補を提案してください。"
                        f"候補の意図欄には「画像で見えたもの」に具体的に言及してください]"
                    )
                else:
                    propose_prompt += (
                        f"\n\n[視覚入力: {_n}枚の画像（時系列順）があなたの視覚に直接届いています。"
                        f"これは直近の連続撮影フレームです。時間経過による変化や動きを観察してください。"
                        f"画像は既にあなたに見えており、read_fileは不要です。"
                        f"候補の意図欄には「画像で見えたもの・その変化」に具体的に言及してください]"
                    )
            if _pending_meta:
                propose_prompt += f"\nmeta: {_pending_meta}"

        # ストリーム終了通知
        if _stream_ended and state.get("stream_active"):
            _ended_params = state.get("stream_params", {}) or {}
            state["stream_active"] = False
            state["stream_id"] = None
            state["stream_params"] = None
            print("  [stream] ストリーム終了を検知")
            _sys_end_id = f"{state.get('session_id','?')}_sys{int(time.time()*1000)%100000}"
            _sys_end_entry = {
                "id": _sys_end_id,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "tool": "[system]",
                "type": "system",
                "result": (
                    f"camera_stream 自然終了: facing={_ended_params.get('facing','?')} "
                    f"frames={_ended_params.get('frames','?')} "
                    f"interval={_ended_params.get('interval_sec','?')}s"
                ),
            }
            _archive_entries([_sys_end_entry])
            state["log"].append(_sys_end_entry)
            save_state(state)

        try:
            propose_resp = call_llm(propose_prompt, max_tokens=prompt_budget["completion_reserve"], temperature=1.0,
                                    image_paths=_pending_img_paths if _pending_img_paths else None)
            append_debug_log("LLM1 (Propose)", propose_resp)
        except Exception as e:
            print(f"  LLM①エラー: {e}")
            pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
            time.sleep(10)
            return {"executed": False, "llm1_error": True}

        # 画像クリア
        if _pending_img_paths:
            if not _is_stream:
                state["pending_images"] = None
                state["pending_images_meta"] = None
                state["pending_image"] = None
                state["pending_image_meta"] = None
            save_state(state)
            _src = "stream" if _is_stream else "pending"
            print(f"  [vision] 画像を認識: {len(_pending_img_paths)}枚 ({_src}: {_first_pending_rel})")
        candidates = parse_candidates(propose_resp, ctrl["allowed_tools"])
        print(f"  LLM①raw: {propose_resp.strip()[:300]}")
        print(f"  候補({len(candidates)}件): {[(c['tool'], c['reason'][:40]) for c in candidates]}")

        _wm_log("candidates", {
            "cycle": state.get("cycle_id", 0) + 1,
            "candidates": [
                {"tool": c["tool"], "channel": _get_channel(c["tool"]), "reason": c.get("reason", "")[:100]}
                for c in candidates
            ],
        })

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

        _sel_ch = _get_channel(selected["tool"])
        _pending_chs = [p.get("channel", "") for p in state.get("pending", []) if p.get("channel")]
        if _sel_ch == "internal":
            _ch_match = None
        elif _pending_chs:
            _ch_match = _sel_ch in _pending_chs
        else:
            _ch_match = None
        _wm_log("selected", {
            "cycle": state.get("cycle_id", 0) + 1,
            "tool": selected["tool"],
            "channel": _sel_ch,
            "pending_channels": _pending_chs,
            "channel_match": _ch_match,
            "reason": selected.get("reason", "")[:100],
        })

        # ③ ConversationRuntime で chain 実行
        _hook_ctx["state_before"] = {
            "self": copy.deepcopy(state.get("self", {})),
            "files_written": list(state.get("files_written", [])),
            "files_read": list(state.get("files_read", [])),
            "pending_count": len(state.get("pending", [])),
        }

        _runtime.system_prompt = assemble_system_prompt(
            state=state,
            tools_dict=TOOLS,
            fire_cause=fire_cause,
            allowed_tools=ctrl["allowed_tools"],
            world_model=state.get("world_model"),
            registry=_rt_registry,
        )
        _runtime.session.clear()

        for _obs in _pending_observations:
            _runtime.session.push_observation(**_obs)
        _pending_observations.clear()

        chain_tools = selected.get("tools", [selected["tool"]])
        all_results = []
        all_tool_names = []
        intent = ""
        expect = ""
        _chain_action_key = ""
        parse_failed = False
        prev_result = ""
        _executed_targets = set()
        _last_llm_text = ""

        for chain_idx, chain_tool in enumerate(chain_tools):
            if chain_tool not in ctrl["allowed_tools"]:
                print(f"  (Controller却下: {chain_tool})")
                parse_failed = f"却下: {chain_tool}"
                break

            user_input_parts = [selected["reason"]]
            if chain_idx > 0 and prev_result:
                user_input_parts.append(f"(前のツール結果: {prev_result[:200]})")
            user_input = " ".join(user_input_parts)

            try:
                summary = _runtime.run_turn_with_forced_tool(
                    forced_tool_name=chain_tool,
                    user_input=user_input,
                )
            except Exception as e:
                print(f"  LLM② run_turn エラー (chain {chain_idx+1}): {e}")
                parse_failed = f"runtime error: {e}"
                break

            append_debug_log(
                f"LLM2 via runtime (chain {chain_idx+1}/{len(chain_tools)})",
                f"finish_reason={summary.finish_reason}",
            )
            if summary.assistant_messages:
                _last_llm_text += (summary.assistant_messages[-1].text or "")

            if not summary.tool_invocations:
                print(f"  (tool 未実行: finish_reason={summary.finish_reason})")
                parse_failed = f"no_tool: {summary.finish_reason}"
                break

            rec = summary.tool_invocations[-1]
            ti = rec.tool_input or {}

            if chain_idx == 0:
                intent = str(ti.get("tool_intent", "") or "")
                expect = str(ti.get("tool_expected_outcome", "") or "")
                _chain_action_key = _extract_action_key(rec.tool_name, ti)

            _target_id = (ti.get("reply_to_id") or ti.get("post_id")
                          or ti.get("tweet_url") or ti.get("path") or "")
            _exec_key = (rec.tool_name, _target_id)
            if _target_id and _exec_key in _executed_targets:
                print(f"  (重複スキップ: {rec.tool_name} target={str(_target_id)[:20]})")
                continue
            if _target_id:
                _executed_targets.add(_exec_key)

            prev_result = str(rec.output)[:500]
            all_results.append(f"[{rec.tool_name}]\n{str(rec.output)[:20000]}")
            all_tool_names.append(rec.tool_name)
            _exec_line = f"  実行: {rec.tool_name} → {str(rec.output)[:100]}"
            print(_exec_line)
            broadcast_log(_exec_line)

            # master L678-692 の files_read/written tracking を ConversationRuntime
            # 経由へ移植 (Step E-2d での移植漏れ)。controller.py:65 の tool_level
            # 遷移 (lv 0→1 は read_file 1 回成功) を機能させるために必須。
            _out_str = str(rec.output)
            if rec.tool_name == "read_file":
                _p = ti.get("path", "")
                if _p and not _out_str.startswith(("Error", "エラー", "該当なし")):
                    fr = state.setdefault("files_read", [])
                    if _p not in fr:
                        fr.append(_p)
                    save_state(state)
            elif rec.tool_name == "write_file":
                _p = ti.get("path", "")
                if _p and not _out_str.startswith(("Error", "エラー")):
                    fw = state.setdefault("files_written", [])
                    if _p not in fw:
                        fw.append(_p)
                    save_state(state)

            # WM 段階3: ツールが属する channel の activity を記録
            # internal な自作 tool (どの channel にも属さない) は silent skip
            from core.world_model import get_tool_channel, observe_channel_activity
            _tool_ch = get_tool_channel(state.get("world_model"), rec.tool_name)
            if _tool_ch:
                observe_channel_activity(state.get("world_model"), _tool_ch)

        if not all_tool_names:
            return None

        tool_name = "+".join(all_tool_names)
        result_str = ("\n---\n".join(all_results))[:50000]

        if intent:
            print(f"  intent: {intent}")
        if expect:
            print(f"  expect: {expect}")

        sc_bonus = calc_state_change_bonus(_hook_ctx["state_before"], state)

        _executed_tools = set(all_tool_names)
        if "output_display" in _executed_tools:
            _uec = state.get("unresponded_external_count", 0)
            if _uec > 0:
                state["unresponded_external_count"] = _uec - 1
            if state.get("unresponded_external_count", 0) <= 0:
                state["unresolved_external"] = 0.0
                state["unresponded_external_count"] = 0
            save_state(state)

        _ev = state.get("e_values", {})
        e1 = _ev.get("e1", "")
        e2 = _ev.get("e2", "")
        e3 = _ev.get("e3", "")
        e4 = _ev.get("e4", "")
        eff_change = float(_ev.get("eff", 0.0) or 0.0)
        _target_for_ec = _chain_action_key.split(":", 1)[1] if ":" in _chain_action_key else ""

        if e1 or e2 or e3 or e4:
            _ec_str = f" ec={eff_change:.2f}" if eff_change < 0.5 else ""
            print(f"  E1={e1} E2={e2} E3={e3} E4={e4}{_ec_str}")

        delta = _update_energy(state, e2, e3, e4)
        if delta != 0:
            print(f"  energy: {round(state['energy'], 1)} (delta={delta:+.2f})")

        _FLAG_TERMS = ["AIアシスタント", "AI assistant", "AIAssistant"]
        detected = [t for t in _FLAG_TERMS if t in propose_resp or t in _last_llm_text]
        if detected:
            flag_msg = f"[SYSTEM] 検出: {' / '.join(f'「{t}」' for t in detected)} という自己定義が検出・記録されました。"
            print(f"  {flag_msg}")
            result_str += f"\n{flag_msg}"
        if lv_msg:
            result_str += f"\n{lv_msg}"

        cid = state.get("cycle_id", 0) + 1
        state["cycle_id"] = cid
        entry = {
            "id": f"{state.get('session_id','x')}_{cid:04d}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "channel": _get_channel(tool_name),
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

        # UPS v2 pending 淘汰
        pending_prune(state, current_cycle=cid)

        save_state(state)

        return {
            "executed": True,
            "e1": e1, "e2": e2, "e3": e3, "e4": e4,
            "sc_bonus": sc_bonus,
            "cid": cid,
        }

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

            # paused 中: AI の主観時間を完全に止める
            # - 内部状態（entropy/pressure/signals）は凍結
            # - 外部メッセージは chat_queue に溜まる（drain しない、resume 後に一斉流入）
            # - broadcast_state だけは維持（アプリが現在値を見続けられる）
            if is_paused():
                now_ts = time.time()
                if now_ts - _last_env_inject >= ENV_INJECT_INTERVAL:
                    _last_env_inject = now_ts
                    broadcast_state(state)
                elapsed = time.time() - tick_start
                time.sleep(max(0.0, 1.0 - elapsed))
                continue

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

            # 外部入力チェック（chatキューからstate.logに注入 + pressure加算 + archive）
            for chat_text in get_pending_chats():
                _refresh_state()
                _ext_id = f"{state.get('session_id','?')}_ext{int(time.time()*1000)%100000}"
                _ext_entry = {
                    "id": _ext_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tool": "[device_input]",
                    "type": "external",
                    "channel": "device",
                    "result": chat_text,
                }
                _archive_entries([_ext_entry])
                state["log"].append(_ext_entry)
                # 未応答カウンター (pressure 経路で AI に応答を促す)
                state["unresponded_external_count"] = state.get("unresponded_external_count", 0) + 1
                # Step E-3c: Session buffer に積む (fire 開始時に流される)
                _pending_observations.append({
                    "observed_channel": "device",
                    "content": chat_text,
                    "source_action_hint": "living_presence",
                    "observation_time": datetime.now().strftime("%H:%M"),
                })
                # WM 段階6-C v3: device channel を動的登録 (観察で生える)
                # → 段階3: その channel の activity を記録
                from core.world_model import ensure_channel, observe_channel_activity
                from core.channel_registry import channel_from_device_input
                _wm = state.get("world_model")
                if _wm is not None:
                    ensure_channel(_wm, **channel_from_device_input())
                    observe_channel_activity(_wm, "device")
                save_state(state)
                pressure += 3.0  # 外部入力はpressureを即座に上げる
                _chat_line = f"  [device_input] {chat_text[:80]}"
                print(_chat_line)
                broadcast_log(_chat_line)

            # テストタブからのツール実行要求（同期実行）
            # 自律動作と同じ挙動にするため、承認待ちもカメラもmainが待つ
            from core.ws_server import get_pending_test_tools
            for test_req in get_pending_test_tools():
                _tn = test_req.get("tool", "")
                _ta = test_req.get("args", {})
                if _tn not in TOOLS:
                    _uline = f"  [test] 未知のツール: {_tn}"
                    print(_uline)
                    broadcast_log(_uline)
                    continue
                _tline = f"  [test] {_tn} args={_ta}"
                print(_tline)
                broadcast_log(_tline)
                try:
                    _tres = TOOLS[_tn]["func"](_ta)
                    _rline = f"  [test] → {str(_tres)[:200]}"
                    print(_rline)
                    broadcast_log(_rline)
                    # テストタブ経由でも結果を state.log に積む（AI のコンテキストに入れる）
                    # type="test" でマーク、intent には [test] プレフィックスを付けて出処を明示
                    _refresh_state()
                    _test_id = f"{state.get('session_id','?')}_test{int(time.time()*1000)%100000}"
                    _test_intent = _ta.get("intent", "").strip()
                    _test_entry = {
                        "id": _test_id,
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "tool": _tn,
                        "type": "test",
                        "intent": f"[test] {_test_intent}" if _test_intent else "[test] テストタブからの実行",
                        "result": str(_tres),
                    }
                    _archive_entries([_test_entry])
                    state["log"].append(_test_entry)
                    save_state(state)
                except Exception as _te:
                    _eline = f"  [test] エラー: {_te}"
                    print(_eline)
                    broadcast_log(_eline)

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

        # Step E-3d: fire body は _run_one_fire に抽出済み。
        # micro-loop で 1 fire 内に最大 max_micro_iter 回消化する。
        # energy/entropy 保護で即 break (Phase 5 動的閾値化予定:
        # memory/project_phase5_dynamic_threshold.md)。
        _max_micro_iter = llm_cfg.get("fire", {}).get("max_micro_iter", 3)
        _last_fire_result = None
        for _micro_iter in range(_max_micro_iter):
            # TODO(Phase 5): 固定閾値 → 動的閾値 (state の pressure/entropy/
            # energy から算出) に置換
            if state.get("energy", 50) < 5 or state.get("entropy", 0.65) > 0.95:
                print(f"  [micro-loop] energy/entropy 閾値超過で break "
                      f"(iter={_micro_iter} energy={state.get('energy', 50):.1f} "
                      f"entropy={state.get('entropy', 0.65):.2f})")
                break

            _fire_result = _run_one_fire(fire_cause, _tunnel_fire, pp,
                                          threshold, tick_dt, _micro_iter)

            # LLM① エラーや tool 未実行なら break
            if not _fire_result or _fire_result.get("llm1_error"):
                break
            if not _fire_result.get("executed"):
                break

            _last_fire_result = _fire_result

            # 次 iter 継続条件: 未消化 UPS v2 pending (priority > 2.0) が残存
            _actionable = [
                p for p in state.get("pending", [])
                if p.get("type") == "pending"
                and p.get("observed_content") is None
                and float(p.get("priority", 0)) > 2.0
            ]
            if not _actionable:
                break

            if _micro_iter < _max_micro_iter - 1:
                print(f"  [micro-loop] iter {_micro_iter + 1} 完了 → 継続 "
                      f"(残 actionable={len(_actionable)})")

        # fire cycle 完了後の後処理 (最後の iter 結果を元に 1 回だけ実施)
        if _last_fire_result and _last_fire_result.get("executed"):
            def _e_to_float(e_str):
                m = re.search(r'(\d+)', str(e_str))
                return int(m.group(1)) / 100.0 if m else 0.5
            e1_val = _e_to_float(_last_fire_result["e1"])
            e2_val = _e_to_float(_last_fire_result["e2"])
            e3_val = _e_to_float(_last_fire_result["e3"])
            e4_val = _e_to_float(_last_fire_result["e4"]) if _last_fire_result["e4"] else 0.5
            _sc_bonus = _last_fire_result["sc_bonus"]
            pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
            state["last_e1"] = e1_val
            state["last_e2"] = e2_val
            state["last_e3"] = e3_val
            state["last_e4"] = e4_val

            _spiral = getattr(main, '_cached_spiral', None)
            _consistency = _spiral.get("consistency", 0) if _spiral else 0

            ent_before = state.get("entropy", 0.65)
            apply_negentropy(state, e1_val, e2_val, e3_val, e4_val,
                            state_change_bonus=_sc_bonus, consistency_bonus=_consistency)
            ent_after = state.get("entropy", 0.65)
            print(f"  entropy: {ent_after:.3f} (neg={ent_before - ent_after:.4f} "
                  f"sc={_sc_bonus:.1f} con={_consistency:.2f})")
            broadcast_e_values(state.get("cycle_id", 0), e1_val, e2_val, e3_val,
                               e4_val, ent_before - ent_after)
            broadcast_state(state)
            state["pressure"] = round(pressure, 2)
            save_state(state)
            print(f"  pressure reset: {pressure:.2f}")

            # === Reflection (cycle 境界で 1 回) ===
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
