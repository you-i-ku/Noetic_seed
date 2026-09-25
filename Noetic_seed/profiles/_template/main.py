"""Noetic_seed"""
# === venv ブートストラップ（初回起動時に自動セットアップ） ===
import sys
import os
from pathlib import Path as _Path

def _bootstrap_venv():
    _here = _Path(__file__).parent
    # 段階12 Step 1.5 (PLAN §3-1 / §11-5-pre): per-profile venv。
    # 旧共通 venv から profile 配下の独立 venv に移行することで、
    # pip install / requirements.txt 編集が iku の身体範囲内に収まり、
    # 自己改変による身体拡張が成立する。
    _venv = _here / ".venv"
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

    _req = _here / "requirements.txt"

    # venv がなければ作成
    if not _venv_python.exists():
        print("[bootstrap] 仮想環境を作成中...")
        subprocess.run([sys.executable, "-m", "venv", str(_venv)], check=True)

    # PLAN §3-3: 身体仕様 (requirements.txt) を single source of truth として
    # 毎回同期する。iku が requirements.txt を編集 → reboot で新ライブラリが
    # 反映される経路 (= 身体拡張の汎用チェーン) を成立させるため、venv 既存でも
    # 必ず実行する。pip は idempotent: 既 install pkg は manifest 検査で skip
    # されるので、全 install 済なら数秒、差分のみ DL する。
    # pip.exe は起動用 stub で、作成時の python.exe の絶対パスが埋め込まれている。
    # profile をコピーすると stub がコピー元の venv を指したままになり、同期が
    # 別の venv に対して行われる (2026-09-24 実例)。この venv の python で
    # `-m pip` を呼び、必ず自分の venv に入れる。
    if _req.exists():
        print("[bootstrap] requirements.txt から身体仕様を同期中...")
        subprocess.run([str(_venv_python), "-m", "pip", "install", "--quiet", "-r", str(_req)],
                       check=True)
        print("[bootstrap] 同期完了。venv で再起動します...\n")
    else:
        print(f"[bootstrap] WARNING: {_req} が見つかりません。空 venv で起動します。\n")

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
import json
from functools import partial
from datetime import datetime

# DualLoggerの設定（printをファイルにも書き出す）
from core.config import (
    DualLogger, RAW_LOG_FILE, STATE_FILE, DEFAULT_PRESSURE_PARAMS,
    ENV_INJECT_INTERVAL, _NOTIFICATION_HOURS, llm_cfg, BASE_DIR, prompt_budget,
    cap_tool_result
)
sys.stdout = DualLogger(RAW_LOG_FILE)

from core.profile_repo import ensure_profile_repo
from core.sanity_check import enforce_sanity_check
from core.state import load_state, save_state, load_pref, save_pref, append_debug_log
from core.llm import call_llm, _get_active_provider_config
from core.embedding import _init_vector, _compare_expect_result
from core.eval import (_calc_e4, _update_energy, eval_with_llm, calc_state_change_bonus,
                       calc_spiral_vector, calc_measured_entropy,
                       calc_effective_change, apply_effective_change_to_e2, EXTERNAL_ACTION_TOOLS,
                       update_unresolved_intents, update_gaps_by_relevance,
                       _extract_action_key, append_action_ledger)
from core.pending_unified import pending_prune, pending_add_response_intent

# Phase 4 Step E-2d: ConversationRuntime 統合用 import
from core.providers.openai_compat import OpenAIProvider
from core.providers.anthropic import AnthropicProvider
from core.providers.claude_code import ClaudeCodeProvider
from core.runtime.registry import ToolRegistry
from core.runtime.conversation import ConversationRuntime
from core.runtime.hooks import (
    HookRunner,
    make_bash_path_guard_hook,
    make_bash_validation_hook,
    make_file_access_guard,
    make_git_auto_stash_hook,
    make_install_command_check_hook,
    make_pre_tool_use_approval_check,
    make_post_body_modify_pending_hook,
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
from core.memory import _record_entry, maybe_compress_log, get_relevant_memories, format_memories_for_prompt, non_empty_subjective_entries
from core import event_emitter
from core.perspective import make_perspective
from core.reflection import should_reflect, reflect, reflect_and_persist
from core.prompt import build_prompt_propose
from core.controller import controller, controller_select, _intent_conditioned_scores

from tools import TOOLS, LEVEL_TOOLS
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
    # 段階12 PLAN §6 / §7 改訂: profile 内独立 git repo の存在確保。
    # 1 個体 = 1 repo = 1 履歴 (autopoiesis 整合)。
    # .git なければ自動 init + 初期 commit、あれば noop (idempotent)。
    # G-2 自動 stash hook (`make_git_auto_stash_hook`) は親方向に .git を
    # 探すため、この後の身体改変 stash は profile 内 repo 上で行われる。
    ensure_profile_repo(BASE_DIR)
    # 段階12 Step 6 (PLAN §10): sanity check + 自動 revert。state load より前。
    # state.json / memory JSON / core.controller + tools の import を検査、
    # 失敗時は最新 iku-auto stash から git stash apply で復元、再検査して
    # 成功なら続行 / 失敗なら sys.exit(1)。「起動できない身体 = 存在できない」
    # の構造化 (PLAN §10-3、§1-5 介入レベル「物理的存在の前提」枠)。
    enforce_sanity_check(profile_root=BASE_DIR, profile_name=BASE_DIR.name)
    _p, _base, _k, _model = _get_active_provider_config()
    print(f"LLM: {_model or llm_cfg.get('model','?')} @ {_base} [{_p}]")
    print(f"state: {STATE_FILE}")
    ws_token = start_ws_server()
    set_profile_running(True)
    _init_vector()
    # 段階11-B Phase 5 Step 5.1: register_standard_tags() 呼出撤去。
    # iku は起動時に登録済 tag ゼロ = 白紙 onboarding 状態。最初の memory_store
    # 呼出で inline register 発火、iku が自己分節の tag を発明する設計。
    # STANDARD_TAGS 辞書 / register_standard_tags() 関数自体は tag_registry.py
    # に残してある (iku が write_file で復活させたい時の handle、撤去 doc 価値)。
    # 既存 smoke / 実運用で標準タグが必要なら、プロファイル起動前に手動で
    # registered_tags.json を seed することで対応 (新プロファイル clean start は
    # PLAN §5 Step 5.3 の手順準拠)。
    print()

    state = load_state()
    state["session_id"] = str(uuid.uuid4())[:8]
    # Slice 2: run_id (full uuid) を起動毎生成。session_id (8 char) と階層化
    # することで「同 run 内の複数 session」(将来の手動再起動シナリオ等)
    # も表現可能。metrics_events.jsonl に記録、replay arena で run 単位識別。
    state["run_id"] = str(uuid.uuid4())
    # F-005: subjective_entries.jsonl は materialized view (raw_events.jsonl が
    # source of truth)、compaction / retro e2 mutation で mark_view_dirty が発火、
    # 以降の save_state() で atomic rewrite + index.json 同期。
    # 将来の同パターン view (goal_shadows / trajectories 等) はここに追加 register。
    from core.view_persistence import register_view
    from core.config import MEMORY_DIR as _MEMORY_DIR_F005
    register_view(
        "subjective_entries",
        _MEMORY_DIR_F005 / "subjective_entries.jsonl",
        "subjective_entries",
        index_path=_MEMORY_DIR_F005 / "index.json",
    )
    # 段階10.5 Fix 1: chain 連結キー entry (tool_A+tool_B 形式) を drop し、
    # tool 単位 entry のみ残す。新 smoke 起点で predictor_confidence をリセット。
    from core.predictor import migrate_chain_keys
    _dropped = migrate_chain_keys(state)
    if _dropped:
        print(f"  [migration] 段階10.5: predictor_confidence から {_dropped} 個の chain 連結 entry を drop")
    # 段階10.5 Fix 2: 旧 pending 形式 (content_observable 欠落) を drop。
    # 案 P 確定、新 smoke 起点で content_observable/content_intent 二層化 pending のみ残す。
    from core.pending_unified import migrate_pending_observable_split
    _p_dropped = migrate_pending_observable_split(state)
    if _p_dropped:
        print(f"  [migration] 段階10.5 Fix 2: 旧形式 pending {_p_dropped} 個を drop (content_observable 欠落)")
    save_state(state)
    # 段階13 Phase 0.2: event_emitter observer pattern。default observer として
    # _record_entry (jsonl emit + state view sync) を subscribe。
    event_emitter.subscribe(_record_entry)
    # 段階13 Phase 0.4: raw event vs subjective entry の意味的 gap 検出 observer。
    # subscribe 順序 snapshot で _record_entry 後に発火、in-memory view 更新済の状態
    # で同 id の raw_part / subj_part を比較、gap >= 0.30 (entity_resolver 閾値流用)
    # で record_ec_prediction_error(source="raw_subj_gap") に流す (check_on_write sibling)。
    from core.reconciliation import check_raw_subjective_gap
    event_emitter.subscribe(check_raw_subjective_gap)
    # 段階13 Phase 2: Layer C JEPA 流予測 observer。subscribe 順序 snapshot で
    # check_raw_subjective_gap 後に発火、subj_part に embedding が enrich 済の
    # 状態で「前回 predicted vs 今回 actual」cosine 距離計算 + 次 predict_next
    # 計算 + train trigger を行う。Phase 0.4 check_raw_subjective_gap と同型の
    # post-write subscriber pattern (entry mutate しない、判断 5 確定)。
    # torch 未 install / sequence 不足 / 例外時は全 graceful skip (Phase 0/1
    # 影響ゼロ保証、判断 4 案 Y 完全独立 module の boundary 維持)。
    # link 接続 (prediction_error → update_link_strength_used) は Phase 3
    # (TE + Bayesian) で「どの link を update するか」を意味的に確定させる
    # 自然構造を壊さないため、Phase 2 では state history に蓄積するのみ
    # (判断 8 案 a deferred、PLAN §4-3 「3 段重ね現実解」に着地)。
    from core.jepa_runtime import jepa_observe_entry
    event_emitter.subscribe(jepa_observe_entry)
    print(f"session: {state['session_id']}  cycle_id: {state['cycle_id']}")
    broadcast_state(state)

    # reflectツールをcall_llm付きで初期化
    # F-005 完了後 hotfix (2026-05-09): 旧実装は `s = load_state()` で別 dict を
    # 読んで mutate → save_state していたため、main scope の `state` には
    # `reflection_cycle = 0` が反映されず、cycle 末 should_reflect(state) が
    # 再 trigger → 自動 reflect 経路が同 cycle で二重発火していた。
    # `reflect_and_persist` 経由で main scope の `state` を closure で直接 in-place
    # mutate する (`_refresh_state` 規約と整合、main.py:305-307 参照)。
    def _tool_reflect(args):
        result = reflect_and_persist(state, call_llm)
        notes = result.get("notes", [])
        return f"内省完了: {len(notes)}件の気づき"
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
        if llm_cfg.get("x_login", {}).get("skip_prompt", False):
            print("  [X] x_login.skip_prompt=true、自動 skip")
            answer = "n"
        else:
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
    elif _provider_name_raw == "claude_code":
        # Claude Code CLI subscription 経由 (claude-agent-sdk + in-process MCP)。
        # LLM① ② ③ ④ 統一 provider、画像 + tool calling フル対応。
        # 詳細: WORLD_MODEL_DESIGN/CLAUDE_CODE_UNIFIED_PROVIDER_PLAN.md
        _rt_provider = ClaudeCodeProvider(
            model=llm_cfg.get("model", "sonnet"),
        )
    else:
        _rt_provider = OpenAIProvider(
            model=llm_cfg.get("model", "gemma-4-26b-a4b-it"),
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "http://localhost:1234/v1"),
            strict_tools=llm_cfg.get("tool_use", {}).get("strict_mode", False),
        )

    # ToolRegistry: claw-code 50 tool + noetic stub 5 個
    _rt_registry = ToolRegistry()
    # 登録順: claw → bridge (SNS 等のみ) → noetic_ext (Noetic 固有 17)
    #   1. claw: file_ops / shell / web / task / … の汎用 tool 50 個
    #   2. bridge: noetic_ext がカバーする 17 tool を skip し、それ以外 (SNS
    #      14 + http_request = 15、段階12 Step 7 で self_modify 撤廃 +
    #      段階13 Phase 6.1 で create_tool / exec_code 撤廃) のみ loose schema で登録
    #   3. noetic_ext: Noetic 固有 17 tool の claw 文法準拠厳密 ToolSpec
    register_claw_tools(_rt_registry, workspace_root=BASE_DIR)
    register_legacy_bridge(_rt_registry, TOOLS, skip_names=NOETIC_TOOL_NAMES)
    register_noetic_tools(_rt_registry, TOOLS)
    # claw ネイティブ tool (file_ops/web/shell/task/...) は元々理由・予想・note なし。
    # Noetic 固有要件として registry 登録後に input_schema へ一括注入する。
    _approval_injected = ensure_approval_props(_rt_registry)
    if _approval_injected:
        print(f"  [approval] 理由・予想・note 注入: {_approval_injected} tool")

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
    _hook_ctx = {"state_before": {}, "evaluations": []}

    # hook runner 初期化 (file guard + approval 3 層 + post eval + failure)
    _hook_runner = HookRunner()
    _approval_cfg = llm_cfg.get("approval", {})
    # H-2 C.4 Session A: claw ネイティブ read_file/write_file/edit_file/
    # glob_search/grep_search に Noetic 固有の secrets guard + profile 外書込禁止
    # を Pre-hook で被せる (legacy _read_file/_write_file/_list_files の代替)
    # 段階12 Step 2 (PLAN §3-2) で「sandbox 外書込禁止」→「profile 外書込禁止」に拡張済
    _hook_runner.register_pre(make_file_access_guard(BASE_DIR))
    # 段階12 Step 3 (PLAN §5): G-2 自動 stash hook。core/* / tools/* /
    # main.py / .mcp.json への write_file / edit_file の直前に profile 配下を
    # git stash で partial 保存、身体改変履歴の safety net + 20 世代 auto drop。
    # git 未初期化環境では noop で安全に通過。
    _hook_runner.register_pre(make_git_auto_stash_hook(BASE_DIR, BASE_DIR.name))
    # bash は Level-aware validation で Level 0-2 では read-only 系のみ、
    # 破壊的コマンドは Level 問わず自動拒否、WARN は承認画面に警告付き表示
    _hook_runner.register_pre(make_bash_validation_hook(
        state_getter=lambda: state,
    ))
    # 段階12 Step 7.5 ② (PLAN §3-5-2 ②): bash 経由の絶対パス profile 外
    # 書込み / 削除系操作を deny。bash_validation の Level 制約に加え、
    # path 境界で「書込み range = profile 内」を構造化。
    _hook_runner.register_pre(make_bash_path_guard_hook(BASE_DIR))
    # 段階12 Step 7.5 ③ (PLAN §3-5-2 ③): pip install で PyPI registry 検証。
    # 存在しない pkg は deny (LLM hallucination 抑止)、人気 pkg と Levenshtein
    # 1-2 は typosquatting 警告。1 時間 in-memory cache、不通時 warning + 続行。
    _hook_runner.register_pre(make_install_command_check_hook())
    # Approval callback (pause_on_await + 3 層 UI + smoke auto_approve_all
    # + 段階13 Phase 6.2 AgentSpec DSL policy_fn)
    # PLAN §5-4: settings.json approval.rules から policy_fn を生成、
    # tool_name + path のみで approval 要否を判定 (LLM 由来 field 影響ゼロ)
    from core.runtime.approval_rules import make_policy_fn
    # ★ workspace_root=BASE_DIR は **本番運用必須** (Codex rescue 7 周目
    # RISK-6 評価): None だと path_resolver が raw fallback で canonical 化
    # されず、path traversal / Windows case / whitespace 等 path quirk family
    # で bypass 発生。test 後方互換 (None) は単体 test 限定。
    _policy_fn = make_policy_fn(
        rules=_approval_cfg.get("rules", []),
        default_action=_approval_cfg.get("default", "approve"),
        workspace_root=BASE_DIR,
    )
    _hook_runner.register_pre(make_pre_tool_use_approval_check(
        missing_field_policy=_approval_cfg.get("missing_field_policy", "deny"),
        auto_approve_all=_approval_cfg.get("auto_approve_all", False),
        policy_fn=_policy_fn,
    ))

    # eval (LLM3 / E 値評価) は claude_code provider 経由なら settings.json
    # の model_overrides で role 別モデル切替が効く。partial で role="llm3"
    # bound 版を渡すことで eval.py / hooks.py 等の call site 側を変えずに
    # settings から制御できる (2026-04-28 hotfix、Sonnet が
    # constraint_circumvention 等の自己観察ラベルに refusal 連発する問題への対策。
    # 他 provider 経路では role 引数自体を伝播しないため自動的に無視される)。
    _base_post_hook = make_post_tool_use_evaluation(
        state=state,
        get_state_before=lambda: _hook_ctx["state_before"],
        call_llm_fn=partial(call_llm, role="llm3"),
        get_cycle_id=lambda: state.get("cycle_id", 0),
        # 段階13 Phase 1: filter 順序 (filter → slice) を non_empty_subjective_entries
        # 経由で集約 (raw event 起源 blank subj 混入時に LLM3 post 評価 context が空に
        # なる同型 bug fix、Codex review 2 周目指摘 + 横断 grep 発見)。
        get_recent_intents=lambda: [
            e["intent"] for e in non_empty_subjective_entries(
                state.get("subjective_entries", [])
            )[-3:] if e.get("intent")
        ],
    )

    def _post_hook_with_sync(tool_name, tool_input, output, tool_id=None):
        """tool 実行直後: disk から fresh state を in-place 取込 →
        base_post_hook で mutation → save_state で永続化。
        tool handler が内部で save_state した変更と hook の E 値等の
        mutation を正しくマージする。採点完了後の E を runtime の tool_id と
        保存する。各 runtime 呼出しで記録をリセットし、ID が欠落・重複して
        一意に対応しない実行は欠測とする。"""
        _refresh_state()
        result = _base_post_hook(tool_name, tool_input, output)
        save_state(state)
        if (not result.failed and not result.denied
                and state.get("e_values", {}).get("scored") is True):
            _hook_ctx["evaluations"].append(
                (tool_id, tool_name, copy.deepcopy(state.get("e_values", {})))
            )
        return result

    _hook_runner.register_post(_post_hook_with_sync, with_tool_id=True)
    # 段階12 Step 5 (PLAN §9): 身体改変反映待ち pending 自動追加。
    # write_file / edit_file が core/* / tools/* / main.py / .mcp.json を
    # 成功書換えしたら「reboot で反映を完了する」内発的 intent を pending 化。
    # match_pattern={"tool_name": "reboot"} で reboot 成功時に自己消化。
    _hook_runner.register_post(make_post_body_modify_pending_hook(
        state_getter=lambda: state,
        get_cycle_id=lambda: state.get("cycle_id", 0),
    ))
    _hook_runner.register_failure(make_post_tool_use_failure_logger(
        state=state,
        get_cycle_id=lambda: state.get("cycle_id", 0),
    ))

    _approval_cb = make_approval_callback(
        pause_on_await=_approval_cfg.get("pause_on_await", True),
        auto_approve_all=_approval_cfg.get("auto_approve_all", False),
        policy_fn=_policy_fn,
    )

    from core.prompt_trace import TracingProvider
    _rt_provider = TracingProvider(_rt_provider)
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

    # 観察者専用。state / prompt / 次回 G 用 snapshot には載せない。
    _outward_seq = 0
    _outward_fire_no = 0
    _outward_pending_inputs = []

    def _record_outward_input(channel, source):
        nonlocal _outward_seq
        try:
            _outward_seq += 1
            _outward_pending_inputs.append({
                "seq": _outward_seq, "channel": channel, "source": source,
                "cycle_at_record": state.get("cycle_id", 0),
            })
        except Exception as e:
            print(f"  [outward] input observation skip: {e}")

    def _observe_outward(event, phase, payload=None):
        """観察失敗は本体に伝播させない。発話と入力は同じ seq を使う。"""
        nonlocal _outward_seq
        try:
            from core.metrics import (
                observe_outward_candidates, observe_outward_selection,
                build_outward_execution, OUTWARD_UTTERANCE_TOOLS,
            )
            if phase == "candidates":
                observe_outward_candidates(event, payload)
            elif phase == "selection":
                observe_outward_selection(event)
            elif phase == "executions":
                planned_tool, invocations = payload
                for rec in invocations:
                    observed = build_outward_execution(
                        rec.tool_name, rec.tool_input or {}, rec.output,
                        rec.is_error, _get_channel(rec.tool_name),
                    )
                    event["exec"].append(observed)
                    if observed["status"] == "ok" and rec.tool_name in OUTWARD_UTTERANCE_TOOLS:
                        _outward_seq += 1
                        event["utterances"].append({
                            "seq": _outward_seq, "channel": observed["channel"],
                            "tool": rec.tool_name,
                        })
                if not any(rec.tool_name == planned_tool for rec in invocations):
                    event["exec"].append({"tool": planned_tool, "channel": _get_channel(planned_tool),
                                          "status": "not_invoked", "connected_at_enqueue": None})
                if not invocations:
                    event["end"] = "not_invoked"
            elif phase in ("runtime_error", "not_invoked"):
                event["end"] = phase
                event["exec"].append({"tool": payload, "channel": _get_channel(payload),
                                      "status": phase, "connected_at_enqueue": None})
        except Exception as e:
            print(f"  [outward] observation skip ({phase}): {e}")

    def _run_observed_fire(*args, **kwargs):
        """fire 出口で1回だけ emit。入力バッファは書き出し成功時だけ消費する。"""
        nonlocal _outward_fire_no
        _outward_fire_no += 1
        event = {
            "event_type": "outward_attempt", "run_id": state.get("run_id", ""),
            "attempt_id": f"{state.get('run_id', '')}_{_outward_fire_no}",
            "cycle_id": state.get("cycle_id", 0),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end": "error", "proposed_act": False, "proposed_sense": False,
            "proposed_undetermined": 0, "selected_act": False,
            "best_outward_g_minus_selected_g": None,
            "exec": [], "utterances": [], "inputs": list(_outward_pending_inputs),
        }
        # 0d-1: 全実行の理由・引数・結果の観察バッファ。entry_id は entry 成立後に入る。
        invocations = {"records": [], "entry_id": None, "cycle_id": None}
        try:
            from core.prompt_trace import trace_fire
            with trace_fire(event):
                result = _run_one_fire(*args, **kwargs, _outward_event=event,
                                       _invocation_buf=invocations)
            if isinstance(result, dict) and result.get("llm1_error"):
                event["end"] = "llm1_error"
            elif event["end"] == "error":
                event["end"] = "completed" if result and result.get("executed") else "not_invoked"
            return result
        finally:
            try:
                from core.metrics import emit_outward_attempt
                emit_outward_attempt(event)
                del _outward_pending_inputs[:len(event["inputs"])]
            except Exception as e:
                print(f"  [outward] emit skip: {e}")
            try:
                if invocations["records"]:
                    from core.metrics import build_tool_invocation_events, emit_tool_invocations
                    emit_tool_invocations(build_tool_invocation_events(
                        invocations["records"], run_id=event["run_id"],
                        attempt_id=event["attempt_id"],
                        entry_id=invocations["entry_id"], cycle_id=invocations["cycle_id"],
                        time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ))
            except Exception as e:
                print(f"  [invocation] emit skip: {e}")

    def _run_one_fire(fire_cause, _tunnel_fire, pp, threshold, tick_dt,
                      _micro_iter=0, fire_candidates=None, _outward_event=None,
                      _invocation_buf=None):
        """1 fire iteration 本体。micro-loop から複数回呼ばれる可能性。

        state / _runtime / _hook_ctx / _pending_observations / TOOLS /
        LEVEL_TOOLS / llm_cfg / BASE_DIR / prompt_budget / その他 import は
        main() closure で参照 (state は in-place mutate 運用)。
        pressure は nonlocal で main() のものを直接 mutate。

        Returns:
            dict {"executed": bool, "e_available": bool,
                  "e1"-"e4": str ("N%") or None, "sc_bonus": float,
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
        ctrl = controller(state, TOOLS, LEVEL_TOOLS)
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
                if llm_cfg.get("x_login", {}).get("skip_prompt", False):
                    print("  [X] x_login.skip_prompt=true、自動 skip")
                    answer = "n"
                else:
                    try:
                        answer = input("  Xにログインする？ [y/N]: ").strip().lower()
                    except EOFError:
                        answer = "n"
                if answer == "y":
                    _x_do_login()
                else:
                    print("  [X] スキップ。X系ツールはセッションなしで動作しません。")
                print()
        print(f"  ctrl: level={new_lv} tools={sorted(allowed)} subj={len(state['subjective_entries'])}件(全件)")

        # ① LLM: 候補提案
        propose_prompt = build_prompt_propose(state, ctrl, TOOLS, fire_cause,
                                               fire_candidates=fire_candidates,
                                               registry=_rt_registry)

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
                "perspective": make_perspective(),  # 段階11-A: system 処理は self/actual
            }
            event_emitter.fire_event(state, _sys_end_entry)
            save_state(state)

        try:
            from core.prompt_trace import trace_llm1
            propose_resp = trace_llm1(call_llm, propose_prompt, cycle_id=state.get("cycle_id", 0),
                                    max_tokens=prompt_budget["completion_reserve"], temperature=1.0,
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
        if _outward_event is not None:
            _observe_outward(_outward_event, "candidates", candidates)
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
        if _outward_event is None:
            selected = controller_select(candidates, ctrl, state)
        else:
            selected = controller_select(candidates, ctrl, state, observe=_outward_event)
            _observe_outward(_outward_event, "selection")
        _sel_line = f"  選択: {selected['tool']} - {selected['reason'][:60]}"
        # 段階14 Step C: penalty (β 値含む) を smoke raw_log で観察可能化
        # (memo line 142-144 「β を第一級観測量として扱う」literal 整合、
        # Codex review P2-2 fix)
        _sel_penalties = selected.get("penalties") or []
        if _sel_penalties:
            _sel_line += f" [{', '.join(_sel_penalties)}]"
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
            fire_candidates=fire_candidates,
            # Step 0.4 hotfix: prompt 用は displayed_tools (affordance 前)、
            # parser 用は allowed_tools (affordance 後)。description は表示、選択は弾く。
            allowed_tools=ctrl.get("displayed_tools", ctrl["allowed_tools"]),
            world_model=state.get("world_model"),
            registry=_rt_registry,
        )
        _runtime.session.clear()

        for _obs in _pending_observations:
            _runtime.session.push_observation(**_obs)
        _pending_observations.clear()

        chain_tools = selected.get("tools", [selected["tool"]])
        from core.metrics import tool_execution_status
        all_results = []
        all_tool_names = []
        intent = ""
        expect = ""
        _chain_action_key = ""
        parse_failed = False
        prev_result = ""
        _executed_targets = set()
        _last_llm_text = ""
        first_args: dict = {}  # 段階8 改善1: 先頭 tool の args を log entry に保存
        # 採点 hook が実行ごとに保存した E を per_tool に対応づける。
        # 1 cycle = 1 log entry + entry["per_tool"] に tool 単位 metadata (Y 方式)。
        per_tool_entries: list = []
        invocation_entries: list = []
        _successful_tools = set()
        _cycle_ev = None
        _cycle_pt = None

        for chain_idx, chain_tool in enumerate(chain_tools):
            if chain_tool not in ctrl["allowed_tools"]:
                if _outward_event is not None:
                    _observe_outward(_outward_event, "not_invoked", chain_tool)
                print(f"  (Controller却下: {chain_tool})")
                parse_failed = f"却下: {chain_tool}"
                break

            user_input_parts = [selected["reason"]]
            if chain_idx > 0 and prev_result:
                user_input_parts.append(f"(前のツール結果: {prev_result[:200]})")
            user_input = " ".join(user_input_parts)

            # 段階11-D Step 4-2 hotfix v6 (ゆう原案 2026-04-28): chain 軸の
            # 二段構造を導入。chain_idx==0 は controller 決定 (LLM① 選択 tool) を
            # 必ず実行する forced 段階 (run_turn_with_forced_tool 経由)。
            # chain_idx>=1 は LLM 自由探索として run_turn (通常版) に切替えて、
            # tool 呼ばない選択も含む LLM のブレを多様性として受容する。
            #
            # Phase 8 hotfix ② の max_iterations 軸二段構造 (iter 0 = forced /
            # iter 1+ = 自由) を chain 軸に拡張した形。chain_idx>=1 では
            # default system_prompt (全 tool 視界) で LLM が自由に判断するため、
            # forced_sp 生成も chain 0 のみで十分。
            # memory/feedback_llm2_iter0_forced_contract.md 参照。
            _hook_ctx["evaluations"] = []
            try:
                from core.prompt_trace import set_chain_position
                set_chain_position(chain_idx, state.get("cycle_id", 0))
                if chain_idx == 0:
                    forced_sp = assemble_system_prompt(
                        state=state,
                        tools_dict=TOOLS,
                        fire_cause=fire_cause,
                        fire_candidates=fire_candidates,
                        allowed_tools={chain_tool},
                        world_model=state.get("world_model"),
                        registry=_rt_registry,
                        force_tool=chain_tool,
                    )
                    summary = _runtime.run_turn_with_forced_tool(
                        forced_tool_name=chain_tool,
                        user_input=user_input,
                        forced_system_prompt=forced_sp,
                    )
                else:
                    summary = _runtime.run_turn(user_input=user_input)
            except Exception as e:
                print(f"  LLM② run_turn エラー (chain {chain_idx+1}): {e}")
                if _outward_event is not None:
                    _observe_outward(_outward_event, "runtime_error", chain_tool)
                parse_failed = f"runtime error: {e}"
                break

            if _outward_event is not None:
                _observe_outward(_outward_event, "executions", (chain_tool, summary.tool_invocations))
            _inv_last = None
            if _invocation_buf is not None:
                try:
                    for _i, _r in enumerate(summary.tool_invocations):
                        _inv_last = {
                            "chain_position": chain_idx, "invocation_position": _i,
                            "tool_id": _r.tool_id, "tool": _r.tool_name,
                            "mode": "forced" if chain_idx == 0 else "free",
                            "tool_input": copy.deepcopy(_r.tool_input or {}),
                            "output": str(_r.output), "is_error": _r.is_error,
                            "in_cycle_result": False,
                        }
                        _invocation_buf["records"].append(_inv_last)
                except Exception as e:
                    _inv_last = None
                    print(f"  [invocation] observation skip: {e}")

            # 段階9 Step 0: LLM② debug log を拡充。
            # 従来は finish_reason だけ。assistant_messages (LLM② 思考) と
            # tool_invocations (選択 tool + 完全 args) も記録して、E値の上の
            # セクションが追跡可能になるようにする。
            _llm2_text = (summary.assistant_messages[-1].text
                          if summary.assistant_messages else "")
            if summary.tool_invocations:
                _rec = summary.tool_invocations[-1]
                try:
                    _tool_call_str = f"{_rec.tool_name}({json.dumps(_rec.tool_input, ensure_ascii=False)})"
                except (TypeError, ValueError):
                    _tool_call_str = f"{_rec.tool_name}({_rec.tool_input!r})"
            else:
                _tool_call_str = "(no tool invocation)"
            append_debug_log(
                f"LLM2 via runtime (chain {chain_idx+1}/{len(chain_tools)})",
                f"finish_reason={summary.finish_reason}\n"
                f"text={_llm2_text}\n"
                f"tool_call={_tool_call_str}",
            )
            if summary.assistant_messages:
                _last_llm_text += (summary.assistant_messages[-1].text or "")

            if not summary.tool_invocations:
                print(f"  (tool 未実行: finish_reason={summary.finish_reason})")
                parse_failed = f"no_tool: {summary.finish_reason}"
                break

            # 各段の先頭だけを chain[i] に対応させる。名前一致かつ採点完了が必要。
            # 追加実行は E のみ保存し、先頭の失敗・拒否・別名・欠測を繰り上げない。
            for invocation_idx, invocation in enumerate(summary.tool_invocations):
                _input = invocation.tool_input or {}
                invocation_entries.append({
                    "tool": invocation.tool_name, "tool_id": invocation.tool_id,
                    "chain_position": chain_idx, "invocation_position": invocation_idx,
                    "args": copy.deepcopy({k: v for k, v in _input.items()
                                           if k not in ("tool_intent", "tool_expected_outcome", "note")}),
                    "status": tool_execution_status(str(invocation.output), invocation.is_error),
                    "in_cycle_result": False,
                    "result": cap_tool_result(str(invocation.output)),
                })
                _matches = [ev for tid, name, ev in _hook_ctx["evaluations"]
                            if tid == invocation.tool_id and name == invocation.tool_name]
                _unique_id = (bool(invocation.tool_id) and
                              sum(r.tool_id == invocation.tool_id
                                  for r in summary.tool_invocations) == 1)
                _ok = (not invocation.is_error and
                       not str(invocation.output).startswith("[REJECTED]"))
                _tool_ev = _matches[0] if _ok and _unique_id and len(_matches) == 1 else None
                pt_entry = {
                    "tool": invocation.tool_name,
                    "tool_id": invocation.tool_id,
                    "chain_position": chain_idx,
                    "invocation_position": invocation_idx,
                    "is_error": invocation.is_error,
                    "intent": str(_input.get("tool_intent", "") or ""),
                    "expect": str(_input.get("tool_expected_outcome", "") or ""),
                    **{k: _tool_ev.get(k) if _tool_ev is not None else None
                       for k in ("e1", "e2", "e2_raw", "e3", "e4", "eff")},
                }
                _chain_list = selected.get("chain")
                _chain_item = (_chain_list[chain_idx]
                               if isinstance(_chain_list, list) and chain_idx < len(_chain_list)
                               else None)
                if (invocation_idx == 0 and _tool_ev is not None
                        and isinstance(_chain_item, dict)
                        and _chain_item.get("tool") == invocation.tool_name):
                    _tool_pe2 = _chain_item.get("predicted_e2")
                    _tool_pec = _chain_item.get("predicted_ec")
                    if isinstance(_tool_pe2, int):
                        pt_entry["predicted_e2"] = _tool_pe2
                        _pt_e2_m = re.search(r'\d+', str(_tool_ev.get("e2", "")))
                        if _pt_e2_m:
                            _pt_actual_e2 = max(0, min(100, int(_pt_e2_m.group(0))))
                            pt_entry["actual_e2"] = _pt_actual_e2
                            pt_entry["prediction_error"] = abs(_tool_pe2 - _pt_actual_e2)
                    if isinstance(_tool_pec, (int, float)):
                        from core.predictor import clamp_ec as _clamp_ec
                        pt_entry["predicted_ec"] = float(_tool_pec)
                        pt_entry["actual_ec"] = _clamp_ec(_tool_ev.get("eff", 0.0))
                        if "prediction_error" in pt_entry:
                            pt_entry["prediction_error_ec"] = abs(
                                float(_tool_pec) - pt_entry["actual_ec"]
                            )
                per_tool_entries.append(pt_entry)
                if _ok:
                    _successful_tools.add(invocation.tool_name)
                    # 最後の成功が採点未完了なら代表 E も欠測。前の採点を流用しない。
                    _cycle_ev = _tool_ev
                    _cycle_pt = pt_entry

            rec = summary.tool_invocations[-1]
            ti = rec.tool_input or {}

            if chain_idx == 0:
                intent = str(ti.get("tool_intent", "") or "")
                expect = str(ti.get("tool_expected_outcome", "") or "")
                _chain_action_key = _extract_action_key(rec.tool_name, ti)
                # 段階8 改善1: 承認 3 層フィールドを除いた tool 固有 args を保存
                first_args = {
                    k: v for k, v in ti.items()
                    if k not in ("tool_intent", "tool_expected_outcome", "note")
                }

            _target_id = (ti.get("reply_to_id") or ti.get("post_id")
                          or ti.get("tweet_url") or ti.get("path") or "")
            _exec_key = (rec.tool_name, _target_id)
            if _target_id and _exec_key in _executed_targets:
                print(f"  (重複スキップ: {rec.tool_name} target={str(_target_id)[:20]})")
                continue
            if _target_id:
                _executed_targets.add(_exec_key)

            prev_result = str(rec.output)[:500]
            # 参照は JSON のバイト位置ではなく、結合した Python 文字列内の位置。
            _result_start = sum(len(part) for part in all_results) + len("\n---\n") * len(all_results)
            _result_start += len(f"[{rec.tool_name}]\n")
            all_results.append(f"[{rec.tool_name}]\n{cap_tool_result(str(rec.output))}")
            invocation_entries[-1]["in_cycle_result"] = True
            invocation_entries[-1]["result_ref"] = {
                "start": _result_start, "length": len(invocation_entries[-1]["result"]),
            }
            if _inv_last is not None:
                _inv_last["in_cycle_result"] = True
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
            # 段階11-D Phase 0 Step 0.4: memory_graph affordance ガード (B2)
            # 自発 memory_store ≥ 1 経験で memory_graph candidate 解除 (controller.py で gate)
            # 失敗 memory_store は count しない (Z2 確定、森のたとえ整合性)
            elif rec.tool_name == "memory_store":
                if not _out_str.startswith(("Error", "エラー")):
                    state["voluntary_memory_store_count"] = (
                        state.get("voluntary_memory_store_count", 0) + 1
                    )
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
        for invocation in invocation_entries:
            ref = invocation.get("result_ref")
            if ref is not None:
                if ref["start"] + ref["length"] <= len(result_str):
                    del invocation["result"]
                else:
                    del invocation["result_ref"]

        if intent:
            print(f"  intent: {intent}")
        if expect:
            print(f"  expect: {expect}")

        sc_bonus = calc_state_change_bonus(_hook_ctx["state_before"], state)

        if "output_display" in _successful_tools:
            _uec = state.get("unresponded_external_count", 0)
            if _uec > 0:
                state["unresponded_external_count"] = _uec - 1
            if state.get("unresponded_external_count", 0) <= 0:
                state["unresolved_external"] = 0.0
                state["unresponded_external_count"] = 0
            save_state(state)

        _ev = _cycle_ev if _cycle_ev is not None else {}
        e1 = _ev.get("e1")
        e2 = _ev.get("e2")
        e3 = _ev.get("e3")
        e4 = _ev.get("e4")
        eff_change = float(_ev.get("eff", 0.0) or 0.0)
        _target_for_ec = _chain_action_key.split(":", 1)[1] if ":" in _chain_action_key else ""

        if e1 or e2 or e3 or e4:
            _ec_str = f" ec={eff_change:.2f}" if eff_change < 0.5 else ""
            print(f"  E1={e1} E2={e2} E3={e3} E4={e4}{_ec_str}")

        if _cycle_ev is not None:
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

        # 段階14 Step D: basin migration 検知 (reflect 経路 / 非 reflect 経路両対応、
        # PLAN §6-2 literal、memo line 56「basin_id 流用元」整合)。
        # reflect 発火時 reflection.py が state に snapshot 保存、ここで pop で消費
        # (1 cycle 限定、永続化しない、Phase 5「非永続 posterior」整合)。
        from core.world_model import update_basin_state
        _clusters_snap = state.pop("last_clusters_snapshot", None)
        _subject_id = state.pop("last_subject_memory_id", None)
        update_basin_state(
            state,
            current_subject_id=_subject_id,
            clusters_snapshot=_clusters_snap,
        )
        # Codex review P1-2 fix: basin migration を smoke raw_log で観察可能化
        # (PLAN §7 K6「basin_state log」literal)。basin 切替 (dwell=1) or pending=True
        # の状態のみ log (毎 cycle 出力じゃなく観察 trigger、smoke noise 抑制)。
        _bs = state.get("basin_state", {})
        _bid = _bs.get("current_basin_id", "")
        _bdwell = _bs.get("current_basin_dwell", 0)
        _bpending = _bs.get("phase_transition_pending", False)
        if _bid and (_bpending or _bdwell == 1):
            _basin_line = (
                f"  [basin] id={_bid[:8]} dwell={_bdwell} pending={_bpending}"
            )
            print(_basin_line)
            broadcast_log(_basin_line)

        cid = state.get("cycle_id", 0) + 1
        state["cycle_id"] = cid
        entry = {
            "id": f"{state.get('session_id','x')}_{cid:04d}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool": tool_name,
            "channel": _get_channel(tool_name),
            "result": result_str,
            "invocations": invocation_entries,
            "perspective": make_perspective(),  # 段階11-A: iku 自身の tool 行動 → self/actual
        }
        if parse_failed:
            entry["parse_error"] = str(parse_failed)[:150]
        if intent:
            entry["intent"] = intent
        if expect:
            entry["expect"] = expect
        if first_args:
            entry["args"] = first_args
        # 段階11-A Step 7: output_display の to_perspective を log entry に刻む。
        # from は entry["perspective"] (self/actual、Step 2 で付与済)、
        # to は args.channel を viewer にした perspective。
        if tool_name == "output_display" and isinstance(first_args, dict):
            _ch = first_args.get("channel")
            if isinstance(_ch, str) and _ch.strip():
                entry["to_perspective"] = make_perspective(
                    viewer=_ch.strip(), viewer_type="actual",
                )
        if e1:
            entry["e1"] = e1
        if e2:
            entry["e2"] = e2
        if e3:
            entry["e3"] = e3
        if e4:
            entry["e4"] = e4
        # cycle の E と同じ実行の予測だけを使用する。先頭の予測で代用しない。
        if _cycle_ev is not None and _cycle_pt is not None:
            for key in ("predicted_e2", "actual_e2", "prediction_error",
                        "predicted_ec", "actual_ec", "prediction_error_ec"):
                if key in _cycle_pt:
                    entry[key] = _cycle_pt[key]
            if "prediction_error" in entry:
                state["last_prediction_error"] = entry["prediction_error"]

        # 段階10.5 Fix 1: 1 cycle = 1 entry + per_tool に tool 単位 metadata (Y 方式)。
        # update_predictor_confidence を chain ループ後の per_tool loop で tool 単位 call。
        # chain 連結文字列 "tool_A+tool_B" ではなく個別 tool 名で β+ 更新される。
        if per_tool_entries:
            entry["per_tool"] = per_tool_entries
            from core.predictor import update_predictor_confidence
            for pt in per_tool_entries:
                if pt.get("prediction_error") is not None:
                    update_predictor_confidence(
                        state, pt["tool"],
                        pt["prediction_error"],
                        pt.get("prediction_error_ec"),
                    )

        event_emitter.fire_event(state, entry)
        if _invocation_buf is not None:
            _invocation_buf["entry_id"] = entry["id"]
            _invocation_buf["cycle_id"] = cid

        maybe_compress_log(state, set(TOOLS.keys()))

        # UPS v2 pending 淘汰
        pending_prune(state, current_cycle=cid)

        # 段階13 Phase 3: TE + Bayesian maintenance (経路 B、毎 cycle 自動実行)
        # graph 全 link の transfer entropy + Beta posterior 更新。
        # PLAN §4-6 重畳: 既存 EMA + 新 Bayesian_TE (α_old/α_new = 0.5/0.5)。
        # reflect 継続原則で例外は catch、メイン処理を止めない。
        try:
            from core.transfer_entropy import maybe_te_maintenance
            maybe_te_maintenance(state, current_cycle=cid)
        except Exception as e:
            print(f"  [transfer_entropy] maintenance skip (error: {e})")

        # 段階11-D Phase 3 Step 3.5: 弱 link の prune (Physarum rule、cycle 終端 batch)
        # idle >= β 半減期 × 4 (≒56 cycle) かつ strength < initial × 0.15 で削除。
        # reflect 継続原則で例外は catch、メイン処理を止めない。
        try:
            from core.memory_links import prune_weak_links
            prune_weak_links(current_cycle=cid)
        except Exception as e:
            print(f"  [memory_links] prune skip (error: {e})")

        # Slice 2: cycle metrics emit (cycle 末、save_state 直前)
        # orchestration §② ゆう確定 emit timing。controller 不変、測定のみ。
        # reflect 継続原則: 例外で metrics 失敗しても cycle 全体は止めない。
        try:
            from core.metrics import emit_cycle_metrics
            emit_cycle_metrics(state, llm_cfg, load_pref())
        except Exception as e:
            print(f"  [metrics] emit skip (error: {e})")

        save_state(state)

        return {
            "executed": True,
            "e_available": _cycle_ev is not None,
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
                # 段階13 Phase 0.1.D: entropy/spiral は tool (raw) + intent (subj) 両層 = merge
                from core.state import merge_log_view as _merge_log_view
                _merged_log = _merge_log_view(state)
                main._cached_measured = calc_measured_entropy(state, _merged_log)
                main._cached_spiral = calc_spiral_vector(state, _merged_log)
                # behavioral_entropy: ツール使用分布の情報エントロピー（tool は raw 直読みで OK）
                from collections import Counter as _Counter
                _recent_tools = [e.get("tool", "unknown") for e in state.get("raw_events", [])[-20:]]
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
            # hotfix 2026-05-12 (Slice 6.5 と独立): clock_base 配線。config.py:71 で
            # DEFAULT_PRESSURE_PARAMS に "clock_base": 0.15 が定義されてたが、本 pressure 更新式に
            # 配線されていなかった (git log -S "clock_base" で 822c0cd commit が定義追加のみ、
            # main.py への配線 commit 0 件確認、= 最初から未配線の placeholder)。
            # 「時間経過だけでも少しずつ圧が溜まる」internal drive baseline として配線、
            # E2=E3=100% 連発などで signal 弱化 (s/u/n 軸ゼロ化) しても pressure が threshold に
            # 到達する path を保証する (feedback_internal_drive literal「外部刺激なしでも動く」整合)。
            pressure = pressure * pp.get("decay", 0.97) + signal_total + pp.get("clock_base", 0.0)
            threshold = calc_dynamic_threshold(state, base_threshold)

            # 外部入力チェック（chatキューからstate.logに注入 + pressure加算 + archive）
            # WM 段階6-C v3: channel spec を channel_registry から引き、spec["tools_in"][0] を tool tag に
            for chat_text in get_pending_chats():
                _refresh_state()
                from core.channel_registry import channel_from_device_input
                from core.world_model import ensure_channel, observe_channel_activity
                _spec = channel_from_device_input()
                _channel_id = _spec["id"]
                _record_outward_input(_channel_id, "chat")
                _input_tag = _spec["tools_in"][0] if _spec["tools_in"] else f"[{_channel_id}_input]"

                _wm = state.get("world_model")
                if _wm is not None:
                    ensure_channel(_wm, **_spec)
                    observe_channel_activity(_wm, _channel_id)

                _ext_id = f"{state.get('session_id','?')}_ext{int(time.time()*1000)%100000}"
                _ext_entry = {
                    "id": _ext_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tool": _input_tag,
                    "type": "external",
                    "channel": _channel_id,
                    "result": chat_text,
                    # 段階11-A G3: 外部入力の viewer は channel_id (channel_registry が
                    # Noetic 主体で決定済の中立 id、feedback_no_user_assistant_frame 整合)
                    "perspective": make_perspective(viewer=_channel_id, viewer_type="actual"),
                }
                event_emitter.fire_event(state, _ext_entry)
                # 未応答カウンター (pressure 経路で AI に応答を促す)
                state["unresponded_external_count"] = state.get("unresponded_external_count", 0) + 1
                # 段階8 改善5 (案 5-A): 外部入力 → iku 内部応答意図 pending 化。
                # match_pattern で output_display (channel 一致時) 消化対応。
                pending_add_response_intent(
                    state=state, channel=_channel_id,
                    text=chat_text, cycle_id=state.get("cycle_id", 0),
                )
                # Step E-3c: Session buffer に積む (fire 開始時に流される)
                _pending_observations.append({
                    "observed_channel": _channel_id,
                    "content": chat_text,
                    "source_action_hint": "living_presence",
                    "observation_time": datetime.now().strftime("%H:%M"),
                })
                save_state(state)
                pressure += 3.0  # 外部入力はpressureを即座に上げる
                # 段階9 fix 2-a: channel 値を併記し、LLM が tool.args.channel
                # に同じ値を渡す推論を繋げやすくする (console/UI 視界も整合)。
                _chat_line = f"  {_input_tag} (channel={_channel_id}) {chat_text[:80]}"
                print(_chat_line)
                broadcast_log(_chat_line)

            # WM 段階6-C v3: pending_mcp_inputs.jsonl consume (MCP client 経由の外部入力)
            # MCP server が書き込み → main.py が cycle 頭で consume + unlink。
            # record に channel_spec があればそれを使う (server 判定済)、
            # なければ client_name から channel_from_mcp_client で fallback 生成。
            _mcp_input_file = BASE_DIR / "pending_mcp_inputs.jsonl"
            if _mcp_input_file.exists():
                try:
                    _lines = _mcp_input_file.read_text(encoding="utf-8").splitlines()
                    _mcp_input_file.unlink()
                    for _line in _lines:
                        if not _line.strip():
                            continue
                        _refresh_state()
                        _rec = _json.loads(_line)
                        # channel_spec 決定 (server 判定優先、なければ registry で生成)
                        from core.channel_registry import channel_from_mcp_client
                        from core.world_model import ensure_channel, observe_channel_activity
                        _spec = _rec.get("channel_spec") or channel_from_mcp_client(
                            _rec.get("client_name", ""))
                        _channel_id = _spec["id"]
                        _record_outward_input(_channel_id, "mcp")
                        _input_tag = _spec["tools_in"][0] if _spec["tools_in"] else f"[{_channel_id}_input]"

                        _wm = state.get("world_model")
                        if _wm is not None:
                            ensure_channel(_wm, **_spec)
                            observe_channel_activity(_wm, _channel_id)

                        _ext_id = _rec.get(
                            "id",
                            f"mcp_{uuid.uuid4().hex[:12]}"
                        )
                        _ext_entry = {
                            "id": _ext_id,
                            "time": _rec.get(
                                "time",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "tool": _input_tag,
                            "type": "external",
                            "channel": _channel_id,
                            "result": _rec.get("content", ""),
                            # 段階11-A G3: MCP 経由の外部入力も channel_id を viewer に。
                            # device_input 側と同じ方針で対称性維持。
                            "perspective": make_perspective(viewer=_channel_id, viewer_type="actual"),
                        }
                        event_emitter.fire_event(state, _ext_entry)
                        state["unresponded_external_count"] = (
                            state.get("unresponded_external_count", 0) + 1
                        )
                        # 段階8 改善5 (案 5-A): MCP 経由の外部入力 → 応答意図 pending
                        pending_add_response_intent(
                            state=state, channel=_channel_id,
                            text=_rec.get("content", ""),
                            cycle_id=state.get("cycle_id", 0),
                        )
                        _pending_observations.append({
                            "observed_channel": _channel_id,
                            "content": _rec.get("content", ""),
                            "source_action_hint": "living_presence",
                            "observation_time": datetime.now().strftime("%H:%M"),
                        })
                        save_state(state)
                        pressure += 3.0
                        # 段階9 fix 2-a: channel 値を併記 (device_input 側と同じ理由)。
                        _line_msg = (
                            f"  {_input_tag} (channel={_channel_id}) "
                            f"{_rec.get('content','')[:80]}"
                        )
                        print(_line_msg)
                        broadcast_log(_line_msg)
                except Exception as _e:
                    print(f"  [mcp_input consume エラー: {_e}]")

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
                        "perspective": make_perspective(),  # 段階11-A: test タブ実行も Noetic 内部扱いで self/actual
                    }
                    event_emitter.fire_event(state, _test_entry)
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
        # 段階13 Phase 4 commit 4 (PLAN §3-1 + §5-5 同時発火階層 literal):
        # pressure 軸 + graph 軸の両軸並走 fire_candidates list を生成、LLM② に
        # selection 委譲 (PLAN §5-5「selection = △ 1 cycle 1 つ、LLM② 判断」literal)。
        # scalar fire_cause は既存 logic 維持 (PLAN spec 対象外、log line / state 用)。
        try:
            from core.dynamic_composition import (
                compute_graph_maturity,
                compute_graph_axis_candidates,
                W_PRESSURE,
            )
            # PLAN §3-1 literal: pressure_axis.candidates(weight=w_pressure)
            pressure_candidates = [
                {"name": k, "score": float(v) * W_PRESSURE, "kind": "pressure"}
                for k, v in (signals or {}).items()
            ]
            w_graph = compute_graph_maturity(state)
            graph_candidates = compute_graph_axis_candidates(state, w_graph)
            fire_candidates = pressure_candidates + graph_candidates
        except Exception as e:
            print(f"  [graph_axis] skip (error: {e})")
            fire_candidates = [
                {"name": k, "score": float(v), "kind": "pressure"}
                for k, v in (signals or {}).items()
            ]

        # primary scalar fire_cause: 既存 logic 維持 (pressure 軸内 max のみ)。
        # selection は LLM② が fire_candidates list を見て判断 (PLAN §5-5 literal)。
        # scalar は log line / state["fire_cause"] 用 metadata のみ、selection の代理ではない。
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

            # tunnel 発火時は fire_candidates=None で渡す: 既存 scalar 経路
            # [発火原因: tunnel] が prompt 表示される (Codex review P2-1 fix、tunnel
            # は PLAN STAGE13 対象外の既存 Noetic 特殊 trigger、backward compat 維持)。
            _fire_result = _run_observed_fire(fire_cause, _tunnel_fire, pp,
                                          threshold, tick_dt, _micro_iter,
                                          fire_candidates=None if _tunnel_fire else fire_candidates)

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
            pressure = max(0.0, pressure * pp.get("post_fire_reset", 0.3))
            if _last_fire_result.get("e_available", False):
                def _e_to_float(e_str):
                    m = re.search(r'(\d+)', str(e_str))
                    return int(m.group(1)) / 100.0 if m else 0.5
                e1_val = _e_to_float(_last_fire_result["e1"])
                e2_val = _e_to_float(_last_fire_result["e2"])
                e3_val = _e_to_float(_last_fire_result["e3"])
                e4_val = _e_to_float(_last_fire_result["e4"]) if _last_fire_result["e4"] else 0.5
                _sc_bonus = _last_fire_result["sc_bonus"]
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
            # 段階11-C hotfix (2026-04-24): 段階11-C smoke 3 段目 baseline で
            # reflect 自動発火ゼロを観察、root cause は「incremented 値が disk
            # に書かれず、次 tool 実行の _refresh_state() で memory 上の +1
            # が毎回消えて reflection_cycle が永遠に 1 止まり」だった。
            # reflect 発火時のみ save (L1274 reset 時) に依存していた race 条件
            # (LLM が reflect tool chain を選ぶ偶発起動が初回書込契機として必要)。
            # 累積値を increment 直後に永続化、reflect tool chain 選択の運に
            # 独立して自動発火経路が機能するようにする。副作用: disk I/O が
            # cycle あたり 1 回増える (同 state object で実害なし)。
            save_state(state)
            _refl_interval = load_pref().get("reflection_interval", 10)
            if should_reflect(state, _refl_interval):
                print("  [reflection] 内省開始...")
                # F-005 完了後 hotfix (2026-05-09): tool 経路と同 helper に集約
                # して reflection_cycle リセット + save_state を 1 経路化。
                # tool 経路で reflect 走った後の同 cycle で should_reflect が
                # True になることはない (state["reflection_cycle"] = 0 が直接
                # 反映されるため) → 二重発火構造的に起きない。
                reflect_and_persist(state, call_llm)

            # 段階11-B Phase 5 Step 5.4: cycle 境界で emergence metric 記録 (reflection 後 state snapshot)
            try:
                from core.tag_emergence_monitor import log_cycle_metrics
                log_cycle_metrics(state.get("cycle_id", 0), state)
            except Exception as _e:
                print(f"  [emergence] log skip: {_e}")

        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Ctrl+C] 終了します。")
        sys.exit(0)
