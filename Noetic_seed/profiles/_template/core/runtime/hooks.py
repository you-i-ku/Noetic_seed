"""Hooks — PreToolUse / PostToolUse / PostToolUseFailure.

claw-code の rust/crates/runtime/src/hooks.rs:22-36 の Python port。

ファイル構成:
  - 上半分 (HookEvent / HookRunResult / HookRunner):
      claw-code 準拠の純粋インフラ。settings 非依存。
  - 下半分 (Noetic 固有 factory 群):
      承認 3 層チェッカー等、Phase 4 で追加される Noetic 固有 handler。
      settings 依存は factory 引数で受け取り、main.py 側で HookRunner に
      register する。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class HookEvent(Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    POST_TOOL_USE_FAILURE = "post_tool_use_failure"


@dataclass
class HookRunResult:
    """hook の戻り値。handler 複数分の集約結果。"""
    denied: bool = False
    failed: bool = False
    permission_override: Optional[str] = None
    updated_input: Optional[dict] = None
    messages: list = field(default_factory=list)

    @classmethod
    def allow(cls, messages: Optional[list] = None) -> "HookRunResult":
        return cls(denied=False, messages=messages or [])

    @classmethod
    def deny(cls, messages: Optional[list] = None) -> "HookRunResult":
        return cls(denied=True, messages=messages or [])

    def merge(self, other: "HookRunResult") -> "HookRunResult":
        return HookRunResult(
            denied=self.denied or other.denied,
            failed=self.failed or other.failed,
            permission_override=(other.permission_override
                                 or self.permission_override),
            updated_input=(other.updated_input
                           if other.updated_input is not None
                           else self.updated_input),
            messages=self.messages + other.messages,
        )


PreHandler = Callable[[str, dict], HookRunResult]
PostHandler = Callable[[str, dict, str], HookRunResult]
FailureHandler = Callable[[str, dict, str], HookRunResult]


class HookRunner:
    """hook event を発火して、登録された handler を順次呼ぶ。"""

    def __init__(self):
        self._pre: list = []
        self._post: list = []
        self._fail: list = []

    def register_pre(self, handler: PreHandler) -> None:
        self._pre.append(handler)

    def register_post(self, handler: PostHandler) -> None:
        self._post.append(handler)

    def register_failure(self, handler: FailureHandler) -> None:
        self._fail.append(handler)

    def run_pre_tool_use(self, tool_name: str,
                         tool_input: dict) -> HookRunResult:
        acc = HookRunResult.allow()
        current_input = tool_input
        for handler in self._pre:
            try:
                r = handler(tool_name, current_input)
            except Exception as e:
                return HookRunResult(
                    denied=False, failed=True,
                    messages=acc.messages + [f"pre_hook exception: {e}"],
                )
            acc = acc.merge(r)
            if r.updated_input is not None:
                current_input = r.updated_input
            if acc.denied:
                break
        if current_input is not tool_input:
            acc.updated_input = current_input
        return acc

    def run_post_tool_use(self, tool_name: str, tool_input: dict,
                          output: str) -> HookRunResult:
        acc = HookRunResult.allow()
        for handler in self._post:
            try:
                r = handler(tool_name, tool_input, output)
            except Exception as e:
                return HookRunResult(
                    failed=True,
                    messages=acc.messages + [f"post_hook exception: {e}"],
                )
            acc = acc.merge(r)
        return acc

    def run_post_tool_use_failure(self, tool_name: str, tool_input: dict,
                                  error: str) -> HookRunResult:
        acc = HookRunResult.allow()
        for handler in self._fail:
            try:
                r = handler(tool_name, tool_input, error)
            except Exception as e:
                return HookRunResult(
                    failed=True,
                    messages=acc.messages + [f"failure_hook exception: {e}"],
                )
            acc = acc.merge(r)
        return acc


# ============================================================
# Noetic 固有 handler (Phase 4 追加)
# ------------------------------------------------------------
# 承認 3 層 (tool_intent / tool_expected_outcome / message) の欠損
# チェックを PreToolUse hook として登録するための factory。
# APPROVAL_PROMPT_SPEC.md §5 の仕様を実装する。
# ============================================================

_APPROVAL_FIELDS = ("tool_intent", "tool_expected_outcome", "message")
_VALID_MISSING_POLICIES = ("deny", "warn", "auto_fill")


def _auto_fill_field(tool_name: str, field_name: str) -> str:
    """tool_name と欠損フィールド名からプレースホルダ文字列を生成。

    Phase 4 Step A 時点の最小実装。将来 tool メタデータベースの推定に
    差し替え可能 (APPROVAL_PROMPT_SPEC.md §9 未決定事項)。
    """
    if field_name == "tool_intent":
        return f"[auto_fill] {tool_name} 実行"
    if field_name == "tool_expected_outcome":
        return f"[auto_fill] {tool_name} の結果取得"
    if field_name == "message":
        return f"[auto_fill] {tool_name} を実行します"
    return "[auto_fill]"


def make_pre_tool_use_approval_check(
    missing_field_policy: str = "deny",
) -> PreHandler:
    """承認 3 層欠損チェッカーを生成。

    tool_input が `tool_intent` / `tool_expected_outcome` / `message`
    を揃えているか検証する PreToolUse hook を返す。

    factory にしている理由: hooks.py を claw-code 準拠の純粋インフラに
    保つため、settings 依存は factory 引数で注入する。main.py 側が
    `settings.approval.missing_field_policy` を読んで register する。

    Args:
        missing_field_policy: 欠損時の動作。"deny" / "warn" / "auto_fill"

    Raises:
        ValueError: 未知の policy が渡された場合 (設定ミスは起動時に検出)
    """
    if missing_field_policy not in _VALID_MISSING_POLICIES:
        raise ValueError(
            f"unknown missing_field_policy={missing_field_policy!r}; "
            f"expected one of {_VALID_MISSING_POLICIES}"
        )

    def _check(tool_name: str, tool_input: dict) -> HookRunResult:
        missing = []
        for field_name in _APPROVAL_FIELDS:
            value = tool_input.get(field_name, "")
            if value is None or not str(value).strip():
                missing.append(field_name)

        if not missing:
            return HookRunResult.allow()

        if missing_field_policy == "deny":
            return HookRunResult.deny([
                f"[approval] tool_input 欠損: {', '.join(missing)}。"
                "tool_intent / tool_expected_outcome / message の 3 層を"
                "揃えて再生成してください。"
            ])

        if missing_field_policy == "warn":
            return HookRunResult(
                denied=False,
                messages=[
                    f"[approval] warning: 欠損フィールド {missing} "
                    "(missing_field_policy=warn のため実行継続)"
                ],
            )

        updated = dict(tool_input)
        for field_name in missing:
            updated[field_name] = _auto_fill_field(tool_name, field_name)
        return HookRunResult(
            denied=False,
            updated_input=updated,
            messages=[f"[approval] 自動補完: {missing}"],
        )

    return _check


# ============================================================
# Noetic 固有 handler — PostToolUse 評価 (Phase 4 Step B)
# ------------------------------------------------------------
# 既存 Noetic の E1-E4 評価 / effective_change / E2 cap /
# Action Ledger / unresolved_intent 更新を PostToolUse hook に
# 配置する。内部ロジック (eval.py の関数群) は一切触らず、
# 呼出場所だけ hook 層に移植する (INTEGRATION_POINTS.md §1.2)。
# ============================================================


def _pct_str(value: float) -> str:
    """0.0-1.0 の float を '70%' 形式に変換。update_unresolved_intents の
    e3_str 引数や state["e_values"] 表示に使う既存 Noetic の流儀。"""
    return f"{int(round(max(0.0, min(1.0, value)) * 100))}%"


def make_post_tool_use_evaluation(
    state: dict,
    get_state_before: Callable[[], dict],
    call_llm_fn: Callable[..., str],
    get_cycle_id: Callable[[], int],
    get_recent_intents: Callable[[], list],
) -> PostHandler:
    """E1-E4 評価 + effective_change + E2 cap + Action Ledger +
    unresolved_intent 更新を一括で行う PostToolUse hook を生成。

    内部は eval.py の既存関数を順次呼ぶ薄い wrapper。戻り値 state は
    factory に渡された参照を破壊的に更新する (Noetic 既存流儀)。

    Args:
        state: メイン state への参照。hook 内で破壊的に更新される。
        get_state_before: tool 実行**前**の state snapshot を返す関数。
            main.py が tool 呼出前に deepcopy した snapshot を保持し、
            この closure で取り出す想定。
        call_llm_fn: LLM 呼出関数。eval_with_llm の 3 引数目に渡される。
            signature: (prompt: str, max_tokens: int, temperature: float) -> str
        get_cycle_id: 現在の cycle_id を返す関数。
        get_recent_intents: 直近 intent list を返す関数。eval_with_llm
            の「recent_intents」引数に使う (最近の行動文脈)。

    Returns:
        PostHandler ((tool_name, tool_input, output) -> HookRunResult)

    Note:
        tool_intent / tool_expected_outcome は tool_input から取る
        (Step A の PreToolUse hook で存在が保証されている前提)。
        欠損 (policy=warn で抜けた等) 時も crash せず空文字で進む。
    """
    from core import eval as _eval

    def _handler(tool_name: str, tool_input: dict,
                 output: str) -> HookRunResult:
        intent = str(tool_input.get("tool_intent", "") or "")
        expect = str(tool_input.get("tool_expected_outcome", "") or "")
        cycle_id = get_cycle_id()
        state_before = get_state_before()
        recent_intents = get_recent_intents() or []
        output_str = str(output)

        # 1. E1-E4 評価 (LLM 0.7 + vec 0.3 ブレンド、失敗時 None)
        scores = _eval.eval_with_llm(
            intent, expect, output_str, recent_intents, call_llm_fn
        ) or {}
        e1 = float(scores.get("e1", 0.5))
        e2_raw = float(scores.get("e2", 0.5))
        e3 = float(scores.get("e3", 0.5))
        e4 = float(scores.get("e4", 0.5))

        # 2. effective_change (5 層)
        target_id = ""
        for key in ("reply_to_id", "tweet_url", "post_id"):
            val = tool_input.get(key, "")
            if val:
                target_id = str(val)
                break
        eff = _eval.calc_effective_change(
            tool_names=[tool_name],
            tool_result=output_str,
            state_before=state_before,
            state_after=state,
            current_intent=intent,
            target_id=target_id,
        )

        # 3. E2 cap: (0.3 + eff*0.7)
        e2 = _eval.apply_effective_change_to_e2(e2_raw, eff)

        # 4. Action Ledger
        action_key = _eval._extract_action_key(tool_name, tool_input)
        _eval.append_action_ledger(
            state=state, tool_name=tool_name, action_key=action_key,
            intent=intent, result=output_str, ec=eff, cycle_id=cycle_id,
        )

        # 5. unresolved_intent 更新 (rate-distortion 容量管理)
        # Step C-2 以降: UPS v2 形式で pending に追加 (source_action=tool_name,
        # lag_kind="cycles", semantic_merge=True)。内部ロジックは不変。
        e3_str = _pct_str(e3)
        _eval.update_unresolved_intents(
            state=state, intent=intent, e3_str=e3_str, cycle_id=cycle_id,
            source_action=tool_name, lag_kind="cycles",
        )

        # 6. 既存 unresolved_intent の gap を relevance で減衰
        _eval.update_gaps_by_relevance(
            state=state, result_str=output_str, ec=eff,
        )

        # 7. state["e_values"] に最新スコアを保存 (Noetic 既存慣習)
        state["e_values"] = {
            "e1": _pct_str(e1),
            "e2": _pct_str(e2),
            "e2_raw": _pct_str(e2_raw),
            "e3": e3_str,
            "e4": _pct_str(e4),
            "eff": round(eff, 4),
        }

        return HookRunResult.allow(messages=[
            f"[post_eval] tool={tool_name} "
            f"e2={_pct_str(e2)} e3={e3_str} e4={_pct_str(e4)} "
            f"eff={eff:.3f}"
        ])

    return _handler


def make_post_tool_use_failure_logger(
    state: dict,
    get_cycle_id: Callable[[], int],
    max_entries: int = 20,
) -> FailureHandler:
    """tool 実行失敗時のエラー記録 hook (簡易版)。

    state['tool_errors'] に追記。上限 max_entries 件で古いものから捨てる。
    E 値評価には走らない (失敗した tool は effective_change が測れない)。
    """
    def _handler(tool_name: str, tool_input: dict,
                 error: str) -> HookRunResult:
        from datetime import datetime
        log = state.setdefault("tool_errors", [])
        log.append({
            "tool": tool_name,
            "error": str(error)[:500],
            "intent": str(tool_input.get("tool_intent", "") or "")[:200],
            "cycle": get_cycle_id(),
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        if len(log) > max_entries:
            state["tool_errors"] = log[-max_entries:]
        return HookRunResult.allow(messages=[
            f"[post_fail] tool={tool_name} error={str(error)[:80]}"
        ])

    return _handler


# ============================================================
# Noetic 固有 factory: ファイルアクセスガード (H-2 C.4 Session A)
# ============================================================
# 背景: H-2 で read_file / write_file / list_files を claw ネイティブ
# (file_ops.read_file / write_file / glob_search) に切替える際、legacy
# 版にあった Noetic 固有のセキュリティガード (secrets.json / sandbox/secrets/
# の保護、sandbox/ 外書込禁止) を PreToolUse hook で再現する。
#
# 将来: memory/project_unified_file_ops_future.md の Phase 5 統合時に
# permission ルール (arg pattern match) 拡張と合わせて settings 化可能。
# ============================================================


def make_file_access_guard(
    workspace_root,
    *,
    sandbox_dir_name: str = "sandbox",
    secrets_subdir: str = "secrets",
    secrets_json_name: str = "secrets.json",
    guarded_write_tools: tuple = ("write_file", "edit_file"),
    guarded_read_tools: tuple = ("read_file", "edit_file", "glob_search", "grep_search"),
    require_write_in_sandbox: bool = True,
) -> PreHandler:
    """ファイル系 tool のアクセスガード PreHandler を生成。

    判定:
      1. path が `<workspace_root>/<sandbox_dir_name>/<secrets_subdir>/` 以下
         → read/write 問わず DENY (secret_read / secret_write 経由に誘導)
      2. path が `<workspace_root>/<secrets_json_name>` と一致
         → read/write 問わず DENY (auth_profile_info に誘導)
      3. tool が guarded_write_tools に含まれ、require_write_in_sandbox=True
         かつ path が sandbox/ 以下でない → DENY (sandbox 外書込禁止)

    path の取得: tool_input の "path" フィールド (read_file/write_file/
    edit_file)、または "pattern"/"query" (glob_search/grep_search)。

    Args:
        workspace_root: プロファイルの workspace root (pathlib.Path)
        sandbox_dir_name: sandbox ディレクトリ名 (default "sandbox")
        secrets_subdir: secrets サブディレクトリ名 (default "secrets")
        secrets_json_name: 保護対象 JSON ファイル名 (default "secrets.json")
        guarded_write_tools: 書込系 tool 名 tuple
        guarded_read_tools: 読取系 tool 名 tuple
        require_write_in_sandbox: True で write_file/edit_file の対象を
            sandbox/ 以下に限定

    Returns:
        PreHandler (tool_name, tool_input) → HookRunResult
    """
    from pathlib import Path

    root = Path(workspace_root).resolve()
    secrets_dir = (root / sandbox_dir_name / secrets_subdir).resolve()
    sandbox_root = (root / sandbox_dir_name).resolve()
    secrets_json = (root / secrets_json_name).resolve()

    def _resolve(path_str: str):
        if not path_str:
            return None
        try:
            return (root / path_str).resolve()
        except Exception:
            return None

    def _is_inside(target, base) -> bool:
        try:
            target.relative_to(base)
            return True
        except ValueError:
            return False

    def _check(tool_name: str, tool_input: dict) -> HookRunResult:
        if tool_name not in set(guarded_read_tools) | set(guarded_write_tools):
            return HookRunResult.allow()

        # tool 別に path / pattern 引数を取得
        if tool_name in ("glob_search", "grep_search"):
            path_arg = str(tool_input.get("path") or tool_input.get("pattern") or "").strip()
        else:
            path_arg = str(tool_input.get("path") or "").strip()

        if not path_arg:
            return HookRunResult.allow()

        target = _resolve(path_arg)
        if target is None:
            return HookRunResult.allow()  # claw 側の boundary check に委譲

        is_write = tool_name in guarded_write_tools

        # 1. secrets.json 保護
        if target == secrets_json:
            return HookRunResult.deny([
                f"[file_guard] secrets.json は直接アクセスできません。"
                f"auth_profile_info を使ってください (tool={tool_name})"
            ])

        # 2. sandbox/secrets/ 保護
        if _is_inside(target, secrets_dir):
            redirect = "secret_write" if is_write else "secret_read"
            return HookRunResult.deny([
                f"[file_guard] sandbox/{secrets_subdir}/ には直接アクセスできません。"
                f"{redirect} を使ってください (tool={tool_name})"
            ])

        # 3. 書込系 tool は sandbox/ 以下限定 (self_modify が legacy 経由で
        #    main.py/pref.json を更新するのは別経路なので影響なし)
        if is_write and require_write_in_sandbox:
            if not _is_inside(target, sandbox_root):
                return HookRunResult.deny([
                    f"[file_guard] {tool_name} は sandbox/ 以下にのみ書き込めます "
                    f"(path={path_arg})"
                ])

        return HookRunResult.allow()

    return _check
