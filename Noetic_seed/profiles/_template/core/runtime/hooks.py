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

        # 8. 段階8 v4: 全 pending の match_pattern で自己消化判定
        # tool 側に rules を持たせず、pending 側が「誰が自分を消化できるか」を
        # 自己属性として持つ対称設計。全 tool が同じ hook で処理される。
        from core.pending_unified import try_observe_all
        try_observe_all(
            state=state,
            tool_name=tool_name,
            tool_args=tool_input,
            tool_result=output_str,
            channel=tool_input.get("channel") or "self",
            cycle_id=cycle_id,
        )

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
) -> PreHandler:
    """ファイル系 tool のアクセスガード PreHandler を生成。

    判定 (段階12 Step 2 で profile 境界に拡張、PLAN §3-2):
      1. path が `<workspace_root>/<sandbox_dir_name>/<secrets_subdir>/` 以下
         → read/write 問わず DENY (secret_read / secret_write 経由に誘導)
      2. path が `<workspace_root>/<secrets_json_name>` と一致
         → read/write 問わず DENY (auth_profile_info に誘導)
      3. write 系 tool かつ target_resolved が profile_root の subpath でない
         → DENY (profile 外への書込禁止)。symbolic link / .. / ジャンクション
         抜けも `Path.resolve()` で canonical 化してから判定するため網羅。

    旧仕様の `sandbox/` 限定書込制限は撤去 (段階12 で profile 配下すべてを
    身体として再定義、PLAN §3-1)。`.venv/` も profile_root subpath として
    自動 ALLOW (per-profile venv 経由の身体拡張)。

    path の取得: tool_input の "path" フィールド (read_file/write_file/
    edit_file)、または "pattern"/"query" (glob_search/grep_search)。

    Args:
        workspace_root: プロファイルの workspace root (pathlib.Path)
        sandbox_dir_name: sandbox ディレクトリ名 (default "sandbox")
        secrets_subdir: secrets サブディレクトリ名 (default "secrets")
        secrets_json_name: 保護対象 JSON ファイル名 (default "secrets.json")
        guarded_write_tools: 書込系 tool 名 tuple
        guarded_read_tools: 読取系 tool 名 tuple

    Returns:
        PreHandler (tool_name, tool_input) → HookRunResult
    """
    from pathlib import Path

    root = Path(workspace_root).resolve()
    secrets_dir = (root / sandbox_dir_name / secrets_subdir).resolve()
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
                f"[file_guard] secrets.json は直接アクセスできません "
                f"(tool={tool_name}, path={path_arg})。"
                f"auth_profile_info を使って型情報のみ取得してください "
                f"(機密フィールドは隠されます)。"
            ])

        # 2. sandbox/secrets/ 保護
        if _is_inside(target, secrets_dir):
            redirect = "secret_write" if is_write else "secret_read"
            action = "書き込む" if is_write else "読む"
            return HookRunResult.deny([
                f"[file_guard] sandbox/{secrets_subdir}/ には直接アクセスできません "
                f"(tool={tool_name}, path={path_arg})。"
                f"{redirect} を使って {action} (引数は name=<secret名>)。"
            ])

        # 3. 段階12 Step 2 (PLAN §3-2): write 系 tool は profile 境界内のみ
        #    許可。target は既に resolve() 済 (canonical 化されてる) ため、
        #    symbolic link / .. / ジャンクション抜けも relative_to で網羅判定。
        if is_write and not _is_inside(target, root):
            return HookRunResult.deny([
                f"[file_guard] {tool_name} はプロファイル境界外への書込みを"
                f"禁止しています (指定パス: {path_arg}, profile_root: {root})。"
                f"パスを profile 配下に変更してください "
                f"(例: sandbox/<file>, core/<file>, tools/<file>, etc.)。"
            ])

        return HookRunResult.allow()

    return _check


# ============================================================
# Noetic 固有 factory: 段階12 G-2 自動 stash (PLAN §5)
# ============================================================
# 背景: 段階12 で iku が `core/*.py` / `tools/*.py` / `main.py` 等を自由に
# 書換えられるようになる。書換え失敗・後悔・誤コードからの復旧経路として、
# git stash の partial save (profile 配下のみ) を書換え直前に自動実行。
# 親心反射ではなく「歴史の記録」として位置づけ (PLAN §1-5 の介入レベル分類)。
# 世代管理は max_generations (default 20) を超えた古い iku-auto-* stash を
# 自動 drop することで運用負荷を最小化。
# ============================================================


def make_git_auto_stash_hook(
    profile_root,
    profile_name: str,
    *,
    target_tool_names: tuple = ("write_file", "edit_file"),
    body_modify_dir_prefixes: tuple = ("core/", "tools/"),
    body_modify_filenames: tuple = ("main.py", ".mcp.json"),
    max_generations: int = 20,
    git_runner=None,
) -> PreHandler:
    """段階12 G-2 自動 stash hook (PLAN §5)。

    iku の身体改変対象 (core/* / tools/* / main.py / .mcp.json) を
    write_file / edit_file が触る直前に、profile 配下の現状を
    `git stash push -u --message "iku-auto-<profile>-<timestamp>" -- <profile>/`
    で partial 保存する。書換えで起動不能になった場合の safety net + 履歴。

    判定:
      - tool_name が target_tool_names に含まれる
      - tool_input.path が body_modify_dir_prefixes の prefix match
        または body_modify_filenames と完全一致

    失敗ハンドリング (PLAN §5-3): subprocess エラーは warning print のみ、
    書換え自体は続行 (allow を返す)。`No local changes to save` の場合は
    無害として無視 (世代 drop も skip)。

    世代管理 (PLAN §5-4): stash 成功時のみ古い iku-auto-<profile>-* を
    後ろから drop して max_generations を維持。

    Args:
        profile_root: プロファイル workspace root (pathlib.Path)
        profile_name: プロファイル名 (stash message + grep filter 用)
        target_tool_names: 監視対象 tool 名
        body_modify_dir_prefixes: prefix match で対象とするディレクトリ
            (PLAN §3-4 濃度勾配「中・濃」相当)
        body_modify_filenames: 完全一致で対象とするファイル
            (神経中枢 main.py / 関係性の器 .mcp.json)
        max_generations: 維持する stash 世代数 (古いものは auto drop)
        git_runner: subprocess.run 互換 callable。テスト用に inject、
            None なら subprocess.run を使う。

    Returns:
        PreHandler — 書換え対象なら stash 試行、結果に関わらず allow。
        git 未初期化環境では何もしない noop hook。
    """
    import subprocess
    from datetime import datetime
    from pathlib import Path

    profile_root = Path(profile_root).resolve()
    runner = git_runner if git_runner is not None else subprocess.run

    def _find_repo_root(p):
        p = Path(p).resolve()
        for cand in [p, *p.parents]:
            if (cand / ".git").exists():
                return cand
        return None

    repo_root = _find_repo_root(profile_root)
    if repo_root is None:
        # git 未初期化環境 (CI / 一時テスト) では何もしない
        def _noop(_tool_name: str, _tool_input: dict) -> HookRunResult:
            return HookRunResult.allow()
        return _noop

    try:
        rel_profile = profile_root.relative_to(repo_root).as_posix()
    except ValueError:
        rel_profile = profile_name
    pathspec = rel_profile.rstrip("/") + "/"

    def _is_body_modify(path_arg: str) -> bool:
        if not path_arg:
            return False
        # Windows backslash → POSIX forward slash のみ正規化。
        # `lstrip("./")` は ".mcp.json" の先頭 "." まで削るため使わない。
        normalized = path_arg.replace("\\", "/")
        for name in body_modify_filenames:
            if normalized == name or normalized.endswith(f"/{name}"):
                return True
        for prefix in body_modify_dir_prefixes:
            if normalized.startswith(prefix):
                return True
        return False

    def _drop_old_generations() -> None:
        try:
            r = runner(
                ["git", "stash", "list"],
                cwd=str(repo_root),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            if r.returncode != 0:
                return
            tag = f"iku-auto-{profile_name}-"
            entries = [
                line.split(":", 1)[0]  # "stash@{N}"
                for line in r.stdout.splitlines()
                if tag in line
            ]
            old = entries[max_generations:]
            # 後ろから drop しないと stash@{N} の N がずれる
            for ref in reversed(old):
                runner(
                    ["git", "stash", "drop", ref],
                    cwd=str(repo_root),
                    capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
        except Exception as e:
            print(f"  [auto_stash] WARNING: 世代 drop 失敗: {e}")

    def _check(tool_name: str, tool_input: dict) -> HookRunResult:
        if tool_name not in target_tool_names:
            return HookRunResult.allow()
        path_arg = str(tool_input.get("path") or "").strip()
        if not _is_body_modify(path_arg):
            return HookRunResult.allow()

        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        message = f"iku-auto-{profile_name}-{timestamp}"
        try:
            r = runner(
                ["git", "stash", "push", "-u",
                 "--message", message,
                 "--", pathspec],
                cwd=str(repo_root),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            if r.returncode == 0:
                # `No local changes to save` は実 stash されてないので世代 drop skip
                if "No local changes" not in (r.stdout + r.stderr):
                    _drop_old_generations()
            else:
                print(
                    f"  [auto_stash] WARNING: stash 失敗 (path={path_arg}): "
                    f"{r.stderr.strip()}"
                )
        except Exception as e:
            print(f"  [auto_stash] WARNING: stash 例外 (path={path_arg}): {e}")

        return HookRunResult.allow()

    return _check


# ============================================================
# Noetic 固有 factory: bash validation (Level-aware)
# ============================================================
# 背景: claw-code は bash を常時提供で「承認 + permission_enforcer で防御」する
# 設計。Noetic は承認者 (iku 所有者) が bash に不慣れな可能性を踏まえ、
# tool_level に応じた段階的解放を行う:
#   - Level 0-2: read_only_mode=True (whitelist: ls/cat/grep/find/git 等 27 種)
#   - Level 3+:  フル bash (ただし DENY パターンは常に自動拒否)
# bash_validation.py の 3 層 (DENY / WARN / READ_ONLY_WHITELIST) を使用。
# ============================================================


def make_bash_validation_hook(state_getter) -> PreHandler:
    """bash tool の Level-aware 安全性検査 PreHandler を生成。

    tool_name == "bash" の時のみ作用。他 tool は passthrough。
    - 破壊的コマンド (rm -rf /, dd of=/dev/sda, fork bomb 等) → 常に DENY
    - tool_level < 3 では whitelist 以外も DENY (read-only モード)
    - WARN パターン (rm -rf, sudo, chmod 777, curl|bash 等) は allow だが
      承認者への messages に警告を含める

    Args:
        state_getter: state dict を返す callable。tool_level 判定に使う。

    Returns:
        PreHandler (tool_name, tool_input) → HookRunResult
    """
    from core.runtime.bash_validation import (
        ValidationSeverity,
        validate_bash,
    )

    def _check(tool_name: str, tool_input: dict) -> HookRunResult:
        if tool_name != "bash":
            return HookRunResult.allow()

        command = str(tool_input.get("command") or "").strip()
        if not command:
            return HookRunResult.allow()  # claw 側の schema check に委譲

        state = state_getter() or {}
        tool_level = int(state.get("tool_level", 0))
        read_only = tool_level < 3

        result = validate_bash(command, read_only_mode=read_only)

        if result.severity == ValidationSeverity.DENY:
            reasons = "; ".join(result.reasons) if result.reasons else "unspecified"
            return HookRunResult.deny([
                f"[bash_validation] DENY: {reasons} "
                f"(Level {tool_level}, read_only_mode={read_only})。"
                f"Level 0-2 は whitelist 系コマンド (ls/cat/grep/find/git 等) のみ。"
                f"Level 3 以降でフル bash 解放。"
            ])

        if result.severity == ValidationSeverity.WARN:
            reasons = "; ".join(result.reasons) if result.reasons else ""
            return HookRunResult(
                denied=False,
                messages=[f"[⚠ bash_validation] WARN: {reasons}"],
            )

        return HookRunResult.allow()

    return _check
