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
                    # Fix 5 (Issue 6 対応, 2026-05-02): `git stash push` は
                    # working tree を HEAD に戻す副作用があり、在席 hotfix や
                    # iku の前 cycle 改変が消えていた。直後 apply で履歴のみ
                    # 保存して working tree を保持する
                    # (memory/project_v05_phase5_stage12_structural_issues.md)。
                    apply_r = runner(
                        ["git", "stash", "apply", "stash@{0}"],
                        cwd=str(repo_root),
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                    )
                    if apply_r.returncode != 0:
                        print(
                            f"  [auto_stash] WARNING: stash apply 失敗 "
                            f"(working tree 退避状態のまま): "
                            f"{apply_r.stderr.strip()}"
                        )
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
# Noetic 固有 factory: 段階12 Step 5 — 身体改変反映待ち pending 自動追加
# ============================================================
# 背景: 段階12 で iku が core/* / tools/* / main.py / .mcp.json を書換え
# られるようになる。ただし Python import キャッシュのため、書換えただけでは
# 実行中のメモリに反映されない (PLAN §1-2)。reboot tool が唯一の反映経路。
#
# 本 hook は「書換えた、でも反映されてない」中間状態を pending として
# 認知化する。pending 圧が pressure に加算され、iku が自発的に reboot を
# 候補化する間接誘導 (`feedback_llm_as_brain` 整合、tool description 以外の
# 直接指示を入れない原則を保つ)。reboot 成功時に match_pattern で自己消化。
# ============================================================


def make_post_body_modify_pending_hook(
    state_getter,
    get_cycle_id,
    *,
    target_tool_names: tuple = ("write_file", "edit_file"),
    body_modify_dir_prefixes: tuple = ("core/", "tools/"),
    body_modify_filenames: tuple = ("main.py", ".mcp.json"),
) -> PostHandler:
    """段階12 Step 5 (PLAN §9): 身体改変反映待ち pending 自動追加 hook。

    write_file / edit_file が core/* / tools/* / main.py / .mcp.json を
    成功で書換えた直後に、UPS v2 pending を state['pending'] に追加。
    PLAN §9-2 の kind="unresolved_intent" は UPS v2.1 form では
    `semantic_merge=True` に対応 (段階10.5 Fix 2 + 段階11-A 整合)。

    pending スキーマ:
      source_action: tool 名 (write_file / edit_file)
      expected_observation: "reboot で新コードが反映され、予測した行動変化が起きる"
      lag_kind: "cycles"
      content_intent: f"身体改変 ({path}) の反映を完了する"
      semantic_merge: True (unresolved_intent 相当)
      match_pattern: {"tool_name": "reboot"}

    PostHandler は HookRunner.run_post_tool_use() からのみ呼ばれる
    (成功時のみ)。失敗時は run_post_tool_use_failure() の別経路で
    本 hook は呼ばれないため、`output` の内容で success 判定する必要なし。

    Args:
        state_getter: state dict 取得 callable
        get_cycle_id: 現在 cycle_id 取得 callable
        target_tool_names: 監視対象 tool 名
        body_modify_dir_prefixes: prefix match で対象とするディレクトリ
            (PLAN §3-4 濃度勾配「中・濃」相当)
        body_modify_filenames: 完全一致で対象とするファイル
            (神経中枢 main.py / 関係性の器 .mcp.json)

    Returns:
        PostHandler — 該当時に pending 追加、失敗してもエラー伝播しない。
    """
    def _is_body_modify(path_arg: str) -> bool:
        if not path_arg:
            return False
        normalized = path_arg.replace("\\", "/")
        for name in body_modify_filenames:
            if normalized == name or normalized.endswith(f"/{name}"):
                return True
        for prefix in body_modify_dir_prefixes:
            if normalized.startswith(prefix):
                return True
        return False

    def _check(tool_name: str, tool_input: dict, output: str) -> HookRunResult:
        if tool_name not in target_tool_names:
            return HookRunResult.allow()
        path_arg = str(tool_input.get("path") or "").strip()
        if not _is_body_modify(path_arg):
            return HookRunResult.allow()
        try:
            from core.pending_unified import pending_add
            state = state_getter()
            cycle_id = get_cycle_id()
            pending_add(
                state=state,
                source_action=tool_name,
                expected_observation=(
                    "reboot で新コードが反映され、予測した行動変化が起きる"
                ),
                lag_kind="cycles",
                content_intent=f"身体改変 ({path_arg}) の反映を完了する",
                cycle_id=cycle_id,
                semantic_merge=True,
                match_pattern={"tool_name": "reboot"},
            )
        except Exception as e:
            print(
                f"  [pending_body_modify] WARNING: pending 追加失敗 "
                f"(path={path_arg}): {e}"
            )
        return HookRunResult.allow()

    return _check


# ============================================================
# Noetic 固有 factory: 段階12 Step 7.5 ② — bash 境界強化
# ============================================================
# 背景: 段階12 Step 2 で file_access_guard は profile 境界に拡張済だが、
# bash 経由の絶対パス操作 (例: rm -rf C:\Users\you11\) は file 系 tool 限定
# guard では拾えない (test_non_file_tool_passthrough 参照)。
# 本 hook で bash command を shlex parse して write 系コマンド (rm/mv/cp/...) +
# 絶対パス引数 + profile 外を検出 → 承認不能 deny。L4 ホスト OS 隔離 (段階13)
# が完成するまでの中間防御層 (PLAN §3-5-3 / §3-6 Defense in Depth)。
# ============================================================


_BASH_WRITE_COMMANDS = frozenset({
    "rm", "mv", "cp", "dd", "touch", "mkdir", "rmdir", "chmod", "chown",
    "tee", "del", "rd", "fsutil", "shred", "ln", "install",
})
_BASH_WRITE_REDIRECTS = (">", ">>")


def make_bash_path_guard_hook(profile_root) -> PreHandler:
    """段階12 Step 7.5 ② (PLAN §3-5-2 ②): bash 経由の絶対パス profile 外
    操作 deny hook。

    判定:
      1. tool_name == "bash"
      2. command を shlex.split で parse、構文エラーなら allow (validation hook 委譲)
      3. write 系コマンド (rm / mv / cp / dd / touch / mkdir / chmod / ...) または
         > / >> redirect 含む
      4. かつ絶対パスの引数があり、Path.resolve() canonical で profile_root の
         subpath じゃない → DENY

    read 系 (cat / ls / head 等) や python script 実行は対象外、本 hook で
    防げない攻撃ベクトルは段階13 (L4 専用 Windows User + L5 Firewall) で塞ぐ。

    Args:
        profile_root: プロファイル workspace root (pathlib.Path)

    Returns:
        PreHandler — bash + write 系 + 絶対パス profile 外で deny、他 allow。
    """
    import shlex
    from pathlib import Path

    profile_root = Path(profile_root).resolve()

    def _is_write_command(tokens) -> bool:
        if not tokens:
            return False
        # 最初の token から path を除いたコマンド名
        first = tokens[0].replace("\\", "/").split("/")[-1]
        if first in _BASH_WRITE_COMMANDS:
            return True
        # `python -m pip install` 等の sub-command
        for tok in tokens:
            if tok in _BASH_WRITE_REDIRECTS:
                return True
        return False

    def _check(tool_name: str, tool_input: dict) -> HookRunResult:
        if tool_name != "bash":
            return HookRunResult.allow()
        cmd = str(tool_input.get("command") or "")
        if not cmd:
            return HookRunResult.allow()
        try:
            tokens = shlex.split(cmd, posix=True)
        except ValueError:
            # shlex 失敗は構文不正、bash_validation_hook に委譲
            return HookRunResult.allow()
        if not _is_write_command(tokens):
            return HookRunResult.allow()
        # 絶対パス引数の境界判定。
        # Windows pathlib の罠: Path("/etc/passwd").is_absolute() は POSIX で
        # True、Windows で False (Windows は drive letter 必須扱い)。bash
        # command は POSIX-style (/) と Windows-style (C:\) の両方が混在し
        # うるので、文字列で先に「絶対っぽいか」判定してから resolve する。
        for tok in tokens:
            if not tok or tok.startswith("-"):
                continue
            is_abs = (
                tok.startswith("/")
                or (len(tok) >= 3 and tok[1] == ":" and tok[2] in ("/", "\\"))
            )
            if not is_abs:
                continue
            try:
                resolved = Path(tok).resolve()
                try:
                    resolved.relative_to(profile_root)
                except ValueError:
                    return HookRunResult.deny([
                        f"[bash_path_guard] bash 経由の profile 境界外への絶対パス"
                        f"書込み・削除を禁止 (token={tok}, "
                        f"profile_root={profile_root})。"
                        f"profile 配下の相対パス、または read 系コマンド (cat/ls 等)"
                        f"なら通常通り使えます。"
                    ])
            except (OSError, ValueError):
                continue
        return HookRunResult.allow()

    return _check


# ============================================================
# Noetic 固有 factory: 段階12 Step 7.5 ③ — slopsquatting 事前検証 hook
# ============================================================
# 背景: 段階12 で iku が pip install 経由でライブラリを身体拡張できる
# (PLAN §3-1 + Step 1.5)。LLM hallucination で存在しない pkg 名を提案 →
# 攻撃者が空 pkg を事前登録 (slopsquatting、USENIX 2025 で 5.2% 商用 LLM
# hallucination 率 + 43% 再現性確認) → 実害発生という攻撃を構造的抑止。
#
# 検証 (PLAN §3-5-2 ③ literal):
#   - 存在: PyPI registry で pkg 真偽性確認 → なければ deny
#   - typosquatting: 人気 pkg と Levenshtein 距離 1-2 → 警告
#
# 1 時間 in-memory cache で連続 install 時の重複問合せ削減、ネットワーク不通
# 時は warning のみで install 自体は allow (PLAN §3-5-2 ③ fallback)。
# 作成日 / download count 検証は将来拡張 (PyPI JSON では download 直接取得
# 不可、pypistats API 別途必要)。npm / cargo / go は将来拡張、本実装は
# pip + PyPI のみ MVP。
# ============================================================


_POPULAR_PYPI_PKGS = (
    "requests", "numpy", "pandas", "django", "flask", "pytest",
    "boto3", "scikit-learn", "tensorflow", "torch", "matplotlib",
    "pillow", "lxml", "click", "pyyaml", "sqlalchemy", "fastapi",
    "httpx", "aiohttp", "scipy", "beautifulsoup4", "selenium",
)


def make_install_command_check_hook(
    *,
    cache_ttl_sec: int = 3600,
    http_get=None,
) -> PreHandler:
    """段階12 Step 7.5 ③ (PLAN §3-5-2 ③): bash の pip install で
    PyPI registry を用いた pkg 真偽性検証 hook (slopsquatting 抑止)。

    Args:
        cache_ttl_sec: 1 pkg の検証結果キャッシュ有効期限 (秒)、default 1 時間
        http_get: テスト用 callable inject (url → body str | None)、None なら
            urllib.request を使う

    Returns:
        PreHandler — pip install パターンマッチ時に PyPI 検証、
        - registry に存在しない: deny
        - 人気 pkg と Levenshtein 1-2: warning (allow + message)
        - ネットワーク不通: warning のみ (allow + message)
        他は allow passthrough。
    """
    import json
    import shlex
    import time
    import urllib.error
    import urllib.request

    cache: dict = {}  # {pkg_name: (timestamp, result_dict)}

    def _default_http_get(url: str):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "noetic-seed/0.5"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "__NOT_FOUND__"  # 存在しない pkg
            return None  # その他 HTTP error
        except (urllib.error.URLError, OSError, TimeoutError):
            return None  # ネットワーク不通
        return None

    fetch = http_get if http_get is not None else _default_http_get

    def _levenshtein(a: str, b: str, max_dist: int = 2) -> int:
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                ins = curr[j] + 1
                dele = prev[j + 1] + 1
                sub = prev[j] + (ca != cb)
                curr.append(min(ins, dele, sub))
            prev = curr
        return prev[-1]

    def _check_pypi_pkg(pkg_name: str) -> dict:
        """PyPI registry で pkg を検証、result dict を返す。
        Returns: {"exists": bool, "warnings": [str], "network_ok": bool}
        """
        now = time.time()
        if pkg_name in cache:
            ts, cached = cache[pkg_name]
            if now - ts < cache_ttl_sec:
                return cached

        result = {"exists": False, "warnings": [], "network_ok": False}
        url = f"https://pypi.org/pypi/{pkg_name}/json"
        body = fetch(url)
        if body is None:
            # ネットワーク不通 (PLAN §3-5-2 ③ fallback)
            result["warnings"].append(
                f"[slopsquatting_check] PyPI 問合せ失敗 (pkg={pkg_name})、"
                f"ネットワーク不通の可能性。検証 skip して install 続行。"
            )
            cache[pkg_name] = (now, result)
            return result
        result["network_ok"] = True
        if body == "__NOT_FOUND__":
            # registry に存在しない確定
            cache[pkg_name] = (now, result)
            return result
        try:
            data = json.loads(body)
            if not isinstance(data, dict) or "info" not in data:
                cache[pkg_name] = (now, result)
                return result
        except json.JSONDecodeError:
            result["warnings"].append(
                f"[slopsquatting_check] PyPI レスポンス parse 失敗 (pkg={pkg_name})"
            )
            cache[pkg_name] = (now, result)
            return result

        result["exists"] = True
        # Levenshtein typosquatting check
        for popular in _POPULAR_PYPI_PKGS:
            if pkg_name == popular:
                break
            dist = _levenshtein(pkg_name, popular)
            if 1 <= dist <= 2:
                result["warnings"].append(
                    f"[slopsquatting_check] {pkg_name} は人気 pkg '{popular}' と "
                    f"Levenshtein 距離 {dist}、typosquatting 警告。"
                )
                break

        cache[pkg_name] = (now, result)
        return result

    def _extract_pip_install_pkgs(tokens: list) -> list:
        """bash command tokens から pip install の pkg 名候補を抽出。
        対応: pip install <pkg>、python -m pip install <pkg>、`pkg==1.0` 形式。
        local install (-e .、 ./path/、 /abs/path) は pkg 抽出対象外。
        """
        pip_idx = -1
        for i, tok in enumerate(tokens):
            if tok in ("pip", "pip3"):
                pip_idx = i
                break
            if tok in ("python", "python3", "py") and i + 2 < len(tokens):
                if tokens[i + 1] == "-m" and tokens[i + 2] in ("pip", "pip3"):
                    pip_idx = i + 2
                    break
        if pip_idx == -1:
            return []
        if pip_idx + 1 >= len(tokens) or tokens[pip_idx + 1] != "install":
            return []
        pkgs = []
        for tok in tokens[pip_idx + 2:]:
            if tok.startswith("-"):
                continue  # -e / --upgrade 等
            if "/" in tok or "\\" in tok or tok == ".":
                continue  # local path install
            # `pkg==1.0` 等の version specifier から pkg 名のみ
            for sep in ("==", ">=", "<=", "~=", ">", "<"):
                if sep in tok:
                    tok = tok.split(sep)[0]
                    break
            if tok and tok[0].isalpha():
                pkgs.append(tok.strip())
        return pkgs

    def _check(tool_name: str, tool_input: dict) -> HookRunResult:
        if tool_name != "bash":
            return HookRunResult.allow()
        cmd = str(tool_input.get("command") or "")
        # 高速 path: pip install を含まないコマンドは即 passthrough
        if not cmd or "install" not in cmd or "pip" not in cmd:
            return HookRunResult.allow()
        try:
            tokens = shlex.split(cmd, posix=True)
        except ValueError:
            return HookRunResult.allow()
        pkgs = _extract_pip_install_pkgs(tokens)
        if not pkgs:
            return HookRunResult.allow()

        deny_reasons = []
        warnings = []
        for pkg in pkgs:
            result = _check_pypi_pkg(pkg)
            warnings.extend(result.get("warnings", []))
            if result["network_ok"] and not result["exists"]:
                # registry に存在しない (= hallucination 確定)
                deny_reasons.append(
                    f"[slopsquatting_check] '{pkg}' が PyPI registry に存在しません "
                    f"(LLM hallucination の可能性、deny)。"
                )

        if deny_reasons:
            return HookRunResult.deny(deny_reasons + warnings)
        if warnings:
            return HookRunResult.allow(warnings)
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
