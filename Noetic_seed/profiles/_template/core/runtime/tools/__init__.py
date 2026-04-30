"""v0.5 Tools (claw-code 厳密準拠)。

Noetic 固有機能 (recall / modify_self / camera_stream 等) は含めない。
それらは別モジュール (将来的に _noetic_ext/) に隔離する。

構成 (12 module / 50 tool):
  file_ops.py   (5)  read_file / write_file / edit_file / glob_search / grep_search
  shell.py      (3)  bash / PowerShell / REPL
  web.py        (3)  WebFetch / WebSearch / RemoteTrigger
  task.py       (6)  TaskCreate/Get/List/Stop/Update/Output
  worker.py     (9)  WorkerCreate/Get/Observe/ResolveTrust/AwaitReady/
                     SendPrompt/Restart/Terminate/ObserveCompletion
  team_cron.py  (5)  TeamCreate/Delete, CronCreate/List/Delete
  lsp.py        (1)  LSP
  mcp.py        (4)  MCP / ListMcpResources / ReadMcpResource / McpAuth
  ui.py         (4)  AskUserQuestion / SendUserMessage /
                     StructuredOutput / Config
  plan.py       (2)  EnterPlanMode / ExitPlanMode
  skill.py      (3)  Skill / Agent / ToolSearch
  util.py       (5)  Sleep / TodoWrite / NotebookEdit /
                     TestingPermission / RunTaskPacket

register_all(registry, workspace_root, settings_path, skill_dirs) で
全 tool を一括登録する。
"""
from pathlib import Path
from typing import Optional

from core.runtime.registry import ToolRegistry


def register_all(
    registry: ToolRegistry,
    workspace_root: Path,
    settings_path: Optional[Path] = None,
    skill_dirs: Optional[list] = None,
) -> None:
    """claw-code の全 tool (50) を registry に登録。

    workspace_root: file_ops の境界
    settings_path:  Config tool の対象 (デフォルト workspace_root/settings.json)
    skill_dirs:     Skill tool の検索パス群
    """
    from core.runtime.tools import (
        file_ops, shell, web, task, worker, team_cron,
        lsp, mcp, ui, plan, skill, util,
    )

    if settings_path is None:
        settings_path = workspace_root / "settings.json"
    if skill_dirs is None:
        skill_dirs = [str(workspace_root / ".claude" / "skills"),
                      str(workspace_root / "skills")]

    file_ops.register(registry, workspace_root)
    shell.register(registry)
    web.register(registry)
    task.register(registry)
    worker.register(registry)
    team_cron.register(registry)
    lsp.register(registry)
    mcp.register(registry)
    ui.register(registry, settings_path)
    plan.register(registry)
    skill.register(registry, skill_dirs)
    util.register(registry, workspace_root)


_NOETIC_FILE_HINTS = {
    "read_file": (
        "[Noetic 制約] secrets.json と sandbox/secrets/ は直接読めません "
        "(auth_profile_info / secret_read を使用)。"
        "その他の workspace 配下は自由に読めます。"
    ),
    "write_file": (
        "[Noetic 制約] 書込先は profile 配下 (= 自分の身体、PLAN §3-1)。"
        "secrets.json / sandbox/secrets/ は secret_write を使用。"
        "profile 外への書込はガードで拒否されます。"
    ),
    "edit_file": (
        "[Noetic 制約] 編集対象は profile 配下。"
        "secrets.json / sandbox/secrets/ は secret_write を使用。"
    ),
    "glob_search": (
        "[Noetic 制約] sandbox/secrets/ と secrets.json は検索対象から除外されます。"
        "pattern は '**/*.py' のような glob を使用 (bash の ** とは挙動が違うので注意)。"
    ),
    "grep_search": (
        "[Noetic 制約] sandbox/secrets/ と secrets.json は検索対象から除外されます。"
    ),
}


_NOETIC_BASH_HINT = (
    "[Noetic 制約] Level-aware 安全性検査あり。"
    "Level 0-2 は read-only モード (ls/cat/grep/find/git/head/tail/wc/echo/"
    "pwd/whoami/date/env/stat/file/du/df/ps/uname 等の whitelist のみ)。"
    "Level 3+ でフル bash 解放。"
    "破壊的コマンド (rm -rf /, dd of=/dev/sd*, fork bomb, mkfs 等) は"
    "Level 問わず常に自動拒否。"
    "注意喚起コマンド (rm -rf, sudo, chmod 777, curl|bash, eval, --force push) は"
    "承認画面に警告付きで表示。"
)


def ensure_noetic_bash_hint(registry: ToolRegistry) -> int:
    """bash tool の description に Noetic 固有 Level-aware 制約 hint を追記。

    claw 本家の bash description は実行方法のみ説明で、Noetic の
    make_bash_validation_hook (hooks.py) が敷く段階的解放ルールを
    LLM に知らせない。事前に制約を description で教えることで、
    Level 0-2 で破壊系コマンド選択 → deny → 再試行のサイクル浪費を防ぐ。

    idempotent: 既に "[Noetic 制約]" が含まれていれば skip。

    Returns:
        hint を注入した tool 数 (0 または 1)。
    """
    spec = registry.get("bash")
    if spec is None:
        return 0
    current = spec.description or ""
    if "[Noetic 制約]" in current:
        return 0
    spec.description = f"{current.rstrip()}\n\n{_NOETIC_BASH_HINT}"
    return 1


def ensure_noetic_file_hints(registry: ToolRegistry) -> int:
    """file 系 claw ネイティブ tool の description に Noetic 固有制約を追記。

    claw 本家の description は claw-code の純粋 file_ops 仕様を説明するだけで、
    Noetic の make_file_access_guard (hooks.py) が敷く追加制約
    (profile 外書込禁止、secrets 保護) は LLM に知らされない。
    LLM が profile 外への書込で deny を食らってから学習するより、
    事前に description で制約を知らせる方がサイクル効率が良い。

    claw 本家ソースには触らず、registry 登録後に ToolSpec.description を
    mutate する後付け設計 (ensure_approval_props と同パターン)。
    idempotent: 既に "[Noetic 制約]" が含まれていれば skip。

    Returns:
        hint を注入した tool 数。
    """
    count = 0
    for name, hint in _NOETIC_FILE_HINTS.items():
        spec = registry.get(name)
        if spec is None:
            continue
        current = spec.description or ""
        if "[Noetic 制約]" in current:
            continue
        spec.description = f"{current.rstrip()}\n\n{hint}"
        count += 1
    return count


def ensure_approval_props(registry: ToolRegistry) -> int:
    """Registry 全 tool の input_schema に承認 3 層を後付け注入する。

    Noetic 固有の承認 3 層 (tool_intent / tool_expected_outcome / message)
    は全 tool で required。claw 本家準拠の tool (file_ops/web/shell/task/...)
    は元々 3 層を持たないため、この関数で registry 登録後に一括注入する。
    noetic_ext / legacy_bridge は既に 3 層持ちなので no-op で skip。

    claw 本家ソース (file_ops.py 等) には触らず、input_schema を実行時に
    書き換えるので claw 準拠との分離を保つ。

    Returns:
        注入対象となった tool 数 (1 つでも field を追加した tool 数)。
    """
    from core.runtime.tools.noetic_ext.cognition import (
        _APPROVAL_REQUIRED,
        _approval_props,
    )

    props_def = _approval_props()
    count = 0
    for name in registry.all_names():
        spec = registry.get(name)
        if spec is None:
            continue
        schema = spec.input_schema
        if not isinstance(schema, dict):
            continue
        props = schema.setdefault("properties", {})
        required = schema.setdefault("required", [])
        touched = False
        for fld, defn in props_def.items():
            if fld not in props:
                props[fld] = defn
                touched = True
        for fld in _APPROVAL_REQUIRED:
            if fld not in required:
                required.append(fld)
                touched = True
        if touched:
            count += 1
    return count
