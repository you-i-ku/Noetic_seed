"""sanity check (段階12 Step 6, PLAN §10)。

iku の起動初頭で、身体 (state / memory / core / tools) の最低限の整合性を
検証する。失敗時は最新の iku-auto-<profile>-* stash から git stash apply で
自動復元を 1 回試行、それでも失敗なら手動介入を要求して exit(1)。

検査項目 (PLAN §10-1):
  1. state.json が JSON parse 可能 + 必須キー (cycle_id / tool_level /
     subjective_entries) (段階13 Phase 0.1.D で log → subjective_entries 移行)
  2. memory/ 配下の JSON が全て parse 可能
  3. import テスト: core.controller / tools が import で副作用なく成功

失敗時 (PLAN §10-2):
  - 最新の iku-auto-<profile>-* stash を git stash apply で復元 (pop じゃない、
    再 apply 可能性のため retain)
  - 再度 sanity check、成功なら起動続行、失敗なら sys.exit(1)

哲学的位置づけ (PLAN §10-3): 「起動できない身体 = 存在できない、選択の対象に
すらならない」が線引き。本 revert は「親心反射」ではなく「存在の前提条件」、
システム制約として許容される (CLAUDE.md 2026-04-18「不可触コア撤回」教訓の
延長線、§1-5 介入レベル分類「物理的存在の前提」枠)。
"""
import importlib
import json
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# 段階13 Phase 0.1.D: 新 schema 強制 — subjective_entries を必須化 (旧 log は撤去済)。
# 既存プロファイル (log のみ存在) は本 check で fail する設計、新規 profile 前提
# (smoke 慣習 = `_template_smokeN` clean profile)。ゆう判断「migration 不要」整合。
REQUIRED_STATE_KEYS: Tuple[str, ...] = ("cycle_id", "tool_level", "subjective_entries")


class SanityCheckError(Exception):
    """sanity check 失敗を示す例外。reason 文字列を含む。"""


def _check_state_json(profile_root: Path) -> None:
    """state.json が JSON parse 可能 + 必須キーが存在するか。"""
    state_file = profile_root / "state.json"
    if not state_file.exists():
        # 初回起動なら state.json なし、load_state が空 dict 返す経路で OK
        return
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SanityCheckError(f"state.json の JSON parse 失敗: {e}")
    if not isinstance(state, dict):
        raise SanityCheckError(
            f"state.json が dict じゃない: {type(state).__name__}"
        )
    missing = [k for k in REQUIRED_STATE_KEYS if k not in state]
    if missing:
        raise SanityCheckError(f"state.json に必須キー欠落: {missing}")


def _check_memory_jsons(profile_root: Path) -> None:
    """memory/ 配下の .json ファイルが全て parse 可能か。"""
    memory_dir = profile_root / "memory"
    if not memory_dir.exists():
        return  # 初回起動相当、問題なし
    for json_file in memory_dir.rglob("*.json"):
        if json_file.is_relative_to(memory_dir / "state_generations"):
            continue
        try:
            json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            rel = json_file.relative_to(memory_dir)
            raise SanityCheckError(f"memory/{rel} の JSON parse 失敗: {e}")


def _enumerate_runtime_modules() -> list:
    """`core.runtime` package 配下のサブモジュール名を動的列挙。

    Issue 5 fix (2026-05-02): `pkgutil.iter_modules` で `core/runtime/*.py` を
    全列挙して import 検査対象に加える。range は **書換え可能 + 起動時必ず
    通る path** に絞り、`core/` 直下 (memory.py / embedding.py 等) は対象外
    (PEP 810 lazy import トレンド整合、副作用 import 回避)。

    Returns:
        ["core.runtime.bash_validation", "core.runtime.compaction", ...]

    Raises:
        SanityCheckError: `core.runtime` package 自体の import に失敗した場合
            (= `core/runtime/__init__.py` が壊れている等)。
    """
    try:
        runtime_pkg = importlib.import_module("core.runtime")
    except Exception as e:
        raise SanityCheckError(
            f"core.runtime package の import 失敗: {type(e).__name__}: {e}"
        )
    return [
        name for _, name, _ in pkgutil.iter_modules(
            runtime_pkg.__path__, prefix="core.runtime."
        )
    ]


def _check_imports() -> None:
    """Check fresh imports in a disposable interpreter, never reload live modules.

    Keep the original import coverage and import-time exception detection. The
    child uses this profile and interpreter; -B prevents cache writes. A failed
    or timed-out check follows the existing SanityCheckError/revert path.
    """
    script = (
        "import importlib\n"
        "from core.sanity_check import _enumerate_runtime_modules\n"
        "for name in ('core.controller', 'tools', *_enumerate_runtime_modules()):\n"
        "    importlib.import_module(name)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SanityCheckError(
            f"import check process failed: {type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output")[-4000:]
        raise SanityCheckError(f"import check failed (exit={result.returncode}): {detail}")



def _run_all_checks(profile_root: Path, *, recover_state=False) -> None:
    """全 sanity check を実行、失敗時は SanityCheckError を伝播。"""
    if not recover_state:
        _check_state_json(profile_root)
    _check_memory_jsons(profile_root)
    _check_imports()


def _git(*args: str, cwd: Path) -> Tuple[int, str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return result.returncode, result.stdout.strip()
    except (FileNotFoundError, OSError):
        return -1, ""


def _find_repo_root(start: Path) -> Optional[Path]:
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return None


def _try_auto_revert_from_stash(
    profile_root: Path, profile_name: str,
) -> bool:
    """最新の iku-auto-<profile>-* stash を git stash apply で復元。

    pop ではなく apply を使うことで、apply 失敗 / 再起動失敗の場合の
    バックアップとして stash を retain する。

    Returns:
        True: 復元成功、False: stash 見つからない / apply 失敗
    """
    repo_root = _find_repo_root(profile_root)
    if repo_root is None:
        return False
    code, out = _git("stash", "list", cwd=repo_root)
    if code != 0:
        return False
    tag = f"iku-auto-{profile_name}-"
    for line in out.splitlines():
        if tag in line:
            stash_ref = line.split(":", 1)[0]  # "stash@{0}"
            apply_code, _ = _git("stash", "apply", stash_ref, cwd=repo_root)
            return apply_code == 0
    return False


def enforce_sanity_check(
    profile_root: Path,
    profile_name: str,
    *,
    auto_revert: bool = True,
    recover_state: bool = False,
) -> None:
    """段階12 Step 6 (PLAN §10): 起動時 sanity check + 自動 revert ガード。

    Args:
        profile_root: プロファイル workspace root
        profile_name: プロファイル名 (stash filter 用)
        auto_revert: True で失敗時に最新 iku-auto stash で自動 revert を 1 回試行
        recover_state: main 起動時のみ True。state の診断を世代復旧に任せる。

    Raises:
        SystemExit(1): revert しても失敗、または auto_revert=False で失敗時。
    """
    check_options = {"recover_state": True} if recover_state else {}
    try:
        _run_all_checks(profile_root, **check_options)
        return  # 全 OK
    except SanityCheckError as e:
        print()
        print(f"[sanity_check] 失敗: {e}")

    if not auto_revert:
        print(f"[sanity_check] auto_revert=False、手動介入してください")
        sys.exit(1)

    print(f"[sanity_check] 最新の iku-auto stash から自動 revert を試行...")
    reverted = _try_auto_revert_from_stash(profile_root, profile_name)
    if not reverted:
        print(
            f"[sanity_check] revert 失敗 (stash 見つからない / apply 失敗)、"
            f"手動介入してください"
        )
        sys.exit(1)

    try:
        _run_all_checks(profile_root, **check_options)
        print(
            f"[sanity_check] revert 成功、起動続行 "
            f"(前回の改変は revert されました)"
        )
    except SanityCheckError as e:
        print(f"[sanity_check] revert 後も失敗: {e}、手動介入してください")
        sys.exit(1)
