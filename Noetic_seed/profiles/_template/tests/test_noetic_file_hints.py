"""_NOETIC_FILE_HINTS の段階12 整合性 test (2026-04-30 hotfix)。

段階12 で write_file / edit_file は profile 配下全体への書込み許可になったが、
hint string が「sandbox/ 以下のみ」という旧仕様のまま残っていた bug を
発見・修正した (ゆう 2026-04-30 gut check で判明、smoke で LLM が hint 通り
core/*.py の書換えを諦めて reboot 直行する現象が観察された)。

逆戻りガード:
  - write_file hint に「profile」が含まれる (新仕様)
  - write_file hint に「sandbox/ 以下のみ」「sandbox/ 外への書込」等の旧記述
    が含まれない
  - edit_file hint に「profile」が含まれる
  - edit_file hint に「sandbox/ 以下のみ」が含まれない

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_noetic_file_hints.py
  (pytest tests/test_noetic_file_hints.py でも動く)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.tools import _NOETIC_FILE_HINTS


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    assert cond, label


def test_write_file_hint_profile_scope():
    """write_file hint が profile 配下を許可していること (段階12 仕様)。"""
    hint = _NOETIC_FILE_HINTS.get("write_file", "")
    _assert(
        "profile" in hint,
        "write_file hint に 'profile' (= 段階12 仕様の身体範囲) が含まれる",
    )


def test_write_file_hint_no_stale_sandbox_only():
    """write_file hint に旧 sandbox-only 制約の記述が残っていないこと。"""
    hint = _NOETIC_FILE_HINTS.get("write_file", "")
    forbidden_patterns = [
        "sandbox/ 以下のみ",
        "sandbox/ 外への書込",
        "sandbox/ 外への書込はガードで拒否",
    ]
    for pat in forbidden_patterns:
        _assert(
            pat not in hint,
            f"write_file hint に旧記述 '{pat}' が残っていない",
        )


def test_edit_file_hint_profile_scope():
    """edit_file hint が profile 配下を許可していること。"""
    hint = _NOETIC_FILE_HINTS.get("edit_file", "")
    _assert(
        "profile" in hint,
        "edit_file hint に 'profile' が含まれる",
    )


def test_edit_file_hint_no_stale_sandbox_only():
    """edit_file hint に旧 sandbox-only 制約の記述が残っていないこと。"""
    hint = _NOETIC_FILE_HINTS.get("edit_file", "")
    _assert(
        "sandbox/ 以下のみ" not in hint,
        "edit_file hint に旧記述 'sandbox/ 以下のみ' が残っていない",
    )


def test_secrets_protection_preserved():
    """secrets 保護記述は維持されている (段階12 でも sandbox/secrets/ + secret_write 経路は不変)。"""
    write_hint = _NOETIC_FILE_HINTS.get("write_file", "")
    edit_hint = _NOETIC_FILE_HINTS.get("edit_file", "")
    _assert(
        "secret_write" in write_hint,
        "write_file hint に secret_write 案内あり",
    )
    _assert(
        "secret_write" in edit_hint,
        "edit_file hint に secret_write 案内あり",
    )


if __name__ == "__main__":
    print("=== test_noetic_file_hints ===")
    test_write_file_hint_profile_scope()
    test_write_file_hint_no_stale_sandbox_only()
    test_edit_file_hint_profile_scope()
    test_edit_file_hint_no_stale_sandbox_only()
    test_secrets_protection_preserved()
    print("=== all green ===")
