"""venv per-profile の path 解決検証 (段階12 Step 1.5、PLAN §3-1 / §11-5-pre)。

実際の venv 構築はせず、`_template/main.py` の `_bootstrap_venv()` が
プロファイル配下の `.venv/` を参照する path に変更されたことだけを
ソース文字列で確認する。

旧共通 venv (`Noetic_seed/.venv/`) 参照: `_here.parent.parent / ".venv"`
新 per-profile venv 参照:                `_here / ".venv"`

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_venv_per_profile.py
  (pytest tests/test_venv_per_profile.py でも動く)
"""
from pathlib import Path

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    assert cond, label


def test_bootstrap_venv_uses_per_profile_path():
    """`_bootstrap_venv()` が per-profile venv path を見ていること。"""
    src = MAIN_PY.read_text(encoding="utf-8")

    # 旧共通 venv path 跡が残っていない
    _assert(
        '_here.parent.parent / ".venv"' not in src,
        "旧共通 venv path (_here.parent.parent / .venv) が消えている",
    )

    # 新 per-profile venv path が導入されている
    _assert(
        '_here / ".venv"' in src,
        "新 per-profile venv path (_here / .venv) が導入されている",
    )


if __name__ == "__main__":
    print("=== test_venv_per_profile ===")
    test_bootstrap_venv_uses_per_profile_path()
    print("=== all green ===")
