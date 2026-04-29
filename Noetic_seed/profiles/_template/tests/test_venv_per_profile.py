"""venv per-profile の path 解決 + requirements.txt 経由化 検証 (段階12 Step 1.5、
PLAN §3-1 / §3-3 / §11-5-pre)。

実際の venv 構築はせず、`_template/main.py` の `_bootstrap_venv()` が
プロファイル配下の `.venv/` を参照し、かつ身体仕様 (= 依存 pkg 群) を
hard-code list ではなく `requirements.txt` から読むことだけを
ソース文字列で確認する。

旧共通 venv (`Noetic_seed/.venv/`) 参照: `_here.parent.parent / ".venv"`
新 per-profile venv 参照:                `_here / ".venv"`
旧 hard-code deps:                       `_deps = [...]` リテラル
新 requirements 経由:                    `pip install -r requirements.txt`

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_venv_per_profile.py
  (pytest tests/test_venv_per_profile.py でも動く)
"""
from pathlib import Path

PROFILE_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROFILE_ROOT / "main.py"
REQ_TXT = PROFILE_ROOT / "requirements.txt"


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


def test_bootstrap_uses_requirements_txt():
    """`_bootstrap_venv()` が hard-code list ではなく requirements.txt を読むこと。

    PLAN §3-3: 「venv は _bootstrap_venv() が requirements.txt から再構築する」。
    iku が requirements.txt を編集 → reboot で身体拡張が反映される経路の前提。
    """
    src = MAIN_PY.read_text(encoding="utf-8")

    # requirements.txt path が main.py で参照されている
    _assert(
        '"requirements.txt"' in src,
        "_bootstrap_venv() が requirements.txt を参照している",
    )

    # pip install -r オプションが使われている
    _assert(
        '"-r"' in src,
        "pip install -r オプションが使われている",
    )

    # 旧 hard-code list (_deps = [...]) が残っていない
    _assert(
        "_deps = [" not in src,
        "旧 hard-code 依存リスト (_deps = [...]) が撤去されている",
    )


def test_requirements_txt_exists_and_nonempty():
    """profile 内に requirements.txt が存在し、空でないこと。"""
    _assert(
        REQ_TXT.exists(),
        f"requirements.txt が profile 内に存在 ({REQ_TXT.name})",
    )

    lines = [
        ln.strip()
        for ln in REQ_TXT.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    _assert(
        len(lines) > 0,
        f"requirements.txt に少なくとも 1 つの pkg 宣言がある ({len(lines)} 個)",
    )


if __name__ == "__main__":
    print("=== test_venv_per_profile ===")
    test_bootstrap_venv_uses_per_profile_path()
    test_bootstrap_uses_requirements_txt()
    test_requirements_txt_exists_and_nonempty()
    print("=== all green ===")
