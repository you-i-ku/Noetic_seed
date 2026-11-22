"""reflect_and_persist 二重発火根治 hotfix test。

旧実装 (main.py:_tool_reflect が `s = load_state()` で別 dict を mutate) では
main scope の state["reflection_cycle"] が更新されず、cycle 末
should_reflect(state) が再 trigger → 同 cycle で自動経路が再発火していた。

本 test は `core.reflection.reflect_and_persist` が **passed state を in-place
mutate する** 性質を機械的に検証する。CLAUDE.md §5 識別力 + §6 docstring 同期遵守。

検証範囲:

A. in-place mutation:
   A1. passed state に `reflection_cycle = 0` が直接書かれる
   A2. save_state_fn には passed state object そのもの (is identity) が渡る
       → 旧実装 (`s = load_state()` で別 dict 作成) なら fail する識別力 fixture
   A3. reflect_fn の戻り値が呼出元に return される

B. 二重発火回避シナリオ:
   B1. reflect_and_persist 実行後、should_reflect(state) が False を返す
       (cycles_since=0 で interval 10 を下回る)
       → 同 cycle 内で auto 経路が再 trigger しない構造を機械的に検証

C. 既定 (引数 None) のフォールバック:
   C1. save_state_fn / reflect_fn を省略しても helper が動作する
       (default は `core.state.save_state` / 本 module の reflect)

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_reflect_double_fire.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reflection import reflect_and_persist, should_reflect


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _fake_reflect_dummy(state: dict, call_llm_fn) -> dict:
    """reflect_fn の mock。state を mutate しない、固定 dict を返す。

    state mutation は reflect_and_persist 自体の責務 (reflection_cycle=0 リセット)
    なので、reflect_fn は副作用なしで結果だけ返す純関数として mock する。
    """
    return {
        "notes": ["mock note"],
        "self_disp_delta": {},
        "attr_disp_delta": {},
        "cluster_mi": 0.0,
        "cluster_inter_ratio": 0.0,
        "cluster_link_pairs": 0,
    }


def _fake_llm(*args, **kwargs):
    return ""


# ============================================================
# A. in-place mutation
# ============================================================

def test_a1_reflection_cycle_reset_on_passed_state():
    """A1: passed state の reflection_cycle が 0 にリセットされる。"""
    print("== A1: passed state の reflection_cycle = 0 ==")
    state = {"reflection_cycle": 5}
    save_calls = []
    reflect_and_persist(
        state, _fake_llm,
        save_state_fn=save_calls.append,
        reflect_fn=_fake_reflect_dummy,
    )
    return _assert(
        state["reflection_cycle"] == 0,
        f"state['reflection_cycle'] == 0 (got {state['reflection_cycle']})",
    )


def test_a2_save_state_receives_same_object_identity():
    """A2: save_state_fn には passed state object そのものが渡る (is identity)。

    識別力 fixture: 旧実装 (`s = load_state()` で別 dict 作成 → save_state(s))
    なら save_calls[0] は passed state とは別 object になり `is` 比較で fail。
    本 test は in-place mutation 設計を機械的に保証する。
    """
    print("== A2: save_state_fn 引数は passed state と同 object (is identity) ==")
    state = {"reflection_cycle": 7, "marker": "passed"}
    save_calls = []
    reflect_and_persist(
        state, _fake_llm,
        save_state_fn=save_calls.append,
        reflect_fn=_fake_reflect_dummy,
    )
    ok = True
    ok &= _assert(len(save_calls) == 1, f"save_state は 1 回呼ばれる (got {len(save_calls)})")
    ok &= _assert(
        save_calls[0] is state,
        "save_calls[0] is state (識別力: 旧実装 `s = load_state()` なら別 object で fail)",
    )
    ok &= _assert(
        save_calls[0].get("marker") == "passed",
        "passed state の固有 field (marker) が見える = 別 object じゃない",
    )
    return ok


def test_a3_reflect_result_returned():
    """A3: reflect_fn の戻り値が helper の戻り値として呼出元に届く。"""
    print("== A3: reflect_fn の result が return される ==")
    state = {"reflection_cycle": 3}
    save_calls = []
    result = reflect_and_persist(
        state, _fake_llm,
        save_state_fn=save_calls.append,
        reflect_fn=_fake_reflect_dummy,
    )
    return _assert(
        result.get("notes") == ["mock note"],
        f"result['notes'] == ['mock note'] (got {result.get('notes')})",
    )


# ============================================================
# B. 二重発火回避シナリオ
# ============================================================

def test_b1_should_reflect_false_after_persist():
    """B1: helper 実行後に should_reflect(state) が False を返す → 二重発火回避を機械検証。

    旧実装シナリオ: tool 経路で reflect 走っても reflection_cycle が main state に
    反映されなかった (`s = load_state()` 由来の別 dict 問題) ため、cycle 末
    should_reflect(state) が True を返し auto 経路が再発火していた。

    新実装: reflect_and_persist が passed state に直接 reflection_cycle=0 を書くので、
    cycle 末 should_reflect(state) は cycles_since=0 で False を返す。
    本 test は **二重発火が構造的に起きない** ことの直接検証。
    """
    print("== B1: helper 後 should_reflect(state) が False (二重発火 構造的回避) ==")
    state = {"reflection_cycle": 12, "last_prediction_error": 0}  # interval=10 を超え
    save_calls = []
    # 事前確認: 旧 state (reflection_cycle=12) なら should_reflect は True
    ok = True
    ok &= _assert(
        should_reflect(state, interval=10) is True,
        "事前: reflection_cycle=12 で should_reflect=True (auto trigger 条件)",
    )
    # helper 実行
    reflect_and_persist(
        state, _fake_llm,
        save_state_fn=save_calls.append,
        reflect_fn=_fake_reflect_dummy,
    )
    # 事後: same state object で should_reflect が False
    ok &= _assert(
        should_reflect(state, interval=10) is False,
        "事後: helper 経由で reflection_cycle=0、should_reflect=False (二重発火回避)",
    )
    return ok


# ============================================================
# C. 既定 (引数 None) フォールバック
# ============================================================

def test_c1_default_args_resolve_via_lazy_import():
    """C1: save_state_fn / reflect_fn 省略時は default に解決される (循環 import 回避)。

    本 test は default 解決経路を mock せず確認するが、reflect 本体を実行すると
    重い (memory load + LLM 呼出 + cluster 推定) ので reflect_fn のみ mock し、
    save_state_fn の default 解決のみ exercise する。
    """
    print("== C1: save_state_fn 省略時は core.state.save_state が default 解決 ==")
    import tempfile
    from pathlib import Path
    import core.state as cs

    tmp = Path(tempfile.mkdtemp(prefix="noetic_reflect_helper_"))
    state_file_orig = cs.STATE_FILE
    cs.STATE_FILE = tmp / "state.json"
    try:
        state = {"reflection_cycle": 4}
        # save_state_fn 省略、reflect_fn のみ mock
        reflect_and_persist(
            state, _fake_llm,
            reflect_fn=_fake_reflect_dummy,
        )
        ok = True
        ok &= _assert(
            state["reflection_cycle"] == 0,
            "default 解決経路でも reflection_cycle=0",
        )
        ok &= _assert(
            cs.STATE_FILE.exists(),
            "default save_state_fn 経由で state.json が書かれた",
        )
        return ok
    finally:
        cs.STATE_FILE = state_file_orig
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    tests = [
        test_a1_reflection_cycle_reset_on_passed_state,
        test_a2_save_state_receives_same_object_identity,
        test_a3_reflect_result_returned,
        test_b1_should_reflect_false_after_persist,
        test_c1_default_args_resolve_via_lazy_import,
    ]
    results = []
    for t in tests:
        try:
            results.append((t.__name__, t()))
        except Exception as e:
            print(f"  [FAIL] {t.__name__} 例外: {e}")
            import traceback
            traceback.print_exc()
            results.append((t.__name__, False))
        print()
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"=== {passed}/{total} passed ===")
    sys.exit(0 if passed == total else 1)
