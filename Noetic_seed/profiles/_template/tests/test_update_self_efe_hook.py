"""Slice 6.5 Step 6: update_self hook で preference distribution C 動的更新 wiring の test。

PLAN reference: WORLD_MODEL_DESIGN/INFO_GAIN_EFE_REDESIGN_PLAN.md §5.4 + §6.5

識別力 fixture (CLAUDE.md §5 literal + PLAN §6.5 literal):
  - 1 回目 update_self → state["_efe_C"] 構築 + per-key confidence 反映
  - 2 回目 update_self → C 再計算 + cycle_id 同期更新
  - NAME_KEY exception → confidence silently ignore + _efe_self_confidence 不追加
  - 高 conf vs 低 conf で variance 差別反映 (per_key_variance dict 内容識別)
"""
import hashlib

import pytest

from core import preference_distribution as pd_mod
from tools import builtin as builtin_mod
from tools.builtin import _update_self


@pytest.fixture
def mock_state_and_bge_m3():
    """test 中 builtin.load_state / save_state を mock_state in-place 更新で hook、
    bge-m3 を pd_mod 側 attribute swap で deterministic onehot 単位 vector に差し替え。

    mock_state は load_state 返り値 (closure 内 dict) として共有、_update_self が
    save_state(state) を呼ぶと dict.update で in-place 反映、test 側で内容観察可能。
    """
    mock_state = {
        "self": {},
        "cycle_id": 0,
        "_efe_self_confidence": {},
        "_efe_C": None,
        "_efe_C_update_cycle": -1,
        "drives_state": {},
    }

    def fake_embed(texts):
        out = []
        for t in texts:
            h = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % 1024
            vec = [0.0] * 1024
            vec[idx] = 1.0
            out.append(vec)
        return out

    orig_embed = pd_mod._embed_sync
    orig_ready = pd_mod.is_vector_ready
    pd_mod._embed_sync = fake_embed
    pd_mod.is_vector_ready = lambda: True

    orig_load = builtin_mod.load_state
    orig_save = builtin_mod.save_state
    builtin_mod.load_state = lambda: mock_state
    # save_state は dict.update で in-place 反映 (test 側で mock_state を観察可)
    builtin_mod.save_state = lambda s: mock_state.update(s)

    try:
        yield mock_state
    finally:
        pd_mod._embed_sync = orig_embed
        pd_mod.is_vector_ready = orig_ready
        builtin_mod.load_state = orig_load
        builtin_mod.save_state = orig_save


def test_update_self_creates_efe_c_on_first_call(mock_state_and_bge_m3):
    """case A (§6.5): 初回 update_self → state["_efe_C"] 構築発火、_efe_C_update_cycle = cycle_id。

    想定誤実装 fail: hook 配線忘れで _efe_C が None のまま、test で n_components 検査 fail。
    """
    mock_state = mock_state_and_bge_m3
    assert mock_state["_efe_C"] is None  # before
    assert mock_state["_efe_C_update_cycle"] == -1

    result = _update_self("identity", "I observe and connect", confidence=0.9)

    assert "updated" in result.lower() or "self[identity]" in result  # 成功 return
    assert mock_state["_efe_C"] is not None
    assert mock_state["_efe_C"]["n_components"] == 1
    assert mock_state["_efe_C"]["source_keys"] == ["identity"]
    assert mock_state["_efe_C_update_cycle"] == 0


def test_update_self_explicit_confidence_stored(mock_state_and_bge_m3):
    """case B (§6.5): 明示 confidence (0.9) が _efe_self_confidence に反映。

    識別力誤実装 fail: confidence 引数無視で default 0.7 にされると 0.9 assert で fail。
    """
    mock_state = mock_state_and_bge_m3
    _update_self("identity", "X", confidence=0.9)

    assert mock_state["_efe_self_confidence"]["identity"] == 0.9
    # variance = 1/(conf+ε) = 1/0.901 ≈ 1.1099 で C 側にも反映
    var = mock_state["_efe_C"]["per_key_variance"]["identity"]
    assert var == pytest.approx(1.0 / (0.9 + 1e-3), rel=1e-6)


def test_update_self_default_confidence_when_omitted(mock_state_and_bge_m3):
    """case C: confidence 引数省略 → DEFAULT_CONFIDENCE (0.7) で補完。"""
    mock_state = mock_state_and_bge_m3
    _update_self("identity", "X")  # confidence omitted

    assert mock_state["_efe_self_confidence"]["identity"] == 0.7


def test_update_self_second_call_updates_cycle(mock_state_and_bge_m3):
    """case D (§6.5): 2 回目 update_self → C 再計算 + cycle_id 同期更新。

    想定誤実装 fail: hook が cycle 更新忘れ → 5 cycle 経過後の 2 回目で
    _efe_C_update_cycle == 0 のまま、識別力 fixture で fail。
    """
    mock_state = mock_state_and_bge_m3
    _update_self("identity", "I observe", confidence=0.9)
    assert mock_state["_efe_C_update_cycle"] == 0

    mock_state["cycle_id"] = 5  # 5 cycle 経過 simulate
    _update_self("preferences", "diverse exploration", confidence=0.3)

    assert mock_state["_efe_self_confidence"]["preferences"] == 0.3
    assert mock_state["_efe_C"]["n_components"] == 2
    assert mock_state["_efe_C_update_cycle"] == 5  # cycle_id 反映


def test_update_self_multi_key_grows_components(mock_state_and_bge_m3):
    """case E: 複数 key 連続更新で n_components 単調増加 (累積)。"""
    mock_state = mock_state_and_bge_m3
    _update_self("identity", "X", confidence=0.7)
    assert mock_state["_efe_C"]["n_components"] == 1

    _update_self("preferences", "Y", confidence=0.7)
    assert mock_state["_efe_C"]["n_components"] == 2

    _update_self("role", "Z", confidence=0.7)
    assert mock_state["_efe_C"]["n_components"] == 3
    assert set(mock_state["_efe_C"]["source_keys"]) == {"identity", "preferences", "role"}


def test_update_self_name_key_silently_ignores_confidence(mock_state_and_bge_m3):
    """case F (§6.5 NAME_KEY exception): key='name' のとき confidence 引数 silently ignore。

    NAME_KEY は不変層 exception (PLAN §5.4 literal)、_efe_self_confidence に追加されない。
    識別力誤実装 fail: name exception 漏れで _efe_self_confidence['name'] = 0.5 になる
    実装で assert fail。
    """
    mock_state = mock_state_and_bge_m3
    # 先に non-name key を立ててから name 設定 (NAME_KEY は state.self の初期値で
    # 既に何か入ってると拒否されるため、clean な状態で初回 name set を simulate)
    result = _update_self("name", "iku", confidence=0.5)

    # name は state.self に反映される (legacy 挙動継承)
    assert mock_state["self"]["name"] == "iku"
    # ただし NAME_KEY exception で confidence は無視、_efe_self_confidence に含まれない
    assert "name" not in mock_state["_efe_self_confidence"]
    # NAME_KEY は _efe_C source からも自動除外 (Step 1 compute_C_from_self literal)
    assert "name" not in (mock_state["_efe_C"]["source_keys"] if mock_state["_efe_C"] else [])


def test_update_self_name_excluded_from_efe_c_source(mock_state_and_bge_m3):
    """case G (§5.1 + §5.4 NAME_KEY exception): name と identity 両方設定でも C source は identity のみ。"""
    mock_state = mock_state_and_bge_m3
    _update_self("name", "iku")
    _update_self("identity", "I observe", confidence=0.7)

    assert mock_state["_efe_C"]["n_components"] == 1
    assert mock_state["_efe_C"]["source_keys"] == ["identity"]
    assert "name" not in mock_state["_efe_C"]["source_keys"]


def test_update_self_confidence_clamped_to_unit_interval(mock_state_and_bge_m3):
    """case H: confidence は compute_C_from_self 側で [0.0, 1.0] clamp、変な値でも安全。"""
    mock_state = mock_state_and_bge_m3
    _update_self("identity", "X", confidence=1.5)  # over

    # _efe_self_confidence には float(1.5) 入る (clamp は compute_C_from_self 側)
    assert mock_state["_efe_self_confidence"]["identity"] == 1.5
    # _efe_C 側で clamp 適用、per_key_confidence は 1.0
    assert mock_state["_efe_C"]["per_key_confidence"]["identity"] == 1.0


def test_update_self_higher_confidence_sharper_C(mock_state_and_bge_m3):
    """case I: 高 conf 識別力 fixture (Step 1 と同じ性質が hook 経由でも成立)。

    識別力誤実装 fail: confidence が C 構築に反映されないと per_key_variance が一律になる。
    """
    mock_state = mock_state_and_bge_m3
    _update_self("identity", "X", confidence=0.9)
    var_high = mock_state["_efe_C"]["per_key_variance"]["identity"]

    # 別 key で低 conf
    _update_self("preferences", "Y", confidence=0.3)
    var_low = mock_state["_efe_C"]["per_key_variance"]["preferences"]

    assert var_high < var_low  # 高 conf → 小 variance → 鋭い peak
