"""WM materialized view テスト (段階7 Step 3)。

成功条件:
  - store_wm_fact: in-memory WM 更新 + memory/wm.jsonl 永続化
  - β+ 学習: 同じ fact 再観察で confidence 上昇 + observation_count +1
  - bitemporal: 矛盾 fact で旧 fact valid_to 凍結 + 新 fact 追加
  - rebuild_wm_from_jsonl: jsonl records から entities 復元
  - rebuild graceful: 不正 record skip
  - 再起動シナリオ: store → rebuild で WM が再現される

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_materialized_view.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.memory as memory_mod
import core.tag_registry as tr
from core.world_model import (
    init_world_model,
    store_wm_fact,
    rebuild_wm_from_jsonl,
    find_fact,
    get_entity,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


def _setup_tmp_memory(tmp_path: Path):
    """各テスト前: MEMORY_DIR を tmp_path に差し替え + tag_registry 初期化。"""
    memory_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)
    tr.register_standard_tags()


def test_store_wm_fact_creates_jsonl(tmp_path: Path):
    print("== store_wm_fact: in-memory 更新 + jsonl 永続化 ==")
    _setup_tmp_memory(tmp_path)
    wm = init_world_model()
    fact = store_wm_fact(wm, "ゆう", "role", "開発者", confidence=0.8)
    jsonl = tmp_path / "wm.jsonl"
    results = [
        _assert(jsonl.exists(), "memory/wm.jsonl 生成"),
        _assert(fact["value"] == "開発者", "fact value"),
        _assert(abs(fact["confidence"] - 0.8) < 1e-6, "fact confidence"),
    ]
    ent = get_entity(wm, "ent_ゆう")
    results.append(_assert(ent is not None, "entity ent_ゆう 生成"))
    results.append(_assert(len(ent["facts"]) == 1, "facts 1 件"))
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    results.append(_assert(len(lines) == 1, "jsonl 1 行"))
    rec = json.loads(lines[0])
    results.append(_assert(rec["network"] == "wm", "network=wm"))
    results.append(_assert(rec["metadata"]["entity_name"] == "ゆう", "metadata entity_name"))
    results.append(_assert(rec["metadata"]["fact_key"] == "role", "metadata fact_key"))
    return all(results)


def test_beta_plus_reinforcement(tmp_path: Path):
    print("== β+: 同じ fact 再観察で confidence 上昇 ==")
    _setup_tmp_memory(tmp_path)
    wm = init_world_model()
    f1 = store_wm_fact(wm, "ゆう", "role", "開発者", confidence=0.7)
    c1 = f1["confidence"]
    f2 = store_wm_fact(wm, "ゆう", "role", "開発者", confidence=0.7)
    c2 = f2["confidence"]
    results = [
        _assert(c2 > c1, f"confidence 上昇 {c1:.4f} → {c2:.4f}"),
        _assert(f2["observation_count"] == 2, "observation_count = 2"),
    ]
    return all(results)


def test_bitemporal_contradiction(tmp_path: Path):
    print("== bitemporal: 矛盾 fact → 旧 fact 凍結 + 新 fact ==")
    _setup_tmp_memory(tmp_path)
    wm = init_world_model()
    store_wm_fact(wm, "ゆう", "role", "開発者", confidence=0.8)
    store_wm_fact(wm, "ゆう", "role", "共生者", confidence=0.8)
    ent = get_entity(wm, "ent_ゆう")
    results = [_assert(len(ent["facts"]) == 2, "facts 2 件 (旧+新)")]
    # 旧 fact: valid_to が設定済
    old_facts = [f for f in ent["facts"] if f.get("valid_to") is not None]
    new_facts = [f for f in ent["facts"] if f.get("valid_to") is None]
    results.append(_assert(len(old_facts) == 1, "凍結 fact 1 件"))
    results.append(_assert(len(new_facts) == 1, "現行 fact 1 件"))
    results.append(_assert(old_facts[0]["value"] == "開発者", "旧 fact value"))
    results.append(_assert(new_facts[0]["value"] == "共生者", "新 fact value"))
    # find_fact は現行のみ返す
    current = find_fact(ent, "role")
    results.append(_assert(current is not None and current["value"] == "共生者",
                          "find_fact は現行 value"))
    return all(results)


def test_rebuild_from_records_pure(tmp_path: Path):
    print("== rebuild_wm_from_jsonl: pure records 経由の再構築 ==")
    _setup_tmp_memory(tmp_path)
    wm = init_world_model()
    records = [
        {"metadata": {"entity_name": "ゆう", "entity_id": "ent_ゆう",
                      "fact_key": "role", "fact_value": "開発者",
                      "confidence": 0.8}},
        {"metadata": {"entity_name": "一ノ瀬", "entity_id": "ent_一ノ瀬",
                      "fact_key": "role", "fact_value": "対話者",
                      "confidence": 0.7}},
    ]
    count = rebuild_wm_from_jsonl(wm, records)
    results = [
        _assert(count == 2, "処理件数 2"),
        _assert(get_entity(wm, "ent_ゆう") is not None, "ゆう 復元"),
        _assert(get_entity(wm, "ent_一ノ瀬") is not None, "一ノ瀬 復元"),
    ]
    return all(results)


def test_rebuild_graceful_invalid_records(tmp_path: Path):
    print("== rebuild_wm_from_jsonl: 不正 record skip ==")
    _setup_tmp_memory(tmp_path)
    wm = init_world_model()
    records = [
        {"metadata": {"entity_name": "ゆう", "fact_key": "role", "fact_value": "開発者"}},
        {"metadata": {}},  # 不正
        {"metadata": {"entity_name": "", "fact_key": "x"}},  # 空 entity_name
        {"metadata": {"entity_name": "x", "fact_key": ""}},  # 空 fact_key
        {},  # metadata 無し
    ]
    count = rebuild_wm_from_jsonl(wm, records)
    return _assert(count == 1, f"1 件だけ成功 (actual: {count})")


def test_roundtrip_store_then_rebuild(tmp_path: Path):
    print("== 再起動シナリオ: store → 新 WM で rebuild ==")
    _setup_tmp_memory(tmp_path)
    wm1 = init_world_model()
    store_wm_fact(wm1, "ゆう", "role", "開発者", confidence=0.8)
    store_wm_fact(wm1, "ゆう", "hobby", "音楽", confidence=0.6)
    store_wm_fact(wm1, "一ノ瀬", "role", "対話者", confidence=0.7)

    # 再起動シミュレーション: jsonl から records を読み込んで新 WM に rebuild
    from core.memory import list_records
    records = list(reversed(list_records("wm", limit=100)))  # list_records は新しい順 → 古い順へ
    wm2 = init_world_model()
    count = rebuild_wm_from_jsonl(wm2, records)

    ent_y = get_entity(wm2, "ent_ゆう")
    ent_i = get_entity(wm2, "ent_一ノ瀬")
    results = [
        _assert(count == 3, f"3 件再構築 (actual: {count})"),
        _assert(ent_y is not None, "ゆう 復元"),
        _assert(ent_i is not None, "一ノ瀬 復元"),
        _assert(len(ent_y["facts"]) == 2, "ゆう facts 2 件"),
        _assert(find_fact(ent_y, "role")["value"] == "開発者", "ゆう role"),
        _assert(find_fact(ent_y, "hobby")["value"] == "音楽", "ゆう hobby"),
    ]
    return all(results)


def test_rebuild_empty_records(tmp_path: Path):
    print("== rebuild_wm_from_jsonl: 空 records → count=0 ==")
    _setup_tmp_memory(tmp_path)
    wm = init_world_model()
    count = rebuild_wm_from_jsonl(wm, [])
    return _assert(count == 0, "空入力で 0")


def run_all():
    print("=" * 60)
    print("test_memory_materialized_view.py (段階7 Step 3)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as td:
        results = []
        for test_fn in [
            test_store_wm_fact_creates_jsonl,
            test_beta_plus_reinforcement,
            test_bitemporal_contradiction,
            test_rebuild_from_records_pure,
            test_rebuild_graceful_invalid_records,
            test_roundtrip_store_then_rebuild,
            test_rebuild_empty_records,
        ]:
            # 各テストで独立した subdir を使う (jsonl のクロス汚染回避)
            sub = Path(td) / test_fn.__name__
            sub.mkdir(exist_ok=True)
            results.append(test_fn(sub))
    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"結果: {passed}/{total} passed")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
