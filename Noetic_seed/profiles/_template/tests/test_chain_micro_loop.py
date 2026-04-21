"""test_chain_micro_loop.py — 段階10.5 Fix 1 (+ Bug fix) の単体テスト。

検証対象:
  - parser.parse_candidates: 新形式 `tool (pe2=X, pec=Y)` の個別抽出
  - parser.parse_candidates: 旧形式 `/ predicted_e2: X / predicted_ec: Y` (chain 全体 1 組) の後方互換
  - candidate["chain"] 構造 (tool 単位 predicted_e2/predicted_ec)
  - candidate["prediction"] は chain[0] 由来の後方互換
  - predictor.migrate_chain_keys: state.predictor_confidence の "+" 含むキー drop
  - predictor.clamp_ec: actual_ec を 0.0-1.0 に clamp

段階10.5 Fix 1 設計判断 (ゆう 2026-04-21 確定):
  - α: LLM① に個別 pe2/pec を出させる (LLM call 回数維持、構造で誘導)
  - Y: 1 cycle = 1 log entry + entry["per_tool"] に tool 単位 metadata
       → log 消費者への影響ゼロ、predictor_confidence だけ tool 粒度化
  - drop: 旧 predictor_confidence の "+" 含むキーは新 smoke 起点で drop

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_chain_micro_loop.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_candidates
from core.predictor import migrate_chain_keys, clamp_ec


results = []


def _assert(cond, msg):
    results.append((bool(cond), msg))
    return bool(cond)


ALLOWED = {"read_file", "glob_search", "output_display", "memory_store",
           "reflect", "write_file", "bash"}


# ============================================================
# Section 1: LLM① 新形式 parse (5 ケース)
# ============================================================
print("=== Section 1: LLM① 新形式 parse ===")

# 1-1: 単一 tool + 新形式
text = "1. [ファイル読む] → read_file (pe2=90, pec=0.3)"
cands = parse_candidates(text, ALLOWED)
_assert(len(cands) == 1, "1-1 candidate 数 = 1")
_assert(cands[0]["tool"] == "read_file", "1-1 tool = read_file")
_assert("chain" in cands[0], "1-1 chain フィールド存在")
_assert(len(cands[0]["chain"]) == 1, "1-1 chain 長 = 1")
_assert(cands[0]["chain"][0]["tool"] == "read_file", "1-1 chain[0].tool")
_assert(cands[0]["chain"][0]["predicted_e2"] == 90, "1-1 chain[0].pe2 = 90")
_assert(cands[0]["chain"][0]["predicted_ec"] == 0.3, "1-1 chain[0].pec = 0.3")

# 1-2: chain 2 tool + 新形式 (個別 pe2/pec)
text = "1. [探索と読] → read_file (pe2=90, pec=0.3) + glob_search (pe2=80, pec=0.6)"
cands = parse_candidates(text, ALLOWED)
_assert(len(cands) == 1, "1-2 candidate 数 = 1")
_assert(cands[0]["tools"] == ["read_file", "glob_search"], "1-2 tools list")
_assert(len(cands[0]["chain"]) == 2, "1-2 chain 長 = 2")
_assert(cands[0]["chain"][0]["predicted_e2"] == 90, "1-2 chain[0].pe2")
_assert(cands[0]["chain"][0]["predicted_ec"] == 0.3, "1-2 chain[0].pec")
_assert(cands[0]["chain"][1]["predicted_e2"] == 80, "1-2 chain[1].pe2")
_assert(cands[0]["chain"][1]["predicted_ec"] == 0.6, "1-2 chain[1].pec")
_assert(cands[0]["chain"][0]["tool"] == "read_file", "1-2 chain[0].tool")
_assert(cands[0]["chain"][1]["tool"] == "glob_search", "1-2 chain[1].tool")

# 1-3: pe2/pec の上限 clamp (0-100 / 0.0-1.0)
text = "1. [test] → read_file (pe2=150, pec=2.0)"
cands = parse_candidates(text, ALLOWED)
_assert(cands[0]["chain"][0]["predicted_e2"] == 100, "1-3 pe2 上限 clamp=100")
_assert(cands[0]["chain"][0]["predicted_ec"] == 1.0, "1-3 pec 上限 clamp=1.0")

# 1-3b: pe2/pec の下限 clamp
text = "1. [test] → read_file (pe2=-10, pec=-0.5)"
cands = parse_candidates(text, ALLOWED)
_assert(cands[0]["chain"][0]["predicted_e2"] == 0, "1-3b pe2 下限 clamp=0")
_assert(cands[0]["chain"][0]["predicted_ec"] == 0.0, "1-3b pec 下限 clamp=0.0")

# 1-4: 新形式 pe2 のみ (pec 欠落)
text = "1. [test] → read_file (pe2=90)"
cands = parse_candidates(text, ALLOWED)
_assert(cands[0]["chain"][0]["predicted_e2"] == 90, "1-4 pe2 のみ")
_assert(cands[0]["chain"][0]["predicted_ec"] is None, "1-4 pec=None")

# 1-5: candidate["prediction"] は chain[0] 由来の後方互換
text = "1. [test] → read_file (pe2=90, pec=0.3) + glob_search (pe2=80, pec=0.6)"
cands = parse_candidates(text, ALLOWED)
_assert(cands[0].get("prediction", {}).get("predicted_e2") == 90,
        "1-5 prediction.pe2 = chain[0].pe2")
_assert(cands[0].get("prediction", {}).get("predicted_ec") == 0.3,
        "1-5 prediction.pec = chain[0].pec")


# ============================================================
# Section 2: 旧形式 parse (後方互換、3 ケース)
# ============================================================
print("=== Section 2: 旧形式 後方互換 parse ===")

# 2-1: 旧形式 chain 全体 1 組 → 全 tool に複製
text = "1. [test] → read_file+glob_search / predicted_e2: 85 / predicted_ec: 0.5"
cands = parse_candidates(text, ALLOWED)
_assert(len(cands[0]["chain"]) == 2, "2-1 chain 長 = 2")
_assert(cands[0]["chain"][0]["predicted_e2"] == 85, "2-1 chain[0].pe2 = 85 (複製)")
_assert(cands[0]["chain"][1]["predicted_e2"] == 85, "2-1 chain[1].pe2 = 85 (複製)")
_assert(cands[0]["chain"][0]["predicted_ec"] == 0.5, "2-1 chain[0].pec = 0.5 (複製)")
_assert(cands[0]["chain"][1]["predicted_ec"] == 0.5, "2-1 chain[1].pec = 0.5 (複製)")

# 2-2: 旧形式 pe2 のみ (pec 欠落)
text = "1. [test] → read_file / predicted_e2: 75"
cands = parse_candidates(text, ALLOWED)
_assert(cands[0]["chain"][0]["predicted_e2"] == 75, "2-2 pe2=75 (旧形式)")
_assert(cands[0]["chain"][0]["predicted_ec"] is None, "2-2 pec=None")

# 2-3: 予測なし (chain は存在、pe2/pec は None)
text = "1. [test] → read_file"
cands = parse_candidates(text, ALLOWED)
_assert("chain" in cands[0], "2-3 chain フィールド存在")
_assert(len(cands[0]["chain"]) == 1, "2-3 chain 長 = 1")
_assert(cands[0]["chain"][0]["predicted_e2"] is None, "2-3 chain[0].pe2=None")
_assert(cands[0]["chain"][0]["predicted_ec"] is None, "2-3 chain[0].pec=None")
_assert("prediction" not in cands[0], "2-3 prediction フィールド欠落")


# ============================================================
# Section 3: candidate chain 構造 (3 ケース)
# ============================================================
print("=== Section 3: candidate chain 構造 ===")

# 3-1: 単一 tool で chain list 長さ 1
text = "1. [test] → read_file (pe2=50, pec=0.5)"
cands = parse_candidates(text, ALLOWED)
_assert(isinstance(cands[0]["chain"], list), "3-1 chain は list")
_assert(len(cands[0]["chain"]) == len(cands[0]["tools"]),
        "3-1 chain 長 = tools 長")

# 3-2: tools 順序 = chain 順序
text = "1. [test] → glob_search (pe2=70, pec=0.4) + read_file (pe2=50, pec=0.3)"
cands = parse_candidates(text, ALLOWED)
_assert(cands[0]["tools"][0] == cands[0]["chain"][0]["tool"],
        "3-2 tools[0] = chain[0].tool")
_assert(cands[0]["tools"][1] == cands[0]["chain"][1]["tool"],
        "3-2 tools[1] = chain[1].tool")

# 3-3: 許容 tool 外は除去、chain も同期
text = "1. [test] → unknown_tool (pe2=90, pec=0.3) + read_file (pe2=80, pec=0.6)"
cands = parse_candidates(text, ALLOWED)
_assert(len(cands) >= 1, "3-3 candidate 生成")
_assert(cands[0]["tools"] == ["read_file"], "3-3 不許可 tool 除去")
_assert(len(cands[0]["chain"]) == 1, "3-3 chain も同期")
_assert(cands[0]["chain"][0]["predicted_e2"] == 80,
        "3-3 除去後の chain[0] は残った tool の pe2")


# ============================================================
# Section 4: migration helper (2 ケース)
# ============================================================
print("=== Section 4: migrate_chain_keys ===")

# 4-1: "+" 含むキー drop、tool 単位キー保持
state = {
    "predictor_confidence": {
        "read_file+glob_search": {"e2_conf": 0.3, "ec_conf": 0.6},
        "write_file+update_self": {"e2_conf": 0.5, "ec_conf": 0.4},
        "read_file": {"e2_conf": 0.7, "ec_conf": 0.7},
        "glob_search": {"e2_conf": 0.65, "ec_conf": 0.55},
    }
}
dropped = migrate_chain_keys(state)
_assert(dropped == 2, f"4-1 drop 件数 = 2 (actual: {dropped})")
pc = state["predictor_confidence"]
_assert("read_file+glob_search" not in pc, "4-1 chain key drop")
_assert("write_file+update_self" not in pc, "4-1 chain key drop (2)")
_assert("read_file" in pc, "4-1 tool key 保持")
_assert("glob_search" in pc, "4-1 tool key 保持 (2)")
_assert(pc["read_file"]["e2_conf"] == 0.7, "4-1 tool key の値不変")

# 4-2: predictor_confidence 未 init でも crash しない + drop 0
state2 = {}
dropped2 = migrate_chain_keys(state2)
_assert(dropped2 == 0, "4-2 未 init で drop=0")
_assert(state2.get("predictor_confidence", {}) == {},
        "4-2 未 init で {} 維持 (または未追加)")


# ============================================================
# Section 5: clamp_ec helper (3 ケース)
# ============================================================
print("=== Section 5: clamp_ec ===")

# 5-1: 上限 clamp
_assert(clamp_ec(1.5) == 1.0, "5-1 上限 clamp: 1.5 → 1.0")
_assert(clamp_ec(100.0) == 1.0, "5-1b 上限 clamp: 100.0 → 1.0")

# 5-2: 下限 clamp + 正常範囲
_assert(clamp_ec(-0.5) == 0.0, "5-2 下限 clamp: -0.5 → 0.0")
_assert(clamp_ec(0.5) == 0.5, "5-2b 正常範囲: 0.5 → 0.5")
_assert(clamp_ec(0.0) == 0.0, "5-2c 境界: 0.0 → 0.0")
_assert(clamp_ec(1.0) == 1.0, "5-2d 境界: 1.0 → 1.0")

# 5-3: 不正値 fallback (None, str, 空)
_assert(clamp_ec(None) == 0.0, "5-3 None → 0.0")
_assert(clamp_ec("invalid") == 0.0, "5-3b str invalid → 0.0")


# ============================================================
# 結果サマリ
# ============================================================
print("\n========== SUMMARY ==========")
passed = sum(1 for ok, _ in results if ok)
failed = [msg for ok, msg in results if not ok]
print(f"passed: {passed}/{len(results)}")
if failed:
    print(f"failed: {len(failed)}")
    for msg in failed:
        print(f"  - {msg}")
    sys.exit(1)
print("all pass")
