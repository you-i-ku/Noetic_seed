"""段階10 柱 C: predicted_ec prompt parsing テスト。

STAGE10 PLAN v1 §3-C の要件:
  - parse_candidates: 行末尾の「/ predicted_ec: 0.XX」を抽出して
    candidate.prediction.predicted_ec に格納 (0.0-1.0 clamp)
  - predicted_e2 と predicted_ec は併記形式 (両方ある行で両方抽出)
  - predicted_ec 欠如時は後方互換 (predicted_e2 のみで prediction 作成)
  - 不正値 (1.5 等) は clamp、NaN は skip
  - tool 抽出ロジックへのノイズ干渉なし

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_ec_prediction_parsing.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.parser import parse_candidates


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


ALLOWED = {"output_display", "read_file", "wait", "glob_search", "bash"}


# ============================================================
# predicted_ec 抽出
# ============================================================

def test_pec_basic_extraction():
    print("== parse: 「predicted_e2 + predicted_ec」併記を両方抽出 ==")
    text = """1. [意図A] → output_display / predicted_e2: 80 / predicted_ec: 0.6
2. [意図B] → read_file / predicted_e2: 50 / predicted_ec: 0.3"""
    cands = parse_candidates(text, ALLOWED)
    return all([
        _assert(len(cands) == 2, f"2 候補抽出 (actual: {len(cands)})"),
        _assert(cands[0]["prediction"]["predicted_e2"] == 80, "cand 0 pe2=80"),
        _assert(abs(cands[0]["prediction"]["predicted_ec"] - 0.6) < 1e-9, "cand 0 pec=0.6"),
        _assert(cands[0]["prediction"]["source"] == "medium", "cand 0 source=medium"),
        _assert(cands[1]["prediction"]["predicted_e2"] == 50, "cand 1 pe2=50"),
        _assert(abs(cands[1]["prediction"]["predicted_ec"] - 0.3) < 1e-9, "cand 1 pec=0.3"),
    ])


def test_pec_missing_keeps_backward_compat():
    print("== parse: predicted_ec 欠如行は predicted_e2 のみで prediction 作成 ==")
    text = "1. [意図] → output_display / predicted_e2: 70"
    cands = parse_candidates(text, ALLOWED)
    p = cands[0].get("prediction", {})
    return all([
        _assert(p.get("predicted_e2") == 70, "pe2=70"),
        _assert("predicted_ec" not in p, "predicted_ec は付加されない (欠如)"),
        _assert(p.get("source") == "medium", "source=medium (pe2 だけで medium 扱い)"),
    ])


def test_pec_clamp_range():
    print("== parse: predicted_ec を [0.0, 1.0] に clamp ==")
    text = """1. [意図X] → output_display / predicted_e2: 50 / predicted_ec: 1.5
2. [意図Y] → read_file / predicted_e2: 50 / predicted_ec: -0.3
3. [意図Z] → wait / predicted_e2: 50 / predicted_ec: 0.5"""
    cands = parse_candidates(text, ALLOWED)
    return all([
        _assert(abs(cands[0]["prediction"]["predicted_ec"] - 1.0) < 1e-9, "1.5 → clamp 1.0"),
        _assert(abs(cands[1]["prediction"]["predicted_ec"] - 0.0) < 1e-9, "-0.3 → clamp 0.0"),
        _assert(abs(cands[2]["prediction"]["predicted_ec"] - 0.5) < 1e-9, "0.5 → 0.5"),
    ])


def test_pec_invalid_value_skipped():
    print("== parse: predicted_ec の不正値は skip (predicted_e2 は保持) ==")
    text = "1. [意図] → output_display / predicted_e2: 80 / predicted_ec: abc"
    cands = parse_candidates(text, ALLOWED)
    p = cands[0].get("prediction", {})
    return all([
        _assert(p.get("predicted_e2") == 80, "pe2=80 保持"),
        _assert("predicted_ec" not in p, "不正 pec は付加されない"),
    ])


def test_pec_tool_extraction_intact():
    print("== parse: pec 併記で tool 抽出が壊れない (chain は tools list に展開) ==")
    # 注: parser 仕様で cand["tool"] は chain 先頭、cand["tools"] が全 list
    # 実際のテスト用に chain は別名 tool 2 つで構成 (重複排除されないこと確認)
    text = """1. [意図] → output_display / predicted_e2: 80 / predicted_ec: 0.5
2. [意図] → read_file+glob_search / predicted_e2: 70 / predicted_ec: 0.4"""
    cands = parse_candidates(text, ALLOWED)
    return all([
        _assert(cands[0]["tool"] == "output_display", f"cand 0 tool=output_display (actual: {cands[0]['tool']})"),
        _assert(cands[1]["tool"] == "read_file", f"cand 1 chain 先頭 tool=read_file (actual: {cands[1]['tool']})"),
        _assert(cands[1].get("tools") == ["read_file", "glob_search"], f"cand 1 tools=[read_file, glob_search] (actual: {cands[1].get('tools')})"),
    ])


def test_pec_decimal_formats():
    print("== parse: 小数表記バリエーション (0.X / .X / 1.0) ==")
    text = """1. [a] → output_display / predicted_e2: 50 / predicted_ec: 0.75
2. [b] → read_file / predicted_e2: 50 / predicted_ec: .4
3. [c] → wait / predicted_e2: 50 / predicted_ec: 1"""
    cands = parse_candidates(text, ALLOWED)
    return all([
        _assert(abs(cands[0]["prediction"]["predicted_ec"] - 0.75) < 1e-9, "0.75"),
        _assert(abs(cands[1]["prediction"]["predicted_ec"] - 0.4) < 1e-9, ".4 → 0.4"),
        _assert(abs(cands[2]["prediction"]["predicted_ec"] - 1.0) < 1e-9, "1 → 1.0"),
    ])


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    tests = [
        test_pec_basic_extraction,
        test_pec_missing_keeps_backward_compat,
        test_pec_clamp_range,
        test_pec_invalid_value_skipped,
        test_pec_tool_extraction_intact,
        test_pec_decimal_formats,
    ]
    results = [t() for t in tests]
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
