"""call_llm の repetition guard (B 層) 動作検証。

A 層 (sampling 設定) で抜けた残りの repetition loop を救うため、
call_llm を _call_llm_inner で wrap し、出力に reptition を検知
したら temperature を上げて自動 retry する。

実 smoke 4 段目 cycle 1 で観察された「として、として、…」
「,note:note:note:...」ループを再現テストで検知する。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# _detect_repetition
# ============================================================

def test_detect_normal_text_no_repetition():
    print("== detect: 健全な日本語 → False ==")
    from core.llm import _detect_repetition
    text = (
        "今日は smoke テストを開始します。"
        "iku は新しい profile で記憶を蓄えていきます。"
        "観察項目は 4 つあって、それぞれ独立しています。"
    )
    return _assert(_detect_repetition(text) is False, "通常文で False")


def test_detect_actual_smoke_loop():
    print("== detect: 実 smoke 4 段目の「として、」ループ → True ==")
    from core.llm import _detect_repetition
    # cycle 1 の実出力を再現
    text = (
        "長期記憶（LTM）に自身のアイデンティティの役割を定義を固定的な、"
        + ("として、" * 20)
    )
    return _assert(_detect_repetition(text) is True,
                   "「として、」20 回反復で True")


def test_detect_short_text_skip():
    print("== detect: 短すぎる文は False (誤検知防止) ==")
    from core.llm import _detect_repetition
    # ngram_size=5, threshold=4 → 20 文字未満は早期 False
    short = "短い文。"
    return _assert(_detect_repetition(short) is False,
                   "短文 False")


def test_detect_natural_japanese_no_false_positive():
    print("== detect: 自然な日本語反復は誤検知しない ==")
    from core.llm import _detect_repetition
    text = (
        "猫は外を見ています。鳥は空を飛んでいます。魚は水を泳いでいます。"
        "犬は道を歩いています。"
    )
    return _assert(_detect_repetition(text) is False,
                   "自然反復で False")


def test_detect_note_repetition():
    print("== detect: ',note:note:note:...' ループ → True ==")
    from core.llm import _detect_repetition
    text = (",note:" * 30)
    return _assert(_detect_repetition(text) is True,
                   "note 反復で True")


# ============================================================
# call_llm: repetition guard wrap
# ============================================================

def test_call_llm_no_repetition_no_retry():
    print("== call_llm: 健全出力なら retry なし ==")
    from core import llm
    call_history: list = []

    def _stub_inner(prompt, max_tokens, temperature, image_path, image_paths):
        call_history.append({"temperature": temperature})
        return "健全な応答です。"

    with patch.object(llm, "_call_llm_inner", side_effect=_stub_inner):
        result = llm.call_llm("hi", temperature=0.7, max_retry=2)
    return all([
        _assert(result == "健全な応答です。", "正常出力 透過"),
        _assert(len(call_history) == 1, f"call 1 回 (実={len(call_history)})"),
        _assert(call_history[0]["temperature"] == 0.7, "temperature 不変"),
    ])


def test_call_llm_repetition_triggers_retry_with_higher_temp():
    print("== call_llm: 反復検知で retry、temperature 上昇 ==")
    from core import llm
    call_history: list = []
    bad_output = "として、" * 20

    def _stub_inner(prompt, max_tokens, temperature, image_path, image_paths):
        call_history.append({"temperature": temperature})
        if len(call_history) == 1:
            return bad_output
        return "正常な再生成出力。"

    with patch.object(llm, "_call_llm_inner", side_effect=_stub_inner):
        result = llm.call_llm("hi", temperature=0.7, max_retry=2)
    return all([
        _assert(result == "正常な再生成出力。", "retry 後の健全出力を返す"),
        _assert(len(call_history) == 2, f"call 2 回 (実={len(call_history)})"),
        _assert(call_history[0]["temperature"] == 0.7, "1 回目 0.7"),
        _assert(abs(call_history[1]["temperature"] - 0.9) < 0.001,
                f"2 回目 0.9 (0.7+0.2、実={call_history[1]['temperature']})"),
    ])


def test_call_llm_max_retry_exhaustion_returns_last():
    print("== call_llm: max_retry 超過で最後の出力を返す (諦め) ==")
    from core import llm
    call_history: list = []
    bad_output = "として、" * 20

    def _stub_inner(prompt, max_tokens, temperature, image_path, image_paths):
        call_history.append({"temperature": temperature})
        return bad_output

    with patch.object(llm, "_call_llm_inner", side_effect=_stub_inner):
        result = llm.call_llm("hi", temperature=0.7, max_retry=2)
    return all([
        _assert(result == bad_output, "諦めて最後の出力を返す"),
        _assert(len(call_history) == 3,
                f"初回 + retry 2 回 = 3 (実={len(call_history)})"),
    ])


def test_call_llm_max_retry_zero_disables_guard():
    print("== call_llm: max_retry=0 で guard 無効 (deterministic) ==")
    from core import llm
    call_history: list = []
    bad_output = "として、" * 20

    def _stub_inner(prompt, max_tokens, temperature, image_path, image_paths):
        call_history.append({"temperature": temperature})
        return bad_output

    with patch.object(llm, "_call_llm_inner", side_effect=_stub_inner):
        result = llm.call_llm("hi", temperature=0.7, max_retry=0)
    return all([
        _assert(result == bad_output, "guard 無効、生出力をそのまま返す"),
        _assert(len(call_history) == 1, f"call 1 回 (実={len(call_history)})"),
    ])


def test_call_llm_temperature_capped_at_1_2():
    print("== call_llm: retry temperature は 1.2 で cap ==")
    from core import llm
    call_history: list = []
    bad_output = "として、" * 20

    def _stub_inner(prompt, max_tokens, temperature, image_path, image_paths):
        call_history.append({"temperature": temperature})
        return bad_output

    with patch.object(llm, "_call_llm_inner", side_effect=_stub_inner):
        llm.call_llm("hi", temperature=1.0, max_retry=3)
    retry_temps = [c["temperature"] for c in call_history[1:]]
    return all([
        _assert(all(t <= 1.2 for t in retry_temps),
                f"全 retry が 1.2 以下 (実={retry_temps})"),
        _assert(retry_temps[-1] == 1.2, f"末尾は 1.2 (実={retry_temps[-1]})"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("detect: 健全文 False", test_detect_normal_text_no_repetition),
        ("detect: 「として、」ループ True", test_detect_actual_smoke_loop),
        ("detect: 短文 False", test_detect_short_text_skip),
        ("detect: 自然反復 False (誤検知防止)",
         test_detect_natural_japanese_no_false_positive),
        ("detect: note ループ True", test_detect_note_repetition),
        ("call_llm: retry なし通過", test_call_llm_no_repetition_no_retry),
        ("call_llm: retry で温度上昇",
         test_call_llm_repetition_triggers_retry_with_higher_temp),
        ("call_llm: max_retry 超過諦め",
         test_call_llm_max_retry_exhaustion_returns_last),
        ("call_llm: max_retry=0 で guard off",
         test_call_llm_max_retry_zero_disables_guard),
        ("call_llm: temperature 1.2 cap",
         test_call_llm_temperature_capped_at_1_2),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
