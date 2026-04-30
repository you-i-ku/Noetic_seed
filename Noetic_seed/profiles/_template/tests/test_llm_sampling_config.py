"""LLM sampling config (settings.json llm_sampling) が
LM Studio Python SDK に正しく camelCase で渡るかの動作検証。

A 層 hotfix (smoke 4 段目前): gemma-4-26b の repetition loop
抑制のため、settings.json の llm_sampling (snake_case) を SDK
config (camelCase) に変換して model.respond に渡す。

責務分離:
  - settings.json 側 (snake_case): Python 慣例で読みやすく
  - SDK 渡し時 (camelCase): LM Studio SDK の LLMPredictionConfigInput に整合
  - section / 個別キー欠落 → SDK default (config dict から落とす)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


class _StubModel:
    """model.respond の引数を捕捉する SDK モック。"""
    def __init__(self, capture: dict):
        self._capture = capture

    def respond(self, chat, config=None, on_prediction_fragment=None):
        self._capture["config"] = config

        class _Result:
            content = "stub-response"
        return _Result()


def _run_with_config(llm_cfg_overrides: dict, *, max_tokens: int = 100,
                     temperature: float = 0.7) -> dict:
    """llm_cfg を一時上書きして _call_lmstudio_native を呼び、config dict を返す。"""
    from core import llm

    capture: dict = {}
    original_cfg = dict(llm.llm_cfg)
    original_cache = dict(llm._lmstudio_model_cache)
    try:
        llm.llm_cfg.clear()
        llm.llm_cfg.update(llm_cfg_overrides)
        llm._lmstudio_model_cache.clear()
        llm._lmstudio_model_cache["test-model"] = _StubModel(capture)

        with patch.object(
            llm, "_get_active_provider_config",
            return_value=("lmstudio", "http://localhost:1234/v1", "", "test-model"),
        ):
            llm._call_lmstudio_native("hi", max_tokens=max_tokens,
                                      temperature=temperature)
    finally:
        llm.llm_cfg.clear()
        llm.llm_cfg.update(original_cfg)
        llm._lmstudio_model_cache.clear()
        llm._lmstudio_model_cache.update(original_cache)

    return capture.get("config", {})


# ============================================================
# 全キー指定 → 全 camelCase 渡る
# ============================================================

def test_full_sampling_all_keys_translated():
    print("== sampling: 全キー snake_case → camelCase 変換 ==")
    cfg = _run_with_config({
        "provider": "lmstudio",
        "model": "test-model",
        "llm_sampling": {
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.01,
            "repetition_penalty": 1.1,
        },
    }, max_tokens=200, temperature=0.5)
    return all([
        _assert(cfg.get("maxTokens") == 200, "maxTokens=200"),
        _assert(cfg.get("temperature") == 0.5, "temperature=0.5"),
        _assert(cfg.get("topPSampling") == 0.9, "topPSampling=0.9"),
        _assert(cfg.get("topKSampling") == 50, "topKSampling=50"),
        _assert(cfg.get("minPSampling") == 0.01, "minPSampling=0.01"),
        _assert(cfg.get("repeatPenalty") == 1.1, "repeatPenalty=1.1"),
    ])


# ============================================================
# section 欠落 → legacy 挙動 (maxTokens + temperature のみ)
# ============================================================

def test_no_sampling_section_legacy_behavior():
    print("== sampling: section 欠落で legacy 挙動 (回帰ガード) ==")
    cfg = _run_with_config({
        "provider": "lmstudio",
        "model": "test-model",
        # llm_sampling 不在
    }, max_tokens=300, temperature=0.6)
    return all([
        _assert(cfg.get("maxTokens") == 300, "maxTokens=300"),
        _assert(cfg.get("temperature") == 0.6, "temperature=0.6"),
        _assert("topPSampling" not in cfg, "topPSampling 不在"),
        _assert("topKSampling" not in cfg, "topKSampling 不在"),
        _assert("minPSampling" not in cfg, "minPSampling 不在"),
        _assert("repeatPenalty" not in cfg, "repeatPenalty 不在"),
        _assert(len(cfg) == 2, f"config キー 2 個のみ (実={list(cfg.keys())})"),
    ])


# ============================================================
# 部分指定 → 指定キーのみ渡る (個別キー欠落は SDK default)
# ============================================================

def test_partial_sampling_only_specified_keys_translated():
    print("== sampling: 部分指定で指定キーのみ camelCase 化 ==")
    cfg = _run_with_config({
        "provider": "lmstudio",
        "model": "test-model",
        "llm_sampling": {
            "repetition_penalty": 1.05,
            # top_p / top_k / min_p は欠落
        },
    })
    return all([
        _assert(cfg.get("repeatPenalty") == 1.05, "repeatPenalty 渡る"),
        _assert("topPSampling" not in cfg, "topPSampling 不在"),
        _assert("topKSampling" not in cfg, "topKSampling 不在"),
        _assert("minPSampling" not in cfg, "minPSampling 不在"),
    ])


# ============================================================
# _comment フィールドは config に混入しない (snake → camel ホワイトリスト)
# ============================================================

def test_comment_field_not_passed_to_sdk():
    print("== sampling: _comment は SDK config に混入しない ==")
    cfg = _run_with_config({
        "provider": "lmstudio",
        "model": "test-model",
        "llm_sampling": {
            "repetition_penalty": 1.05,
            "_comment": "explanatory note",
        },
    })
    return all([
        _assert(cfg.get("repeatPenalty") == 1.05, "repeatPenalty 渡る"),
        _assert("_comment" not in cfg, "_comment 不在"),
    ])


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("全キー snake → camelCase", test_full_sampling_all_keys_translated),
        ("section 欠落で legacy 挙動", test_no_sampling_section_legacy_behavior),
        ("部分指定で指定キーのみ", test_partial_sampling_only_specified_keys_translated),
        ("_comment は SDK に混入しない", test_comment_field_not_passed_to_sdk),
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
