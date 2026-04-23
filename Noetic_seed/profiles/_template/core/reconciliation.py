"""reconciliation — 段階11-B Phase 3 Step 3.2-3.5。

memory_store 書込時に既存 fact との矛盾を LLM judge で検出し、EC 予測誤差として
state に記録する pressure-driven reconciliation。A-MEM (NeurIPS 2025) / SSGM
(2026) を参考に、bitemporal 凍結 (既存 fact を書き換えない) + pressure 加算
(段階10 EC 経路流用、新規マジックナンバー 0) で設計。

定期 align は採用しない (3 重哲学違反: feedback_internal_drive /
feedback_no_biological_mimicry / P2 metacognition_as_affordance)。矛盾は
EC 予測誤差として記録し、段階10 w_prediction_error が pressure 加算、iku が
候補選択 (affordance)。Free Energy Principle の minimalist 実装。
"""
import json
import re
from typing import Callable, Optional


def _build_contradict_prompt(new_content: str, existing_content: str) -> str:
    """矛盾 judge 用 LLM prompt (PLAN §5 Phase 3 Step 3.2 準拠、軽量 JSON 出力)。"""
    return (
        "以下 2 つの fact を比較し、矛盾の有無と度合いを判定してください:\n"
        f"\nFact A (既存): {existing_content}\n"
        f"Fact B (新規): {new_content}\n"
        "\n判定基準 (severity は 0.0-1.0 の float):\n"
        "- 矛盾なし / 無関係: severity ~0.0\n"
        "- 部分矛盾 (文脈依存): severity 0.1-0.5\n"
        "- 直接対立: severity 0.6-1.0\n"
        "\n出力は JSON のみ (他の文字を含めない):\n"
        '{"is_contradict": bool, "severity": float, "reason": str}\n'
        "- reason は 1 文の判定根拠 (smoke 分析用)"
    )


def _parse_contradict_response(response: str) -> dict:
    """LLM 応答から {is_contradict, severity, reason} を抽出 (robust parse)。

    失敗時は非矛盾 default 返却 (graceful fallback — 判断不能で矛盾記録しない)。
    """
    default = {"is_contradict": False, "severity": 0.0, "reason": ""}
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if not m:
            return default
        data = json.loads(m.group(0))
        sev = float(data.get("severity", 0.0))
        sev = max(0.0, min(1.0, sev))  # clamp 0-1
        return {
            "is_contradict": bool(data.get("is_contradict", False)),
            "severity": sev,
            "reason": str(data.get("reason", ""))[:200],
        }
    except Exception:
        return default


def _llm_judge_contradiction(new_entry: dict, existing_entry: dict,
                              llm_call_fn: Optional[Callable] = None) -> dict:
    """Tier 1/2/3 候補に対して LLM に矛盾判定させる。

    llm_call_fn=None なら core.llm.call_llm をデフォルト使用。
    LLM 呼出 / parse 失敗は非矛盾扱い (graceful、memory 書込継続原則)。
    """
    if llm_call_fn is None:
        from core.llm import call_llm
        llm_call_fn = call_llm
    prompt = _build_contradict_prompt(
        new_entry.get("content", ""),
        existing_entry.get("content", ""),
    )
    try:
        response = llm_call_fn(prompt, max_tokens=200, temperature=0.2)
        return _parse_contradict_response(response)
    except Exception as e:
        print(f"  [reconciliation] LLM judge skip (error: {e})")
        return {"is_contradict": False, "severity": 0.0, "reason": ""}


def check_on_write(new_entry: dict,
                   state: dict, *,
                   embed_fn: Optional[Callable] = None,
                   cosine_fn: Optional[Callable] = None,
                   llm_call_fn: Optional[Callable] = None,
                   limit: int = 50) -> list:
    """memory_store 書込後に既存 fact との矛盾を検出し、EC 予測誤差として記録。

    段階4 Entity Resolver 3 段 (find_similar_facts) で候補取得、同 content は
    early skip、異 content は LLM judge で矛盾判定、矛盾 severity > 0 の時
    record_ec_prediction_error(source="reconciliation") で state に記録。

    bitemporal 凍結原則: 既存 fact は書き換えない (A-MEM neighbor 書換罠回避)。
    新 fact は通常通り追加、矛盾情報は EC 誤差として独立記録される。

    Args:
        new_entry: memory_store が生成した新 entry
        state: 記録先 state dict (破壊更新)
        embed_fn / cosine_fn: 段階4 embedding 依存注入 (Tier 2/3 有効化)
        llm_call_fn: LLM judge 依存注入 (test mock 用、None でデフォルト)
        limit: find_similar_facts 走査上限

    Returns:
        [(existing_entry, tier, verdict_dict), ...] 矛盾検出した候補のみ
        (smoke 分析用、実運用は戻り値無視で state 更新のみが重要)
    """
    from core.entity_resolver import find_similar_facts
    from core.entropy import record_ec_prediction_error

    candidates = find_similar_facts(
        new_entry,
        tiers=(1, 2, 3),
        embed_fn=embed_fn,
        cosine_fn=cosine_fn,
        limit=limit,
    )

    contradictions = []
    for cand, tier in candidates:
        # 同 content は早期 skip (Tier 1 での重複、矛盾ではない)
        if new_entry.get("content", "") == cand.get("content", ""):
            continue

        verdict = _llm_judge_contradiction(new_entry, cand, llm_call_fn=llm_call_fn)

        if verdict.get("is_contradict") and verdict.get("severity", 0.0) > 0.0:
            record_ec_prediction_error(
                state,
                source="reconciliation",
                magnitude=verdict["severity"],
                reason=verdict.get("reason", ""),
                context={
                    "new_entry_id": new_entry.get("id", ""),
                    "existing_entry_id": cand.get("id", ""),
                    "tier": tier,
                },
            )
            contradictions.append((cand, tier, verdict))

    return contradictions
