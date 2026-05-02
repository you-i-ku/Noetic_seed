"""reconciliation — 段階11-B Phase 3 Step 3.2-3.5 + 段階13 Phase 0.4。

memory_store 書込時に既存 fact との矛盾を LLM judge で検出し、EC 予測誤差として
state に記録する pressure-driven reconciliation。A-MEM (NeurIPS 2025) / SSGM
(2026) を参考に、bitemporal 凍結 (既存 fact を書き換えない) + pressure 加算
(段階10 EC 経路流用、新規マジックナンバー 0) で設計。

定期 align は採用しない (3 重哲学違反: feedback_internal_drive /
feedback_no_biological_mimicry / P2 metacognition_as_affordance)。矛盾は
EC 予測誤差として記録し、段階10 w_prediction_error が pressure 加算、iku が
候補選択 (affordance)。Free Energy Principle の minimalist 実装。

段階13 Phase 0.4: check_raw_subjective_gap 追加。tool 実行直後の raw event
(物理 fact = result + args) と subjective entry (LLM 解釈 = intent + expect)
の意味的 gap を bge-m3 cosine 距離で検出、check_on_write の sibling として
pressure 経路に流す。閾値は entity_resolver の Tier (0.85 / 0.70) 流用 =
新規マジックナンバー 0 維持。
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


# ============================================================
# 段階13 Phase 0.4: raw event vs subjective entry gap 検出
# ============================================================

def check_raw_subjective_gap(state: dict, entry: dict, *,
                             embed_fn: Optional[Callable] = None,
                             cosine_fn: Optional[Callable] = None) -> Optional[dict]:
    """tool 実行直後に raw event と subjective entry の意味的 gap を検出。

    event_emitter (Phase 0.2) の subscriber として配線され、_record_entry で
    raw_events / subjective_entries 両側に append された直後に発火 (subscribe
    順序 snapshot 経由)。同 id の raw_part / subj_part を取得し、bge-m3 で
    embedding 化、cosine 距離で意味的 gap を測る。

    比較対象 (LLM の視界には入らない、embedding 入力専用 text):
      raw 側 text  = result + " " + str(args)        # 物理 fact = 何が起きたか
      subj 側 text = intent + " " + expect            # LLM 解釈 = 何のつもりで

    注意 (Codex review 0.4 WARN 反映): text 組み立ては Phase 0.4 lean。`str(args)`
    が辞書 repr で raw_text を支配したり、長い `expect` が subj_text を支配したり
    する可能性あり。smoke で false positive 観察したら field 選択や正規化
    (args の重要 key 抽出、expect 文字数 cap 等) を Phase 1+ で調整予定。

    順序契約 (Codex review 0.4 WARN 反映): event_emitter subscribe snapshot で
    _record_entry が先に発火する前提。_record_entry が例外で失敗した場合は
    event_emitter の isolation で本関数も呼ばれるが、raw_part / subj_part が
    in-memory view に反映されてない状態 = 下記 graceful skip (part 不在 → None)
    で吸収される。Phase 0.5+ で critical observer 機構 (_record_entry を例外伝播
    させる) を入れるなら、本関数も同期保証強化される。

    閾値 (entity_resolver.py Tier 流用、新規マジックナンバー 0):
      gap = 1 - cosine_similarity(embed(raw_text), embed(subj_text))
      - gap >= (1 - EMBEDDING_DIFFERENT_THRESHOLD) ≒ 0.30: record (確定 gap)
      - (1 - EMBEDDING_SAME_THRESHOLD) ≒ 0.15 <= gap < 0.30: Tier 3 ambiguous zone
        (LLM judge 推奨ゾーン、Phase 0.4 lean では skip、Phase 1+ で拡張)
      - gap < 0.15: 同義扱い、skip

    記録 (check_on_write sibling): gap >= 0.30 で record_ec_prediction_error
    (source="raw_subj_gap") を呼び、段階10 EC 経路 (state["prediction_error_history_ec"])
    に magnitude 記録。pressure 加算 (w_prediction_error) で iku が gap を体感する
    affordance 経路。

    Args:
        state: 記録先 state dict (破壊更新)
        entry: event_emitter から渡される entry (id 経由で raw_part / subj_part 検索)
        embed_fn: embedding 依存注入 (test mock 用、None で _embed_sync)
        cosine_fn: cosine 依存注入 (test mock 用、None で cosine_similarity)

    Returns:
        verdict dict (id, gap, raw_excerpt, subj_excerpt) or None
        - None: skip (id 不在 / part 不在 / 比較材料空 / embedding 不在 / gap 閾値未満)
        - dict: gap 検出して record した
    """
    from core.embedding import is_vector_ready, _embed_sync, cosine_similarity
    from core.entity_resolver import EMBEDDING_SAME_THRESHOLD, EMBEDDING_DIFFERENT_THRESHOLD
    from core.entropy import record_ec_prediction_error

    # 軸 9: subscribe 順序 snapshot 既存契約 = _record_entry 後に発火
    # 軸 1+2: 閾値 entity_resolver 流用、gap = 1 - cosine
    record_threshold = 1.0 - EMBEDDING_DIFFERENT_THRESHOLD  # ≒ 0.30
    # ambiguous_threshold は Phase 0.4 lean では未使用 (Tier 3 zone は丸ごと skip)。
    # Phase 1+ で LLM judge 拡張する時に `if ambiguous_threshold <= gap < record_threshold`
    # の判定で使う予定の予約変数として明示残置 (削除すると拡張時再導入忘れる)。
    ambiguous_threshold = 1.0 - EMBEDDING_SAME_THRESHOLD   # ≒ 0.15
    _ = ambiguous_threshold  # noqa: F841 (Phase 1+ 予約、削除しない)

    entry_id = entry.get("id")
    if not entry_id:
        return None  # id 不在で対応取れず skip

    # 同 id の raw_part / subj_part を末尾から探す (event_emitter で直前 append された)
    raw_events = state.get("raw_events", [])
    subj_entries = state.get("subjective_entries", [])
    raw_part = next((e for e in reversed(raw_events) if e.get("id") == entry_id), None)
    subj_part = next((e for e in reversed(subj_entries) if e.get("id") == entry_id), None)
    if not raw_part or not subj_part:
        return None  # 軸 7 graceful: part 不在で skip

    # 軸 3: raw=result+args, subj=intent+expect (LLM 視界外、embedding 入力専用)
    raw_text = (str(raw_part.get("result", "")) + " " + str(raw_part.get("args", ""))).strip()
    subj_text = (str(subj_part.get("intent", "")) + " " + str(subj_part.get("expect", ""))).strip()

    if not raw_text or not subj_text:
        return None  # 軸 8 graceful: 比較材料空で skip

    if not is_vector_ready():
        return None  # 軸 7 graceful: embedding 不在で skip

    # 軸 2: gap = 1 - cosine_similarity (bge-m3 既存基盤流用)
    if embed_fn is None:
        embed_fn = _embed_sync
    if cosine_fn is None:
        cosine_fn = cosine_similarity
    try:
        vecs = embed_fn([raw_text, subj_text])
    except Exception as exc:
        print(f"  [reconciliation] gap embed skip (error: {exc})")
        return None
    if not vecs or len(vecs) != 2:
        return None
    sim = cosine_fn(vecs[0], vecs[1])
    gap = max(0.0, min(1.0, 1.0 - sim))

    # 軸 4: Phase 0.4 lean = gap >= record_threshold のみ record、
    # ambiguous_threshold <= gap < record_threshold は Tier 3 LLM judge ゾーンで skip
    if gap < record_threshold:
        return None

    # 軸 6: record_ec_prediction_error (check_on_write sibling)
    record_ec_prediction_error(
        state,
        source="raw_subj_gap",
        magnitude=gap,
        reason=f"raw vs subjective gap (id={entry_id})",
        context={
            "id": entry_id,
            "raw_excerpt": raw_text[:100],
            "subj_excerpt": subj_text[:100],
        },
    )

    return {
        "id": entry_id,
        "gap": gap,
        "raw_excerpt": raw_text[:100],
        "subj_excerpt": subj_text[:100],
    }
