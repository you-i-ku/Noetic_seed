"""Controller（制御層）+ controller_select + intent-conditioned scoring"""
import re
import random
from core.config import SANDBOX_TOOLS_DIR
from core.state import load_pref
from core.embedding import _vector_ready, _embed_sync, cosine_similarity
from core.eval import predict_result_novelty


def controller(state: dict, tools_dict: dict, level_tools: dict, ai_created_tools: dict, dangerous_patterns: list, run_ai_tool_fn) -> dict:
    """E値とenergyから構造的制約を導出。ツール解放レベルを判定。"""
    energy = state.get("energy", 50)
    log = state["log"]

    # --- sandbox/tools/ をスキャンしてAI製ツールを動的ロード ---
    if SANDBOX_TOOLS_DIR.exists():
        for tool_path in sorted(SANDBOX_TOOLS_DIR.glob("*.py")):
            tname = tool_path.stem
            if tname in tools_dict:
                continue
            try:
                code = tool_path.read_text(encoding="utf-8")
                dangerous = [p for p in dangerous_patterns if p in code]
                if dangerous:
                    print(f"  [scan] {tname}: 危険パターン検出、スキップ {dangerous}")
                    continue
                namespace: dict = {}
                exec(compile(code, str(tool_path), "exec"), namespace)
                func = namespace.get("run") or namespace.get(tname)
                if func and callable(func):
                    tdesc = namespace.get("DESCRIPTION", tname)
                    ai_created_tools[tname] = func
                    tools_dict[tname] = {
                        "desc": f"[AI製] {tdesc}",
                        "func": lambda a, f=func: run_ai_tool_fn(f, a),
                    }
            except Exception as e:
                print(f"  [scan] {tname}: 読み込み失敗 ({e})")

    # --- ツール順序: 各ツールの過去E2平均で並べる ---
    tool_e2 = {}
    for entry in log:
        tool = entry.get("tool", "")
        m = re.search(r'(\d+)%', str(entry.get("e2", "")))
        if m and tool in tools_dict:
            tool_e2.setdefault(tool, []).append(int(m.group(1)))
    tool_avg = {t: sum(vs) / len(vs) for t, vs in tool_e2.items() if vs}
    for t in tools_dict:
        if t not in tool_avg:
            tool_avg[t] = 50

    pref = load_pref()
    for t in tools_dict:
        if t in pref:
            tool_avg[t] = round(min(100, max(0, tool_avg[t] * (pref[t] / 50.0))), 1)

    ranked = sorted(tools_dict.keys(), key=lambda t: tool_avg[t], reverse=True)

    # --- tool_level による段階解放 ---
    fr = set(state.get("files_read", []))
    fw = set(state.get("files_written", []))
    lv = state.get("tool_level", 0)
    new_lv = lv
    tc = state.get("tools_created", [])
    if lv == 0 and len(fr) >= 1:
        new_lv = 1
    elif lv == 1 and len(fr) >= 2:
        new_lv = 2
    elif lv == 2 and len(fr) >= 1 and len(fw) >= 1 and len(fr) + len(fw) >= 5:
        new_lv = 3
    elif lv == 3 and any(f.endswith(".py") for f in fw):
        new_lv = 4
    elif lv == 4 and len(tc) >= 1:
        new_lv = 5

    # Level 6: self_modify
    if lv == 5:
        ec_entries = [e for e in log if e.get("tool") == "exec_code"]
        ct_entries = [e for e in log if e.get("tool") == "create_tool"]
        if len(ec_entries) + len(ct_entries) >= 7 and len(ec_entries) >= 2 and len(ct_entries) >= 2:
            if tool_avg.get("exec_code", 0) >= 65 and tool_avg.get("create_tool", 0) >= 65:
                def _e2_list(entries):
                    result = []
                    for e in entries:
                        m = re.search(r'(\d+)%', str(e.get("e2", "")))
                        if m:
                            result.append(int(m.group(1)))
                    return result
                def _std(vals):
                    if len(vals) < 2:
                        return 0.0
                    mean = sum(vals) / len(vals)
                    return (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
                ec_std = _std(_e2_list(ec_entries[-3:]))
                ct_std = _std(_e2_list(ct_entries[-3:]))
                if ec_std < 20 and ct_std < 20:
                    def _err_rate(entries, tool):
                        valid = [e for e in entries if not str(e.get("result", "")).startswith("キャンセル")]
                        if not valid:
                            return 1.0
                        if tool == "exec_code":
                            errs = [e for e in valid if
                                    str(e.get("result", "")).startswith("タイムアウト") or
                                    "[stderr]" in str(e.get("result", ""))]
                        else:
                            errs = [e for e in valid if
                                    str(e.get("result", "")).startswith(("コンパイルエラー", "エラー:"))]
                        return len(errs) / len(valid)
                    if _err_rate(ec_entries, "exec_code") <= 0.3 and _err_rate(ct_entries, "create_tool") <= 0.3:
                        new_lv = 6

    allowed = set(level_tools[new_lv])

    # Xセッションがなければ X系ツールを除外
    from tools.x_tools import X_SESSION_PATH
    if not X_SESSION_PATH.exists():
        _x_tools = {"x_post", "x_reply", "x_timeline", "x_search", "x_quote", "x_like", "x_get_notifications"}
        allowed -= _x_tools

    return {
        "allowed_tools": allowed,
        "tool_rank": {t: round(tool_avg[t], 1) for t in ranked},
        "tool_level": new_lv,
        "tool_level_prev": lv,
    }


def _intent_conditioned_scores(candidates: list, state: dict) -> list:
    """候補ごとに、過去の類似intent×同toolのE2加重平均を返す。"""
    log = state.get("log", [])
    if not log or not _vector_ready:
        return [50.0] * len(candidates)

    past = []
    for e in log:
        intent = e.get("intent", "")
        tool = e.get("tool", "")
        m = re.search(r'(\d+)%', str(e.get("e2", "")))
        if intent and tool and m:
            past.append({"intent": intent, "tool": tool, "e2": int(m.group(1))})
    if not past:
        return [50.0] * len(candidates)

    candidate_texts = [c.get("reason", "") or c.get("tool", "") for c in candidates]
    past_texts = [p["intent"] for p in past]
    all_texts = candidate_texts + past_texts
    all_vecs = _embed_sync(all_texts)
    if not all_vecs or len(all_vecs) != len(all_texts):
        return [50.0] * len(candidates)

    nc = len(candidates)
    cand_vecs = all_vecs[:nc]
    past_vecs = all_vecs[nc:]

    scores = []
    for i, c in enumerate(candidates):
        tool = c["tool"]
        weighted_sum = 0.0
        weight_total = 0.0
        for j, p in enumerate(past):
            if p["tool"] != tool:
                continue
            sim = cosine_similarity(cand_vecs[i], past_vecs[j])
            if sim > 0.3:
                weighted_sum += sim * p["e2"]
                weight_total += sim
        if weight_total > 0:
            scores.append(weighted_sum / weight_total)
        else:
            scores.append(50.0)
    return scores


def _pending_priority_boost(state: dict, candidate: dict) -> float:
    """candidate が未消化 UPS v2 pending を解消しそうなら boost 倍率を返す。

    UPS v2 (Phase 4 Step E-1): 未 observed な pending のうち、
      - source_action == candidate_tool (直接対応)
      - または output_display × device channel (対等協力者への応答)
    にマッチするものの priority 最大値から倍率を算出。

    Returns:
        1.0 〜 3.0 の倍率。該当 pending なしなら 1.0。
    """
    pending = state.get("pending", [])
    candidate_tool = candidate.get("tool", "")
    if not candidate_tool:
        return 1.0

    max_pri = 0.0
    for p in pending:
        if p.get("type") != "pending":
            continue
        if p.get("observed_content") is not None:
            continue
        source = p.get("source_action", "")
        direct_match = source == candidate_tool
        device_response_match = (
            candidate_tool == "output_display"
            and (p.get("observed_channel") == "device"
                 or p.get("expected_channel") == "device")
        )
        if direct_match or device_response_match:
            pri = float(p.get("priority", 0.0))
            if pri > max_pri:
                max_pri = pri

    if max_pri <= 0.0:
        return 1.0
    # priority 想定最大 ≈ 12.0 → [1.0, 3.0] に正規化
    return 1.0 + min(2.0, max_pri / 6.0)


def controller_select(candidates: list, ctrl: dict, state: dict) -> dict:
    """D-4設計 + intent-conditioned scoring + entropy認知品質 + UPS v2 priority"""
    energy = state.get("energy", 50) / 100.0
    entropy = state.get("entropy", 0.65)
    tool_rank = ctrl.get("tool_rank", {})
    n = len(candidates)

    intent_scores = _intent_conditioned_scores(candidates, state)

    sharpness = (1 - energy) * (1 - entropy)

    weights = []
    for i, c in enumerate(candidates):
        base = tool_rank.get(c["tool"], 50) / 100.0
        ics = intent_scores[i] / 100.0
        score = (base + ics) / 2.0
        w = score * sharpness + (1.0 / n) * (1 - sharpness)
        # 事前シミュレーション（報酬予測誤差）: 予測結果新規性が低い → 動機が生まれない
        novelty = predict_result_novelty(state, c["tool"], c.get("reason", ""))
        w *= max(0.05, novelty)
        # UPS v2 priority boost: 未消化 pending を解消する候補を優先
        w *= _pending_priority_boost(state, c)
        if novelty < 0.5:
            try:
                from core.config import RESOLUTION_LOG
                with open(RESOLUTION_LOG, "a", encoding="utf-8") as _f:
                    _f.write(f"  [predict] {c['tool']} novelty={novelty:.2f} "
                             f"reason={c.get('reason','')[:40]}\n")
            except Exception:
                pass
        weights.append(w)

    total = sum(weights)
    r = random.random() * total
    cumul = 0.0
    for i, w in enumerate(weights):
        cumul += w
        if r <= cumul:
            return candidates[i]
    return candidates[-1]
