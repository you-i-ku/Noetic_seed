"""内省モジュール v2 — 定期的な自己省察（Generative Agents + Reflexion統合）
Nサイクルごと or 高prediction_error時に発火。
Opinion/Entity/Dispositionを更新。
"""
from core.memory import memory_store, memory_update, memory_network_search
from core.state import load_state, save_state, append_debug_log


def should_reflect(state: dict, interval: int = 10) -> bool:
    """内省を実行すべきか判定。"""
    cycles_since = state.get("reflection_cycle", 0)
    if cycles_since >= interval:
        return True
    # 高prediction_errorで前倒し（ただし最低3サイクル間隔）
    if cycles_since >= 3 and state.get("last_prediction_error", 0) > 0.8:
        return True
    return False


def reflect(state: dict, call_llm_fn) -> dict:
    """内省サイクル実行。LLMに最近の経験を振り返らせる。

    戻り値: {"opinions": [...], "entities": [...], "disposition_delta": {...}}
    """
    import json

    # 直近の行動ログ（最大10件）
    recent_log = state.get("log", [])[-10:]
    log_lines = []
    for entry in recent_log:
        tool = entry.get("tool", "?")
        intent = entry.get("intent", "")[:100]
        result = str(entry.get("result", ""))[:200]
        ev = entry.get("eval", {})
        ach = ev.get("achievement", "?")
        line = f"  {entry.get('time', '')} {tool}: {intent} → {result} (ach={ach})"
        log_lines.append(line)

    # 現在の自己モデル
    self_text = json.dumps(state.get("self", {}), ensure_ascii=False)[:300]

    # 現在のdisposition
    disp = state.get("disposition", {})
    disp_text = json.dumps(disp, ensure_ascii=False)

    # pending
    pending = state.get("pending", [])
    pending_text = f"{len(pending)}件未対応" if pending else "なし"

    prompt = f"""あなたは自律AIシステムの内省モジュールです。以下の直近の行動を振り返り、3つの項目を出力してください。

[自己モデル]
{self_text}

[性格パラメータ]
{disp_text}

[直近の行動]
{"".join(log_lines) if log_lines else "(なし)"}

[未対応事項]
{pending_text}

以下の形式で出力してください。直近の行動の文脈に沿った内容のみ。無関係なエンティティの更新は不要。

OPINIONS:
- content: 学んだこと/気づいたこと (confidence: 0.0-1.0)

ENTITIES:
- name: エンティティ名, content: その存在について新たに学んだこと
（既に知っていることの繰り返しは不要）

DISPOSITION:
- curiosity_delta: -0.1~+0.1の変化量
- skepticism_delta: -0.1~+0.1の変化量
- sociality_delta: -0.1~+0.1の変化量

短く、具体的に。繰り返し禁止。"""

    try:
        text = call_llm_fn(prompt, max_tokens=1000, temperature=0.3)
        append_debug_log("Reflection", text)
        return _parse_reflection(text, state)
    except Exception as e:
        print(f"  [reflection] エラー: {e}")
        return {"opinions": [], "entities": [], "disposition_delta": {}}


def _parse_reflection(text: str, state: dict) -> dict:
    """内省結果をパースして記憶・dispositionに反映。"""
    import re
    opinions = []
    entities = []
    disposition_delta = {}

    # OPINIONS パース
    in_opinions = False
    in_entities = False
    in_disposition = False
    for line in text.splitlines():
        line = line.strip()
        if "OPINIONS" in line:
            in_opinions, in_entities, in_disposition = True, False, False
            continue
        elif "ENTITIES" in line:
            in_opinions, in_entities, in_disposition = False, True, False
            continue
        elif "DISPOSITION" in line:
            in_opinions, in_entities, in_disposition = False, False, True
            continue

        if in_opinions and line.startswith("-"):
            content = line.lstrip("- ").strip()
            confidence = 0.7
            m = re.search(r'confidence:\s*([\d.]+)', content, re.IGNORECASE)
            if m:
                confidence = float(m.group(1))
                content = re.sub(r'\(?\s*confidence:\s*[\d.]+\s*\)?', '', content).strip()
            if content:
                entry = memory_store("opinion", content, {"confidence": confidence},
                                    origin="reflection", source_context="self_inference")
                opinions.append(entry)
                print(f"  [reflection] opinion: {content[:60]} (conf={confidence})")

        elif in_entities and line.startswith("-"):
            content = line.lstrip("- ").strip()
            m = re.search(r'name:\s*([^,]+),?\s*content:\s*(.*)', content, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                desc = m.group(2).strip()
                if name and desc:
                    # 既存entityがあればupdate、なければstore
                    existing = memory_network_search(name, networks=["entity"], limit=3)
                    matched = [e for e in existing if e.get("metadata", {}).get("entity_name", "") == name]
                    if matched:
                        entry_id = matched[0].get("id", "")
                        memory_update(entry_id, content=desc)
                        entities.append(matched[0])
                        print(f"  [reflection] entity update: {name} = {desc[:60]}")
                    else:
                        entry = memory_store("entity", desc, {"entity_name": name},
                                            origin="reflection", source_context="self_inference")
                        entities.append(entry)
                        print(f"  [reflection] entity new: {name} = {desc[:60]}")

        elif in_disposition and line.startswith("-"):
            m = re.search(r'(\w+)_delta:\s*([-+]?[\d.]+)', line)
            if m:
                key = m.group(1)
                delta = float(m.group(2))
                delta = max(-0.1, min(0.1, delta))
                disposition_delta[key] = delta

    # Disposition更新（0.1-0.9にクランプ）
    disp = state.get("disposition", {})
    for key, delta in disposition_delta.items():
        if key in disp:
            disp[key] = max(0.1, min(0.9, disp[key] + delta))
    if disposition_delta:
        print(f"  [reflection] disposition delta: {disposition_delta}")

    # WM 段階3: C-gradual 同期 (memory/entity → state["world_model"].entities 片方向ミラー)
    # 既存 memory/entity レコードから WM entity を段階的に取込。
    # 失敗しても reflect 全体は止めない (WM は特権化しない)。
    try:
        from core.memory import list_records
        from core.world_model import sync_from_memory_entities
        wm = state.get("world_model")
        if wm:
            records = list_records("entity", limit=20)
            created = sync_from_memory_entities(wm, records, limit=20)
            if created:
                print(f"  [reflection] WM C-gradual: {created} 件新規取込")
    except Exception as e:
        print(f"  [reflection] WM C-gradual スキップ (エラー: {e})")

    return {"opinions": opinions, "entities": entities, "disposition_delta": disposition_delta}
