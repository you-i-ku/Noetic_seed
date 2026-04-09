"""長期記憶管理（アーカイブ・要約・圧縮 + Entity/Opinionネットワーク）"""
import json
import re
import uuid
from datetime import datetime
from core.config import MEMORY_DIR, LOG_HARD_LIMIT, LOG_KEEP, SUMMARY_HARD_LIMIT, META_SUMMARY_RAW
from core.state import load_pref, save_pref
from core.llm import call_llm
from core.embedding import _vector_ready, _embed_sync, cosine_similarity

# === Entity/Opinion Network ===
_VALID_NETWORKS = {"world", "experience", "opinion", "entity"}


def _network_file(network: str):
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / f"{network}.jsonl"


def memory_store(network: str, content: str, metadata: dict = None) -> dict:
    """記憶を保存。"""
    if network not in _VALID_NETWORKS:
        raise ValueError(f"Invalid network: {network}")
    entry = {
        "id": f"mem_{uuid.uuid4().hex[:12]}",
        "network": network,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(_network_file(network), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def memory_update(memory_id: str, content: str = None, metadata: dict = None) -> str:
    """既存記憶を更新。"""
    for network in _VALID_NETWORKS:
        fpath = _network_file(network)
        if not fpath.exists():
            continue
        lines = fpath.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("id") == memory_id:
                    if content is not None:
                        entry["content"] = content
                    if metadata is not None:
                        entry["metadata"].update(metadata)
                    entry["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    updated = True
                new_lines.append(json.dumps(entry, ensure_ascii=False))
            except Exception:
                new_lines.append(line)
        if updated:
            fpath.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            return f"更新完了: {memory_id}"
    return f"エラー: {memory_id} が見つかりません"


def memory_forget(memory_id: str) -> str:
    """記憶を削除。"""
    for network in _VALID_NETWORKS:
        fpath = _network_file(network)
        if not fpath.exists():
            continue
        lines = fpath.read_text(encoding="utf-8").splitlines()
        new_lines = [l for l in lines if l.strip() and memory_id not in l]
        if len(new_lines) < len(lines):
            fpath.write_text("\n".join(new_lines) + "\n" if new_lines else "", encoding="utf-8")
            return f"削除完了: {memory_id}"
    return f"エラー: {memory_id} が見つかりません"


def memory_network_search(query: str, networks: list = None, limit: int = 5) -> list:
    """Entity/Opinionネットワークをベクトル検索。"""
    if not networks:
        networks = list(_VALID_NETWORKS)
    all_entries = []
    for network in networks:
        if network not in _VALID_NETWORKS:
            continue
        fpath = _network_file(network)
        if not fpath.exists():
            continue
        for line in fpath.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                all_entries.append(json.loads(line))
            except Exception:
                pass
    if not all_entries:
        return []
    if _vector_ready:
        try:
            texts = [e.get("content", "")[:400] for e in all_entries]
            vecs = _embed_sync([query] + texts)
            if vecs and len(vecs) == 1 + len(all_entries):
                q_vec = vecs[0]
                scored = [(cosine_similarity(q_vec, vecs[i + 1]), all_entries[i])
                          for i in range(len(all_entries))]
                scored.sort(key=lambda x: x[0], reverse=True)
                return [{"score": s, **e} for s, e in scored[:limit]]
        except Exception:
            pass
    # フォールバック: キーワード
    query_tokens = set(re.findall(r'\w+', query.lower()))
    scored = []
    for entry in all_entries:
        tokens = set(re.findall(r'\w+', entry.get("content", "").lower()))
        if query_tokens & tokens:
            scored.append((len(query_tokens & tokens) / max(len(query_tokens), 1), entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, **e} for s, e in scored[:limit]]


def get_relevant_memories(state: dict, limit: int = 8) -> list:
    """プロンプト用: 直近intentに関連する記憶を全ネットワークから取得。"""
    recent_intents = [e.get("intent", "") for e in state.get("log", [])[-5:] if e.get("intent")]
    if not recent_intents:
        return []
    query = " ".join(recent_intents)[:500]
    return memory_network_search(query, limit=limit)


def format_memories_for_prompt(memories: list, max_chars: int = 2000) -> str:
    """記憶をプロンプト用テキストに整形。"""
    if not memories:
        return ""
    lines = []
    total = 0
    for m in memories:
        network = m.get("network", "?")
        content = m.get("content", "")[:200]
        meta = m.get("metadata", {})
        if network == "entity" and "entity_name" in meta:
            line = f"  [entity:{meta['entity_name']}] {content}"
        elif network == "opinion" and "confidence" in meta:
            line = f"  [opinion] {content} (確度:{meta['confidence']})"
        else:
            line = f"  [{network}] {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def _archive_entries(entries: list):
    """エントリ群をmemory/archive_YYYYMMDD.jsonlに追記しindex.jsonを更新"""
    MEMORY_DIR.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    archive_file = MEMORY_DIR / f"archive_{today}.jsonl"
    index_file = MEMORY_DIR / "index.json"
    with open(archive_file, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    index = {}
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    fname = archive_file.name
    if fname not in index:
        index[fname] = {"count": 0, "from": "", "to": ""}
    index[fname]["count"] += len(entries)
    if not index[fname]["from"] and entries:
        index[fname]["from"] = entries[0].get("time", "")
    if entries:
        index[fname]["to"] = entries[-1].get("time", "")
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize_entries(entries: list, label: str = "要約") -> dict:
    """LLMでエントリ群を200字以内に要約して1件のsummaryエントリを返す"""
    lines = []
    for e in entries:
        if e.get("type") in ("system", "environment"):
            continue
        line = f"{e.get('time','')} {e.get('tool','')}"
        if e.get("intent"): line += f" [{e['intent'][:80]}]"
        if e.get("result"): line += f" → {str(e['result'])[:120]}"
        e_str = " ".join(f"{k}={e[k]}" for k in ("e2","e3","e4") if e.get(k))
        if e_str: line += f" ({e_str})"
        lines.append(line)
    prompt = f"""以下は自律AIの行動ログ（{len(entries)}件）です。200字以内で要約してください。
「何を試みたか」「何が起きたか」「energyの傾向」を中心に。

{"  ".join(lines[:30])}

200字以内で要約（日本語）:"""
    ids = [e.get("id", "") for e in entries if e.get("id")]
    try:
        text = call_llm(prompt, max_tokens=400).strip()[:500]
    except Exception:
        tools_used = list(set(e.get("tool", "") for e in entries))
        text = f"{len(entries)}件({entries[0].get('time','')}〜{entries[-1].get('time','')}): ツール={tools_used}"
    sgid = f"sg_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "type": "summary",
        "summary_group_id": sgid,
        "label": label,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "covers_ids": ids,
        "covers_from": entries[0].get("time", "") if entries else "",
        "covers_to": entries[-1].get("time", "") if entries else "",
        "text": text,
    }


def _archive_summary(summary: dict):
    """要約をmemory/summaries.jsonlに書き出し、rawエントリとの紐付けをarchiveに追記する"""
    MEMORY_DIR.mkdir(exist_ok=True)
    with open(MEMORY_DIR / "summaries.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    today = datetime.now().strftime("%Y%m%d")
    archive_file = MEMORY_DIR / f"archive_{today}.jsonl"
    sgid = summary.get("summary_group_id", "")
    with open(archive_file, "a", encoding="utf-8") as f:
        for raw_id in summary.get("covers_ids", []):
            f.write(json.dumps({
                "type": "summary_ref",
                "summary_group_id": sgid,
                "raw_id": raw_id,
                "time": summary.get("time", ""),
            }, ensure_ascii=False) + "\n")


def maybe_compress_log(state: dict, tool_names: set = None):
    """
    Trigger1: log >= 150 → 古い51件を要約 → summaries[]に追加 → log = 99件
    Trigger2: summaries >= 10 → メタ要約（10件 + min(41,len(log))件raw） → summaries = [1件]
    """
    state.setdefault("summaries", [])

    if len(state["log"]) >= LOG_HARD_LIMIT:
        to_summarize = state["log"][:51]
        pref = load_pref()
        ema = pref.get("_ema", {})
        _tool_names = tool_names or set()
        for entry in to_summarize:
            if entry.get("type") in ("system", "environment"):
                continue
            t = entry.get("tool", "")
            m = re.search(r'(\d+)%', str(entry.get("e2", "")))
            if m and t in _tool_names:
                old = ema.get(t, 50.0)
                ema[t] = round(old * 0.8 + int(m.group(1)) * 0.2, 1)
        pref["_ema"] = ema
        save_pref(pref)
        summary = _summarize_entries(to_summarize, "L1要約")
        _archive_summary(summary)
        state["summaries"].append(summary)
        state["log"] = state["log"][51:]
        print(f"  [memory] Trigger1: 51件→要約, log={len(state['log'])}件, summaries={len(state['summaries'])}件")

    if len(state["summaries"]) >= SUMMARY_HARD_LIMIT:
        n_raw = min(META_SUMMARY_RAW, len(state["log"]))
        raw_for_meta = state["log"][:n_raw]
        meta_input = []
        for s in state["summaries"]:
            meta_input.append({
                "time": s.get("time", ""),
                "tool": f"[{s.get('label','')}]",
                "intent": s.get("text", "")[:200],
                "result": f"{s.get('covers_from','')}〜{s.get('covers_to','')}",
            })
        meta_input.extend(raw_for_meta)
        meta_summary = _summarize_entries(meta_input, "L2メタ要約")
        meta_summary["covers_summaries"] = len(state["summaries"])
        meta_summary["covers_raw"] = n_raw
        _archive_summary(meta_summary)
        state["summaries"] = [meta_summary]
        state["log"] = state["log"][n_raw:]
        print(f"  [memory] Trigger2: メタ要約, log={len(state['log'])}件, summaries=1件")
