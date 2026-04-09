"""記憶システム v2 — Hindsight型4ネットワーク + A-Mem自律管理
World: 客観事実
Experience: 一人称体験
Opinion: 主観+信頼度（0-1）
Entity: エンティティ別要約（関係性、属性）
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from core.config import MEMORY_DIR, LOG_HARD_LIMIT
from core.embedding import _vector_ready, _embed_sync, cosine_similarity

_VALID_NETWORKS = {"world", "experience", "opinion", "entity"}


def _network_file(network: str) -> Path:
    MEMORY_DIR.mkdir(exist_ok=True)
    return MEMORY_DIR / f"{network}.jsonl"


def _archive_file() -> Path:
    MEMORY_DIR.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    return MEMORY_DIR / f"archive_{today}.jsonl"


# === CRUD ===

def memory_store(network: str, content: str, metadata: dict = None) -> dict:
    """記憶を保存。entryを返す。"""
    if network not in _VALID_NETWORKS:
        raise ValueError(f"Invalid network: {network}. Use: {_VALID_NETWORKS}")
    entry = {
        "id": f"mem_{uuid.uuid4().hex[:12]}",
        "network": network,
        "content": content,
        "metadata": metadata or {},
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "access_count": 0,
    }
    with open(_network_file(network), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def memory_update(memory_id: str, content: str = None, metadata: dict = None) -> str:
    """既存記憶を更新（ネットワークファイルを書き換え）。"""
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


# === 検索 ===

def memory_search(query: str, networks: list[str] = None, limit: int = 5) -> list[dict]:
    """ベクトル検索。複数ネットワークを横断。"""
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

    # ベクトル検索
    if _vector_ready and len(all_entries) > 0:
        try:
            texts = [e.get("content", "")[:400] for e in all_entries]
            vecs = _embed_sync([query] + texts)
            if vecs and len(vecs) == 1 + len(all_entries):
                q_vec = vecs[0]
                scored = [
                    (cosine_similarity(q_vec, vecs[i + 1]), all_entries[i])
                    for i in range(len(all_entries))
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                return [{"score": s, **e} for s, e in scored[:limit]]
        except Exception:
            pass

    # フォールバック: キーワード検索
    import re
    query_tokens = set(re.findall(r'\w+', query.lower()))
    scored = []
    for entry in all_entries:
        text = entry.get("content", "").lower()
        tokens = set(re.findall(r'\w+', text))
        if query_tokens & tokens:
            score = len(query_tokens & tokens) / max(len(query_tokens), 1)
            scored.append((score, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": s, **e} for s, e in scored[:limit]]


# === アーカイブ（行動ログ保存）===

def archive_action(entry: dict):
    """行動ログエントリをarchiveファイルに保存。"""
    MEMORY_DIR.mkdir(exist_ok=True)
    with open(_archive_file(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # index.json更新
    index_file = MEMORY_DIR / "index.json"
    index = {}
    if index_file.exists():
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    fname = _archive_file().name
    if fname not in index:
        index[fname] = {"count": 0, "from": "", "to": ""}
    index[fname]["count"] += 1
    if not index[fname]["from"]:
        index[fname]["from"] = entry.get("time", "")
    index[fname]["to"] = entry.get("time", "")
    index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def prune_log(state: dict):
    """ログが上限に達したら古いエントリをexperienceネットワークに移動。LLM不要。"""
    if len(state["log"]) < LOG_HARD_LIMIT:
        return

    to_move = state["log"][:LOG_HARD_LIMIT - 30]
    state["log"] = state["log"][LOG_HARD_LIMIT - 30:]

    for entry in to_move:
        if entry.get("type") in ("system", "external", "environment"):
            continue
        intent = entry.get("intent", "")
        result = str(entry.get("result", ""))[:300]
        tool = entry.get("tool", "")
        time = entry.get("time", "")
        content = f"[{time}] {tool}: {intent}" + (f" → {result}" if result else "")
        memory_store("experience", content, {
            "source_id": entry.get("id", ""),
            "source_time": time,
        })

    print(f"  [memory] prune: {len(to_move)}件 → experience network, log={len(state['log'])}件")


# === ユーティリティ ===

def get_relevant_memories(state: dict, limit: int = 10) -> list[dict]:
    """プロンプト構築用: 直近のintentに関連する記憶を全ネットワークから取得。"""
    recent_intents = [e.get("intent", "") for e in state.get("log", [])[-5:] if e.get("intent")]
    if not recent_intents:
        return []
    query = " ".join(recent_intents)[:500]
    return memory_search(query, limit=limit)


def format_memories_for_prompt(memories: list[dict], max_chars: int = 2000) -> str:
    """記憶をプロンプト用テキストに整形。"""
    if not memories:
        return ""
    lines = []
    total = 0
    for m in memories:
        network = m.get("network", "?")
        content = m.get("content", "")[:200]
        score = m.get("score", 0)
        meta = m.get("metadata", {})
        line = f"  [{network}] {content}"
        if network == "opinion" and "confidence" in meta:
            line += f" (確度:{meta['confidence']})"
        if network == "entity" and "entity_name" in meta:
            line = f"  [{meta['entity_name']}] {content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)
