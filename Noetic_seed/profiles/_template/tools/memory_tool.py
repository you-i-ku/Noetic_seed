"""記憶操作ツール — archive検索（v1） + Entity/Opinionネットワーク管理（A-Mem方式）"""
import json
import re
from core.config import MEMORY_DIR
from core.embedding import _vector_ready, _embed_sync, cosine_similarity
from core.memory import memory_store, memory_update, memory_forget, memory_network_search

_VALID_NETWORKS = {"world", "experience", "opinion", "entity"}


def _search_memory(args):
    """v1互換: memory/archive_*.jsonlからエントリをベクトル/キーワード検索"""
    query = args.get("query", "")
    search_id = args.get("id", "")
    n = min(int(args.get("max_results", "") or "5"), 20)

    MEMORY_DIR.mkdir(exist_ok=True)
    archive_files = sorted(MEMORY_DIR.glob("archive_*.jsonl"), reverse=True)
    if not archive_files:
        return "記憶ファイルがまだありません"

    if search_id:
        for f in archive_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if search_id in entry.get("id", ""):
                        return (f"id={entry.get('id','')} time={entry.get('time','')} "
                                f"tool={entry.get('tool','')} intent={entry.get('intent','')[:200]} "
                                f"result={str(entry.get('result',''))[:200]}")
                except Exception:
                    pass
        return f"ID '{search_id}' に一致するエントリなし"

    if not query:
        return "エラー: queryまたはidを指定してください"

    all_entries = []
    for f in archive_files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                all_entries.append(json.loads(line))
                if len(all_entries) >= 1000:
                    break
        except Exception:
            pass
        if len(all_entries) >= 1000:
            break

    if not all_entries:
        return "記憶ファイルが空です"

    # ベクトル検索
    if _vector_ready:
        try:
            texts = [f"{e.get('intent','')} {str(e.get('result',''))}"[:400] for e in all_entries]
            vecs = _embed_sync([query] + texts)
            if vecs and len(vecs) == 1 + len(all_entries):
                q_vec = vecs[0]
                scored = sorted(
                    [(cosine_similarity(q_vec, vecs[i+1]), i, all_entries[i]) for i in range(len(all_entries))],
                    key=lambda x: x[0], reverse=True
                )[:n]
                return "\n".join(
                    f"[{round(s*100)}%] id={e.get('id','')} time={e.get('time','')} "
                    f"tool={e.get('tool','')} intent={e.get('intent','')[:100]}"
                    for s, _, e in scored
                )
        except Exception:
            pass

    # フォールバック
    query_tokens = set(re.findall(r'\w+', query.lower()))
    scored = []
    for idx, entry in enumerate(all_entries):
        text = f"{entry.get('intent','')} {str(entry.get('result',''))}".lower()
        tokens = set(re.findall(r'\w+', text))
        if query_tokens & tokens:
            scored.append((len(query_tokens & tokens) / max(len(query_tokens), 1), idx, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return f"'{query}' に一致するエントリなし"
    return "\n".join(
        f"[{round(s*100)}%] id={e.get('id','')} time={e.get('time','')} "
        f"tool={e.get('tool','')} intent={e.get('intent','')[:100]}"
        for s, _, e in scored[:n]
    )


def _tool_memory_store(args):
    """記憶を保存する。"""
    network = args.get("network", "").strip()
    content = args.get("content", "").strip()
    if not network or not content:
        return "エラー: networkとcontentを指定してください"
    if network not in _VALID_NETWORKS:
        return f"エラー: networkは {'/'.join(_VALID_NETWORKS)} のいずれか"

    metadata = {}
    if network == "opinion":
        confidence = args.get("confidence", "0.5")
        try:
            metadata["confidence"] = float(confidence)
        except ValueError:
            metadata["confidence"] = 0.5
    if network == "entity":
        entity_name = args.get("entity_name", "")
        if entity_name:
            metadata["entity_name"] = entity_name
        relationship = args.get("relationship", "")
        if relationship:
            metadata["relationship"] = relationship

    entry = memory_store(network, content, metadata)
    return f"記憶保存完了: [{network}] {content[:60]} (id={entry['id']})"


def _tool_memory_update(args):
    """既存の記憶を更新する。"""
    memory_id = args.get("memory_id", "") or args.get("id", "")
    content = args.get("content", "")
    if not memory_id:
        return "エラー: memory_idを指定してください"
    metadata = {}
    confidence = args.get("confidence", "")
    if confidence:
        try:
            metadata["confidence"] = float(confidence)
        except ValueError:
            pass
    return memory_update(memory_id, content or None, metadata or None)


def _tool_memory_forget(args):
    """記憶を削除する。"""
    memory_id = args.get("memory_id", "") or args.get("id", "")
    if not memory_id:
        return "エラー: memory_idを指定してください"
    return memory_forget(memory_id)


def _tool_search_memory(args):
    """記憶を検索する。全ネットワーク横断。"""
    query = args.get("query", "")
    if not query:
        return "エラー: queryを指定してください"
    networks = None
    net_str = args.get("networks", "")
    if net_str:
        networks = [n.strip() for n in net_str.split(",") if n.strip() in _VALID_NETWORKS]
    limit = min(int(args.get("max_results", "") or "5"), 20)

    results = memory_network_search(query, networks=networks, limit=limit)
    if not results:
        return f"'{query}' に一致する記憶なし"

    lines = []
    for r in results:
        score = round(r.get("score", 0) * 100)
        network = r.get("network", "?")
        content = r.get("content", "")[:150]
        mid = r.get("id", "")
        meta = r.get("metadata", {})
        line = f"[{score}%] [{network}] {content} (id={mid})"
        if network == "opinion" and "confidence" in meta:
            line += f" 確度={meta['confidence']}"
        if network == "entity" and "entity_name" in meta:
            line = f"[{score}%] [entity:{meta['entity_name']}] {content} (id={mid})"
        lines.append(line)
    return "\n".join(lines)
