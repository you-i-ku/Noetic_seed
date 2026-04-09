"""記憶操作ツール — AIが自律的に記憶を管理する（A-Mem方式）"""
import json
from core.memory import memory_store, memory_update, memory_forget, memory_search

_VALID_NETWORKS = {"world", "experience", "opinion", "entity"}


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

    results = memory_search(query, networks=networks, limit=limit)
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
