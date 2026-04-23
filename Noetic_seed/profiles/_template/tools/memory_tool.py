"""記憶操作ツール — archive検索（v1） + Entity/Opinionネットワーク管理（A-Mem方式）"""
import json
import re
from core.config import MEMORY_DIR
from core.embedding import _vector_ready, _embed_sync, cosine_similarity
from core.memory import memory_store, memory_update, memory_forget, memory_network_search
from core.tag_registry import is_tag_registered, list_registered_tags

# 段階7: _VALID_NETWORKS 撤去 → tag_registry で動的検証
_WORLD_DEPRECATION_WARNED = False


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
    """記憶を保存する。段階7: 未登録タグは rules 付きで inline 登録。"""
    global _WORLD_DEPRECATION_WARNED
    network = args.get("network", "").strip()
    content = args.get("content", "").strip()
    if not network or not content:
        return "エラー: networkとcontentを指定してください"
    # 段階7 Step 6: world → wm リダイレクト (back-compat、段階8 で削除)
    if network == "world":
        if not _WORLD_DEPRECATION_WARNED:
            print("  [memory_store] 'world' は段階7 で 'wm' に統合済。リダイレクトします。")
            _WORLD_DEPRECATION_WARNED = True
        network = "wm"
    # 段階7 Step 5: 未登録タグは rules 必須で inline 登録
    if not is_tag_registered(network):
        rules = args.get("rules")
        if not isinstance(rules, dict):
            # 段階11-B Phase 5 Step 5.2: hint を例示表現に一般化 (schema 拡張は
            # Phase 1 で c_gradual_source、Phase 2' で write_protected を追加済。
            # 最小書式は beta_plus / bitemporal の 2 key、iku は必要に応じて拡張可)
            return (f"エラー: 未登録タグ '{network}' には rules 必須 "
                    "(例: {\"beta_plus\": bool, \"bitemporal\": bool})")
        display_format = args.get("display_format", "") or ""
        from core.tag_registry import register_tag
        try:
            register_tag(
                network,
                learning_rules=rules,
                display_format=display_format,
                origin="dynamic",
                intent=args.get("tool_intent"),
            )
        except ValueError as e:
            return f"エラー: タグ登録失敗 ({e})"

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

    entry = memory_store(network, content, metadata,
                         origin="tool:memory_store", source_context="deliberate")
    # 段階10 Step 4 付帯 D: Fix 5 精神で content truncation 撤去。
    # iku が保存した記憶内容を「60 字で切れた」と次 cycle で誤認するリスク回避。
    return f"記憶保存完了: [{network}] {content} (id={entry['id']})"


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
        networks = [n.strip() for n in net_str.split(",") if is_tag_registered(n.strip())]
    limit = min(int(args.get("max_results", "") or "5"), 20)

    results = memory_network_search(query, networks=networks, limit=limit)
    if not results:
        return f"'{query}' に一致する記憶なし"

    from core.tag_registry import get_tag_rules
    lines = []
    for r in results:
        score = round(r.get("score", 0) * 100)
        network = r.get("network", "?")
        content = r.get("content", "")[:150]
        mid = r.get("id", "")
        meta = r.get("metadata", {})
        rules = get_tag_rules(network)
        fmt = (rules or {}).get("display_format", "") or f"[{network}] {{content}}"
        fmt_kwargs = {
            "content": content,
            "tag": network,
            "entity_name": meta.get("entity_name", "?"),
            "confidence": meta.get("confidence", "?"),
        }
        for k, v in meta.items():
            if k not in fmt_kwargs:
                fmt_kwargs[k] = v
        try:
            body = fmt.format(**fmt_kwargs)
        except (KeyError, IndexError, ValueError):
            body = f"[{network}] {content}"
        lines.append(f"[{score}%] {body} (id={mid})")
    return "\n".join(lines)
