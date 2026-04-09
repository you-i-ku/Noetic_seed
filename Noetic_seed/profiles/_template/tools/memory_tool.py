"""記憶検索ツール"""
import json
import re
from core.config import MEMORY_DIR
from core.embedding import _vector_ready, _embed_sync, cosine_similarity


def _search_memory(args):
    """memory/archive_*.jsonlからエントリをベクトル検索またはキーワード検索する"""
    query = args.get("query", "")
    search_id = args.get("id", "")
    n = min(int(args.get("max_results", "") or "5"), 20)

    MEMORY_DIR.mkdir(exist_ok=True)
    archive_files = sorted(MEMORY_DIR.glob("archive_*.jsonl"), reverse=True)
    if not archive_files:
        return "記憶ファイルがまだありません"

    # ID検索
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

    # 全ファイルからエントリ収集（最大1000件）
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

    # フォールバック: キーワード検索
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
