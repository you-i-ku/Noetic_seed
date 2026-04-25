"""長期記憶管理（アーカイブ・要約・圧縮 + Entity/Opinionネットワーク）"""
import json
import re
import uuid
from datetime import datetime
from typing import Optional
from core.config import MEMORY_DIR, LOG_HARD_LIMIT, LOG_KEEP, SUMMARY_HARD_LIMIT, META_SUMMARY_RAW
from core.state import load_pref, save_pref
from core.llm import call_llm
from core.embedding import is_vector_ready, _embed_sync, cosine_similarity
from core.tag_registry import is_tag_registered, list_registered_tags, get_tag_rules
from core.perspective import Perspective, default_self_perspective

# === Entity/Opinion Network (段階7: tag_registry で動的管理) ===
# 段階6-C まで: _VALID_NETWORKS = {"experience", "opinion", "entity"} (ハードコード)
# 段階7: tag_registry.is_tag_registered で動的検証 (標準タグ含めて register 経由)


UNTAGGED_NETWORK = "_untagged"


def _network_file(network):
    """network jsonl ファイルパスを返す。

    段階11-D Phase 1 (Step 1.1): network=None or UNTAGGED_NETWORK で
    `_untagged.jsonl` を返す (tag 廃止移行の保存先、untagged memory 専用)。
    """
    MEMORY_DIR.mkdir(exist_ok=True)
    if network is None or network == UNTAGGED_NETWORK:
        return MEMORY_DIR / f"{UNTAGGED_NETWORK}.jsonl"
    return MEMORY_DIR / f"{network}.jsonl"


def _build_metadata_prompt(content: str, network) -> str:
    """段階11-B Phase 3 Step 3.1: keywords + contextual_description 生成 prompt (軽量)。

    段階11-D Phase 1 (Step 1.7 v1 調整、A-MEM paper 版 literal 整合):
    A-MEM (WujiangXu memory_layer.py + agiresearch memory_system.py 共通) の
    実プロンプト literal は「at least three keywords, but don't be too redundant」+
    「one sentence summarizing」(個数上限・文字数 literal なし)。Noetic は
    日本語訳でこの literal に揃え、上限を Noetic 側で課さない (cognitive
    richness 優先、`feedback_llm_as_brain` + ゆう gut「マジックナンバー回避」
    + ゆう gut「o 制約解放」 三原則整合)。

    段階11-D Phase 1 (Step 1.1): network=None 対応 (untagged の場合は
    "(untagged)" ラベル表示、LLM への文脈提示は維持)。
    """
    tag_label = network if network else "(untagged)"
    return (
        "以下の記憶 entry から keywords と contextual_description を生成してください:\n"
        f"content: {content}\n"
        f"tag: {tag_label}\n"
        "出力は JSON のみ (他の文字を含めない):\n"
        '{"keywords": [str, ...], "contextual_description": str}\n'
        "- keywords: 3 個以上 (冗長にならない範囲で)、content の主要概念を抽出 (日本語可)\n"
        "- contextual_description: 1 文、この memory が想起される文脈を要約"
    )


def _parse_metadata_response(response: str) -> dict:
    """LLM 応答から keywords / contextual_description を抽出 (robust parse)。

    段階11-D Phase 1 (Step 1.7 v1 調整、A-MEM paper 版 literal 整合):
    上限値 (旧 keywords[:7] / desc[:500]) を撤去。A-MEM 実 literal に
    上限指定なし、Noetic も上限を課さない (cognitive richness 優先)。
    型安全化のみ実施。
    """
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if not m:
            return {"keywords": [], "contextual_description": ""}
        data = json.loads(m.group(0))
        kws = data.get("keywords", [])
        desc = data.get("contextual_description", "")
        if not isinstance(kws, list):
            kws = []
        if not isinstance(desc, str):
            desc = str(desc) if desc is not None else ""
        return {
            "keywords": [str(k) for k in kws],
            "contextual_description": desc,
        }
    except Exception:
        return {"keywords": [], "contextual_description": ""}


def _generate_memory_metadata(content: str, network: str) -> dict:
    """memory_store 同期呼出で keywords + contextual_description を LLM 生成。

    LLM 呼出 / parse 失敗時は空 dict を返す (memory_store 側で fallback)。
    """
    from core.llm import call_llm
    prompt = _build_metadata_prompt(content, network)
    response = call_llm(prompt, max_tokens=300, temperature=0.3)
    return _parse_metadata_response(response)


def memory_store(network: Optional[str] = None, content: str = "", metadata: dict = None,
                 origin: str = "unknown", source_context: str = "",
                 perspective: Optional[Perspective] = None, *,
                 keywords: Optional[list] = None,
                 contextual_description: Optional[str] = None,
                 _auto_metadata: bool = True,
                 _state: Optional[dict] = None,
                 _reconcile_embed_fn: Optional[object] = None,
                 _reconcile_cosine_fn: Optional[object] = None,
                 _reconcile_llm_fn: Optional[object] = None,
                 _link_generation_enabled: bool = True) -> dict:
    """記憶を保存。origin=生成きっかけ、source_context=根拠の出処。

    段階11-A: perspective kwarg を専用キーとして entry に昇格 (metadata と並列、
    型安全優先)。None なら default_self_perspective() (self/actual) で補完、
    呼び出し側は意識不要で後方互換。既存 jsonl entry (perspective 欠落) は
    読み出し側 (rebuild_wm / list_records) が default で解釈する。

    段階11-B Phase 3 Step 3.1: keywords / contextual_description を keyword-only
    引数として追加。両方 None + _auto_metadata=True なら LLM 同期呼出で自動生成
    (A-MEM paper 版 = WujiangXu memory_layer.py 哲学整合: atomic enrich on store
    で reconciliation の前提条件を満たす)。LLM 失敗時は graceful fallback
    (空値で保存、memory 書込は継続)。test 用途で `_auto_metadata=False` 指定で
    LLM skip 可能。

    段階11-D Phase 1 (Step 1.1-1.2): network=None 許可 (tag 廃止移行)。
    network=None なら is_tag_registered チェック skip、UNTAGGED_NETWORK
    マーカーで内部統一して `_untagged.jsonl` に保存する。
    """
    if network is not None:
        if not is_tag_registered(network):
            raise ValueError(f"Invalid network: {network}")
        _rules = get_tag_rules(network) or {}
        if _rules.get("learning_rules", {}).get("write_protected", False):
            raise ValueError(
                f"tag '{network}' is write_protected (pseudo-tag, meta-section only)"
            )
    if perspective is None:
        perspective = default_self_perspective()

    # Phase 3 Step 3.1: keywords / contextual_description の生成 / 補完
    if _auto_metadata and keywords is None and contextual_description is None:
        try:
            _meta = _generate_memory_metadata(content, network)
            keywords = _meta.get("keywords", [])
            contextual_description = _meta.get("contextual_description", "")
        except Exception as e:
            print(f"  [memory] metadata 自動生成 skip (error: {e})")
            keywords = []
            contextual_description = ""
    else:
        keywords = keywords if keywords is not None else []
        contextual_description = contextual_description if contextual_description is not None else ""

    entry = {
        "id": f"mem_{uuid.uuid4().hex[:12]}",
        "network": network if network is not None else UNTAGGED_NETWORK,
        "content": content,
        "origin": origin,
        "source_context": source_context,
        "metadata": metadata or {},
        "perspective": perspective,
        "keywords": keywords,
        "contextual_description": contextual_description,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(_network_file(network), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 段階11-B Phase 3/4 hook (state 渡された時のみ発火、reflect 継続原則で
    # exception は catch して memory 書込を止めない)。
    # embed_fn / cosine_fn は Phase 3 reconciliation と Phase 4 link 生成で
    # 共有 (同 近傍取得ロジック、cost 冗長防止)。
    if _state is not None:
        _ef = _reconcile_embed_fn
        _cf = _reconcile_cosine_fn
        if _ef is None and is_vector_ready():
            _ef = _embed_sync
            _cf = cosine_similarity

        # Phase 3 Step 3.3: reconciliation (矛盾検出 → EC 誤差)
        # bitemporal 凍結原則: 既存 fact は書き換えない
        try:
            from core.reconciliation import check_on_write
            check_on_write(
                entry, _state,
                embed_fn=_ef,
                cosine_fn=_cf,
                llm_call_fn=_reconcile_llm_fn,
            )
        except Exception as e:
            print(f"  [reconciliation] skip (error: {e})")

        # Phase 4 Step 4.3: memory_links 同期生成 (近傍 top-K)
        if _link_generation_enabled:
            try:
                from core.memory_links import generate_links_for
                generate_links_for(
                    entry,
                    embed_fn=_ef,
                    cosine_fn=_cf,
                    llm_call_fn=_reconcile_llm_fn,
                )
            except Exception as e:
                print(f"  [memory_links] skip (error: {e})")

    return entry


def memory_update(memory_id: str, content: str = None, metadata: dict = None) -> str:
    """既存記憶を更新。

    段階11-D Phase 1 (Step 1.1): UNTAGGED_NETWORK も走査対象に追加。
    """
    for network in list(list_registered_tags()) + [UNTAGGED_NETWORK]:
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
    """記憶を削除。

    段階11-D Phase 1 (Step 1.1): UNTAGGED_NETWORK も走査対象に追加。
    """
    for network in list(list_registered_tags()) + [UNTAGGED_NETWORK]:
        fpath = _network_file(network)
        if not fpath.exists():
            continue
        lines = fpath.read_text(encoding="utf-8").splitlines()
        new_lines = [l for l in lines if l.strip() and memory_id not in l]
        if len(new_lines) < len(lines):
            fpath.write_text("\n".join(new_lines) + "\n" if new_lines else "", encoding="utf-8")
            return f"削除完了: {memory_id}"
    return f"エラー: {memory_id} が見つかりません"


def load_all_memories() -> list:
    """全 network (UNTAGGED 含む) の memory entry を 1 list に集めて返す。

    段階11-D Phase 5 Step 5.2: cluster 推定 (estimate_clusters) の入力源。
    Phase 6 metric 等の他 consumer でも再利用想定。

    cap なし (Q2 ゆう判断 2026-04-26): 「先に固定値を決めず観察 log で
    cap 必要性を実証判断」=PLAN §11-4「マジックナンバー 0」精神の延長。
    OOM 安全網は cluster_estimation.estimate_clusters 内の debug print
    (N + vectors.nbytes) と smoke 後の grep で別レイヤー確保。

    Returns:
        全 memory entry の list (順序は network 順 → 各 jsonl の新しい順)。
    """
    all_entries: list = []
    networks = list(list_registered_tags()) + [UNTAGGED_NETWORK]
    for network in networks:
        if network != UNTAGGED_NETWORK and not is_tag_registered(network):
            continue
        fpath = _network_file(network)
        if not fpath.exists():
            continue
        try:
            lines = fpath.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                all_entries.append(json.loads(line))
            except Exception:
                continue
    return all_entries


def list_records(network, limit: int = 20) -> list:
    """指定ネットワークの jsonl を新しい順に読んで直近 limit 件を返す。
    WM の C-gradual 同期 (段階3) 等、検索ではなく全件走査系の消費者向け。

    段階11-D Phase 1 (Step 1.1): network=UNTAGGED_NETWORK は untagged
    jsonl を走査 (is_tag_registered チェック skip)。
    """
    if network != UNTAGGED_NETWORK and not is_tag_registered(network):
        return []
    fpath = _network_file(network)
    if not fpath.exists():
        return []
    try:
        lines = fpath.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    records = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
        if len(records) >= limit:
            break
    return records


def memory_network_search(query: str, networks: list = None, limit: int = 5,
                           view_filter: Optional[dict] = None) -> list:
    """Entity/Opinionネットワークをベクトル検索。

    段階11-A Step 6: view_filter kwarg 追加 (perspective filter)。
      None → 全視点 (既存挙動、デフォルト)
      {"viewer": "self"} → self 視点の entry のみ
      {"viewer_type": "actual"} → 仮想視点除外
    perspective 欠落 entry (旧形式) は default_self_perspective 相当で判定、
    view_filter={"viewer":"self"} 等で拾われる (backward compat)。
    """
    if not networks:
        # 段階11-D Phase 1 (Step 1.1): networks=None なら全登録タグ +
        # UNTAGGED_NETWORK を走査対象にする (untagged memory も検索対象)。
        networks = list(list_registered_tags()) + [UNTAGGED_NETWORK]
    all_entries = []
    for network in networks:
        # 段階11-D Phase 1: UNTAGGED_NETWORK は is_tag_registered で False
        # を返すが、untagged jsonl は明示的に走査対象に含める。
        if network != UNTAGGED_NETWORK and not is_tag_registered(network):
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

    # 段階11-A Step 6: view_filter 適用 (embedding search 前に候補絞り込み)
    if view_filter is not None:
        def _match(e: dict) -> bool:
            p = e.get("perspective") or default_self_perspective()
            for k, v in view_filter.items():
                if p.get(k) != v:
                    return False
            return True
        all_entries = [e for e in all_entries if _match(e)]
        if not all_entries:
            return []
    if is_vector_ready():
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


def get_relevant_memories(
    state: dict,
    limit: int = 8,
    *,
    use_links: bool = False,
    link_depth: int = 1,
    link_top_n: int = 3,
) -> list:
    """プロンプト用: 直近intentに関連する記憶を取得。
    4ネットワーク検索 + archive の直近外部入力を合わせて返す。
    外部入力は優先的に先頭へ入れる（外部からの会話は忘却耐性を与える）。

    段階11-C G-lite Phase 1: use_links=True で memory_links graph 経由の
    近傍 memory を merge (既存 semantic search と併用、opt-in)。
    use_links=False (デフォルト) で挙動不変。
    """
    recent_intents = [e.get("intent", "") for e in state.get("log", [])[-5:] if e.get("intent")]

    # archive からの最近の [external] 入力を優先的に取得
    external_mems = _recent_externals_from_archive(limit=3)

    query_parts = [i for i in recent_intents if i]
    # external 原文も query に混ぜて類似度検索の精度を上げる
    query_parts.extend(str(m.get("content", "")) for m in external_mems)
    if not query_parts:
        return external_mems

    query = " ".join(query_parts)[:500]
    network_mems = memory_network_search(query, limit=limit)

    # 外部入力を先頭に（重複除去）
    seen_ids = {m.get("id") for m in external_mems if m.get("id")}
    merged = list(external_mems)
    for m in network_mems:
        mid = m.get("id")
        if mid and mid in seen_ids:
            continue
        merged.append(m)
        if mid:
            seen_ids.add(mid)  # 段階11-C Phase 1: link merge 側で seen チェックに使うため明示追加
        if len(merged) >= limit + len(external_mems):
            break

    # 段階11-C G-lite Phase 1: link 経由の近傍 memory を merge (opt-in)
    # 段階11-D Phase 3 Step 3.3: 経由 link の strength を update (Physarum rule)
    if use_links and network_mems:
        from core.memory_links import follow_links, update_link_strength_used
        current_cycle = state.get("cycle_id") if isinstance(state, dict) else None
        for origin_mem in network_mems[:link_top_n]:
            origin_id = origin_mem.get("id")
            if not origin_id:
                continue
            reached = follow_links(
                origin_id,
                depth=link_depth,
                top_n_per_depth=link_top_n,
            )
            for r in reached:
                entry = r.get("memory_entry") or {}
                eid = entry.get("id")
                if not eid or eid in seen_ids:
                    continue
                # log 集計用の内部属性付与 (§11-d 確定: prompt 表示 marker は省略)
                entry = dict(entry)
                entry["_retrieval_via"] = "link"
                entry["_retrieval_depth"] = r.get("depth", 1)
                entry["_retrieval_strength_hint"] = r.get("strength_hint", 0.0)
                # 段階11-D Phase 3 Step 3.3: 使用された link の strength を up
                # (lazy decay 込み、case Q)。reflect 継続原則で例外は catch
                via_link = r.get("via_link") or {}
                link_id = via_link.get("id")
                if link_id:
                    try:
                        update_link_strength_used(link_id, current_cycle=current_cycle)
                    except Exception as e:
                        print(f"  [memory_links] strength update skip (error: {e})")
                merged.append(entry)
                seen_ids.add(eid)

    return merged


def _recent_externals_from_archive(limit: int = 3, days_back: int = 7) -> list:
    """archive の jsonl を逆順走査して直近 N 件の [external] エントリを取得。
    外部入力は永続保存されるが、通常 related_memory は network しか見ないので、
    ここで明示的に archive から拾って優先表示する。"""
    from datetime import timedelta
    MEMORY_DIR.mkdir(exist_ok=True)
    externals: list = []
    now = datetime.now()
    for offset in range(days_back):
        day = (now - timedelta(days=offset)).strftime("%Y%m%d")
        fpath = MEMORY_DIR / f"archive_{day}.jsonl"
        if not fpath.exists():
            continue
        try:
            lines = fpath.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        # 逆順で走査（新しい順）
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("type") == "external":
                externals.append({
                    "id": e.get("id", ""),
                    "network": "external",
                    "content": str(e.get("result", ""))[:400],
                    "metadata": {"time": e.get("time", "")},
                })
                if len(externals) >= limit:
                    return externals
        if len(externals) >= limit:
            break
    return externals


def format_memories_for_prompt(memories: list, max_chars: int = 2000) -> str:
    """記憶をプロンプト用テキストに整形 (段階7: display_format 駆動)。

    段階11-D Phase 1 (Step 1.3): UNTAGGED_NETWORK は `[untagged] {content}`
    形式で表示 (smoke 4 段目で表示形式の観察 → tune 予定、PLAN §5 Phase 1
    Step 1.3)。
    """
    from core.tag_registry import get_tag_rules
    if not memories:
        return ""
    lines = []
    total = 0
    for m in memories:
        network = m.get("network", "?")
        content = m.get("content", "")[:300]
        meta = m.get("metadata", {})
        # external は archive 由来で tag_registry 非登録、特別扱い
        if network == "external":
            t = meta.get("time", "")
            line = f"  [external voice {t}] {content}"
        elif network == UNTAGGED_NETWORK:
            # 段階11-D Phase 1: untagged memory の表示
            line = f"  [untagged] {content}"
        else:
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
                line = f"  {fmt.format(**fmt_kwargs)}"
            except (KeyError, IndexError, ValueError):
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
    """LLMでエントリ群を要約して1件のsummaryエントリを返す。
    外部入力（[external]）は原文に近い形で保持する方針。"""
    lines = []
    external_snippets = []
    for e in entries:
        if e.get("type") in ("system", "environment"):
            continue
        if e.get("type") == "external":
            # 外部入力は原文を別枠で保持（要約で消えないように）
            external_snippets.append(f"{e.get('time','')} {str(e.get('result',''))[:300]}")
            continue
        line = f"{e.get('time','')} {e.get('tool','')}"
        if e.get("intent"): line += f" [{e['intent'][:120]}]"
        if e.get("result"): line += f" → {str(e['result'])[:200]}"
        e_str = " ".join(f"{k}={e[k]}" for k in ("e2","e3","e4") if e.get(k))
        if e_str: line += f" ({e_str})"
        lines.append(line)

    ext_block = ""
    if external_snippets:
        ext_block = "\n\n【外部入力（原文優先、必ず要約に含める）】\n" + "\n".join(external_snippets[:10])

    prompt = f"""以下は自律AIの行動ログ（{len(entries)}件）です。800字以内で要約してください。

以下を優先して含めてください：
1. **外部入力（外部からのメッセージ）があれば必ず原文に近い形で記録**
   - 名前・役割・役割の変化・要望・伝えられた事実・環境の前提 等
2. 受動的に明らかになった事実（APIの有無、設定状態、既知の制約、開発モード等）
3. 何を試みて何が起きたか（行動パターン）
4. energy / entropy の傾向

表面的な言い換えは避け、「誰が何を言ったか」「何が事実として確定したか」を保存することを優先してください。

{chr(10).join(lines[:40])}{ext_block}

800字以内で要約（日本語、重要事実を先頭に）:"""
    ids = [e.get("id", "") for e in entries if e.get("id")]
    try:
        text = call_llm(prompt, max_tokens=1500).strip()[:1200]
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
    Trigger1: log >= LOG_HARD_LIMIT(150) → 古い (LOG_HARD_LIMIT - LOG_KEEP) 件を要約
              → 直近 LOG_KEEP(120) 件を保持
              ※ [external] エントリは要約対象外で保持（外部からの会話は高価値情報として永続）
    Trigger2: summaries >= 10 → メタ要約（全 summary + 直近 raw 数件） → summaries = [1件]
    """
    state.setdefault("summaries", [])

    if len(state["log"]) >= LOG_HARD_LIMIT:
        compress_count = max(1, LOG_HARD_LIMIT - LOG_KEEP)
        old_section = state["log"][:compress_count]
        # [external] 入力は要約せず保持（外部からの会話の永続化）
        to_preserve = [e for e in old_section if e.get("type") == "external"]
        to_summarize = [e for e in old_section if e.get("type") != "external"]

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

        if to_summarize:
            summary = _summarize_entries(to_summarize, "L1要約")
            _archive_summary(summary)
            state["summaries"].append(summary)

        # 残り = 保持対象 external + 直近 LOG_KEEP 件
        state["log"] = to_preserve + state["log"][compress_count:]
        print(
            f"  [memory] Trigger1: {len(to_summarize)}件→要約 "
            f"({len(to_preserve)}件のexternal保持), "
            f"log={len(state['log'])}件, summaries={len(state['summaries'])}件"
        )

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
