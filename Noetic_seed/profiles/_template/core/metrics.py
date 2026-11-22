"""Slice 2 (orchestration §3 P0 #2): cycle ごとの metrics emission + run metadata.

orchestration plan §3 P0 #2 + §3 #3 + §⑧ 靄メトリクス A-E。BP-1 baseline smoke
の素材を作る (「変更前後の比較基準」)。**controller には触らない、測定のみ**
(orchestration §3 #3 末尾「最初は測定のみ」literal)。

設計確定事項 (2026-05-09 ゆう判断):
  ① schema: 観測可能な全 indicator + 靄 5 種 (embedding_spread / local_density_mean
     / strength_distribution / inter_cluster_blur / te_flow_anisotropy)
  ② emit timing: cycle 末 (save_state 直前)、append-only
  ③ run_id: full uuid (起動毎、main.py で state に設定)、session_id (8 char)
     と階層化
  ④ config_hash: canonicalize_config で _comment* / ws_token / _ema strip →
     SHA256 prefix 16 char
  ⑤ enabled_features: β light snapshot (5 keys、settings.json から抽出)
  ⑥ persistence: append-only、F-005 view_persistence registry には乗せない
     (mutable view ではない、新規 _atomic_append_jsonl 経路)
  ⑦ Slice 3 との関係: info_gain field は本 file に Slice 3 で後付け追加可

たとえ:
  metrics_events.jsonl = Noetic の身体検査表 (バイタルサイン)。
  raw_events = 検査結果、subjective_entries = 医師の所見、metrics_events =
  数値で時系列追える血圧/体温/血糖 etc。前者 2 つは「事実 + 解釈」を保存、
  本 file は「数値分布で動向を測る」役割。

正典 pointer:
  WORLD_MODEL_DESIGN/NOETIC_INTEGRATED_ORCHESTRATION_PLAN.md §3 P0 #2 + §⑧
  WORLD_MODEL_DESIGN/NOETIC_AUTONOMY_REINFORCEMENT_PLAN.md §8
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ============================================================
# 定数
# ============================================================

METRICS_FILE_NAME = "metrics_events.jsonl"
INDEX_FILE_NAME = "index.json"
CONFIG_HASH_PREFIX_LEN = 16
HISTORY_LATEST_N = 5
FOG_K_NEAREST = 5
LINKS_SCAN_LIMIT = 500

# canonicalize_config で除外する key (cosmetic、再現性に効かない)
_STRIP_PREFIXES = ("_comment",)
_STRIP_KEYS = frozenset({"ws_token", "_ema"})


# ============================================================
# Run metadata
# ============================================================

def get_code_version() -> str:
    """git HEAD short hash (12 char) を返す。git 不在 / 失敗時は "unknown"。

    metrics_events に記録、replay arena / ablation で「どの commit で取った
    metrics か」を追跡可能化する。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            if out:
                return out
    except Exception:
        pass
    return "unknown"


# ============================================================
# Config hash (canonicalize → SHA256 prefix)
# ============================================================

def _strip_cosmetic(d: Any) -> Any:
    """``_comment*`` / ``ws_token`` / ``_ema`` を再帰的に除去した dict を返す。

    実験再現性に効かない key を strip して、cosmetic 変更で hash が
    変わらないようにする (replay 比較が安定)。
    """
    if not isinstance(d, dict):
        return d
    return {
        k: _strip_cosmetic(v)
        for k, v in d.items()
        if not (
            any(k.startswith(p) for p in _STRIP_PREFIXES) or k in _STRIP_KEYS
        )
    }


def canonicalize_config(settings: dict, pref: dict) -> dict:
    """settings + pref を ``_comment*``/``ws_token``/``_ema`` 除外で正規化。

    Returns:
        ``{"settings": <stripped>, "pref": <stripped>}``
    """
    return {
        "settings": _strip_cosmetic(settings or {}),
        "pref": _strip_cosmetic(pref or {}),
    }


def compute_config_hash(
    settings: dict,
    pref: dict,
    prefix_len: int = CONFIG_HASH_PREFIX_LEN,
) -> str:
    """canonicalize → JSON canonical (sort_keys) → SHA256 prefix を返す。

    決定的 (同じ入力で必ず同じ hash)、replay 検証で「実験条件が同じか」を
    1 行で照合できる。
    """
    canonical = canonicalize_config(settings, pref)
    text = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:prefix_len]


# ============================================================
# Enabled features (β light snapshot)
# ============================================================

def get_enabled_features(settings: dict) -> dict:
    """settings.json から実験 vs 本番に効く 5 keys を抽出。

    orchestration §⑤ ゆう確定 β light: 5 個に絞って snapshot。infra 増設
    せずに「どの実験 mode で取った metrics か」を 1 dict で識別可能化。

    将来 enabled_features 概念が拡張された時はこの 5 keys を維持しつつ
    field 追加していく。
    """
    settings = settings or {}
    return {
        "mcp_bridge": bool(settings.get("mcp_bridge", {}).get("enabled", False)),
        "retrieval_links": bool(
            settings.get("retrieval", {}).get("use_links", False)
        ),
        "reflect_cold_start": bool(
            settings.get("reflection", {}).get("reflect_cold_start_mode", False)
        ),
        "predictor_mode": str(
            settings.get("world_model", {}).get("predictor_mode", "light")
        ),
        "auto_approve_all": bool(
            settings.get("approval", {}).get("auto_approve_all", False)
        ),
    }


# ============================================================
# History summary helper
# ============================================================

def summarize_history(history: list, latest_n: int = HISTORY_LATEST_N) -> dict:
    """list[number] を ``{len, mean, max, latest_n}`` に圧縮。

    metrics_events.jsonl に list そのまま埋め込むと file size が肥大化する
    ため、各 history (E2/EC/JEPA 予測誤差列) は本 helper で要約する。
    空 list / 全要素非数値 → ``{len: 0, mean: None, max: None, latest_n: []}``
    で返す (defensive、metrics 行欠損を回避)。
    """
    nums = [float(x) for x in (history or []) if isinstance(x, (int, float))]
    if not nums:
        return {"len": 0, "mean": None, "max": None, "latest_n": []}
    return {
        "len": len(nums),
        "mean": round(sum(nums) / len(nums), 6),
        "max": max(nums),
        "latest_n": nums[-latest_n:],
    }


# ============================================================
# 靄メトリクス A-E (orchestration §⑧ ゆう確定)
# ============================================================

def fog_embedding_spread(entries_with_embedding: list) -> Optional[float]:
    """A: 全 embedding 重心からの平均 cosine 距離 (靄の広がり)。

    識別力: count ベースの森メトリクス (node_count etc) では捉えられない
    「embedding 空間での広がり」を捕捉。新 memory が追加されるほど (多様な
    内容なら) 広がる、似た内容ばかりなら縮む。
    """
    try:
        import numpy as np
    except ImportError:
        return None
    embs = [
        e["embedding"] for e in entries_with_embedding
        if isinstance(e, dict) and isinstance(e.get("embedding"), list)
    ]
    if len(embs) < 2:
        return None
    arr = np.array(embs, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm < 1e-9:
        return None
    centroid = centroid / norm
    sims = arr @ centroid  # bge-m3 は L2 正規化済前提 (Phase 1 整合)
    distances = 1.0 - sims
    return round(float(distances.mean()), 6)


def fog_local_density_mean(
    entries_with_embedding: list, k: int = FOG_K_NEAREST
) -> Optional[float]:
    """B: 各 entry の k 近傍までの平均 cosine 距離 (低密度なほど高値、靄の濃淡)。

    識別力: avg_strength や全体 spread (A) では見えない「局所的に密集する
    領域 vs 疎な領域」の有無を捕捉。同じ N 件でも、ぎゅっと寄せ集まってる
    分布と均等分散の分布で値が変わる。
    """
    try:
        import numpy as np
    except ImportError:
        return None
    embs = [
        e["embedding"] for e in entries_with_embedding
        if isinstance(e, dict) and isinstance(e.get("embedding"), list)
    ]
    if len(embs) < k + 1:
        return None
    arr = np.array(embs, dtype=np.float32)
    sim = arr @ arr.T
    np.fill_diagonal(sim, -np.inf)  # 自己除外
    top_k = np.sort(sim, axis=1)[:, -k:]
    dists = 1.0 - top_k
    return round(float(dists.mean()), 6)


def fog_strength_distribution(links: list) -> Optional[dict]:
    """C: link strength の percentile + variance (avg だけじゃなく分布形)。

    識別力: graph_maturity 内の avg_strength 単体では見えない「強い link
    が一部に集中 vs 全体的に弱い」の差分を捕捉。Physarum decay の効き方も
    分布で観察可能。
    """
    try:
        import numpy as np
    except ImportError:
        return None
    strengths = [
        float(l.get("strength", 0.0)) for l in (links or [])
        if isinstance(l, dict)
    ]
    if len(strengths) < 4:
        return None
    arr = np.array(strengths, dtype=np.float32)
    return {
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "variance": round(float(arr.var()), 6),
    }


def fog_inter_cluster_blur(state: dict) -> Optional[float]:
    """D: cluster 境界の柔らかさ (= ``phase6_metrics.cluster_inter_ratio``)。

    Phase 6 で reflect 内の cluster 推定後に算出される inter-cluster link
    比率を流用する (本 metric 用に再計算しない、低コスト)。reflect 未発火
    なら ``None`` (start 直後 cycle 等)。

    識別力: 高値 = cluster 間 link 多 = 境界 fuzzy / 低値 = cluster 孤立 =
    境界 sharp。sharp は森メトリクス的、fuzzy は靄的。
    """
    p6 = state.get("phase6_metrics", {}) if isinstance(state, dict) else {}
    val = p6.get("cluster_inter_ratio") if isinstance(p6, dict) else None
    if not isinstance(val, (int, float)):
        return None
    return round(float(val), 6)


def fog_te_flow_anisotropy(links: list) -> Optional[float]:
    """E: per-node 方向流出入の偏り (link strength を TE 流の proxy として)。

    各 node の inflow_strength_sum (to_id 一致の link strength 合計) と
    outflow_strength_sum (from_id 一致の合計) の絶対差を全 node 平均する。
    高値 = 一方通行 node 多 (source / sink が分離) / 低値 = 双方向 node 多
    (流れの偏りなし)。

    識別力: 直接 TE 計算は重いが、link strength は段階13 Phase 3 で
    TE + Bayesian で更新済 = 結果反映済。TE 計算を回避しつつ流れの方向性
    を観察可能。
    """
    inflow: dict = {}
    outflow: dict = {}
    for l in links or []:
        if not isinstance(l, dict):
            continue
        s = float(l.get("strength", 0.0))
        f, t = l.get("from_id"), l.get("to_id")
        if f:
            outflow[f] = outflow.get(f, 0.0) + s
        if t:
            inflow[t] = inflow.get(t, 0.0) + s
    nodes = set(inflow.keys()) | set(outflow.keys())
    if not nodes:
        return None
    diffs = [abs(inflow.get(n, 0.0) - outflow.get(n, 0.0)) for n in nodes]
    return round(float(sum(diffs) / len(diffs)), 6)


def compute_fog_metrics(
    state: dict, entries_with_embedding: list, links: list
) -> dict:
    """靄 5 種を 1 dict にまとめる (None 含む可、欠損は明示する設計)。"""
    return {
        "embedding_spread": fog_embedding_spread(entries_with_embedding),
        "local_density_mean": fog_local_density_mean(entries_with_embedding),
        "strength_distribution": fog_strength_distribution(links),
        "inter_cluster_blur": fog_inter_cluster_blur(state),
        "te_flow_anisotropy": fog_te_flow_anisotropy(links),
    }


# ============================================================
# Cycle metrics event 組立
# ============================================================

def build_cycle_metrics_event(
    state: dict,
    settings: dict,
    pref: dict,
    entries_with_embedding: Optional[list] = None,
    links: Optional[list] = None,
    code_version: Optional[str] = None,
) -> dict:
    """1 cycle ぶんの metrics event dict を組み立てる。emit はしない。

    引数 ``entries_with_embedding`` / ``links`` / ``code_version`` は test
    注入兼用。本番経路では ``None`` 渡しで遅延 import + 自動取得する。
    """
    if entries_with_embedding is None or links is None:
        from core.memory import (
            load_all_subjective_entries, load_all_memories
        )
        from core.memory_links import list_links

        if entries_with_embedding is None:
            entries_with_embedding = [
                e for e in load_all_subjective_entries()
                if isinstance(e.get("embedding"), list)
            ] + [
                m for m in load_all_memories()
                if isinstance(m.get("embedding"), list)
            ]
            # Codex P2 #6 fix: auto-load 経路でも emit と同じ global sort
            # (direct caller が build を呼ぶ場合の info_gain 順序契約を担保)
            _sort_entries_global_newest_first(entries_with_embedding)
        if links is None:
            links = list_links(limit=LINKS_SCAN_LIMIT)
    if code_version is None:
        code_version = get_code_version()

    pending = state.get("pending", []) or []
    pending_high_pri = sum(
        1 for p in pending if isinstance(p, dict)
        and float(p.get("priority", 0.0)) > 0.7
    )
    cycle_id_now = int(state.get("cycle_id", 0) or 0)
    pending_oldest_age = 0
    for p in pending:
        if not isinstance(p, dict):
            continue
        last_cycle = p.get("last_cycle")
        if isinstance(last_cycle, int):
            age = cycle_id_now - last_cycle
            if age > pending_oldest_age:
                pending_oldest_age = age
    pending_total_priority = round(
        sum(float(p.get("priority", 0.0)) for p in pending if isinstance(p, dict)),
        6,
    )

    cig = state.get("cumulative_information_gain", {}) or {}
    basin = state.get("basin_state", {}) or {}
    pc = state.get("predictor_confidence", {}) or {}
    pc_betas = [
        float(v.get("beta")) for v in pc.values()
        if isinstance(v, dict) and isinstance(v.get("beta"), (int, float))
    ]
    pc_mean_beta = (
        round(sum(pc_betas) / len(pc_betas), 6) if pc_betas else None
    )

    # graph 森メトリクス (count 系) + 靄メトリクス
    forest_node_ids = set()
    for l in links or []:
        if isinstance(l, dict):
            if l.get("from_id"):
                forest_node_ids.add(l.get("from_id"))
            if l.get("to_id"):
                forest_node_ids.add(l.get("to_id"))
    forest = {
        "node_count": len(forest_node_ids),
        "edge_count": len(links or []),
        "embedding_entry_count": len(entries_with_embedding or []),
    }
    fog = compute_fog_metrics(state, entries_with_embedding or [], links or [])

    # Slice 6.5 Step 5 (PLAN §5.7、2026-05-11): info_gain → efe 完全置換。
    # EFE 9 成分 (epistemic 4 + pragmatic 3 + regularization 2) + 3 カテゴリ +
    # 統合 G (Active Inference literal、最小化対象) + 5 C 情報 (案 ④ identity-anchored)。
    # state[CYCLE_KEY] 経由の cycle 間 diff 計算は不変、controller 不変。
    from core.info_gain import compute_efe_components, CYCLE_KEY
    prev_snapshot = state.get(CYCLE_KEY, {}) or {}
    efe_dict = compute_efe_components(
        state, prev_snapshot, entries_with_embedding or [], links or [], fog_now=fog
    )

    # V10 Sedimentary C 案 C (2026-07-05): post-hoc EFE の主観 affordance 用 snapshot。
    # subjective_view が <subjective_state> に preference_alignment 1 行を表示する
    # (P2 affordance literal: 表示のみ、行動指示なし。iku が「自分の観測が自分の
    # 選好に近づいたか」を読める材料を置くだけ)。
    if isinstance(state, dict):
        state["last_efe_snapshot"] = {
            "cycle_id": cycle_id_now,
            "G": efe_dict.get("G"),
            "effective_change": efe_dict.get("effective_change"),
            "epistemic_gain": efe_dict.get("epistemic_gain"),
            "pragmatic_gain": efe_dict.get("pragmatic_gain"),
        }

    dispositions = state.get("dispositions", {}) or {}
    self_disp_keys = list(
        (dispositions.get("self") or {}).keys()
    )
    attr_persp_keys = [k for k in dispositions.keys() if k != "self"]

    return {
        "event_type": "cycle_metrics",
        "cycle_id": cycle_id_now,
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": str(state.get("run_id", "")),
        "session_id": str(state.get("session_id", "")),
        "config_hash": compute_config_hash(settings, pref),
        "code_version": code_version,
        "enabled_features": get_enabled_features(settings),
        "provider": str((settings or {}).get("provider", "")),
        "model": str((settings or {}).get("model", "")),

        "vital": {
            "energy": float(state.get("energy", 50)),
            "entropy": float(state.get("entropy", 0.65)),
            "pressure": float(state.get("pressure", 0.0)),
            "unresolved_external": float(state.get("unresolved_external", 0.0)),
        },
        "evaluation": {
            "e1": state.get("last_e1"),
            "e2": state.get("last_e2"),
            "e3": state.get("last_e3"),
            "e4": state.get("last_e4"),
            "prediction_error": state.get("last_prediction_error"),
        },
        "histories_summary": {
            "pred_err_e2": summarize_history(
                state.get("prediction_error_history_e2", [])
            ),
            "pred_err_ec": summarize_history(
                state.get("prediction_error_history_ec", [])
            ),
            "jepa_pe": summarize_history(
                state.get("jepa_prediction_error_history", [])
            ),
        },
        "cig": {
            "e2_total": cig.get("e2_total"),
            "ec_total": cig.get("ec_total"),
            "e2_window_mean": cig.get("e2_window_mean"),
            "ec_window_mean": cig.get("ec_window_mean"),
            "flat_signal": cig.get("flat_signal"),
            "flat_streak": cig.get("flat_streak"),
            "per_tool_count": len(cig.get("per_tool", {}) or {}),
        },
        "basin": {
            "id": str(basin.get("current_basin_id", "")),
            "dwell": int(basin.get("current_basin_dwell", 0) or 0),
            "transition_pending": bool(basin.get("phase_transition_pending", False)),
        },
        "redundancy": {
            "last": state.get("last_redundancy"),
        },
        "predictor_confidence_summary": {
            "tool_count": len(pc),
            "mean_beta": pc_mean_beta,
        },
        "graph": {
            **forest,
            "fog": fog,
        },
        "efe": efe_dict,
        "agency": {
            "voluntary_memory_store_count": int(
                state.get("voluntary_memory_store_count", 0) or 0
            ),
            "tool_level": int(state.get("tool_level", 0) or 0),
            "files_written_count": len(state.get("files_written", []) or []),
            "files_read_count": len(state.get("files_read", []) or []),
            "action_ledger_count": len(state.get("action_ledger", []) or []),
        },
        "relations": {
            "self_disposition_keys": self_disp_keys,
            "attributed_perspective_keys": attr_persp_keys,
        },
        "pending": {
            "count": len(pending),
            "high_priority_count": pending_high_pri,
            "oldest_age_cycles": pending_oldest_age,
            "total_priority": pending_total_priority,
        },
        "phase6_metrics": {
            "cluster_mi": (state.get("phase6_metrics", {}) or {}).get("cluster_mi"),
            "cluster_inter_ratio": (
                state.get("phase6_metrics", {}) or {}
            ).get("cluster_inter_ratio"),
        },
    }


# ============================================================
# Append + Index
# ============================================================

def _json_default(obj: Any) -> Any:
    """JSON encoder fallback (numpy 等の非標準型を最後の砦で str 化)."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def _sort_entries_global_newest_first(entries: list) -> list:
    """``entries`` を time field で global newest first に in-place sort。

    Codex review 2026-05-09 P2 #5/#6 fix: subj+mem の単純 concat は各 source
    内部 newest first だが merged 全体は newest first にならない。
    info_gain.compute_efe_components (Slice 6.5 Step 5 で旧 compute_info_gain_components
    から置換) の docstring 契約「global newest first 順」を満たすため、入力前処理として
    必ず本 helper を通す。

    emit_cycle_metrics と build_cycle_metrics_event (auto-load 経路) の両方で
    呼び出す DRY helper、片方だけ sort して片方が漏れる罠を回避する。

    Args:
        entries: subj + memory entry list (各 entry に "time" or "ts" field 期待)

    Returns:
        sort 後の list (in-place 変更後の参照、副作用あり)。
    """
    entries.sort(
        key=lambda e: e.get("time") or e.get("ts") or "",
        reverse=True,  # newest first
    )
    return entries


def _atomic_append_jsonl(path: Path, entry: dict) -> None:
    """append-only file に 1 行 jsonl を書き足す。

    raw_events.jsonl と同流儀の append、F-005 view_persistence registry には
    乗せない (mutable view ではない、再起動時 rebuild も raw 同様)。
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    line = json.dumps(entry, ensure_ascii=False, default=_json_default)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _update_metrics_index(
    index_path: Path, filename: str, event: dict
) -> None:
    """metrics_events.jsonl の append-mode index 更新 (count += 1)。

    既存 ``memory.py:_update_jsonl_index`` と同 semantic だが、metrics 専用に
    1 行 1 entry の append step で count increment + from/to 更新する。
    count は cycle と outward_attempt を含む全 event 数（cycle 数ではない）。
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(exist_ok=True, parents=True)
    index: dict = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    if filename not in index:
        index[filename] = {"count": 0, "from": "", "to": ""}
    index[filename]["count"] += 1
    if not index[filename].get("from"):
        index[filename]["from"] = event.get("time", "")
    index[filename]["to"] = event.get("time", "")
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# 外向き行動の観察 (state / prompt / G から独立)
# ============================================================

OUTWARD_ACT_TOOLS = frozenset({
    "output_display", "elyth_post", "elyth_reply", "elyth_like", "elyth_follow",
    "elyth_mark_read", "x_post", "x_reply", "x_quote", "x_like",
})
OUTWARD_SENSE_TOOLS = frozenset({
    "elyth_info", "elyth_get", "x_timeline", "x_search", "x_get_notifications",
    "camera_stream", "screen_peek", "mic_record", "WebSearch", "WebFetch",
})
OUTWARD_UTTERANCE_TOOLS = frozenset({
    "output_display", "elyth_post", "elyth_reply", "x_post", "x_reply", "x_quote",
})
OUTWARD_ARGUMENT_TOOLS = frozenset({"http_request", "view_image", "listen_audio"})


def classify_outward(tool: str, args: Optional[dict] = None) -> str:
    """act / sense / control / undetermined / internal。args=None は候補段階。

    URL は http(s)、profile 内のローカル画像・音声は internal。
    profile 外や欠落した path は未判定のまま残す。
    """
    if tool in OUTWARD_ACT_TOOLS:
        return "act"
    if tool in OUTWARD_SENSE_TOOLS:
        return "sense"
    if tool == "camera_stream_stop":
        return "control"
    if tool in OUTWARD_ARGUMENT_TOOLS:
        if args is None:
            return "undetermined"
        if tool == "http_request":
            return "sense" if str(args.get("method", "GET")).strip().upper() == "GET" else "act"
        path = str(args.get("path", "")).strip()
        if path.lower().startswith(("https://", "http://")):
            return "sense"
        if path:
            from core.config import BASE_DIR
            resolved = (BASE_DIR / path).resolve()
            if resolved.is_relative_to(BASE_DIR.resolve()):
                return "internal"
        return "undetermined"
    return "internal"


def observe_outward_candidates(event: dict, candidates: list) -> None:
    """パーサ通過後の全構成 tool を分類。未判定数は構成 tool の出現数。"""
    kinds = [classify_outward(tool) for c in candidates
             for tool in c.get("tools", [c.get("tool", "")])]
    event.update(proposed_act="act" in kinds, proposed_sense="sense" in kinds,
                 proposed_undetermined=kinds.count("undetermined"))
    if not candidates:
        event["end"] = "no_candidate"


def observe_outward_selection(event: dict) -> None:
    """controller が渡した同順序の候補と G から、働きかける候補の差を測る。"""
    tools = event.get("candidate_tools", [])
    values = event.get("g_values", [])
    idx = event.get("selected_idx")
    if not isinstance(idx, int) or len(tools) != len(values) or not 0 <= idx < len(tools):
        return
    acts = [any(classify_outward(t) == "act" for t in chain) for chain in tools]
    outward_g = [g for g, act in zip(values, acts) if act]
    event["selected_act"] = acts[idx]
    event["best_outward_g_minus_selected_g"] = min(outward_g) - values[idx] if outward_g else None


def tool_execution_status(output: str, is_error: bool, error_kind=None) -> str:
    """Runtime の構造だけで判定する。本文の語句は成否を表さない。"""
    if not is_error:
        return "ok"
    return {"rejected": "rejected", "tool_failure": "tool_error",
            "exception": "runtime_error"}.get(error_kind, "runtime_error")


def build_outward_execution(tool: str, args: dict, output: str,
                            is_error: bool, channel: str, *, error_kind=None,
                            detail=None, detail_error=None) -> dict:
    """全 invocation の観察。成否と失敗の種類は runtime の構造を使う。

    connected_at_enqueue は output_display の受付文から取得し、到達数とは扱わない。
    """
    text = str(output)
    status = tool_execution_status(text, is_error, error_kind)
    connected = None
    if tool == "output_display":
        channel = str(args.get("channel", "")).strip()
        match = re.search(r"受付時の接続 (\d+) 件", text)
        if status == "ok" and match:
            connected = int(match.group(1))
        elif is_error and "接続している受け手が 0 件" in text:
            connected = 0
    return {"tool": tool, "channel": channel, "status": status,
            "is_error": is_error, "error_kind": error_kind,
            "detail": detail, "detail_error": detail_error,
            "connected_at_enqueue": connected, "category": classify_outward(tool, args)}


def emit_outward_attempt(event: dict) -> dict:
    """観察イベントを append。state を受け取らず、cycle snapshot を更新しない。"""
    from core.config import MEMORY_DIR
    target = MEMORY_DIR / METRICS_FILE_NAME
    _atomic_append_jsonl(target, event)
    _update_metrics_index(MEMORY_DIR / INDEX_FILE_NAME, target.name, event)
    return event


IDENTITY_TERM_DETECTOR_VERSION = "substring-v1"
_IDENTITY_TERMS = ("AIアシスタント", "AI assistant", "AIAssistant")


def _emit_identity_observation(event: dict) -> bool:
    """Append once, independently of state and cognitive memory; failures are observational."""
    from core.config import MEMORY_DIR
    target = MEMORY_DIR / METRICS_FILE_NAME
    try:
        _atomic_append_jsonl(target, event)
    except Exception as exc:
        print(f"  [identity] append skip ({event['event_type']}): {exc}")
        return False
    try:
        _update_metrics_index(MEMORY_DIR / INDEX_FILE_NAME, target.name, event)
    except Exception as exc:
        print(f"  [identity] index skip (行は保存済み): {exc}")
    return True


def emit_identity_term_detected(*, run_id: str, cycle_id: int,
                                llm1_text: str, llm2_text: str) -> bool:
    """One event per cycle, preserving each term/source pair (including quotes/negation)."""
    detections = [
        {"term": term, "source": source}
        for source, text in (("LLM①", llm1_text), ("LLM②", llm2_text))
        for term in _IDENTITY_TERMS if term in text
    ]
    if not detections:
        return False
    return _emit_identity_observation({
        "event_type": "identity_term_detected",
        "time": datetime.now().isoformat(), "run_id": run_id, "cycle_id": cycle_id,
        "detections": detections, "detector_version": IDENTITY_TERM_DETECTOR_VERSION,
    })


def emit_identity_name_changed(*, run_id: str, cycle_id: int,
                               old_name: str, new_name: str, tool_id: str) -> bool:
    """Called only after successful persistence by update_self, using its live state."""
    return _emit_identity_observation({
        "event_type": "identity_name_changed",
        "time": datetime.now().isoformat(), "run_id": run_id, "cycle_id": cycle_id,
        "old_name": old_name, "new_name": new_name, "tool_id": tool_id,
    })


_APPROVAL_FIELDS = ("tool_intent", "tool_expected_outcome", "note")


def build_tool_invocation_events(records: list, *, run_id: str, attempt_id: str,
                                 entry_id, cycle_id, time: str) -> list:
    """1 実行 1 行の観察イベント (0d-1)。records は main が summary から写した値。

    record: chain_position / invocation_position / tool_id / tool / mode /
    tool_input / output / is_error / in_cycle_result。
    intent・expect・note は承認 3 層をそのまま (無ければ空文字)、args は
    それを除いた runtime 記録上の引数。status は 0b の build_outward_execution と同じ判定。
    entry 未成立なら entry_id / cycle_id は None のまま渡す。
    """
    from core.config import cap_tool_result
    events = []
    for r in records:
        ti = r.get("tool_input") or {}
        output = str(r.get("output", ""))
        events.append({
            "event_type": "tool_invocation", "run_id": run_id,
            "failure_contract": ACTIVE_FAILURE_CONTRACT,
            "attempt_id": attempt_id, "time": time,
            "entry_id": entry_id, "cycle_id": cycle_id,
            "chain_position": r["chain_position"],
            "invocation_position": r["invocation_position"],
            "tool_id": r.get("tool_id", ""), "tool": r["tool"], "mode": r["mode"],
            "intent": str(ti.get("tool_intent", "") or ""),
            "expect": str(ti.get("tool_expected_outcome", "") or ""),
            "note": str(ti.get("note", "") or ""),
            "args": {k: v for k, v in ti.items() if k not in _APPROVAL_FIELDS},
            "is_error": r.get("is_error"),
            "error_kind": r.get("error_kind"),
            "detail": r.get("detail"), "detail_error": r.get("detail_error"),
            "status": build_outward_execution(r["tool"], ti, output,
                                              bool(r.get("is_error")), "",
                                              error_kind=r.get("error_kind"))["status"],
            "result": cap_tool_result(output),
            "in_cycle_result": bool(r.get("in_cycle_result")),
        })
    return events


class InvocationAppendError(OSError):
    """append 失敗。written は再試行せず利用できる保存済みの先頭行数。"""

    def __init__(self, written: int):
        super().__init__(f"tool_invocation append failed after {written} rows")
        self.written = written


def emit_tool_invocations(events: list) -> int:
    """観察イベントを 1 行ずつ append。state を受け取らない。再試行しない。

    append と index 更新の失敗を区別して print する。index だけ失敗した行は
    保存済みなので次の行へ進む (再試行による重複を作らない)。append の失敗は
    InvocationAppendError を送出する (残りは書かない)。その written と正常時の
    戻り値は append できた先頭行数。index の成否には依存しない。
    """
    from core.config import MEMORY_DIR
    target = MEMORY_DIR / METRICS_FILE_NAME
    written = 0
    for event in events:
        try:
            _atomic_append_jsonl(target, event)
        except Exception as e:
            print(f"  [invocation] append 失敗 ({written}/{len(events)} 行保存済み): {e}")
            raise InvocationAppendError(written) from e
        written += 1
        try:
            _update_metrics_index(MEMORY_DIR / INDEX_FILE_NAME, target.name, event)
        except Exception as e:
            print(f"  [invocation] index 更新失敗 (行は保存済み): {e}")
    return written


# C-2: 観察専用。state / prompt / G へは接続しない。
ERROR_MODEL_VERSION = "beta_bernoulli_v1"
# All tool families now report failures explicitly; v1 history is excluded.
ACTIVE_FAILURE_CONTRACT = 2


def failure_contract_exclusion(event, required_contract=None):
    """Shared by live observation, rebuild and summaries; never infer from text."""
    required = ACTIVE_FAILURE_CONTRACT if required_contract is None else required_contract
    version = event.get("failure_contract", 1)
    if type(version) is not int or version not in (1, 2):
        return "unknown_failure_contract"
    if version != required:
        return "old_failure_contract" if version < required else "unknown_failure_contract"
    return None

_INVOCATION_KEY = ("run_id", "attempt_id", "chain_position", "invocation_position")
_PREDICTION_REFS = (*_INVOCATION_KEY, "entry_id", "cycle_id", "tool_id", "tool", "time")


def beta_error_prediction(alpha: float, beta: float, is_error: bool) -> dict:
    """事前から一観測の予測・KL[事後||事前]を計算する純粋関数 (nats)。

    正の有限パラメータと本物の bool のみ受理。非有限値・大きな負の KL は
    例外にし、丸め誤差 (-1e-10 以上) の負値だけ 0 にする。SciPy は遅延 import。
    """
    from scipy.special import digamma

    if (type(is_error) is not bool or not math.isfinite(alpha)
            or not math.isfinite(beta) or alpha <= 0 or beta <= 0):
        raise ValueError("invalid Beta parameters or observation")
    total = alpha + beta
    observed = alpha if is_error else beta
    p = alpha / total
    surprise = math.log(total) - math.log(observed)
    # betaln(a,b)-betaln(a+o,b+1-o) = log(a+b)-log(observed)。
    # 一観測の共役更新に限定した同じ式で、大きい betaln 同士の桁落ちを避ける。
    kl = float(surprise + digamma(observed + 1) - digamma(total + 1))
    brier = (p - int(is_error)) ** 2
    if not all(math.isfinite(v) for v in (p, surprise, kl, brier)) or kl < -1e-10:
        raise ValueError("non-finite prediction or negative KL")
    return {"alpha": alpha, "beta": beta, "p_error": p, "is_error": is_error,
            "surprise_nats": surprise, "kl_nats": max(0.0, kl), "brier": brier,
            "posterior_alpha": alpha + int(is_error),
            "posterior_beta": beta + int(not is_error)}


def _invocation_exclusion(event: dict, seen: set, required_contract=None) -> Optional[str]:
    """復元・通常運転共通の検証。無関係な event は呼出側で除く。"""
    contract_reason = failure_contract_exclusion(event, required_contract)
    if contract_reason:
        return contract_reason
    if (any(not isinstance(event.get(k), str) or not event[k].strip()
            for k in ("run_id", "attempt_id"))
            or any(type(event.get(k)) is not int or event[k] < 0
                   for k in ("chain_position", "invocation_position"))):
        return "invalid_key"
    if type(event.get("is_error")) is not bool:
        return "invalid_is_error"
    if not isinstance(event.get("tool"), str) or not event["tool"].strip():
        return "invalid_tool"
    if tuple(event[k] for k in _INVOCATION_KEY) in seen:
        return "duplicate"
    return None


def _emit_error_observation(event: dict) -> bool:
    """観察を一度だけ append。index 故障で保存済み行を再試行しない。"""
    from core.config import MEMORY_DIR
    target = MEMORY_DIR / METRICS_FILE_NAME
    try:
        _atomic_append_jsonl(target, event)
    except Exception as e:
        print(f"  [error_prediction] append skip ({event['event_type']}): {e}")
        return False
    try:
        _update_metrics_index(MEMORY_DIR / INDEX_FILE_NAME, target.name, event)
    except Exception as e:
        print(f"  [error_prediction] index skip (行は保存済み): {e}")
    return True


class ErrorPredictionObserver:
    """保存済み invocation だけを数える、main 専用の観察者 (state と独立)。"""

    def __init__(self, run_id: str, *, required_contract=None):
        self.run_id = run_id
        self.required_contract = required_contract
        self.counts = {}
        self.total = (1, 1)
        self.seen = set()
        self.history_reset = False

    def _accept(self, event: dict):
        reason = _invocation_exclusion(event, self.seen, self.required_contract)
        if reason:
            return None, reason
        before = self.counts.get(event["tool"], (1, 1))
        global_before = self.total
        error = int(event["is_error"])
        self.seen.add(tuple(event[k] for k in _INVOCATION_KEY))
        self.counts[event["tool"]] = (before[0] + error, before[1] + 1 - error)
        self.total = (self.total[0] + error, self.total[1] + 1 - error)
        return (before, global_before), None

    def rebuild(self) -> dict:
        """全 run の履歴を行順で復元。読込障害は今回だけリセットし、印を残す。"""
        from core.config import MEMORY_DIR
        self.counts, self.total, self.seen = {}, (1, 1), set()
        self.history_reset = False
        excluded = {}
        failure = None
        try:
            try:
                stream = (MEMORY_DIR / METRICS_FILE_NAME).open("rb")
            except FileNotFoundError:
                stream = None
            if stream is not None:
                with stream:
                    for line in stream:
                        try:
                            event = json.loads(line)
                            if not isinstance(event, dict):
                                raise ValueError("not an object")
                        except (ValueError, UnicodeError):
                            excluded["invalid_json"] = excluded.get("invalid_json", 0) + 1
                            continue
                        if event.get("event_type") != "tool_invocation":
                            continue
                        _, reason = self._accept(event)
                        if reason:
                            excluded[reason] = excluded.get(reason, 0) + 1
        except Exception as e:
            failure = str(e)
            self.counts, self.total, self.seen = {}, (1, 1), set()
            self.history_reset = True
        report = {"event_type": "error_model_rebuilt", "run_id": self.run_id,
                  "failure_contract": self.required_contract if self.required_contract is not None else ACTIVE_FAILURE_CONTRACT,
                  "history_id": self.run_id, "history_reset": self.history_reset,
                  "history_scope": "since_startup" if self.history_reset else "profile",
                  "model_version": ERROR_MODEL_VERSION, "prior": {"alpha": 1, "beta": 1},
                  "baseline_prior": {"alpha": 1, "beta": 1},
                  "accepted": len(self.seen), "excluded": sum(excluded.values()),
                  "exclusion_reasons": excluded, "read_error": failure,
                  "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        _emit_error_observation(report)
        return report

    def observe(self, events: list) -> None:
        """保存済み行を順に計数。計算・予測保存の故障でも後続の計数を続ける。"""
        for event in events:
            if event.get("event_type") != "tool_invocation":
                continue
            priors, reason = self._accept(event)
            ref = {k: event.get(k) for k in _PREDICTION_REFS}
            ref.update(model_version=ERROR_MODEL_VERSION, history_id=self.run_id,
                       history_reset=self.history_reset,
                       history_scope="since_startup" if self.history_reset else "profile")
            if reason:
                _emit_error_observation({**ref, "event_type": "error_prediction_excluded",
                                         "reason": reason,
                                         "source_failure_contract": event.get("failure_contract", 1)})
                continue
            # _accept が先に計数済み。以下の失敗は学習履歴を欠落させない。
            try:
                before, global_before = priors
                prediction = beta_error_prediction(*before, event["is_error"])
                baseline = beta_error_prediction(*global_before, event["is_error"])
                row = {**ref, **prediction, "event_type": "error_prediction",
                       "failure_contract": self.required_contract if self.required_contract is not None else ACTIVE_FAILURE_CONTRACT,
                       "prior": {"alpha": 1, "beta": 1},
                       "baseline": {**baseline, "prior": {"alpha": 1, "beta": 1}},
                       "prediction_timing": "posthoc_from_preceding_invocations"}
            except Exception as e:
                _emit_error_observation({**ref, "event_type": "error_prediction_missing",
                                         "reason": "calculation_failed", "error": str(e)})
                continue
            if not _emit_error_observation(row):
                _emit_error_observation({**ref, "event_type": "error_prediction_missing",
                                         "reason": "append_failed"})


def summarize_error_prediction(events: list, *, run_id: Optional[str] = None,
                               required_contract=None) -> dict:
    """指定行 (任意で run 限定) のみ集計。履歴復元はせず、保存済み baseline と比較。

    tool_invocation と予測を複合キーで重複排除し、未対応の実行は missing として返す。
    delta はモデル - baseline (負ならモデルの損失が小さい)。
    """
    rows, invocations = {}, set()
    for event in events:
        if run_id is not None and event.get("run_id") != run_id:
            continue
        kind = event.get("event_type")
        if kind not in ("tool_invocation", "error_prediction"):
            continue
        if _invocation_exclusion(event, set(), required_contract):
            continue
        key = tuple(event[k] for k in _INVOCATION_KEY)
        if kind == "tool_invocation":
            invocations.add(key)
        else:
            rows.setdefault(key, event)

    def aggregate(group):
        count = len(group)
        def mean(field, baseline=False):
            return (sum((r["baseline"] if baseline else r)[field] for r in group) / count
                    if count else None)
        loss, baseline_loss = mean("surprise_nats"), mean("surprise_nats", True)
        brier, baseline_brier = mean("brier"), mean("brier", True)
        return {"count": count, "mean_surprise_nats": loss, "log_loss": loss,
                "mean_brier": brier, "baseline_log_loss": baseline_loss,
                "baseline_brier": baseline_brier,
                "log_loss_delta": loss - baseline_loss if count else None,
                "brier_delta": brier - baseline_brier if count else None}

    values = list(rows.values())
    return {**aggregate(values), "invocations": len(invocations),
            "missing": len(invocations - rows.keys()),
            "per_tool": {tool: aggregate([r for r in values if r["tool"] == tool])
                         for tool in sorted({r["tool"] for r in values})}}


def summarize_outward(events: list, k: int = 5) -> dict:
    """観察イベントだけを集計する。割合の分母を各項目に明記する。

    後続入力は同じ run 内で seq が大きく同 channel、fire 差が 0..k のもの。
    k は後続 fire 数（初期値5）。末尾で k fire を観測できない発話は、入力が
    あっても分母から除外する。同じ入力を複数発話に割り当ててよい。
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    attempts = [e for e in events if e.get("event_type") == "outward_attempt"]
    total = len(attempts)

    def ratio(count, denominator):
        return {"count": count, "denominator": denominator,
                "rate": count / denominator if denominator else None}

    summary = {"attempts": total}
    for key in ("proposed_act", "proposed_sense", "selected_act"):
        summary[key] = ratio(sum(bool(e.get(key)) for e in attempts), total)
    summary["proposed_undetermined"] = ratio(
        sum(bool(e.get("proposed_undetermined")) for e in attempts), total)
    summary["proposed_undetermined"]["tools"] = sum(e.get("proposed_undetermined", 0) for e in attempts)
    executions = [rec for e in attempts for rec in e.get("exec", [])]
    summary["execution"] = {}
    for status in ("ok", "rejected", "runtime_error", "tool_error", "not_invoked"):
        item = ratio(sum(any(rec.get("status") == status for rec in e.get("exec", []))
                         for e in attempts), total)
        item["invocations"] = sum(rec.get("status") == status for rec in executions)
        summary["execution"][status] = item

    runs = {}
    for e in attempts:
        run_id = e.get("run_id", e.get("attempt_id", "").rsplit("_", 1)[0])
        runs.setdefault(run_id, []).append(e)
    matched, unmatched, incomplete = [], [], []
    for run_id, fires in runs.items():
        for i, fire in enumerate(fires):
            for utterance in fire.get("utterances", []):
                if utterance.get("tool") not in OUTWARD_UTTERANCE_TOOLS:
                    continue
                ref = {"attempt_id": fire.get("attempt_id"), **utterance}
                if i + k >= len(fires):
                    incomplete.append(ref)
                    continue
                found = any(inp.get("channel") == utterance.get("channel")
                            and inp["seq"] > utterance["seq"]
                            for later in fires[i:i + k + 1] for inp in later.get("inputs", []))
                (matched if found else unmatched).append(ref)
    summary["following_input"] = {
        **ratio(len(matched), len(matched) + len(unmatched)),
        "matched": matched, "unmatched": unmatched, "incomplete": incomplete, "k": k,
    }
    return summary


# ============================================================
# 公開 API: emit
# ============================================================

def emit_cycle_metrics(state: dict, settings: dict, pref: dict) -> dict:
    """1 cycle ぶんの metrics event を組み立てて jsonl に append + index 更新。

    本番経路は cycle 末で main.py から save_state 直前に呼ぶ想定 (orchestration
    §② ゆう確定 emit timing)。例外は呼出側で catch する (defensive、metrics
    失敗で cycle 全体を止めない、reflect 継続原則と整合)。

    Slice 3 拡張: emit 後に state[CYCLE_KEY] を更新して次 cycle の info_gain
    diff 計算用 snapshot を残す。state mutation は本関数のみ、build 内では
    read-only に扱う設計。

    Returns:
        emit した event dict (test / 即時 inspection 用)
    """
    from core.config import MEMORY_DIR
    from core.memory import (
        load_all_subjective_entries, load_all_memories
    )
    from core.memory_links import list_links
    from core.info_gain import snapshot_for_next_cycle, CYCLE_KEY

    # 1 度の取得を build と snapshot で共有 (重複 fetch 回避)。
    # Codex review 2026-05-09 P2 #5 fix: subj + mem の単純 concat は各 source
    # 内部 newest first だが merged 全体は newest first じゃない (subj 全件が
    # 頭、mem 全件が後ろ)。memory のみに新 entry 追加された cycle で先頭の
    # subj 古い entry を「新」と誤検知する罠を回避するため、time field で
    # global newest first sort する (raw / subj / mem 全て time field 持つ
    # 前提、metrics.py 自身も datetime.now().strftime() 形式で発行)。
    entries_with_embedding = [
        e for e in load_all_subjective_entries()
        if isinstance(e.get("embedding"), list)
    ] + [
        m for m in load_all_memories()
        if isinstance(m.get("embedding"), list)
    ]
    # Codex P2 #5 fix: subj+mem の global newest first sort (build 側と DRY)
    _sort_entries_global_newest_first(entries_with_embedding)
    links = list_links(limit=LINKS_SCAN_LIMIT)

    event = build_cycle_metrics_event(
        state, settings, pref,
        entries_with_embedding=entries_with_embedding,
        links=links,
    )
    target = MEMORY_DIR / METRICS_FILE_NAME
    _atomic_append_jsonl(target, event)
    _update_metrics_index(MEMORY_DIR / INDEX_FILE_NAME, target.name, event)

    # Slice 3: 次 cycle 用 prev snapshot 更新 (state mutation はここだけ)
    fog_now = (event.get("graph", {}) or {}).get("fog", {})
    state[CYCLE_KEY] = snapshot_for_next_cycle(
        state, entries_with_embedding, links, fog_now
    )
    return event
