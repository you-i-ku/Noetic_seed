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
import os
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
# 公開 API: emit
# ============================================================

def emit_cycle_metrics(state: dict, settings: dict, pref: dict) -> dict:
    """1 cycle ぶんの metrics event を組み立てて jsonl に append + index 更新。

    本番経路は cycle 末で main.py から save_state 直前に呼ぶ想定 (orchestration
    §② ゆう確定 emit timing)。例外は呼出側で catch する (defensive、metrics
    失敗で cycle 全体を止めない、reflect 継続原則と整合)。

    Returns:
        emit した event dict (test / 即時 inspection 用)
    """
    from core.config import MEMORY_DIR

    event = build_cycle_metrics_event(state, settings, pref)
    target = MEMORY_DIR / METRICS_FILE_NAME
    _atomic_append_jsonl(target, event)
    _update_metrics_index(MEMORY_DIR / INDEX_FILE_NAME, target.name, event)
    return event
