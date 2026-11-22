"""Slice 2 metrics test (orchestration §3 P0 #2 + §⑧ 靄メトリクス)。

CLAUDE.md §5 識別力 + §6 docstring 同期 遵守。各 helper / fog metric / 統合
build / append emit に discriminating fixture を仕込んで、旧/誤実装が fail
する設計。

検証範囲:

A. canonicalize_config + config_hash:
   A1. _comment* / ws_token / _ema が strip される (再帰)
   A2. cosmetic 変更 (_comment 値変更) で config_hash 不変 (識別力)
   A3. non-cosmetic 変更 (model 名) で config_hash 変化

B. enabled_features:
   B1. 5 keys 全て揃う (mcp_bridge / retrieval_links / reflect_cold_start
                       / predictor_mode / auto_approve_all)
   B2. settings 欠損時の default 値 (False / "light")

C. summarize_history:
   C1. 通常 list → {len, mean, max, latest_n}
   C2. 空 list → 全 None / 空 latest_n (defensive)
   C3. latest_n が len より大 → list 全体を返す

D. 靄 fog_embedding_spread (A):
   D1. 同方向 vector × 多 → spread 小 (≈0)
   D2. 直交 vector → spread 大 (>0)
   D3. データ < 2 → None

E. 靄 fog_local_density_mean (B):
   E1. clustered → 小値、spread → 大値 (識別力)
   E2. データ < k+1 → None

F. 靄 fog_strength_distribution (C):
   F1. p25 <= p50 <= p75 (percentile 順序整合)
   F2. variance >= 0
   F3. データ < 4 → None

G. 靄 fog_inter_cluster_blur (D):
   G1. state["phase6_metrics"]["cluster_inter_ratio"] あり → その値
   G2. phase6 なし or 値 None → None (識別力: 違う key 見る実装で fail)

H. 靄 fog_te_flow_anisotropy (E):
   H1. 一方通行 links → 高 anisotropy
   H2. 双方向 同強度 links → 低 anisotropy (識別力: from/to 反転実装で fail)

I. build_cycle_metrics_event:
   I1. 必須 field 全て含む (run_id / config_hash / cig / basin / graph.fog 等)
   I2. summarize_history が histories 全部に適用されてる

J. emit_cycle_metrics + index 更新:
   J1. jsonl に 1 行 append、複数回 emit で行数増加 (識別力: overwrite で fail)
   J2. index.json count = 実 jsonl 行数

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_metrics.py
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.metrics as cm


def _assert(cond, label):
    """assert helper。pytest 互換のため失敗時は AssertionError を raise する。

    Slice 3 自己点検 (2026-05-09): Codex review P2 #4 同型 bug を本 file の
    `_assert` でも発見 → test_info_gain.py と同 pattern で修正。`return False`
    だけだと pytest で PytestReturnNotNoneWarning 出るだけで test pass 化する。
    AssertionError raise + 既存 runner 側 try/except で fail 認識化。
    """
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        raise AssertionError(label)
    return cond


# ============================================================
# A. canonicalize_config + config_hash
# ============================================================

def test_a1_strip_cosmetic_recursive():
    """A1: _comment* / ws_token / _ema が再帰的に strip される。"""
    print("== A1: canonicalize_config strip ==")
    settings = {
        "model": "x",
        "ws_token": "secret",
        "_comment": "top",
        "_comment_extra": "more",
        "nested": {
            "value": 1,
            "_comment": "nested",
        },
    }
    pref = {"_ema": {"x": 50.0}, "drives": {"a": 0.5}}
    canonical = cm.canonicalize_config(settings, pref)
    ok = True
    s = canonical["settings"]
    p = canonical["pref"]
    ok &= _assert("ws_token" not in s, "ws_token strip")
    ok &= _assert("_comment" not in s, "_comment strip")
    ok &= _assert("_comment_extra" not in s, "_comment* prefix strip")
    ok &= _assert("_comment" not in s["nested"], "nested _comment strip (再帰)")
    ok &= _assert("model" in s and s["model"] == "x", "non-strip key 残存")
    ok &= _assert("_ema" not in p, "pref _ema strip")
    ok &= _assert("drives" in p, "pref non-strip 残存")
    return ok


def test_a2_hash_stable_against_cosmetic():
    """A2: cosmetic 変更で config_hash は変わらない (識別力: canonicalize なし実装で fail)."""
    print("== A2: cosmetic 変更で hash 不変 ==")
    base = {"model": "x", "approval": {"auto_approve_all": True}}
    h_base = cm.compute_config_hash(base, {})
    # _comment 追加 → hash 不変
    with_comment = {**base, "_comment": "explanation"}
    h_with_comment = cm.compute_config_hash(with_comment, {})
    # ws_token 変更 → hash 不変
    with_ws = {**base, "ws_token": "different_token"}
    h_with_ws = cm.compute_config_hash(with_ws, {})
    ok = True
    ok &= _assert(h_base == h_with_comment,
                  f"_comment 追加で hash 不変 (旧実装で fail、{h_base} == {h_with_comment})")
    ok &= _assert(h_base == h_with_ws,
                  "ws_token 変更で hash 不変")
    return ok


def test_a3_hash_changes_on_substantive_diff():
    """A3: 非 cosmetic 変更 (model 名 etc) で hash 変化。"""
    print("== A3: model 変更で hash 変化 ==")
    h1 = cm.compute_config_hash({"model": "x"}, {})
    h2 = cm.compute_config_hash({"model": "y"}, {})
    return _assert(h1 != h2, f"model 違うと hash 違う ({h1} != {h2})")


# ============================================================
# B. enabled_features
# ============================================================

def test_b1_enabled_features_full_5_keys():
    """B1: 5 keys 全て出る (mcp_bridge / retrieval_links / reflect_cold_start
        / predictor_mode / auto_approve_all)."""
    print("== B1: enabled_features 5 keys ==")
    settings = {
        "mcp_bridge": {"enabled": True},
        "retrieval": {"use_links": True},
        "reflection": {"reflect_cold_start_mode": True},
        "world_model": {"predictor_mode": "medium"},
        "approval": {"auto_approve_all": False},
    }
    f = cm.get_enabled_features(settings)
    expected_keys = {
        "mcp_bridge", "retrieval_links", "reflect_cold_start",
        "predictor_mode", "auto_approve_all",
    }
    ok = True
    ok &= _assert(set(f.keys()) == expected_keys,
                  f"5 keys 完全一致 (got {set(f.keys())})")
    ok &= _assert(f["mcp_bridge"] is True, "mcp_bridge=True")
    ok &= _assert(f["predictor_mode"] == "medium", "predictor_mode='medium'")
    ok &= _assert(f["auto_approve_all"] is False, "auto_approve_all=False")
    return ok


def test_b2_enabled_features_defaults():
    """B2: settings 欠損時の default (False / 'light' のみ)."""
    print("== B2: 欠損時の default 値 ==")
    f = cm.get_enabled_features({})
    ok = True
    ok &= _assert(f["mcp_bridge"] is False, "default False")
    ok &= _assert(f["predictor_mode"] == "light",
                  f"predictor_mode default 'light' (got {f['predictor_mode']})")
    return ok


# ============================================================
# C. summarize_history
# ============================================================

def test_c1_summarize_normal_list():
    """C1: 通常 list → 各 field 計算正しい。"""
    print("== C1: summarize_history 通常 list ==")
    s = cm.summarize_history([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], latest_n=3)
    ok = True
    ok &= _assert(s["len"] == 7, f"len=7 (got {s['len']})")
    ok &= _assert(abs(s["mean"] - 4.0) < 1e-6, f"mean=4.0 (got {s['mean']})")
    ok &= _assert(s["max"] == 7.0, f"max=7.0 (got {s['max']})")
    ok &= _assert(s["latest_n"] == [5.0, 6.0, 7.0],
                  f"latest_n 末尾 3 個 (got {s['latest_n']})")
    return ok


def test_c2_summarize_empty():
    """C2: 空 list → 全 None / 空 latest_n。"""
    print("== C2: 空 list で defensive None ==")
    s = cm.summarize_history([])
    ok = True
    ok &= _assert(s["len"] == 0, "len=0")
    ok &= _assert(s["mean"] is None, "mean=None")
    ok &= _assert(s["max"] is None, "max=None")
    ok &= _assert(s["latest_n"] == [], "latest_n=[]")
    return ok


def test_c3_summarize_latest_n_overflow():
    """C3: latest_n > len なら list 全体を返す (overflow handle)."""
    print("== C3: latest_n > len で全体返す ==")
    s = cm.summarize_history([1.0, 2.0], latest_n=10)
    return _assert(s["latest_n"] == [1.0, 2.0],
                   f"latest_n 全体 (got {s['latest_n']})")


# ============================================================
# D. fog_embedding_spread (A)
# ============================================================

def test_d1_fog_embedding_spread_clustered_low():
    """D1: 同方向 vector 多 → spread ~ 0 (clustered = 小)."""
    print("== D1: 同方向 vector で spread 小 ==")
    # L2 正規化された同方向 vector を 5 つ
    embs = [
        {"embedding": [1.0, 0.0, 0.0]} for _ in range(5)
    ]
    spread = cm.fog_embedding_spread(embs)
    return _assert(
        spread is not None and spread < 0.01,
        f"spread < 0.01 (got {spread})",
    )


def test_d2_fog_embedding_spread_orthogonal_high():
    """D2: 直交 vector → spread > 0 (識別力: clustered 検出と区別)."""
    print("== D2: 直交 vector で spread 大 ==")
    embs = [
        {"embedding": [1.0, 0.0, 0.0]},
        {"embedding": [0.0, 1.0, 0.0]},
        {"embedding": [0.0, 0.0, 1.0]},
    ]
    spread = cm.fog_embedding_spread(embs)
    return _assert(
        spread is not None and spread > 0.1,
        f"spread > 0.1 (識別力: clustered 実装なら ~0 で fail、got {spread})",
    )


def test_d3_fog_embedding_spread_too_few():
    """D3: データ < 2 → None。"""
    print("== D3: データ不足で None ==")
    return _assert(
        cm.fog_embedding_spread([{"embedding": [1.0, 0.0]}]) is None,
        "1 件で None",
    )


# ============================================================
# E. fog_local_density_mean (B)
# ============================================================

def test_e1_fog_local_density_clustered_vs_spread():
    """E1: clustered 群 と spread 群 で値差 (識別力)."""
    print("== E1: clustered vs spread で local_density 差 ==")
    # clustered: ほぼ同じ方向 6 個
    clustered = [
        {"embedding": [1.0 if i == 0 else 0.0 + 0.001 * j
                       for i in range(8)]}
        for j in range(6)
    ]
    # spread: 直交軸を使った 6 個
    import math
    spread = []
    for j in range(6):
        v = [0.0] * 8
        v[j] = 1.0
        spread.append({"embedding": v})
    d_clustered = cm.fog_local_density_mean(clustered, k=3)
    d_spread = cm.fog_local_density_mean(spread, k=3)
    ok = True
    ok &= _assert(d_clustered is not None and d_spread is not None,
                  "両方 None じゃない")
    ok &= _assert(d_clustered < d_spread,
                  f"clustered ({d_clustered}) < spread ({d_spread}) (識別力: 計算誤りで反転)")
    return ok


def test_e2_fog_local_density_too_few():
    """E2: データ < k+1 → None。"""
    print("== E2: データ < k+1 で None ==")
    embs = [{"embedding": [1.0, 0.0]} for _ in range(3)]
    return _assert(
        cm.fog_local_density_mean(embs, k=5) is None,
        "k=5 だが 3 件で None",
    )


# ============================================================
# F. fog_strength_distribution (C)
# ============================================================

def test_f1_strength_distribution_percentile_order():
    """F1: p25 <= p50 <= p75、variance >= 0。"""
    print("== F1: percentile 順序 + variance 非負 ==")
    links = [
        {"strength": 0.1}, {"strength": 0.3}, {"strength": 0.5},
        {"strength": 0.7}, {"strength": 0.9},
    ]
    d = cm.fog_strength_distribution(links)
    ok = True
    ok &= _assert(d is not None, "結果あり")
    ok &= _assert(d["p25"] <= d["p50"] <= d["p75"],
                  f"順序整合 ({d['p25']} <= {d['p50']} <= {d['p75']})")
    ok &= _assert(d["variance"] >= 0.0, f"variance>=0 ({d['variance']})")
    return ok


def test_f2_strength_distribution_too_few():
    """F2: links < 4 → None。"""
    print("== F2: links < 4 で None ==")
    return _assert(
        cm.fog_strength_distribution([{"strength": 0.5}, {"strength": 0.6}]) is None,
        "2 件で None",
    )


# ============================================================
# G. fog_inter_cluster_blur (D)
# ============================================================

def test_g1_inter_cluster_blur_from_phase6():
    """G1: state["phase6_metrics"]["cluster_inter_ratio"] あり → その値。"""
    print("== G1: phase6_metrics 経由で値取得 ==")
    state = {"phase6_metrics": {"cluster_inter_ratio": 0.42}}
    val = cm.fog_inter_cluster_blur(state)
    return _assert(
        val is not None and abs(val - 0.42) < 1e-6,
        f"0.42 (got {val})",
    )


def test_g2_inter_cluster_blur_missing():
    """G2: phase6 なし or 値 None → None (識別力: 別 key 見る実装で fail)."""
    print("== G2: phase6 欠損で None ==")
    ok = True
    ok &= _assert(cm.fog_inter_cluster_blur({}) is None, "state 空 → None")
    ok &= _assert(
        cm.fog_inter_cluster_blur({"phase6_metrics": {}}) is None,
        "phase6 内 key 欠損 → None",
    )
    return ok


# ============================================================
# H. fog_te_flow_anisotropy (E)
# ============================================================

def test_h1_anisotropy_unidirectional_high():
    """H1: 一方通行 (A→B, A→C, A→D) → 高 anisotropy。

    識別力 fixture: A は完全 source (out=3, in=0)、B/C/D は完全 sink
    (out=0, in=1)。期待: 全 node の |inflow - outflow| 平均は 1.5 程度。
    旧実装が from/to を入れ替えても同じ値が出るため discrimination は H2 で。
    """
    print("== H1: 一方通行 links で 高 anisotropy ==")
    links = [
        {"from_id": "A", "to_id": "B", "strength": 1.0},
        {"from_id": "A", "to_id": "C", "strength": 1.0},
        {"from_id": "A", "to_id": "D", "strength": 1.0},
    ]
    a = cm.fog_te_flow_anisotropy(links)
    # node count = 4 (A, B, C, D)
    # |outflow A - inflow A| = |3 - 0| = 3
    # |outflow B/C/D - inflow B/C/D| = |0 - 1| = 1 each
    # mean = (3 + 1 + 1 + 1) / 4 = 1.5
    return _assert(
        a is not None and abs(a - 1.5) < 1e-6,
        f"anisotropy=1.5 (got {a})",
    )


def test_h2_anisotropy_balanced_low():
    """H2: 双方向同強度 (A↔B) → 低 anisotropy = 0 (識別力)。

    旧実装が from / to を区別せず単純合計するなら、A↔B でも anisotropy が
    出てしまう (合計だけ見る)。本 test は in/out を明確に区別する必要を
    discriminating する。
    """
    print("== H2: 双方向同強度で 0 (識別力) ==")
    links = [
        {"from_id": "A", "to_id": "B", "strength": 1.0},
        {"from_id": "B", "to_id": "A", "strength": 1.0},
    ]
    a = cm.fog_te_flow_anisotropy(links)
    # 両 node とも in=1, out=1 → diff=0、mean=0
    return _assert(
        a is not None and abs(a) < 1e-9,
        f"anisotropy=0 (識別力: from/to 区別漏れ実装で != 0、got {a})",
    )


# ============================================================
# I. build_cycle_metrics_event 統合
# ============================================================

def test_i1_event_has_required_fields():
    """I1: 必須 field 全て含む (top-level + nested)."""
    print("== I1: event 必須 field 全部ある ==")
    state = {
        "cycle_id": 5,
        "run_id": "test-run",
        "session_id": "abc12345",
        "energy": 50,
        "entropy": 0.65,
        "pressure": 0.1,
        "last_e1": 80, "last_e2": 70, "last_e3": 60, "last_e4": 50,
        "last_prediction_error": 10,
        "prediction_error_history_e2": [10, 20, 30],
        "prediction_error_history_ec": [0.1, 0.2],
        "jepa_prediction_error_history": [],
        "cumulative_information_gain": {
            "e2_total": 100, "e2_window_mean": 15, "flat_signal": False,
            "flat_streak": 0, "per_tool": {"x": {}, "y": {}},
        },
        "basin_state": {
            "current_basin_id": "c1", "current_basin_dwell": 3,
            "phase_transition_pending": False,
        },
        "predictor_confidence": {
            "x": {"beta": 0.4}, "y": {"beta": 0.6},
        },
        "voluntary_memory_store_count": 2,
        "tool_level": 1,
        "files_written": [], "files_read": [], "action_ledger": [],
        "dispositions": {
            "self": {"curiosity": {"value": 0.8}},
            "ゆう": {"open_sharing": {"value": 0.7}},
        },
        "pending": [
            {"id": "p1", "priority": 0.5, "last_cycle": 3},
            {"id": "p2", "priority": 0.8, "last_cycle": 1},
        ],
        "phase6_metrics": {"cluster_mi": 0.3, "cluster_inter_ratio": 0.4},
    }
    settings = {
        "model": "test-model", "provider": "lmstudio",
        "world_model": {"predictor_mode": "medium"},
    }
    pref = {"reflection_interval": 10}
    event = cm.build_cycle_metrics_event(
        state, settings, pref,
        entries_with_embedding=[],  # test fixture: 空でも event 構造は出る
        links=[],
        code_version="test-version",
    )
    required_top = {
        "event_type", "cycle_id", "time", "run_id", "session_id",
        "config_hash", "code_version", "enabled_features",
        "vital", "evaluation", "histories_summary", "cig", "basin",
        "redundancy", "predictor_confidence_summary", "graph",
        "agency", "relations", "pending", "phase6_metrics",
    }
    ok = True
    ok &= _assert(
        required_top.issubset(set(event.keys())),
        f"必須 top field 全部 (missing: {required_top - set(event.keys())})",
    )
    ok &= _assert(
        event["event_type"] == "cycle_metrics",
        "event_type=cycle_metrics",
    )
    ok &= _assert(event["run_id"] == "test-run", "run_id 反映")
    ok &= _assert(event["code_version"] == "test-version", "code_version 反映")
    ok &= _assert("fog" in event["graph"], "graph.fog ある (識別力)")
    ok &= _assert(
        event["pending"]["count"] == 2 and event["pending"]["high_priority_count"] == 1,
        "pending count + high_priority (priority>0.7) 識別",
    )
    ok &= _assert(
        event["pending"]["oldest_age_cycles"] == 4,  # cycle 5 - last_cycle 1
        f"pending oldest_age=4 (got {event['pending']['oldest_age_cycles']})",
    )
    ok &= _assert(
        event["predictor_confidence_summary"]["mean_beta"] is not None
        and abs(event["predictor_confidence_summary"]["mean_beta"] - 0.5) < 1e-6,
        "mean_beta=0.5",
    )
    ok &= _assert(
        event["histories_summary"]["pred_err_e2"]["len"] == 3,
        "pred_err_e2 history summarize",
    )
    ok &= _assert(
        event["relations"]["self_disposition_keys"] == ["curiosity"],
        "self disposition keys",
    )
    ok &= _assert(
        event["relations"]["attributed_perspective_keys"] == ["ゆう"],
        "attributed perspective keys",
    )
    return ok


# ============================================================
# J. emit_cycle_metrics + index 更新
# ============================================================

def test_j1_emit_appends_jsonl_and_updates_index():
    """J1: 複数回 emit で jsonl 行数増加、index count 一致 (識別力: overwrite で fail)."""
    print("== J1: emit 複数回で append + index 一致 ==")
    tmp = Path(tempfile.mkdtemp(prefix="noetic_metrics_"))
    import core.config as cc
    orig = cc.MEMORY_DIR
    cc.MEMORY_DIR = tmp
    try:
        state = {
            "cycle_id": 1, "run_id": "r1", "session_id": "s1",
            "energy": 50, "entropy": 0.65, "pressure": 0.0,
            "last_e1": None, "last_e2": None, "last_e3": None, "last_e4": None,
            "last_prediction_error": 0,
            "prediction_error_history_e2": [],
            "prediction_error_history_ec": [],
            "jepa_prediction_error_history": [],
            "cumulative_information_gain": {},
            "basin_state": {}, "predictor_confidence": {},
            "voluntary_memory_store_count": 0, "tool_level": 0,
            "files_written": [], "files_read": [], "action_ledger": [],
            "dispositions": {}, "pending": [],
            "phase6_metrics": {},
            # entries_with_embedding / links は空 (test 用 minimal)
        }
        settings = {"model": "x"}
        pref = {}
        # 直接 emit、entries / links 自動取得 (空 profile では空)
        # 注意: load_all_subjective_entries は MEMORY_DIR を見るので tmp に
        # ファイルが無い → 空 list 返る (graceful)
        ev1 = cm.emit_cycle_metrics(state, settings, pref)
        state["cycle_id"] = 2
        ev2 = cm.emit_cycle_metrics(state, settings, pref)
        state["cycle_id"] = 3
        cm.emit_cycle_metrics(state, settings, pref)

        target = tmp / cm.METRICS_FILE_NAME
        ok = True
        ok &= _assert(target.exists(), "metrics_events.jsonl 作成")
        lines = target.read_text(encoding="utf-8").splitlines()
        ok &= _assert(
            len(lines) == 3,
            f"3 行 append (識別力: overwrite なら 1 行で fail、got {len(lines)})",
        )

        index_path = tmp / cm.INDEX_FILE_NAME
        ok &= _assert(index_path.exists(), "index.json 作成")
        idx = json.loads(index_path.read_text(encoding="utf-8"))
        ok &= _assert(
            idx[cm.METRICS_FILE_NAME]["count"] == 3,
            f"index count=3 (got {idx[cm.METRICS_FILE_NAME]['count']})",
        )
        # 内容も最新 cycle_id を反映
        last = json.loads(lines[-1])
        ok &= _assert(last["cycle_id"] == 3, "最新行 cycle_id=3")
        return ok
    finally:
        cc.MEMORY_DIR = orig
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# K. _sort_entries_global_newest_first (Codex review 2026-05-09 P2 #5/#6 fix)
# ============================================================

def test_k1_sort_entries_global_newest_first():
    """K1: time field で global newest first に sort する (P2 #5/#6 回帰防止)。

    fixture: 古い subj entry + 新しい mem entry を「subj→mem の concat 順」で
    渡す (= emit_cycle_metrics の concat 順を再現)。
    旧実装 (sort なし): 入力順維持 = subj (古) が先頭。
    新実装 (sort あり): time newest first = mem (新) が先頭。

    識別力: helper が time field を参照しない実装で fail (順序不変)。
    """
    print("== K1: sort entries global newest first ==")
    entries = [
        {"id": "subj-old", "time": "2026-05-01T00:00:00Z", "embedding": [1, 0]},
        {"id": "mem-new",  "time": "2026-05-09T12:00:00Z", "embedding": [0, 1]},
    ]
    result = cm._sort_entries_global_newest_first(entries)
    ok = True
    ok &= _assert(result[0]["id"] == "mem-new", f"head=mem-new (newest): {result[0]['id']}")
    ok &= _assert(result[1]["id"] == "subj-old", f"next=subj-old (older): {result[1]['id']}")
    return ok


# ============================================================
# Runner
# ============================================================

if __name__ == "__main__":
    tests = [
        test_a1_strip_cosmetic_recursive,
        test_a2_hash_stable_against_cosmetic,
        test_a3_hash_changes_on_substantive_diff,
        test_b1_enabled_features_full_5_keys,
        test_b2_enabled_features_defaults,
        test_c1_summarize_normal_list,
        test_c2_summarize_empty,
        test_c3_summarize_latest_n_overflow,
        test_d1_fog_embedding_spread_clustered_low,
        test_d2_fog_embedding_spread_orthogonal_high,
        test_d3_fog_embedding_spread_too_few,
        test_e1_fog_local_density_clustered_vs_spread,
        test_e2_fog_local_density_too_few,
        test_f1_strength_distribution_percentile_order,
        test_f2_strength_distribution_too_few,
        test_g1_inter_cluster_blur_from_phase6,
        test_g2_inter_cluster_blur_missing,
        test_h1_anisotropy_unidirectional_high,
        test_h2_anisotropy_balanced_low,
        test_i1_event_has_required_fields,
        test_j1_emit_appends_jsonl_and_updates_index,
        test_k1_sort_entries_global_newest_first,
    ]
    results = []
    for t in tests:
        try:
            results.append((t.__name__, t()))
        except Exception as e:
            print(f"  [FAIL] {t.__name__} 例外: {e}")
            import traceback
            traceback.print_exc()
            results.append((t.__name__, False))
        print()
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"=== {passed}/{total} passed ===")
    sys.exit(0 if passed == total else 1)
