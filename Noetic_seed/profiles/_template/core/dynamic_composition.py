"""段階13 Phase 4: pressure 軸 + graph 軸の動的合成 (PLAN §3-1〜§3-2)。

graph_maturity ∈ [0, 1] を 5 入力 sigmoid 合成で算出する。全入力は段階11-D
Phase 1-7 + 段階13 Phase 2 で実装済 metric から導出 (新規計算なし、glue 層)。

PLAN §3-2 literal:
    density      = sigmoid(link_count / N_normalize)
    structure    = sigmoid(small_world_sigma - baseline)
    anomaly      = sigmoid(recent_anomaly_rate)
    frontier     = sigmoid(frontier_count / N_normalize)
    avg_strength = sigmoid(mean_link_strength)

    graph_maturity = weighted_avg([density, structure, anomaly, frontier, avg_strength])

PLAN §3-4 (d) "graph 周縁の未踏領域" の frontier 定義 (ゆう gut 確定 2026-05-03):
    leaf node (無向視 degree=1) を frontier として数える。
    isolated (degree=0) は除外 (= "そのへんに浮いてるもの" は周縁ではない)。
    "相互で link してる先がない" のうち、graph 末端 (1 隣接のみ) が真の frontier。

graph 軸 fire 候補 4 種は commit 2 で追加 (PLAN §3-4 (a)-(d))。
fire_candidates list 構造化 + main.py 配線は commit 3 (PLAN §3-1 + §5-5 同時発火階層)。
"""

import math


# ------------------------------------------------------------
# 定数 (PLAN §3-2 sigmoid 入力の正規化点 / baseline、smoke で tune)
# ------------------------------------------------------------

# link_count 正規化点。link 100 本で sigmoid(1.0)≈0.731 を transition pivot とする
# (sigmoid 中心点 0.5 は count=0 に対応、半飽和ではなく正規化スケール定義)。
N_NORMALIZE_LINK: float = 100.0

# frontier_count 正規化点。leaf 20 個で sigmoid(1.0)≈0.731 が transition pivot。
N_NORMALIZE_FRONTIER: float = 20.0

# small-world sigma > 1 で small-world (Watts-Strogatz 1998 reference)。
SMALL_WORLD_BASELINE: float = 1.0

# anomaly_rate 直近 window。jepa_prediction_error_history (cap=100, Phase 2 確立) 対応。
ANOMALY_HISTORY_WINDOW: int = 20

# graph_maturity 5 入力の重み (uniform、PLAN §3-2 literal の weighted_avg)。
# 合計 1.0 = maturity の上限保証、smoke 後 tune 候補。
WEIGHT_DENSITY: float = 0.2
WEIGHT_STRUCTURE: float = 0.2
WEIGHT_ANOMALY: float = 0.2
WEIGHT_FRONTIER: float = 0.2
WEIGHT_AVG_STRENGTH: float = 0.2


# ------------------------------------------------------------
# 純粋関数: sigmoid
# ------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """numpy 不要の純粋 sigmoid。overflow 安全 clamp 付き。"""
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ------------------------------------------------------------
# 5 sub-score 関数 (PLAN §3-2 literal、各 docstring に式 + source field 明記)
# ------------------------------------------------------------

def _density_score(state: dict) -> float:
    """density = sigmoid(link_count / N_NORMALIZE_LINK)。

    PLAN §3-2 literal: link 量の純粋指標。link_type 区別なしで raw 件数を採用
    (構造的整合性は structure_score で別軸測定)。

    想定誤実装 D4 (link_type='none' sanitize 漏れ) は **density には不適用**:
    density は意図的に 'none' 含む raw count 仕様 (PLAN §3-2 "link 量の純粋指標"
    literal 整合)。D4 sanitize 想定は avg_strength_score / _frontier_count 側のみ。

    Graceful skip (PLAN §3-3 「空 graph で graph_maturity ≈ 0.0」literal 整合):
    link 0 件 → return 0.0 (寄与ゼロ、sigmoid 中央値 0.5 ではない)。
    「データなし」と「中庸の構造度」を概念分離し、空 graph で graph 軸寄与を
    完全ゼロにすることで pressure 軸支配 (§3-3 推移表) を数学的に純粋表現。

    既存 helper: core.memory_links.list_links。
    state 引数は signature 統一のため受取のみ (現状未参照)。
    """
    from core.memory_links import list_links, LINK_SCAN_LIMIT
    links = list_links(limit=LINK_SCAN_LIMIT)
    if not links:
        return 0.0
    return _sigmoid(len(links) / N_NORMALIZE_LINK)


def _structure_score(state: dict) -> float:
    """structure = sigmoid(small_world_sigma - SMALL_WORLD_BASELINE)。

    既存 helper: core.tag_emergence_monitor._compute_small_world_metrics
                 (段階11-D Phase 6 確立、Watts-Strogatz 1998 reference)。
    sigma > 1 で small-world、baseline=1.0 を引いて sigmoid 通すと
    small-world graph で >0.5、random graph で <0.5。

    Sentinel 解釈 (既存 _compute_small_world_metrics literal 整合):
    helper は以下の条件で sigma=0.0 を return する (= 算出不能 sentinel、
    「random graph 以下の構造度」ではない):
      (i)  memory_count<2 or not links  (line 435-441 早期 return)
      (ii) valid_link_count==0 (sanitize-out 後の adj 空、line 442-454)
      (iii) C_rand or L_rand or L が 0 (line 481-484、計算不能)

    structure_score 側で sigma=0.0 を全部 sentinel 扱い → 0.0 graceful skip
    (5 sub-score 統一の「データなし → 寄与ゼロ」pattern、PLAN §3-3 整合)。
    早期 (i) check は不要呼出回避の最適化、(ii)(iii) は `sigma == 0.0` 共通検出。

    state 引数は signature 統一のため受取のみ (現状未参照)。
    """
    from core.memory_links import list_links, LINK_SCAN_LIMIT
    from core.tag_emergence_monitor import _compute_small_world_metrics
    from core.memory import load_all_memories

    links = list_links(limit=LINK_SCAN_LIMIT)
    memories = load_all_memories()
    # 早期 graceful skip: helper の (i) 条件と同等、不要呼出回避
    if not links or len(memories) < 2:
        return 0.0
    metrics = _compute_small_world_metrics(links, len(memories))
    sigma = float(metrics.get("small_world_sigma", 0.0))
    # sigma==0.0 sentinel: helper の (ii)(iii) sanitize-out 全部 / 計算不能経路
    if sigma == 0.0:
        return 0.0
    return _sigmoid(sigma - SMALL_WORLD_BASELINE)


def _anomaly_score(state: dict) -> float:
    """anomaly = sigmoid(recent_anomaly_rate)。

    PLAN §3-2 literal: 予測誤差の存在感。Phase 2 で確立した JEPA cosine 距離
    (1024D embedding 空間) = Layer C 直接由来 = graph 軸 anomaly の本筋。
    Phase 11-D の prediction_error_history_e2/ec も候補だが、jepa は
    Layer B + Layer C 統合の純粋空間距離指標 (PLAN §3 Layer C literal 整合)。

    入力: state["jepa_prediction_error_history"] 直近 ANOMALY_HISTORY_WINDOW 平均。

    Graceful skip (PLAN §3-3 整合): history 空 (Phase 2 起動前 / cycle 初期) で
    return 0.0 (寄与ゼロ)。「予測誤差データなし」と「中庸の anomaly 度」を
    概念分離。
    """
    history = state.get("jepa_prediction_error_history", []) if isinstance(state, dict) else []
    if not history:
        return 0.0
    window = history[-ANOMALY_HISTORY_WINDOW:]
    rate = sum(window) / len(window)
    return _sigmoid(rate)


def _frontier_count(state: dict) -> int:
    """graph 周縁の未踏領域 = leaf node 数 (無向視で degree=1)。

    PLAN §3-4 (d) "graph 周縁の未踏領域" 解釈 (ゆう gut 確定 2026-05-03):
      - isolated (degree=0) は除外: "そのへんに浮いてるもの" は周縁ではない。
      - leaf (degree=1): 1 本だけ link、相互参照を持たない、graph 末端で
        まだ繋がりが伸びていける余地 = "未踏" の真の意味。
      - 無向視構築は **adjacency-set pattern** (set でユニーク化)、
        tag_emergence_monitor._compute_small_world_metrics:442-454 と完全同形。
        重複 link (A→B 2 回) や 相互 link (A→B + B→A) でも、A の隣接 set は
        {B} で degree=1 = leaf 扱い (Noetic 慣習継承、新規マジックナンバーゼロ)。

    link_type="none" / from==to / from_id 不在 / to_id 不在 は除外
    (small_world と同 sanitize 規則)。
    state 引数は signature 統一のため受取のみ (現状未参照)。
    """
    from core.memory_links import list_links, LINK_SCAN_LIMIT

    links = list_links(limit=LINK_SCAN_LIMIT)
    adj: dict = {}  # node_id → set(neighbor_ids) で重複/相互 link をユニーク化
    for link in links:
        if link.get("link_type", "none") == "none":
            continue
        from_id = link.get("from_id")
        to_id = link.get("to_id")
        if not from_id or not to_id or from_id == to_id:
            continue
        adj.setdefault(from_id, set()).add(to_id)
        adj.setdefault(to_id, set()).add(from_id)

    return sum(1 for neighbors in adj.values() if len(neighbors) == 1)


def _frontier_score(state: dict) -> float:
    """frontier = sigmoid(frontier_count / N_NORMALIZE_FRONTIER)。

    Graceful skip (PLAN §3-3 整合): frontier_count == 0 (leaf なし) なら return 0.0。
    leaf 0 は「探索余地なし」と「データなし」の両解釈ありうるが、
    PLAN §3-3「空 graph で graph_maturity ≈ 0.0」整合で 0.0 graceful skip。
    """
    count = _frontier_count(state)
    if count == 0:
        return 0.0
    return _sigmoid(count / N_NORMALIZE_FRONTIER)


def _avg_strength_score(state: dict) -> float:
    """avg_strength = sigmoid(mean_link_strength)。

    PLAN §3-2 literal: 構造の堅さ。各 link の strength (段階11-B Phase 2 確立、
    初期値=confidence、Phase 3 Bayesian で更新) 平均を sigmoid に通す。
    link_type="none" は除外 (small_world / frontier_count と整合)。

    既存 helper: core.memory_links.list_links + _link_strength
                 (Phase 2 以前 link は confidence fallback、backward compat)。

    Graceful skip (PLAN §3-3 整合): valid link 0 件なら return 0.0 (寄与ゼロ)。
    「strength データなし」と「中庸の strength」を概念分離。
    state 引数は signature 統一のため受取のみ (現状未参照)。
    """
    from core.memory_links import list_links, _link_strength, LINK_SCAN_LIMIT

    links = list_links(limit=LINK_SCAN_LIMIT)
    valid_strengths = [
        _link_strength(link) for link in links
        if link.get("link_type", "none") != "none"
    ]
    if not valid_strengths:
        return 0.0
    mean = sum(valid_strengths) / len(valid_strengths)
    return _sigmoid(mean)


# ------------------------------------------------------------
# Public API: graph_maturity
# ------------------------------------------------------------

def compute_graph_maturity(state: dict) -> float:
    """段階13 Phase 4: graph 育成度 [0, 1] を 5 入力 sigmoid 合成で算出 (PLAN §3-2)。

    全入力は段階11-D Phase 1-7 + 段階13 Phase 2 で実装済 metric から導出。
    新規計算なし、glue 層のみ (PLAN §18-A4 (iv') 流用方針整合)。

    Graceful skip 統一 (PLAN §3-3 整合、ゆう gut 確定 2026-05-03):
    全 5 sub-score で「データなし → 0.0 寄与ゼロ」pattern 統一。空 graph
    (cycle 0) で graph_maturity = 0.0 = w_graph = 0.0 = graph 軸寄与ゼロ =
    pressure 軸完全支配 (PLAN §3-3「空で 0.0、pressure 軸のみ動く」literal 整合)。
    sigmoid(0) = 0.5 を「データなし sentinel」として誤読しない設計。

    Args:
        state: state dict (jepa_prediction_error_history 等を参照)

    Returns:
        graph_maturity ∈ [0.0, 1.0]。
        weight 合計 1.0 + 各 sub-score [0, 1] で範囲保証。

    PLAN §3-3 概念推移 (実数値推移、graceful skip 統一後):
        空 graph (cycle 0)        ≈ 0.0   (全 sub-score=0.0、pressure 軸支配)
        育成中 (cycle 5-20)       ≈ 0.2-0.5 (sub-score 育つ、両軸混合)
        豊穣 graph (cycle 100+)   ≈ 0.9   (graph 軸支配)
    """
    density = _density_score(state)
    structure = _structure_score(state)
    anomaly = _anomaly_score(state)
    frontier = _frontier_score(state)
    avg_strength = _avg_strength_score(state)

    return (
        WEIGHT_DENSITY * density
        + WEIGHT_STRUCTURE * structure
        + WEIGHT_ANOMALY * anomaly
        + WEIGHT_FRONTIER * frontier
        + WEIGHT_AVG_STRENGTH * avg_strength
    )


# ============================================================
# graph 軸 fire 候補 4 種 (PLAN §3-4 (a)-(d)、commit 3)
#
# 各 candidate format: {"name": str, "score": float, "kind": "graph"}
# 統合 entry: compute_graph_axis_candidates(state, w_graph) -> list[dict]
#
# 新規マジックナンバーゼロ (ゆう gut 確定 2026-05-03):
#   既存定数 (ANOMALY_HISTORY_WINDOW / LINK_SCAN_LIMIT) 流用 + 相対比正規化
#   (degree / avg_degree、cluster_inter_ratio 等) で固定数値追加なし。
# ============================================================


def _anomaly_candidates(state: dict) -> list:
    """PLAN §3-4 (a) anomaly: 予測誤差大の状態シグナル。

    Phase 2 jepa_prediction_error_history (scalar list) は entry id 紐付けが
    ないため、graph 軸全体の anomaly 度を **1 candidate** として返す
    (link/entry 単位細粒度は Phase 4 後半 / Phase 7 smoke 後で改善余地)。

    score = sigmoid(直近 ANOMALY_HISTORY_WINDOW の error mean)。
    history 空なら graceful skip (空 list return、PLAN §3-3 整合)。
    """
    history = state.get("jepa_prediction_error_history", []) if isinstance(state, dict) else []
    if not history:
        return []
    window = history[-ANOMALY_HISTORY_WINDOW:]
    rate = sum(window) / len(window)
    return [{
        "name": "anomaly:recent_history",
        "score": _sigmoid(rate),
        "kind": "graph",
    }]


def _structural_tension_candidates(state: dict) -> list:
    """PLAN §3-4 (b) structural tension: cluster 間の link 不均衡。

    既存 state["phase6_metrics"] (段階11-D Phase 6、reflect 時更新) の
    cluster_inter_ratio を流用、低い値 (cluster 間 link 少) = 緊張高い。
    score = sigmoid(1.0 - inter_ratio) で緊張度表現。

    phase6_metrics 未確立 (reflect 未走行) or inter_ratio 0 (= cluster なし
    or 全孤立) では graceful skip (空 list、PLAN §3-3 整合)。
    """
    if not isinstance(state, dict):
        return []
    phase6 = state.get("phase6_metrics", {})
    if not phase6:
        return []
    inter_ratio = float(phase6.get("cluster_inter_ratio", 0.0))
    # inter_ratio == 0 = cluster なし or 全孤立、structural tension 概念成立せず
    if inter_ratio <= 0.0:
        return []
    # tension = 1 - inter_ratio (cluster 間連結が薄い = 緊張)
    tension = 1.0 - inter_ratio
    return [{
        "name": "structural_tension:cluster_inter_ratio",
        "score": _sigmoid(tension),
        "kind": "graph",
    }]


def _relational_salience_candidates(state: dict) -> list:
    """PLAN §3-4 (c) relational salience: 特定 entity 中心の link 集中。

    無向視 degree / avg_degree の **相対比** で正規化、avg を超える node
    のみ candidate (hub 性、ego view ベース)。新規マジックナンバーなし
    (avg_degree は graph 動的派生)。

    Graceful skip: 有効 link 0 / avg_degree 0 → 空 list (PLAN §3-3 整合)。
    """
    from core.memory_links import list_links, LINK_SCAN_LIMIT

    links = list_links(limit=LINK_SCAN_LIMIT)
    adj: dict = {}
    for link in links:
        if link.get("link_type", "none") == "none":
            continue
        from_id = link.get("from_id")
        to_id = link.get("to_id")
        if not from_id or not to_id or from_id == to_id:
            continue
        adj.setdefault(from_id, set()).add(to_id)
        adj.setdefault(to_id, set()).add(from_id)

    if not adj:
        return []
    total_deg = sum(len(n) for n in adj.values())
    node_count = len(adj)
    if node_count == 0 or total_deg == 0:
        return []
    avg_deg = total_deg / node_count
    if avg_deg <= 0.0:
        return []

    candidates = []
    for node, neighbors in adj.items():
        deg = len(neighbors)
        relative = deg / avg_deg
        # avg を超える hub 性のある node のみ candidate
        if relative <= 1.0:
            continue
        # sigmoid(relative - 1.0): avg ちょうどで 0.5、avg の 2 倍で sigmoid(1)≈0.731
        candidates.append({
            "name": f"relational_salience:{node}",
            "score": _sigmoid(relative - 1.0),
            "kind": "graph",
        })
    return candidates


def _predictive_frontier_candidates(state: dict) -> list:
    """PLAN §3-4 (d) predictive frontier: graph 周縁の未踏領域 (leaf)。

    Phase 4 commit 1 で確立した leaf 定義 (無向視 degree=1) 流用。
    各 leaf node に candidate 1 件、score = sigmoid(その link の strength)。
    leaf なし (空 graph or 三角形以上の密 graph) → 空 list (PLAN §3-3 整合)。
    """
    from core.memory_links import list_links, _link_strength, LINK_SCAN_LIMIT

    links = list_links(limit=LINK_SCAN_LIMIT)
    adj: dict = {}
    # link strength を pair 単位で集約 (重複時は max、_frontier_count と整合)
    pair_strength: dict = {}
    for link in links:
        if link.get("link_type", "none") == "none":
            continue
        from_id = link.get("from_id")
        to_id = link.get("to_id")
        if not from_id or not to_id or from_id == to_id:
            continue
        adj.setdefault(from_id, set()).add(to_id)
        adj.setdefault(to_id, set()).add(from_id)
        s = _link_strength(link)
        for pair in [(from_id, to_id), (to_id, from_id)]:
            pair_strength[pair] = max(pair_strength.get(pair, 0.0), s)

    candidates = []
    for node, neighbors in adj.items():
        if len(neighbors) != 1:
            continue
        nbr = next(iter(neighbors))
        strength = pair_strength.get((node, nbr), 0.0)
        candidates.append({
            "name": f"predictive_frontier:{node}",
            "score": _sigmoid(strength),
            "kind": "graph",
        })
    return candidates


def compute_graph_axis_candidates(state: dict, w_graph: float) -> list:
    """段階13 Phase 4 commit 3: graph 軸 fire 候補統合 entry (PLAN §3-1 + §3-4)。

    PLAN §3-1 literal: fire_candidates = pressure_axis(weight=w_pressure) +
    graph_axis(weight=w_graph)。本関数は graph 軸 list を返し、各 candidate
    の score に w_graph を multiplicative 適用済 (graph 育成度に応じた重み)。

    w_graph <= 0 (空 graph、graph_maturity = 0.0) では graph 軸寄与ゼロ =
    空 list return (PLAN §3-3「空で 0.0、pressure 軸のみ動く」literal 整合)。

    Args:
        state: state dict
        w_graph: ∈ [0.0, 1.0]、compute_graph_maturity の結果 (重み)

    Returns:
        list[dict]: 4 candidate 関数の結果連結 + score *= w_graph 適用済。
        各 entry format: {"name": str, "score": float, "kind": "graph"}。
        commit 4 (main.py 配線) で render 時に件数絞る、本関数では無制限 list。
    """
    if w_graph <= 0.0:
        return []

    candidates: list = []
    candidates.extend(_anomaly_candidates(state))
    candidates.extend(_structural_tension_candidates(state))
    candidates.extend(_relational_salience_candidates(state))
    candidates.extend(_predictive_frontier_candidates(state))

    # w_graph multiplicative 重み付け
    for c in candidates:
        c["score"] = c["score"] * w_graph

    return candidates
