"""Goal Shadow Observer テスト (v0.5 Phase 5 Slice 5).

設計: orchestration §5 Slice 5 + ゆう判断 + Codex 5 周目 review。
観測のみ、controller 不変、LLM judge 経由なし。

テスト群 (CLAUDE.md §5 識別力 + §6 docstring/spec/test 同期):
  (a) schema: _new_goal_shadow / make_evidence_ref が schema literal 通り
  (b) observe: embedding cosine 0.85 同定 / label fallback / centroid EMA + L2 normalize
  (c) status 遷移: latent→active / active→cooling / cooling→realized
                   files_delta依存 / cooling→abandoned / 終端非再活性
                   + delta=0 で realized 不発火 (Codex 追加指摘 #1)
                   + raw/subj 同 id の source 区別 (Codex 追加指摘 evidence_refs)
                   + subj_count_prev snapshot で古い entry filter
  (d) metrics 露出: summarize_for_metrics dict 構造 / build_cycle_metrics_event 配線

使い方:
  cd Noetic_seed/profiles/_template
  python tests/test_goal_shadow.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import goal_shadow as gs
import core.memory as cm
import core.state as cs


# ============================================================
# (a) Schema tests
# ============================================================

def test_a_schema_new_goal_shadow():
    """新規 GoalShadow が orchestration §5 schema literal 通り。"""
    s = gs._new_goal_shadow(label="hello", origin="intent", cycle_id=5)
    expected_keys = {
        "id", "label", "status", "origin", "evidence_refs", "embedding",
        "activation", "persistence", "competence", "learning_progress",
        "interestingness", "risk", "last_seen_cycle", "created_cycle",
    }
    actual = set(s.keys())
    assert actual == expected_keys, (
        f"schema mismatch: missing={expected_keys - actual}, "
        f"extra={actual - expected_keys}"
    )
    assert s["status"] == "latent"
    assert s["origin"] == "intent"
    assert s["activation"] == 0.0
    assert s["last_seen_cycle"] == 5
    assert s["created_cycle"] == 5
    assert s["id"].startswith("goal_")
    assert s["embedding"] == []
    print("PASS: test_a_schema_new_goal_shadow")


def test_a_schema_evidence_ref():
    """EvidenceRef が {source, id, cycle_id} 3 field 厳密 (role は YAGNI)。"""
    ref = gs.make_evidence_ref("subjective_entry", "abc123", 7)
    assert set(ref.keys()) == {"source", "id", "cycle_id"}
    assert ref["source"] == "subjective_entry"
    assert ref["id"] == "abc123"
    assert ref["cycle_id"] == 7
    # type coercion (defensive)
    ref2 = gs.make_evidence_ref("pending", 999, "13")
    assert ref2["id"] == "999"
    assert ref2["cycle_id"] == 13
    print("PASS: test_a_schema_evidence_ref")


# ============================================================
# (b) Observe tests
# ============================================================

def _dot_cosine(a, b):
    """正規化済 vector 用 dot product (test fixture cosine_fn)."""
    return sum(x * y for x, y in zip(a, b))


def test_b_observe_cosine_identification():
    """embedding cosine ≥ 0.85 で既存 shadow 合致、未満で新規 (識別力 fixture)."""
    shadow_a = gs._new_goal_shadow("A", "intent", 1, embedding=[1.0, 0.0, 0.0])
    shadows = [shadow_a]

    # cosine ≈ 0.99 ≥ 0.85 → match
    m1 = gs.find_matching_shadow(
        shadows, "different_label", [0.99, 0.14, 0.0], cosine_fn=_dot_cosine
    )
    assert m1 is shadow_a, "cosine ≈ 0.99 should match"

    # cosine ≈ 0.7 < 0.85 → no match (label も違う)
    m2 = gs.find_matching_shadow(
        shadows, "different_label", [0.7, 0.7, 0.14], cosine_fn=_dot_cosine
    )
    assert m2 is None, "cosine ≈ 0.7 < 0.85 should not match"

    # cosine = 0 → no match
    m3 = gs.find_matching_shadow(
        shadows, "different_label", [0.0, 1.0, 0.0], cosine_fn=_dot_cosine
    )
    assert m3 is None
    print("PASS: test_b_observe_cosine_identification")


def test_b_observe_no_label_fallback_when_embedding_active():
    """embedding 経路 active 時に cosine 不足で label fallback しない (Codex 1 周目 P2 fix).

    識別力 fixture (CLAUDE.md §5):
      - shadow_a: label="same_label", embedding=[1, 0, 0]
      - signal: label="same_label" (同じ), embedding=[0, 1, 0] (cosine = 0 < 0.85)
    誤実装 (label fallback に流す) なら shadow_a を返してしまう。
    本 test が fail する誤実装 = 「embedding 経路で best=None でも label 一致 fallback する」
    実装 (= Codex P2 指摘の修正前 fall-through 動作)。

    なぜ fix が必要か:
      docstring contract「label fallback は embedding 不在時のみ」literal 違反、
      semantic 距離の遠い goal が同 truncated label で誤 merge される silent bug。
    """
    shadow_a = gs._new_goal_shadow(
        "same_label", "intent", 1, embedding=[1.0, 0.0, 0.0]
    )
    # cosine = 0 < 0.85 で同 label 持ち
    m = gs.find_matching_shadow(
        [shadow_a], "same_label", [0.0, 1.0, 0.0], cosine_fn=_dot_cosine
    )
    assert m is None, (
        "embedding 経路 active + cosine < 0.85 なら同 label でも match しない "
        "(Codex 1 周目 P2 fix: docstring contract 整合)"
    )

    # 反証 control: embedding 不在時 (cosine_fn=None) は同 label で fallback match する
    m_fallback = gs.find_matching_shadow(
        [shadow_a], "same_label", None, cosine_fn=None
    )
    assert m_fallback is shadow_a, (
        "embedding 不在時 (cosine_fn=None) は label fallback で match する (graceful)"
    )
    print("PASS: test_b_observe_no_label_fallback_when_embedding_active")


def test_b_observe_empty_centroid_upgrade_on_label_match():
    """empty centroid shadow + label 一致で embedding 経路の upgrade 候補に
    なる (Codex 2 周目 P2 fix: vector_ready 後の重複生成 silent bug 防止).

    シナリオ (実 smoke で起こる流れ):
      cycle N (vector_ready=False): label="task_x" の shadow_a 生成、embedding=[]
      cycle N+M (vector_ready=True): label="task_x" の signal が embedding 経路に入る
      → 既存 shadow_a の centroid が空、cosine ループでスキップ
      → 1 周目 P2 fix で label fallback も skip
      → caller が新規 shadow 作って重複発生

    両立解:
      empty centroid + label 一致を embedding 経路の upgrade 候補として追跡。
      cosine match なし時 fallback で返す (label distance ≠ semantic 距離なので、
      embedding 経路 active 時の通常の label fallback は依然 skip)。

    識別力 fixture (CLAUDE.md §5):
      - shadow_a: empty centroid + label="task_x"
      - shadow_b: full centroid + label="task_y" (距離遠い)
      - signal_match: label="task_x" + emb=[1,0,0]   → shadow_a を upgrade
      - signal_diff:  label="task_z" + emb=[1,0,0]   → label 不一致 → None
    """
    shadow_a = gs._new_goal_shadow("task_x", "intent", 0)  # embedding=[] (空)
    shadow_b = gs._new_goal_shadow(
        "task_y", "intent", 0, embedding=[0.0, 0.0, 1.0]
    )  # 距離遠い
    shadows = [shadow_a, shadow_b]

    # signal_match: same label + embedding → empty centroid shadow_a を upgrade 候補で返す
    m1 = gs.find_matching_shadow(
        shadows, "task_x", [1.0, 0.0, 0.0], cosine_fn=_dot_cosine
    )
    assert m1 is shadow_a, (
        "empty centroid + label match → upgrade 候補として返すべき "
        "(Codex 2 周目 P2 fix)"
    )

    # signal_diff: 異 label → None (1 周目 P2 fix も保持: label 距離だけで誤 merge しない)
    m2 = gs.find_matching_shadow(
        shadows, "task_z", [1.0, 0.0, 0.0], cosine_fn=_dot_cosine
    )
    assert m2 is None, (
        "empty centroid + label 不一致 → None (1 周目 P2 fix 保持)"
    )

    # cosine match 優先: shadow_b の cosine が閾値超なら label より優先
    m3 = gs.find_matching_shadow(
        shadows, "task_x", [0.0, 0.0, 0.99], cosine_fn=_dot_cosine
    )
    assert m3 is shadow_b, "cosine match > empty centroid label upgrade"
    print("PASS: test_b_observe_empty_centroid_upgrade_on_label_match")


def test_b_observe_label_fallback():
    """embedding 不在時 label 完全一致で fallback 同定。"""
    shadow_a = gs._new_goal_shadow("hello", "intent", 1)
    shadows = [shadow_a]

    m1 = gs.find_matching_shadow(shadows, "hello", None, cosine_fn=None)
    assert m1 is shadow_a, "label exact match → fallback-match"

    m2 = gs.find_matching_shadow(shadows, "different", None, cosine_fn=None)
    assert m2 is None, "label mismatch → None"

    # cosine_fn あっても evidence_emb=None → label fallback
    m3 = gs.find_matching_shadow(shadows, "hello", None, cosine_fn=_dot_cosine)
    assert m3 is shadow_a
    print("PASS: test_b_observe_label_fallback")


def test_b_observe_centroid_ema_normalize():
    """update_centroid が EMA 後 L2 normalize する (Codex 追加指摘 #2)。

    識別力: normalize なしだと magnitude drift で cosine 0.85 閾値が不安定。
    本 test が fail する誤実装 = 「EMA だけして normalize 抜けてる」実装。
    """
    old = [1.0, 0.0, 0.0]
    new_emb = [0.0, 1.0, 0.0]
    centroid = gs.update_centroid(old, new_emb, alpha=0.7)
    norm = sum(x * x for x in centroid) ** 0.5
    assert abs(norm - 1.0) < 1e-6, (
        f"EMA centroid not L2 normalized: norm={norm} (Codex 追加指摘 #2 違反)"
    )

    # 旧 centroid が空 → 新値を normalize して返す
    c2 = gs.update_centroid([], [3.0, 4.0, 0.0])
    norm2 = sum(x * x for x in c2) ** 0.5
    assert abs(norm2 - 1.0) < 1e-6, "from-empty case still normalized"

    # 異 dim → 旧 centroid そのまま (defensive)
    c3 = gs.update_centroid([1.0, 0.0], [1.0, 0.0, 0.0])
    assert c3 == [1.0, 0.0]

    # ゼロ + 空 → 空
    c4 = gs.update_centroid([], [])
    assert c4 == []
    print("PASS: test_b_observe_centroid_ema_normalize")


# ============================================================
# (c) Status transition tests
# ============================================================

def test_c_status_latent_to_active():
    """activation ≥ ACTIVATION_TO_ACTIVE で latent → active。"""
    s = gs._new_goal_shadow("X", "intent", 1)
    s["activation"] = gs.ACTIVATION_TO_ACTIVE - 0.1
    gs._apply_status_transition(s, cycle_id=2, files_delta=0)
    assert s["status"] == "latent", "below threshold → still latent"

    s["activation"] = gs.ACTIVATION_TO_ACTIVE
    gs._apply_status_transition(s, cycle_id=2, files_delta=0)
    assert s["status"] == "active", "at threshold → active"
    print("PASS: test_c_status_latent_to_active")


def test_c_status_active_to_cooling():
    """idle > COOLING_IDLE_CYCLES で active → cooling (向き = cid - last_seen)。"""
    s = gs._new_goal_shadow("X", "intent", 0)
    s["status"] = "active"
    s["last_seen_cycle"] = 0

    # idle = 5 = COOLING_IDLE_CYCLES → false (>)
    gs._apply_status_transition(s, cycle_id=5, files_delta=0)
    assert s["status"] == "active"

    # idle = 6 > 5 → true
    gs._apply_status_transition(s, cycle_id=6, files_delta=0)
    assert s["status"] == "cooling"
    print("PASS: test_c_status_active_to_cooling")


def test_c_status_cooling_to_realized_requires_delta():
    """cooling → realized は files_delta ≥ 1 + persistence ≥ 1 必須 (識別力ある AND).

    Codex 追加指摘 #1 検証: snapshot を update 前に取らないと delta=0 で
    realized が永久に発火しない silent bug を test で防ぐ。

    識別力 fixture (CLAUDE.md §5 AND filter):
      - persistence=0, delta=1 → 発火しない
      - persistence=1, delta=0 → 発火しない (← Codex 指摘の核心)
      - persistence=1, delta=1 → 発火 (両条件満たす)
      - persistence=0, delta=0 → 発火しない
    各 case で結果が区別される = §5 識別力ある fixture
    """
    def make():
        s = gs._new_goal_shadow("X", "intent", 0)
        s["status"] = "cooling"
        s["last_seen_cycle"] = 5
        return s

    # case 1: persistence=0, delta=1 → 発火しない
    s = make(); s["persistence"] = 0.0
    gs._apply_status_transition(s, cycle_id=6, files_delta=1)
    assert s["status"] == "cooling", "persistence < 1 should not realize"

    # case 2: persistence=1, delta=0 → 発火しない (Codex 指摘の核心)
    s = make(); s["persistence"] = 1.0
    gs._apply_status_transition(s, cycle_id=6, files_delta=0)
    assert s["status"] == "cooling", (
        "delta=0 should not realize (Codex 追加指摘 #1 silent bug 防止)"
    )

    # case 3: persistence=1, delta=1 → realized
    s = make(); s["persistence"] = 1.0
    gs._apply_status_transition(s, cycle_id=6, files_delta=1)
    assert s["status"] == "realized", "both conditions met → realized"

    # case 4: persistence=0, delta=0 → 発火しない (control)
    s = make(); s["persistence"] = 0.0
    gs._apply_status_transition(s, cycle_id=6, files_delta=0)
    assert s["status"] == "cooling"
    print("PASS: test_c_status_cooling_to_realized_requires_delta")


def test_c_status_cooling_to_abandoned():
    """idle > ABANDON_IDLE_CYCLES で cooling → abandoned (delta なし条件)。"""
    s = gs._new_goal_shadow("X", "intent", 0)
    s["status"] = "cooling"
    s["last_seen_cycle"] = 0

    # idle = 15 ちょうど → false (>)
    gs._apply_status_transition(s, cycle_id=15, files_delta=0)
    assert s["status"] == "cooling"

    gs._apply_status_transition(s, cycle_id=16, files_delta=0)
    assert s["status"] == "abandoned"
    print("PASS: test_c_status_cooling_to_abandoned")


def test_c_status_terminal_no_reactivation():
    """realized / abandoned から再活性化しない (本 Slice 範囲の終端契約)。"""
    for terminal in ("realized", "abandoned"):
        s = gs._new_goal_shadow("X", "intent", 0)
        s["status"] = terminal
        s["activation"] = 100.0
        gs._apply_status_transition(s, cycle_id=20, files_delta=10)
        assert s["status"] == terminal, f"{terminal} should not reactivate"
    print("PASS: test_c_status_terminal_no_reactivation")


def test_c_decay_unmentioned():
    """当該 cycle 未言及で activation × DECAY_RATE、言及済 / 終端は除外。"""
    a = gs._new_goal_shadow("A", "intent", 0)
    a["activation"] = 10.0
    a["last_seen_cycle"] = 5  # cur=10 で 5 < 10 → 未言及

    b = gs._new_goal_shadow("B", "intent", 0)
    b["activation"] = 10.0
    b["last_seen_cycle"] = 10  # cur=10 で touch → 除外

    c = gs._new_goal_shadow("C", "intent", 0)
    c["activation"] = 10.0
    c["status"] = "realized"
    c["last_seen_cycle"] = 5  # 終端 → 除外

    gs._decay_unmentioned([a, b, c], cycle_id=10)
    assert abs(a["activation"] - 9.5) < 1e-6, (
        f"unmentioned should decay × {gs.DECAY_RATE}: {a['activation']}"
    )
    assert b["activation"] == 10.0, "mentioned this cycle → no decay"
    assert c["activation"] == 10.0, "terminal → no decay"
    print("PASS: test_c_decay_unmentioned")


def test_c_update_signal_intent():
    """update_goal_shadows: 1 cycle subj entry signal で activation += 1 + evidence."""
    state = {
        "subjective_entries": [
            {"id": "e1", "intent": "explore controller", "expect": "find tool_level"},
        ],
        "pending": [],
        "goal_shadows": [],
    }
    shadows = gs.update_goal_shadows(
        state, cycle_id=1, files_written_delta=0, subj_count_prev=0,
        embed_fn=None, cosine_fn=None,
    )
    assert len(shadows) == 1
    s = shadows[0]
    assert s["activation"] == 1.0
    assert s["origin"] == "intent"
    assert s["last_seen_cycle"] == 1
    assert len(s["evidence_refs"]) == 1
    assert s["evidence_refs"][0]["source"] == gs.SOURCE_SUBJECTIVE_ENTRY
    assert s["evidence_refs"][0]["id"] == "e1"
    assert s["evidence_refs"][0]["cycle_id"] == 1
    print("PASS: test_c_update_signal_intent")


def test_c_update_evidence_source_distinguishes_same_id():
    """raw/subj/pending が同じ id を持っても source field で区別される。

    Codex 追加指摘 (raw/subj id keyspace 共有設計): union list だと曖昧、
    {source, id} schema で追跡性確保 = 本 test の核心。
    """
    state = {
        "subjective_entries": [
            {"id": "shared_x", "intent": "task_subj", "expect": ""},
        ],
        "pending": [
            {
                "id": "shared_x", "content_intent": "task_pending",
                "content_observable": "", "last_cycle": 1,
            },
        ],
        "goal_shadows": [],
    }
    shadows = gs.update_goal_shadows(
        state, cycle_id=1, files_written_delta=0, subj_count_prev=0,
    )
    # 別 label なので別 shadow → 2 件
    assert len(shadows) == 2
    sources = {s["evidence_refs"][0]["source"] for s in shadows}
    assert sources == {gs.SOURCE_SUBJECTIVE_ENTRY, gs.SOURCE_PENDING}, (
        f"sources={sources}"
    )
    # 同 id でも source 違いで両方記録される (id 衝突しても情報残る)
    refs_by_source = {
        s["evidence_refs"][0]["source"]: s["evidence_refs"][0]["id"]
        for s in shadows
    }
    assert refs_by_source[gs.SOURCE_SUBJECTIVE_ENTRY] == "shared_x"
    assert refs_by_source[gs.SOURCE_PENDING] == "shared_x"
    print("PASS: test_c_update_evidence_source_distinguishes_same_id")


def test_c_update_files_delta_persistence():
    """files_written_delta が非終端 shadow の persistence に加算、終端は除外。"""
    state = {
        "subjective_entries": [],
        "pending": [],
        "goal_shadows": [
            gs._new_goal_shadow("A", "intent", 0),
            gs._new_goal_shadow("B", "intent", 0),
        ],
    }
    state["goal_shadows"][1]["status"] = "realized"

    shadows = gs.update_goal_shadows(
        state, cycle_id=2, files_written_delta=3, subj_count_prev=0,
    )
    assert shadows[0]["persistence"] == 3.0, "non-terminal shadow gets delta"
    assert shadows[1]["persistence"] == 0.0, "realized shadow excluded"
    print("PASS: test_c_update_files_delta_persistence")


def test_c_subj_count_prev_filters_old_entries():
    """subj_count_prev で古い subjective_entries は signal 化されない (snapshot delta)."""
    state = {
        "subjective_entries": [
            {"id": "old1", "intent": "old task 1", "expect": ""},
            {"id": "old2", "intent": "old task 2", "expect": ""},
            {"id": "new1", "intent": "new task", "expect": ""},
        ],
        "pending": [],
        "goal_shadows": [],
    }
    # subj_count_prev=2 → 末尾 1 件 (new1) のみ signal 化
    shadows = gs.update_goal_shadows(
        state, cycle_id=5, files_written_delta=0, subj_count_prev=2,
    )
    assert len(shadows) == 1
    assert shadows[0]["label"] == "new task"
    print("PASS: test_c_subj_count_prev_filters_old_entries")


# ============================================================
# (d) Metrics output tests
# ============================================================

def test_d_summarize_for_metrics_empty():
    """空 list → schema literal 通り、count=0、status_distribution={}."""
    summary = gs.summarize_for_metrics([])
    assert summary == {
        "count": 0,
        "status_distribution": {},
        "total_evidence_refs": 0,
        "top_active": [],
    }
    print("PASS: test_d_summarize_for_metrics_empty")


def test_d_summarize_for_metrics_mixed():
    """複数 shadow で count / status_distribution / total_refs / top_active が整合。

    識別力 fixture (CLAUDE.md §5): 異 status × 異 activation × 異 evidence count
    で summary の各 field が独立に検証される。誤実装 (例: top_active を昇順、
    status_dist を全部 latent カウント等) では fail する。
    """
    s1 = gs._new_goal_shadow("a", "intent", 1)
    s1["status"] = "active"
    s1["activation"] = 5.0
    s1["evidence_refs"] = [
        gs.make_evidence_ref("subjective_entry", "x1", 1),
        gs.make_evidence_ref("subjective_entry", "x2", 2),
    ]
    s2 = gs._new_goal_shadow("b", "intent", 1)
    s2["status"] = "latent"
    s2["activation"] = 1.0
    s2["evidence_refs"] = [gs.make_evidence_ref("pending", "p1", 3)]
    s3 = gs._new_goal_shadow("c", "intent", 1)
    s3["status"] = "realized"
    s3["activation"] = 100.0

    summary = gs.summarize_for_metrics([s1, s2, s3], top_n=2)
    assert summary["count"] == 3
    assert summary["status_distribution"] == {
        "active": 1, "latent": 1, "realized": 1,
    }
    assert summary["total_evidence_refs"] == 3
    # top_active = activation 降順 + top_n=2 → c (100), s1 (5)
    assert len(summary["top_active"]) == 2
    assert summary["top_active"][0]["label"] == "c"
    assert summary["top_active"][1]["label"] == "a"
    assert summary["top_active"][0]["evidence_count"] == 0
    assert summary["top_active"][1]["evidence_count"] == 2
    print("PASS: test_d_summarize_for_metrics_mixed")


def _setup_tmp_state_dirs():
    """tmp dir を作って core.memory + core.state の MEMORY_DIR / STATE_FILE を差し替え。

    test_view_rebuild.py と同 pattern。load_state を独立稼働可能にする。
    """
    tmp = Path(tempfile.mkdtemp(prefix="noetic_slice5_state_"))
    originals = (cm.MEMORY_DIR, cs.MEMORY_DIR, cs.STATE_FILE)
    cm.MEMORY_DIR = tmp
    cs.MEMORY_DIR = tmp
    cs.STATE_FILE = tmp / "state.json"
    return tmp, originals


def _restore_state_dirs(originals):
    cm.MEMORY_DIR, cs.MEMORY_DIR, cs.STATE_FILE = originals


def test_d_load_state_subj_snapshot_after_rebuild():
    """Codex 3 周目 P2 fix integration test: state.json なし + jsonl に履歴あり →
    load_state 後 _subjective_entries_count_prev は jsonl 件数に init される。

    シナリオ (実 smoke で起こる復旧パス):
      - state.json 削除されてる (or 初回起動)
      - subjective_entries.jsonl に過去履歴 N 件残ってる
      - load_state で _rebuild_views_from_jsonl が subjective_entries を N 件で rebuild
      - rebuild 前 setdefault (修正前バグ) だと _subjective_entries_count_prev=0
      - 次 cycle で update_goal_shadows が subj_count_prev=0 → jsonl 全 N 件を「新規」と誤判定
      - 全履歴に対して goal_shadow 量産、metrics 歪む

    識別力 fixture (CLAUDE.md §5):
      - state.json 不在
      - subjective_entries.jsonl に 3 件
    誤実装 (rebuild 前 setdefault) なら _subjective_entries_count_prev=0 で fail。
    本 test は rebuild 後 setdefault が正しく動いてることを確認する。
    """
    tmp, originals = _setup_tmp_state_dirs()
    try:
        subj_jsonl = tmp / "subjective_entries.jsonl"
        records = [
            {"id": "e1", "intent": "task1", "expect": ""},
            {"id": "e2", "intent": "task2", "expect": ""},
            {"id": "e3", "intent": "task3", "expect": ""},
        ]
        with open(subj_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        state = cs.load_state()
        # rebuild が効いて subjective_entries が jsonl 全件
        assert len(state["subjective_entries"]) == 3, (
            f"rebuild 後 subjective_entries={len(state['subjective_entries'])}, "
            f"expected 3"
        )
        # snapshot は rebuild 後の件数で init される (Codex 3 周目 P2 fix の核心)
        assert state["_subjective_entries_count_prev"] == 3, (
            f"_subjective_entries_count_prev={state['_subjective_entries_count_prev']}, "
            f"expected 3 (Codex 3 周目 P2 fix: rebuild 後に snapshot init すべき)"
        )
        print("PASS: test_d_load_state_subj_snapshot_after_rebuild")
    finally:
        _restore_state_dirs(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_d_load_state_subj_snapshot_preserves_existing():
    """既存 state.json で _subjective_entries_count_prev に前 cycle 末値があれば
    setdefault でスキップされ、rebuild 後でも上書きされない (前 cycle の正しい snapshot 維持).

    識別力 fixture (CLAUDE.md §5):
      state.json: _subjective_entries_count_prev=2, _files_written_count_prev=5
      subjective_entries.jsonl: 3 件
    誤実装 (常に上書き) なら 3 で上書きされて fail。
    本 test は setdefault が既値を尊重してることを確認 = 前 cycle 末の正しい snapshot
    が rebuild で破壊されないことを担保。
    """
    tmp, originals = _setup_tmp_state_dirs()
    try:
        # state.json で前 cycle 末値を持つ
        state_data = {
            "raw_events": [],
            "subjective_entries": [],
            "_subjective_entries_count_prev": 2,
            "_files_written_count_prev": 5,
            "goal_shadows": [],
        }
        cs.STATE_FILE.write_text(
            json.dumps(state_data, ensure_ascii=False), encoding="utf-8"
        )
        # jsonl に 3 件 (rebuild すると state["subjective_entries"] が 3 になる)
        subj_jsonl = tmp / "subjective_entries.jsonl"
        records = [
            {"id": "e1", "intent": "task1", "expect": ""},
            {"id": "e2", "intent": "task2", "expect": ""},
            {"id": "e3", "intent": "task3", "expect": ""},
        ]
        with open(subj_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        state = cs.load_state()
        # rebuild 効いてる
        assert len(state["subjective_entries"]) == 3
        # 既値 2 が尊重 (rebuild 後の 3 で上書きされない)
        assert state["_subjective_entries_count_prev"] == 2, (
            f"既存値継承されるべき: got {state['_subjective_entries_count_prev']}, "
            f"expected 2 (setdefault が前 cycle 末値を尊重)"
        )
        assert state["_files_written_count_prev"] == 5, (
            f"files_written 既値も維持: got {state['_files_written_count_prev']}"
        )
        print("PASS: test_d_load_state_subj_snapshot_preserves_existing")
    finally:
        _restore_state_dirs(originals)
        shutil.rmtree(tmp, ignore_errors=True)


def test_d_metrics_event_includes_goal_shadows():
    """build_cycle_metrics_event の戻り dict に goal_shadows field がある。

    Slice 5 の metrics 露出契約 (BP-3 判定基準: state["goal_shadows"] が成長 /
    status 遷移 / evidence binding) を build 経路全体で検証。
    """
    from core.metrics import build_cycle_metrics_event
    state = {
        "cycle_id": 1,
        "subjective_entries": [],
        "raw_events": [],
        "pending": [],
        "files_written": [],
        "goal_shadows": [
            gs._new_goal_shadow("test", "intent", 1),
        ],
    }
    event = build_cycle_metrics_event(
        state, settings={}, pref={},
        entries_with_embedding=[], links=[], code_version="test",
    )
    assert "goal_shadows" in event, "Slice 5 metrics 配線 (build_cycle_metrics_event)"
    summary = event["goal_shadows"]
    assert "count" in summary
    assert "status_distribution" in summary
    assert "total_evidence_refs" in summary
    assert "top_active" in summary
    assert summary["count"] == 1
    print("PASS: test_d_metrics_event_includes_goal_shadows")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_a_schema_new_goal_shadow()
    test_a_schema_evidence_ref()
    test_b_observe_cosine_identification()
    test_b_observe_no_label_fallback_when_embedding_active()
    test_b_observe_empty_centroid_upgrade_on_label_match()
    test_b_observe_label_fallback()
    test_b_observe_centroid_ema_normalize()
    test_c_status_latent_to_active()
    test_c_status_active_to_cooling()
    test_c_status_cooling_to_realized_requires_delta()
    test_c_status_cooling_to_abandoned()
    test_c_status_terminal_no_reactivation()
    test_c_decay_unmentioned()
    test_c_update_signal_intent()
    test_c_update_evidence_source_distinguishes_same_id()
    test_c_update_files_delta_persistence()
    test_c_subj_count_prev_filters_old_entries()
    test_d_summarize_for_metrics_empty()
    test_d_summarize_for_metrics_mixed()
    test_d_load_state_subj_snapshot_after_rebuild()
    test_d_load_state_subj_snapshot_preserves_existing()
    test_d_metrics_event_includes_goal_shadows()
    print("\nAll goal_shadow tests passed (22 件).")
