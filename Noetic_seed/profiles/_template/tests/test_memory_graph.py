"""memory_graph tool テスト (段階11-D Phase 0 Step 0.2: ego view MVP).

成功条件:
  - _self_to_virtual_entries: state.self の各 key を facet として変換
  - _self_to_virtual_entries: 空値はスキップ、空 self で空 list
  - _build_self_node: id="self" 不変、kind="self"、facets/metrics 構造
  - _compute_self_to_memory_edges: empty case (memory なし) で空
  - _compute_memory_edges: empty case (link なし) で空
  - _memory_graph: view="ego" で必須 key を持つ JSON 出力
  - _memory_graph: view 不正値で error response
  - _memory_graph: depth args の解釈

案 ③ (純粋仮想化、永続化なし) の特性:
  - self_node の id は cycle に紐付かず "self" 不変
  - virtual_entries は描画毎に on-the-fly 生成、永続化されない

使い方:
  cd Noetic_seed/profiles/_template
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_memory_graph.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.memory_graph_tool import (
    _self_to_virtual_entries,
    _build_self_node,
    _compute_self_to_memory_edges,
    _compute_memory_edges,
    _compute_trace,
    _memory_graph,
)


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


# ============================================================
# _self_to_virtual_entries
# ============================================================

def test_self_to_virtual_entries_basic():
    print("== _self_to_virtual_entries: 各 key を facet として変換 ==")
    state = {
        "self": {
            "name": "iku",
            "identity": "Active Inference エージェント",
            "strategy": "exploration",
        }
    }
    entries = _self_to_virtual_entries(state)
    facets = [e["facet"] for e in entries]
    return all([
        _assert(len(entries) == 3, f"3 entries (actual: {len(entries)})"),
        _assert("name" in facets, "facet=name"),
        _assert("identity" in facets, "facet=identity"),
        _assert("strategy" in facets, "facet=strategy"),
        _assert(all(e["kind"] == "self" for e in entries), "kind=self 全エントリ"),
        _assert(all(e["id"].startswith("self.") for e in entries),
                "id は self. prefix"),
    ])


def test_self_to_virtual_entries_empty_skip():
    print("== _self_to_virtual_entries: 空値はスキップ ==")
    state = {
        "self": {
            "name": "",            # 空文字 → スキップ
            "identity": "real",     # 残る
            "empty_dict": {},       # 空 dict (falsy) → スキップ
            "zero": 0,              # 0 (falsy) → スキップ
            "real_value": "x",     # 残る
        }
    }
    entries = _self_to_virtual_entries(state)
    facets = [e["facet"] for e in entries]
    return all([
        _assert(len(entries) == 2, f"残るは identity と real_value のみ (actual: {len(entries)})"),
        _assert("identity" in facets, "identity 残る"),
        _assert("real_value" in facets, "real_value 残る"),
        _assert("name" not in facets, "name (空文字) はスキップ"),
    ])


def test_self_to_virtual_entries_no_self_key():
    print("== _self_to_virtual_entries: state に self なしで空 list ==")
    return all([
        _assert(_self_to_virtual_entries({}) == [], "空 state で空 list"),
        _assert(_self_to_virtual_entries({"self": None}) == [],
                "self=None で空 list"),
        _assert(_self_to_virtual_entries({"self": {}}) == [],
                "self={} で空 list"),
    ])


def test_self_to_virtual_entries_content_stringified():
    print("== _self_to_virtual_entries: dict / int 値も str() 化 (JSON-encoded string も生扱い MVP) ==")
    state = {
        "self": {
            "json_str": '{"key": "value"}',   # JSON-encoded string
            "dict_val": {"a": 1},              # 生 dict
            "int_val": 42,
        }
    }
    entries = _self_to_virtual_entries(state)
    contents = {e["facet"]: e["content"] for e in entries}
    return all([
        _assert(contents.get("json_str") == '{"key": "value"}',
                "JSON 文字列は parse せず生で保持"),
        _assert("a" in contents.get("dict_val", ""), "dict は str() 化"),
        _assert(contents.get("int_val") == "42", "int は str() 化"),
    ])


# ============================================================
# _build_self_node
# ============================================================

def test_build_self_node_id_invariant():
    print("== _build_self_node: id='self' は cycle に紐付かず不変 (continuous self) ==")
    state1 = {"cycle_id": 5, "self": {"a": "x"}}
    state2 = {"cycle_id": 100, "self": {"b": "y"}}
    node1 = _build_self_node(state1, _self_to_virtual_entries(state1))
    node2 = _build_self_node(state2, _self_to_virtual_entries(state2))
    return all([
        _assert(node1["id"] == "self", "node1 id='self'"),
        _assert(node2["id"] == "self", "node2 id='self' (cycle 違っても同じ)"),
        _assert(node1["id"] == node2["id"], "id は cycle 跨いで不変"),
    ])


def test_build_self_node_metrics_live():
    print("== _build_self_node: metrics は state から live 取得 ==")
    state = {
        "cycle_id": 39,
        "entropy": 0.50,
        "pressure": 3.17,
        "energy": 60.32,
        "self": {"name": "iku"},
    }
    node = _build_self_node(state, _self_to_virtual_entries(state))
    metrics = node.get("metrics", {})
    return all([
        _assert(metrics.get("cycle") == 39, "cycle 取れる"),
        _assert(metrics.get("entropy") == 0.50, "entropy 取れる"),
        _assert(metrics.get("pressure") == 3.17, "pressure 取れる"),
        _assert(metrics.get("energy") == 60.32, "energy 取れる"),
        _assert(node.get("kind") == "self", "kind=self"),
        _assert(node.get("facets") == ["name"], "facets list 反映"),
    ])


def test_build_self_node_missing_metrics_graceful():
    print("== _build_self_node: state に metrics 欠落でも graceful ==")
    state = {"self": {"a": "x"}}  # cycle_id / entropy / etc なし
    node = _build_self_node(state, _self_to_virtual_entries(state))
    metrics = node.get("metrics", {})
    return all([
        _assert(node.get("id") == "self", "id 出る"),
        _assert(metrics.get("cycle") is None, "cycle=None で graceful"),
        _assert(metrics.get("entropy") is None, "entropy=None で graceful"),
    ])


# ============================================================
# _compute_self_to_memory_edges (永続化なし、on-the-fly)
# ============================================================

def test_self_to_memory_edges_empty_inputs():
    print("== _compute_self_to_memory_edges: 入力空で edges 空 ==")
    return all([
        _assert(_compute_self_to_memory_edges([], []) == [],
                "両方空"),
        _assert(_compute_self_to_memory_edges([{"id": "self.x", "content": "y"}], []) == [],
                "memory なし"),
        _assert(_compute_self_to_memory_edges([], [{"id": "m1", "content": "z"}]) == [],
                "self なし"),
    ])


# ============================================================
# _compute_memory_edges (永続 link 参照)
# ============================================================

def test_compute_memory_edges_no_links():
    print("== _compute_memory_edges: link 無しでも空 list (例外でない) ==")
    # memory_links.jsonl が無い / 空 / none-only でも空 list を返す
    edges = _compute_memory_edges()
    return all([
        _assert(isinstance(edges, list), "list 返却"),
        # link が永続化されてれば >= 0 件、なければ 0 件、どちらでも OK
        _assert(all(isinstance(e, dict) for e in edges),
                "全 edge が dict"),
        _assert(all("from" in e and "to" in e and "relation" in e for e in edges),
                "全 edge に from/to/relation"),
    ])


# ============================================================
# _compute_trace
# ============================================================

def test_compute_trace_basic():
    print("== _compute_trace: 総数返却 ==")
    all_memory = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
    edges = [{"from": "m1", "to": "m2"}]
    trace = _compute_trace(all_memory, edges)
    return all([
        _assert(trace.get("memory_total") == 3, "memory_total=3"),
        _assert(trace.get("link_total") == 1, "link_total=1"),
    ])


# ============================================================
# _memory_graph (tool entry point)
# ============================================================

def test_memory_graph_ego_view_shape():
    print("== _memory_graph: view=ego で必須 key を持つ JSON ==")
    result = _memory_graph({"view": "ego"})
    try:
        data = json.loads(result)
    except Exception as e:
        return _assert(False, f"JSON parse 失敗: {e}")
    return all([
        _assert(data.get("view") == "ego", "view=ego"),
        _assert("self" in data, "self key"),
        _assert("edges_self_to_memory" in data, "edges_self_to_memory key"),
        _assert("edges_memory_to_memory" in data, "edges_memory_to_memory key"),
        _assert("trace_recent" in data, "trace_recent key"),
        _assert("clusters" not in data, "clusters key なし (Phase 5 reserved)"),
        _assert("frontier" not in data, "frontier key なし (Phase 4 reserved)"),
        _assert(data["self"].get("id") == "self", "self.id='self'"),
        _assert(data["self"].get("kind") == "self", "self.kind='self'"),
    ])


def test_memory_graph_default_view():
    print("== _memory_graph: view 未指定で ego (default) ==")
    result = _memory_graph({})
    data = json.loads(result)
    return _assert(data.get("view") == "ego", "default view=ego")


def test_memory_graph_unsupported_view_error():
    print("== _memory_graph: view=global / both で error response (Step 0.3 b' future_views 含む) ==")
    for v in ["global", "both", "invalid"]:
        result = _memory_graph({"view": v})
        try:
            data = json.loads(result)
        except Exception:
            return _assert(False, f"view={v} で JSON parse 失敗")
        if "error" not in data:
            return _assert(False, f"view={v} で error key なし")
        if "supported_views" not in data:
            return _assert(False, f"view={v} で supported_views なし")
    # Step 0.3 b': future_views slot で Phase 4/5 降臨予定を明示
    result_global = json.loads(_memory_graph({"view": "global"}))
    result_both = json.loads(_memory_graph({"view": "both"}))
    return all([
        _assert("future_views" in result_global, "global error に future_views"),
        _assert("global" in result_global.get("future_views", []),
                "future_views に global 含む"),
        _assert("both" in result_global.get("future_views", []),
                "future_views に both 含む"),
        _assert("future_views" in result_both, "both error にも future_views"),
    ])


def test_memory_graph_placeholder_args_accepted():
    print("== _memory_graph: PLAN §6-6 placeholder args 受取 (Step 0.3 b') ==")
    # Step 0.2 では未使用だが、PLAN §6-6 signature 互換のため reject しない
    result = _memory_graph({
        "view": "ego",
        "depth": 2,
        "focus_node": "self",
        "cluster_count": 5,
        "frontier_count": 3,
    })
    try:
        data = json.loads(result)
    except Exception as e:
        return _assert(False, f"JSON parse 失敗: {e}")
    return all([
        _assert("error" not in data, "placeholder args で error にならない"),
        _assert(data.get("view") == "ego", "view=ego の正常出力"),
    ])


def test_memory_graph_depth_arg():
    print("== _memory_graph: depth 引数の解釈 ==")
    r1 = json.loads(_memory_graph({"view": "ego", "depth": 3}))
    r2 = json.loads(_memory_graph({"view": "ego", "depth": "5"}))
    r3 = json.loads(_memory_graph({"view": "ego", "depth": "invalid"}))
    return all([
        _assert(r1.get("depth") == 3, "depth=3 (int)"),
        _assert(r2.get("depth") == 5, "depth='5' (string でも int 化)"),
        _assert(r3.get("depth") == 2, "depth='invalid' で default=2"),
    ])


def test_memory_graph_output_no_natural_language_keys():
    print("== _memory_graph: 出力 key に自然言語比喩なし (中立技術用語のみ) ==")
    result = _memory_graph({"view": "ego"})
    data = json.loads(result)
    forbidden = ["森", "心拍", "俯瞰", "観測", "存在", "日記", "蜘蛛"]
    found_forbidden = []
    def scan(obj):
        if isinstance(obj, dict):
            for k in obj.keys():
                for f in forbidden:
                    if f in str(k):
                        found_forbidden.append(f"key='{k}' contains '{f}'")
            for v in obj.values():
                scan(v)
        elif isinstance(obj, list):
            for item in obj:
                scan(item)
    scan(data)
    return _assert(not found_forbidden,
                   f"禁止語彙なし (found: {found_forbidden})")


# ============================================================
# 実行
# ============================================================

if __name__ == "__main__":
    groups = [
        ("_self_to_virtual_entries: 基本変換", test_self_to_virtual_entries_basic),
        ("_self_to_virtual_entries: 空値スキップ", test_self_to_virtual_entries_empty_skip),
        ("_self_to_virtual_entries: state self なしで空", test_self_to_virtual_entries_no_self_key),
        ("_self_to_virtual_entries: content 文字列化", test_self_to_virtual_entries_content_stringified),
        ("_build_self_node: id 不変 (continuous)", test_build_self_node_id_invariant),
        ("_build_self_node: metrics live read", test_build_self_node_metrics_live),
        ("_build_self_node: metrics 欠落 graceful", test_build_self_node_missing_metrics_graceful),
        ("_compute_self_to_memory_edges: empty", test_self_to_memory_edges_empty_inputs),
        ("_compute_memory_edges: list 返却", test_compute_memory_edges_no_links),
        ("_compute_trace: 総数", test_compute_trace_basic),
        ("_memory_graph: ego view shape", test_memory_graph_ego_view_shape),
        ("_memory_graph: default view", test_memory_graph_default_view),
        ("_memory_graph: 未対応 view で error", test_memory_graph_unsupported_view_error),
        ("_memory_graph: placeholder args (Step 0.3 b')", test_memory_graph_placeholder_args_accepted),
        ("_memory_graph: depth 引数", test_memory_graph_depth_arg),
        ("_memory_graph: 自然言語 key なし", test_memory_graph_output_no_natural_language_keys),
    ]
    results = []
    for _label, fn in groups:
        print()
        ok = fn()
        results.append((_label, ok))
    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for _label, ok in results:
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {_label}")
    print(f"\n  {passed}/{total} groups passed")
    sys.exit(0 if passed == total else 1)
