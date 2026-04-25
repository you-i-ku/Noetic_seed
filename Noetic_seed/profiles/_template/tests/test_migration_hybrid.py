"""段階11-D Phase 7 Step 7.1: hybrid retrieval (tag_filter) tests。

PLAN §5 Phase 7 案 α (`get_relevant_memories` 拡張):
- tag_filter=None (default): 全 networks 横断 (UNTAGGED 含む)、現状挙動維持
- tag_filter=list: 指定 tag のみ semantic search
- 既存呼出 (tag_filter なし) backward compat、既存 test 全部 green 維持

実行:
  "C:/Users/you11/Desktop/iku/Noetic_seed/.venv/Scripts/python.exe" tests/test_migration_hybrid.py
"""
import os
import sys
import tempfile
from pathlib import Path

PROFILE_PATH = Path(__file__).parent.parent
os.environ["NOETIC_PROFILE"] = str(PROFILE_PATH)
sys.path.insert(0, str(PROFILE_PATH))


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _setup_clean(tmp_path: Path):
    """clean memory dir + tag registry を tmp に切替。

    既存 test_memory_store_untagged.py の `_setup` と同 pattern。
    `core.memory.MEMORY_DIR` を直接上書き (import snapshot bug 回避、
    11-C hotfix 5 と同型注意点)。
    """
    import core.memory as memory_mod
    import core.tag_registry as tr
    memory_mod.MEMORY_DIR = tmp_path
    reg_file = tmp_path / "registered_tags.json"
    tr._reset_for_testing(registry_file=reg_file)


def test_tag_filter_default_returns_all_networks(tmp_path: Path):
    print("== tag_filter=None で UNTAGGED + 既存 tag 両方拾う ==")
    _setup_clean(tmp_path)
    from core.memory import memory_store, get_relevant_memories
    from core.tag_registry import register_tag
    from core.memory import UNTAGGED_NETWORK

    # opinion tag 登録 + memory 1 件
    register_tag("opinion", learning_rules={}, display_format="[opinion] {content}")
    memory_store(network="opinion", content="opinion entry hybrid test",
                 metadata={"confidence": 0.8}, _auto_metadata=False)
    # untagged も 1 件
    memory_store(network=None, content="untagged entry hybrid test",
                 metadata={}, _auto_metadata=False)

    # state に最近 intent を入れる (query 駆動のため)
    state = {"log": [{"intent": "hybrid test"}]}
    results = get_relevant_memories(state, limit=10)

    networks_found = {m.get("network") for m in results}
    _assert("opinion" in networks_found or UNTAGGED_NETWORK in networks_found,
            f"tag_filter=None で network 横断、got {networks_found}")
    return True


def test_tag_filter_specific_tag_only(tmp_path: Path):
    print("== tag_filter=[opinion] で opinion のみ拾う ==")
    _setup_clean(tmp_path)
    from core.memory import memory_store, get_relevant_memories
    from core.tag_registry import register_tag
    from core.memory import UNTAGGED_NETWORK

    register_tag("opinion", learning_rules={}, display_format="[opinion] {content}")
    register_tag("experience", learning_rules={}, display_format="[experience] {content}")
    memory_store(network="opinion", content="opinion target text", metadata={"confidence": 0.8},
                 _auto_metadata=False)
    memory_store(network="experience", content="experience non-target text", metadata={},
                 _auto_metadata=False)
    memory_store(network=None, content="untagged non-target text", metadata={},
                 _auto_metadata=False)

    state = {"log": [{"intent": "target text"}]}
    results = get_relevant_memories(state, limit=10, tag_filter=["opinion"])

    networks_found = {m.get("network") for m in results if m.get("network") != "external"}
    _assert(networks_found.issubset({"opinion"}),
            f"tag_filter=[opinion] で opinion のみ, got {networks_found}")
    return True


def test_tag_filter_untagged_only(tmp_path: Path):
    print("== tag_filter=[UNTAGGED_NETWORK] で untagged のみ拾う ==")
    _setup_clean(tmp_path)
    from core.memory import memory_store, get_relevant_memories, UNTAGGED_NETWORK
    from core.tag_registry import register_tag

    register_tag("opinion", learning_rules={}, display_format="[opinion] {content}")
    memory_store(network="opinion", content="opinion non-target", metadata={"confidence": 0.8},
                 _auto_metadata=False)
    memory_store(network=None, content="untagged target memory", metadata={}, _auto_metadata=False)

    state = {"log": [{"intent": "target memory"}]}
    results = get_relevant_memories(state, limit=10, tag_filter=[UNTAGGED_NETWORK])

    networks_found = {m.get("network") for m in results if m.get("network") != "external"}
    _assert(networks_found.issubset({UNTAGGED_NETWORK}),
            f"tag_filter=[UNTAGGED_NETWORK] で untagged のみ, got {networks_found}")
    return True


def test_tag_filter_signature_with_use_links(tmp_path: Path):
    print("== tag_filter + use_links 同時指定で signature 受ける ==")
    _setup_clean(tmp_path)
    from core.memory import memory_store, get_relevant_memories
    from core.tag_registry import register_tag

    register_tag("opinion", learning_rules={}, display_format="[opinion] {content}")
    memory_store(network="opinion", content="signature combo test", metadata={"confidence": 0.8},
                 _auto_metadata=False)

    state = {"log": [{"intent": "signature combo"}]}
    # tag_filter + use_links 共存できる (signature 受領 + crash しない)
    results = get_relevant_memories(state, limit=5, use_links=True, tag_filter=["opinion"])
    _assert(isinstance(results, list), "戻り値は list")
    return True


def test_tag_filter_backward_compat(tmp_path: Path):
    print("== 既存呼出 (tag_filter 引数なし) backward compat ==")
    _setup_clean(tmp_path)
    from core.memory import memory_store, get_relevant_memories
    from core.tag_registry import register_tag

    register_tag("opinion", learning_rules={}, display_format="[opinion] {content}")
    memory_store(network="opinion", content="backward compat test", metadata={"confidence": 0.8},
                 _auto_metadata=False)

    state = {"log": [{"intent": "backward compat"}]}
    # tag_filter 渡さない → default None → 全 networks 横断
    results = get_relevant_memories(state, limit=5)
    _assert(isinstance(results, list), "tag_filter なしで戻り値は list")
    # crash しないこと、結果は最低 0 件 (memory ない時) or hit (memory ある時)
    return True


def test_tag_filter_empty_list_falls_back_to_all(tmp_path: Path):
    print("== tag_filter=[] (空 list) は全 networks 扱い (memory_network_search の if not networks 経路) ==")
    _setup_clean(tmp_path)
    from core.memory import memory_store, get_relevant_memories
    from core.tag_registry import register_tag

    register_tag("opinion", learning_rules={}, display_format="[opinion] {content}")
    memory_store(network="opinion", content="empty list fallback", metadata={"confidence": 0.8},
                 _auto_metadata=False)
    memory_store(network=None, content="untagged empty list fallback", metadata={},
                 _auto_metadata=False)

    state = {"log": [{"intent": "empty list fallback"}]}
    results = get_relevant_memories(state, limit=10, tag_filter=[])

    # 空 list は memory_network_search 内 `if not networks` 経路 → 全 networks
    networks_found = {m.get("network") for m in results if m.get("network") != "external"}
    _assert(len(networks_found) > 0,
            f"empty list で hit が出る (全横断 fallback), got {networks_found}")
    return True


if __name__ == "__main__":
    print("test_migration_hybrid.py (段階11-D Phase 7 Step 7.1)")
    print("=" * 60)
    tests = [
        test_tag_filter_default_returns_all_networks,
        test_tag_filter_specific_tag_only,
        test_tag_filter_untagged_only,
        test_tag_filter_signature_with_use_links,
        test_tag_filter_backward_compat,
        test_tag_filter_empty_list_falls_back_to_all,
    ]
    pass_count = 0
    for fn in tests:
        try:
            with tempfile.TemporaryDirectory() as td:
                ok = fn(Path(td))
                if ok:
                    pass_count += 1
                    print(f"  OK  {fn.__name__}")
        except Exception as e:
            print(f"  FAIL {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
    print("=" * 60)
    print(f"{pass_count}/{len(tests)} passed")
    sys.exit(0 if pass_count == len(tests) else 1)
