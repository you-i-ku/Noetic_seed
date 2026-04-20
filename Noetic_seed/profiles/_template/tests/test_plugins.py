"""Plugin system tests。"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.plugins import (
    PluginState, PluginMetadata, PluginContext, PluginRecord,
    PluginManager,
    discover_plugins, load_plugin, activate_plugin,
    healthcheck, deactivate_plugin,
)
from core.runtime.registry import ToolRegistry
from core.runtime.commands.dispatcher import CommandDispatcher


def _assert(cond, label):
    status = "OK " if cond else "FAIL"
    print(f"  [{status}] {label}")
    return cond


TMPDIR: Path = None


def _setup():
    global TMPDIR
    TMPDIR = Path(tempfile.mkdtemp(prefix="clawcode_plugin_"))
    pdir = TMPDIR / "plugins"
    pdir.mkdir()

    # 単一モジュール型 plugin
    (pdir / "simple_plugin.py").write_text(
        "PLUGIN_METADATA = {\n"
        "    'name': 'simple',\n"
        "    'version': '0.1',\n"
        "    'description': 'a simple test plugin',\n"
        "}\n\n"
        "_state = {'registered': False, 'healthy': True}\n\n"
        "def register(ctx):\n"
        "    _state['registered'] = True\n"
        "    ctx.settings['simple_loaded'] = True\n\n"
        "def healthcheck():\n"
        "    return _state['healthy']\n\n"
        "def deactivate(ctx):\n"
        "    _state['registered'] = False\n",
        encoding="utf-8",
    )

    # ディレクトリ型 plugin
    d = pdir / "fancy"
    d.mkdir()
    (d / "plugin.py").write_text(
        "def register(ctx):\n"
        "    if ctx.tool_registry is not None:\n"
        "        from core.runtime.tool_schema import ToolSpec\n"
        "        from core.runtime.permissions import PermissionMode\n"
        "        ctx.tool_registry.register(ToolSpec(\n"
        "            name='fancy_noop',\n"
        "            description='no-op from plugin',\n"
        "            input_schema={'type': 'object'},\n"
        "            required_permission=PermissionMode.READ_ONLY,\n"
        "            handler=lambda i: 'fancy!',\n"
        "        ))\n",
        encoding="utf-8",
    )

    # 壊れた plugin (import 時に例外)
    (pdir / "broken.py").write_text(
        "raise RuntimeError('boom at import')\n",
        encoding="utf-8",
    )

    # アンダースコア付きは無視される
    (pdir / "_internal.py").write_text(
        "def register(ctx): pass\n",
        encoding="utf-8",
    )


def _teardown():
    if TMPDIR and TMPDIR.exists():
        shutil.rmtree(TMPDIR)


def _ctx():
    return PluginContext(
        tool_registry=ToolRegistry(),
        command_dispatcher=CommandDispatcher(),
        settings={},
        workspace_root=TMPDIR,
    )


# ============================================================
# Discovery
# ============================================================

def test_discover():
    print("== discover_plugins ==")
    found = discover_plugins([str(TMPDIR / "plugins")])
    names = {r.metadata.name for r in found}
    return all([
        _assert("simple_plugin" in names, "simple_plugin ファイル"),
        _assert("fancy" in names, "fancy ディレクトリ"),
        _assert("broken" in names, "broken も discover はされる"),
        _assert("_internal" not in names, "_internal は無視"),
    ])


def test_discover_empty_dirs():
    print("== discover: 存在しないディレクトリは無視 ==")
    found = discover_plugins(["/no/such", "", str(TMPDIR / "plugins")])
    return _assert(len(found) >= 3, ">= 3 件発見")


# ============================================================
# Load
# ============================================================

def test_load_simple():
    print("== load_plugin: simple ==")
    records = discover_plugins([str(TMPDIR / "plugins")])
    simple = next(r for r in records if r.metadata.name == "simple_plugin")
    ok = load_plugin(simple)
    return all([
        _assert(ok, "load 成功"),
        _assert(simple.state == PluginState.LOADED, "LOADED"),
        _assert(simple.metadata.version == "0.1",
                "PLUGIN_METADATA 反映"),
        _assert(simple.metadata.name == "simple",
                "name はメタデータの値"),
    ])


def test_load_broken():
    print("== load_plugin: broken → FAILED ==")
    records = discover_plugins([str(TMPDIR / "plugins")])
    broken = next(r for r in records if r.metadata.name == "broken")
    ok = load_plugin(broken)
    return all([
        _assert(not ok, "load 失敗"),
        _assert(broken.state == PluginState.FAILED, "FAILED"),
        _assert("boom at import" in broken.last_error,
                "例外メッセージ保持"),
    ])


# ============================================================
# Activate
# ============================================================

def test_activate_simple():
    print("== activate_plugin: register 呼出 ==")
    records = discover_plugins([str(TMPDIR / "plugins")])
    simple = next(r for r in records if r.metadata.name == "simple_plugin")
    load_plugin(simple)
    ctx = _ctx()
    ok = activate_plugin(simple, ctx)
    return all([
        _assert(ok, "activate 成功"),
        _assert(simple.state == PluginState.ACTIVATED, "ACTIVATED"),
        _assert(ctx.settings.get("simple_loaded") is True,
                "register が ctx を変更"),
    ])


def test_activate_fancy_registers_tool():
    print("== activate_plugin: fancy が tool を登録 ==")
    records = discover_plugins([str(TMPDIR / "plugins")])
    fancy = next(r for r in records if r.metadata.name == "fancy")
    load_plugin(fancy)
    ctx = _ctx()
    activate_plugin(fancy, ctx)
    return all([
        _assert(ctx.tool_registry.has("fancy_noop"),
                "fancy_noop 登録"),
        _assert(ctx.tool_registry.execute("fancy_noop", {}) == "fancy!",
                "handler 動作"),
    ])


# ============================================================
# Healthcheck / Deactivate
# ============================================================

def test_healthcheck():
    print("== healthcheck ==")
    records = discover_plugins([str(TMPDIR / "plugins")])
    simple = next(r for r in records if r.metadata.name == "simple_plugin")
    load_plugin(simple)
    activate_plugin(simple, _ctx())
    ok = healthcheck(simple)
    return all([
        _assert(ok, "健康"),
        _assert(simple.state == PluginState.HEALTHY, "HEALTHY"),
    ])


def test_deactivate():
    print("== deactivate ==")
    records = discover_plugins([str(TMPDIR / "plugins")])
    simple = next(r for r in records if r.metadata.name == "simple_plugin")
    load_plugin(simple)
    ctx = _ctx()
    activate_plugin(simple, ctx)
    ok = deactivate_plugin(simple, ctx)
    return all([
        _assert(ok, "deactivate 成功"),
        _assert(simple.state == PluginState.DEACTIVATED,
                "DEACTIVATED"),
    ])


# ============================================================
# Manager (run_all)
# ============================================================

def test_manager_run_all():
    print("== PluginManager.run_all ==")
    ctx = _ctx()
    mgr = PluginManager(ctx, search_dirs=[str(TMPDIR / "plugins")])
    states = mgr.run_all()
    # manager の dict key は discovery 時の name (= ファイル/ディレクトリ名)
    return all([
        _assert(states.get("simple_plugin") == "healthy",
                f"simple_plugin healthy (got {states.get('simple_plugin')})"),
        _assert(states.get("fancy") in ("healthy", "activated"),
                f"fancy activated/healthy (got {states.get('fancy')})"),
        _assert(states.get("broken") == "failed",
                f"broken failed (got {states.get('broken')})"),
        _assert(ctx.tool_registry.has("fancy_noop"),
                "fancy tool も登録済"),
    ])


def test_manager_deactivate_all():
    print("== PluginManager.deactivate_all ==")
    ctx = _ctx()
    mgr = PluginManager(ctx, search_dirs=[str(TMPDIR / "plugins")])
    mgr.run_all()
    n = mgr.deactivate_all()
    return _assert(n >= 1, f">= 1 deactivated (got {n})")


def main():
    _setup()
    try:
        tests = [
            test_discover, test_discover_empty_dirs,
            test_load_simple, test_load_broken,
            test_activate_simple, test_activate_fancy_registers_tool,
            test_healthcheck, test_deactivate,
            test_manager_run_all, test_manager_deactivate_all,
        ]
        print(f"Running {len(tests)} test groups...\n")
        passed = 0
        for t in tests:
            if t():
                passed += 1
            print()
        print(f"=== Result: {passed}/{len(tests)} test groups passed ===")
        return 0 if passed == len(tests) else 1
    finally:
        _teardown()


if __name__ == "__main__":
    sys.exit(main())
