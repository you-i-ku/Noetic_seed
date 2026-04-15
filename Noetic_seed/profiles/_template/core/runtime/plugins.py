"""Plugin System — claw-code 準拠。

claw-code 参照:
  - rust/crates/plugins/src/
  - rust/crates/runtime/src/plugin_lifecycle.rs

プラグインは Python パッケージ or 単一モジュールとして配布され、
以下の entry point を持つ:

    def register(ctx: PluginContext) -> None: ...
    # または
    PLUGIN_METADATA = {"name": "...", "version": "...", ...}
    def activate(ctx: PluginContext) -> list[ToolSpec | CommandSpec]: ...

検索パス (優先順):
  1. <project>/.claw/plugins/<name>/plugin.py or <name>.py
  2. ~/.claude/plugins/<name>/
  3. installed python packages with entry point group "claw_code_plugins"

状態遷移:
  Discovered -> Loaded -> Activated -> (Healthy | Unhealthy) -> Deactivated
"""
import importlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class PluginState(Enum):
    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVATED = "activated"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEACTIVATED = "deactivated"
    FAILED = "failed"


@dataclass
class PluginMetadata:
    name: str
    version: str = "0.0.0"
    description: str = ""
    author: str = ""
    requires: list = field(default_factory=list)


@dataclass
class PluginContext:
    """plugin activate() に渡される。

    plugin は自由に tool_registry / command_dispatcher に register できる。
    """
    tool_registry: object = None       # ToolRegistry
    command_dispatcher: object = None  # CommandDispatcher
    skill_registry: object = None      # SkillRegistry
    hook_runner: object = None         # HookRunner
    settings: dict = field(default_factory=dict)
    workspace_root: Optional[Path] = None
    # plugin 側から呼ぶヘルパ
    log: Callable = None


@dataclass
class PluginRecord:
    metadata: PluginMetadata
    source_path: Optional[Path] = None
    module: object = None
    state: PluginState = PluginState.DISCOVERED
    last_error: Optional[str] = None
    registered_tools: list = field(default_factory=list)
    registered_commands: list = field(default_factory=list)


# ============================================================
# Discovery
# ============================================================

def discover_plugins(search_dirs: list) -> list:
    """search_dirs から plugin 候補を探す。

    パターン:
      - <dir>/<name>/plugin.py
      - <dir>/<name>.py (単一モジュール)
    """
    found: list = []
    seen_names = set()
    for d in search_dirs or []:
        if not d:
            continue
        root = Path(d)
        if not root.exists() or not root.is_dir():
            continue
        # ディレクトリ型 plugin
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            candidate = sub / "plugin.py"
            if candidate.exists() and sub.name not in seen_names:
                found.append(PluginRecord(
                    metadata=PluginMetadata(name=sub.name),
                    source_path=candidate,
                    state=PluginState.DISCOVERED,
                ))
                seen_names.add(sub.name)
        # 単一モジュール型
        for py in sorted(root.glob("*.py")):
            if py.name.startswith("_"):
                continue
            name = py.stem
            if name in seen_names:
                continue
            found.append(PluginRecord(
                metadata=PluginMetadata(name=name),
                source_path=py,
                state=PluginState.DISCOVERED,
            ))
            seen_names.add(name)
    return found


# ============================================================
# Load + Activate
# ============================================================

def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_claw_plugin_{module_name}"] = module
    spec.loader.exec_module(module)
    return module


def load_plugin(record: PluginRecord) -> bool:
    """plugin を import する。成功で state=LOADED。"""
    if record.source_path is None:
        record.state = PluginState.FAILED
        record.last_error = "no source path"
        return False
    try:
        mod = _load_module_from_path(record.metadata.name,
                                      record.source_path)
    except Exception as e:
        record.state = PluginState.FAILED
        record.last_error = (f"{type(e).__name__}: {e}\n"
                             + traceback.format_exc(limit=3))
        return False

    # PLUGIN_METADATA が定義されていればマージ
    pm = getattr(mod, "PLUGIN_METADATA", None)
    if isinstance(pm, dict):
        record.metadata = PluginMetadata(
            name=pm.get("name", record.metadata.name),
            version=pm.get("version", record.metadata.version),
            description=pm.get("description", ""),
            author=pm.get("author", ""),
            requires=list(pm.get("requires") or []),
        )
    record.module = mod
    record.state = PluginState.LOADED
    return True


def activate_plugin(record: PluginRecord,
                    ctx: PluginContext) -> bool:
    """plugin の register() または activate() を呼ぶ。"""
    if record.module is None:
        record.state = PluginState.FAILED
        record.last_error = "module not loaded"
        return False

    entry = (getattr(record.module, "register", None)
             or getattr(record.module, "activate", None))
    if entry is None:
        record.state = PluginState.FAILED
        record.last_error = "no entry point (register/activate)"
        return False

    try:
        entry(ctx)
    except Exception as e:
        record.state = PluginState.FAILED
        record.last_error = (f"activate failed: {type(e).__name__}: {e}\n"
                             + traceback.format_exc(limit=3))
        return False

    record.state = PluginState.ACTIVATED
    return True


def healthcheck(record: PluginRecord) -> bool:
    """plugin が healthcheck() を持てば呼ぶ。"""
    if record.module is None:
        return False
    fn = getattr(record.module, "healthcheck", None)
    if fn is None:
        record.state = PluginState.HEALTHY  # default 健康
        return True
    try:
        ok = bool(fn())
    except Exception as e:
        record.state = PluginState.UNHEALTHY
        record.last_error = f"healthcheck failed: {e}"
        return False
    record.state = PluginState.HEALTHY if ok else PluginState.UNHEALTHY
    return ok


def deactivate_plugin(record: PluginRecord,
                      ctx: PluginContext) -> bool:
    """plugin の deactivate() を呼ぶ (任意)。"""
    if record.module is None:
        return False
    fn = getattr(record.module, "deactivate", None)
    if fn is not None:
        try:
            fn(ctx)
        except Exception as e:
            record.last_error = f"deactivate failed: {e}"
            return False
    record.state = PluginState.DEACTIVATED
    return True


# ============================================================
# Manager
# ============================================================

class PluginManager:
    """複数 plugin の一括管理。"""

    def __init__(self, ctx: PluginContext,
                 search_dirs: Optional[list] = None):
        self.ctx = ctx
        self.search_dirs = list(search_dirs or [])
        self._plugins: dict = {}   # name -> PluginRecord

    def discover(self) -> list:
        found = discover_plugins(self.search_dirs)
        for r in found:
            if r.metadata.name not in self._plugins:
                self._plugins[r.metadata.name] = r
        return list(self._plugins.values())

    def load_all(self) -> int:
        ok = 0
        for r in self._plugins.values():
            if r.state == PluginState.DISCOVERED:
                if load_plugin(r):
                    ok += 1
        return ok

    def activate_all(self) -> int:
        ok = 0
        for r in self._plugins.values():
            if r.state == PluginState.LOADED:
                if activate_plugin(r, self.ctx):
                    ok += 1
        return ok

    def healthcheck_all(self) -> dict:
        out: dict = {}
        for name, r in self._plugins.items():
            if r.state in (PluginState.ACTIVATED,
                            PluginState.HEALTHY,
                            PluginState.UNHEALTHY):
                out[name] = healthcheck(r)
        return out

    def deactivate_all(self) -> int:
        ok = 0
        for r in self._plugins.values():
            if r.state in (PluginState.ACTIVATED,
                            PluginState.HEALTHY,
                            PluginState.UNHEALTHY):
                if deactivate_plugin(r, self.ctx):
                    ok += 1
        return ok

    def list_all(self) -> list:
        return list(self._plugins.values())

    def get(self, name: str) -> Optional[PluginRecord]:
        return self._plugins.get(name)

    def run_all(self) -> dict:
        """discover → load → activate → healthcheck を一括実行。"""
        self.discover()
        self.load_all()
        self.activate_all()
        self.healthcheck_all()
        return {
            name: r.state.value
            for name, r in self._plugins.items()
        }
