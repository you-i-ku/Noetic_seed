"""Sandbox — bash 実行時の分離制御 (Linux unshare / fs / network)。

claw-code 参照: rust/crates/runtime/src/sandbox.rs:1-385

Phase では config 構造 + コマンドラッパの構築だけ行う。
実際の unshare 実行は Linux でしか動かないので platform guard 付き。
Windows / macOS では no-op (ラッパなし、そのまま実行)。
"""
import os
import platform
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FilesystemMode(Enum):
    OFF = "off"
    WORKSPACE_ONLY = "workspace_only"
    ALLOW_LIST = "allow_list"


@dataclass
class SandboxConfig:
    """bash sandbox の構成。"""
    filesystem: FilesystemMode = FilesystemMode.OFF
    workspace_root: Optional[str] = None
    allow_list: list = field(default_factory=list)
    network_isolated: bool = False
    allow_network_hosts: list = field(default_factory=list)


def detect_container() -> Optional[str]:
    """実行環境が Docker/Podman/Kubernetes かを検知。"""
    # Docker
    if os.path.exists("/.dockerenv"):
        return "docker"
    # Podman
    if os.path.exists("/run/.containerenv"):
        return "podman"
    # Kubernetes
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return "kubernetes"
    # cgroup からの推定
    try:
        with open("/proc/1/cgroup", "r", encoding="utf-8") as f:
            cg = f.read()
            if "docker" in cg:
                return "docker"
            if "kubepods" in cg:
                return "kubernetes"
    except Exception:
        pass
    return None


def unshare_available() -> bool:
    """Linux の unshare コマンドが使えるか。"""
    if platform.system() != "Linux":
        return False
    return shutil.which("unshare") is not None


def wrap_command(args: list, config: SandboxConfig) -> list:
    """bash 実行時の subprocess args に sandbox ラッパをかける。

    Linux + unshare 利用可のときのみ実際にラップ、それ以外はそのまま。
    """
    if config.filesystem == FilesystemMode.OFF and not config.network_isolated:
        return list(args)
    if not unshare_available():
        return list(args)

    cmd = ["unshare"]
    if config.network_isolated:
        cmd.append("--net")
    # NOTE: fs mount 分離は unshare --mount + bind mount セットアップが必要。
    # 簡易版として Phase では --mount のみ付与、bind mount は将来実装。
    if config.filesystem != FilesystemMode.OFF:
        cmd.append("--mount")
    cmd.append("--")
    return cmd + list(args)
