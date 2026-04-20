"""Git Context — system prompt への自動注入用。

claw-code 参照: rust/crates/runtime/src/git_context.rs:1-100
"""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GitContext:
    is_repo: bool = False
    branch: Optional[str] = None
    head_commit: Optional[str] = None
    recent_commits: list = field(default_factory=list)  # ["sha msg", ...]
    staged_files: list = field(default_factory=list)
    modified_files: list = field(default_factory=list)
    untracked_files: list = field(default_factory=list)

    @classmethod
    def detect(cls, cwd: Optional[Path] = None,
               recent_limit: int = 5) -> "GitContext":
        ctx = cls()
        cwd_str = str(cwd) if cwd else None

        if not cls._run(["rev-parse", "--is-inside-work-tree"], cwd_str):
            return ctx
        ctx.is_repo = True

        ctx.branch = cls._run(["rev-parse", "--abbrev-ref", "HEAD"],
                              cwd_str)
        ctx.head_commit = cls._run(["rev-parse", "--short", "HEAD"],
                                   cwd_str)

        log = cls._run(
            ["log", f"-{recent_limit}", "--oneline", "--no-decorate"],
            cwd_str,
        )
        if log:
            ctx.recent_commits = [l for l in log.splitlines() if l.strip()]

        status = cls._run(["status", "--porcelain=v1"], cwd_str)
        if status:
            for line in status.splitlines():
                if len(line) < 3:
                    continue
                flag = line[:2]
                fname = line[3:]
                if flag[0] in ("A", "M", "R", "C", "D") and flag[0] != " ":
                    ctx.staged_files.append(fname)
                if flag[1] == "M" or flag[1] == "D":
                    ctx.modified_files.append(fname)
                if flag == "??":
                    ctx.untracked_files.append(fname)
        return ctx

    @staticmethod
    def _run(args: list, cwd: Optional[str]) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                cwd=cwd, timeout=10,
            )
        except FileNotFoundError:
            return None
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def to_prompt_block(self) -> str:
        """system_prompt 注入用の短い context。"""
        if not self.is_repo:
            return ""
        lines = ["[git context]"]
        if self.branch:
            head = f" @ {self.head_commit}" if self.head_commit else ""
            lines.append(f"  branch: {self.branch}{head}")
        if self.recent_commits:
            lines.append("  recent:")
            for c in self.recent_commits:
                lines.append(f"    {c}")
        if self.staged_files:
            lines.append(f"  staged ({len(self.staged_files)}): "
                         f"{', '.join(self.staged_files[:10])}")
        if self.modified_files:
            lines.append(f"  modified ({len(self.modified_files)}): "
                         f"{', '.join(self.modified_files[:10])}")
        if self.untracked_files:
            lines.append(f"  untracked ({len(self.untracked_files)})")
        return "\n".join(lines)
