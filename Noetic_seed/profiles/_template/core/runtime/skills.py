"""Skills Registry & Loader — claw-code 準拠。

claw-code 参照:
  - rust/crates/runtime/src/skill_registry.rs
  - rust/crates/runtime/src/skill_loader.rs

Skill 定義フォーマット:
  ---
  name: skill-name
  description: 1 行説明 (自動発火用)
  triggers: ["keyword", "/command"]
  allowed_tools: ["read_file", "bash"]
  ---
  # Skill Body
  ... markdown 本文 ...

検索パス (優先順):
  1. <project>/.claw/skills/
  2. ~/.claude/skills/
  3. $CODEX_HOME/skills/
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    """skill 1 件分。"""
    name: str
    description: str = ""
    triggers: list = field(default_factory=list)
    allowed_tools: list = field(default_factory=list)
    body: str = ""
    source_path: Optional[Path] = None


# ============================================================
# Parser (YAML frontmatter + markdown)
# ============================================================

_SIMPLE_YAML_LIST_RE = re.compile(r"^\[(.*)\]$")


def _parse_yaml_value(raw: str):
    """簡易 YAML 値パース: string / [a, b, c] / quoted string。"""
    v = raw.strip()
    # list 形式
    m = _SIMPLE_YAML_LIST_RE.match(v)
    if m:
        inner = m.group(1).strip()
        if not inner:
            return []
        items = []
        for part in inner.split(","):
            part = part.strip().strip('"').strip("'")
            if part:
                items.append(part)
        return items
    # quote 剥がし
    if (v.startswith('"') and v.endswith('"')) or \
       (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    return v


def parse_skill_file(path: Path) -> Optional[Skill]:
    """.md ファイルを skill として読み込む。"""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    frontmatter: dict = {}
    body = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end > 0:
            fm_text = text[3:end].strip()
            body = text[end + 3:].lstrip("\n")
            for line in fm_text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    frontmatter[k.strip()] = _parse_yaml_value(v)

    name = frontmatter.get("name") or path.stem
    if isinstance(name, list):
        name = name[0] if name else path.stem

    return Skill(
        name=str(name),
        description=str(frontmatter.get("description", "")),
        triggers=list(frontmatter.get("triggers") or []),
        allowed_tools=list(frontmatter.get("allowed_tools") or []),
        body=body,
        source_path=path,
    )


# ============================================================
# Registry
# ============================================================

class SkillRegistry:
    """skill の集中管理。

    load(skill_dirs) でディレクトリから読込。
    get(name) / list_all() / find_by_trigger(keyword) で検索。
    """

    def __init__(self):
        self._skills: dict = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def has(self, name: str) -> bool:
        return name in self._skills

    def list_all(self) -> list:
        return list(self._skills.values())

    def find_by_trigger(self, keyword: str) -> list:
        """triggers / name / description のどれかに keyword を含む skill を返す。"""
        kw = keyword.lower()
        out: list = []
        for sk in self._skills.values():
            if any(kw in str(t).lower() for t in sk.triggers):
                out.append(sk)
                continue
            if kw in sk.name.lower() or kw in sk.description.lower():
                out.append(sk)
        return out

    def load(self, skill_dirs: list) -> int:
        """複数ディレクトリから *.md / */SKILL.md を読込。戻り値: 新規登録数。

        先に来たディレクトリが優先 (同名 skill は上書きしない)。
        """
        count = 0
        for d in skill_dirs or []:
            if not d:
                continue
            root = Path(d)
            if not root.exists() or not root.is_dir():
                continue
            # トップレベルの *.md
            for p in sorted(root.glob("*.md")):
                sk = parse_skill_file(p)
                if sk and sk.name not in self._skills:
                    self._skills[sk.name] = sk
                    count += 1
            # サブディレクトリの SKILL.md / skill.md
            for p in sorted(root.glob("*/SKILL.md")) + \
                     sorted(root.glob("*/skill.md")):
                sk = parse_skill_file(p)
                if sk and sk.name not in self._skills:
                    self._skills[sk.name] = sk
                    count += 1
        return count


# ============================================================
# Default search paths
# ============================================================

def default_skill_dirs(workspace_root: Optional[Path] = None) -> list:
    dirs: list = []
    if workspace_root:
        dirs.append(str(workspace_root / ".claw" / "skills"))
    home = Path.home()
    dirs.append(str(home / ".claude" / "skills"))
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        dirs.append(str(Path(codex_home) / "skills"))
    return dirs
