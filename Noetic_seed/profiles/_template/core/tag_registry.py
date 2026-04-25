"""タグレジストリ — 記憶ネットワークの動的タグ管理 (段階7)。

`STAGE7_UNIFIED_MEMORY_STORE_PLAN.md` §4-2 / §5-1 / §5-2 / §5-4 の実装。

設計指針:
- **config = function**: タグごとの学習特性 (β+ / bitemporal / display_format) をメタデータで管理
- **標準タグも register_tag 経由**: wm / experience / opinion / entity は起動時 register_standard_tags() で登録 (特権化しない)
- **動的タグは memory_store の inline 拡張経由**: 未登録タグで store した瞬間に登録 (段階7 Step 5、feedback_no_individual_tools 準拠)
- **consumer は get_tag_rules(name) 参照**: `if tag == "x"` ハードコード禁止
- **永続化**: memory/registered_tags.json (atomic write)、起動時 _load_from_disk で復元
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import MEMORY_DIR
from core.state import _atomic_write


_REGISTRY_VERSION = 1
_REGISTRY_FILE: Path = MEMORY_DIR / "registered_tags.json"

_REGISTERED: dict = {}
_LOADED: bool = False


STANDARD_TAGS: dict = {
    "wm": {
        "learning_rules": {"beta_plus": True, "bitemporal": True},
        "display_format": "[wm:{entity_name}] {content}",
    },
    "experience": {
        "learning_rules": {"beta_plus": False, "bitemporal": False},
        "display_format": "[experience] {content}",
    },
    "opinion": {
        "learning_rules": {"beta_plus": True, "bitemporal": False},
        "display_format": "[opinion] {content} (確度:{confidence})",
        # DEPRECATED 段階11-D Phase 5 (Step 5.2): _build_reflect_sections 機構撤去済。
        # この reflect_section dict は **dead data** として Phase 7 migration まで残置
        # (撤去予定、PLAN §5 Phase 5 Step 5.4 + §10「段階13+ で完全撤去」)。
        # 残置理由: ① test_memory_store_perspective.py Section 3 の存在 assert
        # を破壊しない、② 既存 jsonl entry (network="opinion") の display_format
        # は依然有効、③ 段階7→11-D の互換性窓を確保。
        "reflect_section": {
            "header": "OPINIONS",
            "template": "- 主張内容 (確度: 0.0-1.0)",
            "enabled_in_reflect": True,
        },
    },
    "entity": {
        "learning_rules": {"beta_plus": True, "bitemporal": True, "c_gradual_source": True},
        "display_format": "[entity:{entity_name}] {content}",
        # DEPRECATED 段階11-D Phase 5 (Step 5.2): 同上、Phase 7 migration で撤去予定。
        "reflect_section": {
            "header": "ENTITIES",
            "template": "- 対象名: 属性記述",
            "enabled_in_reflect": True,
        },
    },
    # 段階11-B Phase 2' scope-down (2026-04-23): tag_consideration pseudo-tag は
    # 撤去。理由: 動的タグ生成は段階7 inline register で既に動く + tool spec に
    # rules 引数が含まれる = iku への「新 tag 作れるよ」リマインダーは冗長
    # (ゆう gut check)。write_protected schema と tag_emergence_monitor は
    # 汎用機構 / 観察基盤として保持、Phase 5 白紙 onboarding で必要性を再評価。
}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_from_disk() -> None:
    """永続化ファイルから _REGISTERED に読み込む。idempotent (1 回のみ)。"""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    if not _REGISTRY_FILE.exists():
        return
    try:
        data = json.loads(_REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    for entry in data.get("tags", []):
        name = entry.get("name")
        if isinstance(name, str) and name:
            _REGISTERED[name] = entry


def _save_to_disk() -> None:
    """in-memory → ファイル (atomic)。"""
    _REGISTRY_FILE.parent.mkdir(exist_ok=True)
    data = {
        "version": _REGISTRY_VERSION,
        "tags": list(_REGISTERED.values()),
    }
    _atomic_write(_REGISTRY_FILE, json.dumps(data, ensure_ascii=False, indent=2))


def register_tag(name: str,
                 learning_rules: dict,
                 display_format: str = "",
                 origin: str = "dynamic",
                 intent: Optional[str] = None,
                 file_path: Optional[str] = None,
                 reflect_section: Optional[dict] = None) -> dict:
    """タグを登録。

    - 未登録 → 新規登録
    - 既存 standard タグに standard 再登録 → idempotent (学習ルール最新化、created_at/origin 保持)
    - dynamic 既存への再登録 → ValueError (上書き事故回避)

    Args:
        name: タグ名 (非空文字列)
        learning_rules: {"beta_plus": bool, "bitemporal": bool,
                         "c_gradual_source": bool, "write_protected": bool}
        display_format: format_memories_for_prompt 用 (省略時 "[{name}] {content}")
        origin: "standard" (起動時) / "dynamic" (AI 発明)
        intent: AI 発明時の tool_intent 記録
        file_path: 保存先 jsonl 相対パス (None なら memory/{name}.jsonl)
        reflect_section: 段階11-A G1 — reflect prompt の動的組立に使う定義 dict。
            {"header": str, "template": str, "enabled_in_reflect": bool}。
            None なら reflect prompt に現れない (opt-in)。段階11-B で AI が tag
            を自由発明する時も同 kwarg で付けられる (抽象化拡張点)。
    """
    _load_from_disk()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name は非空文字列")
    name = name.strip()
    if not isinstance(learning_rules, dict):
        raise ValueError("learning_rules は dict")
    rules_norm = {
        "beta_plus": bool(learning_rules.get("beta_plus", False)),
        "bitemporal": bool(learning_rules.get("bitemporal", False)),
        "c_gradual_source": bool(learning_rules.get("c_gradual_source", False)),
        "write_protected": bool(learning_rules.get("write_protected", False)),
    }
    existing = _REGISTERED.get(name)
    if existing is not None:
        if existing.get("origin") == "standard" and origin == "standard":
            existing["learning_rules"] = rules_norm
            if display_format:
                existing["display_format"] = display_format
            if reflect_section is not None:
                existing["reflect_section"] = reflect_section
            _save_to_disk()
            return existing
        raise ValueError(
            f"tag '{name}' は既に登録済 (origin={existing.get('origin')})"
        )
    entry = {
        "name": name,
        "learning_rules": rules_norm,
        "display_format": display_format or f"[{name}] {{content}}",
        "file_path": file_path,
        "origin": origin,
        "created_at": _now(),
        "intent": intent,
    }
    if reflect_section is not None:
        entry["reflect_section"] = reflect_section
    _REGISTERED[name] = entry
    _save_to_disk()
    return entry


def get_tag_rules(name: str) -> Optional[dict]:
    """タグの rule dict を返す。未登録は None。"""
    _load_from_disk()
    return _REGISTERED.get(name)


def get_tags_with_rule(rule_name: str) -> list:
    """指定 learning_rule が True の tag 名 list を返す (段階11-B Phase 1)。

    Phase 1 用途: c_gradual_source を持つ tag (現状 entity のみ) を動的取得、
    reflection.py の entity lookup / C-gradual WM sync に使用。
    Phase 5 (白紙 onboarding) で registered_tags が空なら [] を返す
    = reflect が entity 抽出・WM sync を自然に skip する挙動に migrate。
    """
    _load_from_disk()
    return [
        name for name, entry in _REGISTERED.items()
        if entry.get("learning_rules", {}).get(rule_name, False)
    ]


def list_registered_tags() -> list:
    """登録済みタグ名のリスト。"""
    _load_from_disk()
    return list(_REGISTERED.keys())


def is_tag_registered(name: str) -> bool:
    """タグが登録済みか確認。"""
    _load_from_disk()
    return name in _REGISTERED


def register_standard_tags() -> None:
    """起動時呼出用: 標準 4 タグを登録 (idempotent)。main.py init で呼ぶ。

    段階11-A: STANDARD_TAGS に `reflect_section` があれば register_tag kwarg
    経由で伝播。standard も dynamic も同じ登録経路 (config=function 哲学整合)。
    """
    _load_from_disk()
    for name, cfg in STANDARD_TAGS.items():
        try:
            register_tag(
                name,
                learning_rules=cfg["learning_rules"],
                display_format=cfg["display_format"],
                origin="standard",
                reflect_section=cfg.get("reflect_section"),
            )
        except ValueError:
            pass


def _reset_for_testing(registry_file: Optional[Path] = None) -> None:
    """テスト専用: in-memory クリア + 永続化ファイル差し替え。"""
    global _REGISTERED, _LOADED, _REGISTRY_FILE
    _REGISTERED = {}
    _LOADED = False
    if registry_file is not None:
        _REGISTRY_FILE = Path(registry_file)
