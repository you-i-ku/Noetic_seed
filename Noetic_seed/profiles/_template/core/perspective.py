"""core/perspective.py — 段階11-A 視点タグ基盤 (Perspective Foundation, v1)

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §3
原則 P1 (drift is not developer error): 本ファイルに drift 検知 / 自動 repair は
  実装しない (視点は記述のみ、強制修正なし)。
原則 P2 (metacognition as affordance): 本ファイルは helper のみで発火機構なし。
  起動は呼び出し側の選択に委ねる。

責務:
  - Perspective schema (TypedDict) の定義
  - Helper 8 関数の提供 (認知 unit に perspective 属性を付与するビルディングブロック)

非責務 (明確に禁止):
  - drift 検知 / 視点強制修正
  - 視点の自動発火 / 閾値起動
  - LLM prompt 経由での視点命令
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, TypedDict


class Perspective(TypedDict, total=False):
    """視点 metadata. 全 field 任意 (total=False)。

    Field 意味:
      viewer: "self" | entity_id (例 "ent_yuu") | "imagined_<label>"
      viewer_type: "actual" | "imagined" | "past_self" | "future_self"
      view_time: ISO8601 UTC 秒精度 "YYYY-MM-DDTHH:MM:SSZ" | None
      confidence: 0.0-1.0 | None (self/actual は None、他視点/仮想は推奨 0.5)
      nested: 再帰 Perspective (深さ無制限、iku が想像するゆうの視点等)
    """
    viewer: str
    viewer_type: str
    view_time: Optional[str]
    confidence: Optional[float]
    nested: Optional[dict]


def _now_iso_utc_seconds() -> str:
    """現在時刻を ISO8601 UTC 秒精度 "YYYY-MM-DDTHH:MM:SSZ" で返す。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_perspective(
    viewer: str = "self",
    viewer_type: str = "actual",
    view_time: Optional[str] = None,
    confidence: Optional[float] = None,
    nested: Optional[dict] = None,
) -> Perspective:
    """Perspective 生成。

    view_time=None 時は現在時刻 UTC 秒精度を自動付与。
    confidence / nested は None の場合キー自体を省略 (type safe、sparse schema)。
    """
    if view_time is None:
        view_time = _now_iso_utc_seconds()
    p: Perspective = {
        "viewer": viewer,
        "viewer_type": viewer_type,
        "view_time": view_time,
    }
    if confidence is not None:
        p["confidence"] = confidence
    if nested is not None:
        p["nested"] = nested
    return p


def default_self_perspective() -> Perspective:
    """既存データ backward compat 用: perspective 欠落 → self/actual/現在時刻。

    memory/*.jsonl の既存 entry (perspective 属性なし) を view 層で読む時、
    本関数の戻り値を default として扱う。
    """
    return make_perspective(viewer="self", viewer_type="actual")


def is_self_view(p: Perspective) -> bool:
    """self 自身が actual に観測している視点かどうか。"""
    return p.get("viewer") == "self" and p.get("viewer_type") == "actual"


def is_actual_view(p: Perspective) -> bool:
    """actual (非仮想) な視点かどうか。imagined/past_self/future_self は False。"""
    return p.get("viewer_type") == "actual"


def is_nested(p: Perspective) -> bool:
    """ネスト視点 (A が見ている B の視点) を持つかどうか。"""
    return p.get("nested") is not None


def perspective_depth(p: Perspective) -> int:
    """ネスト深さ。0=フラット、1=A から見た B、2=A から見た B から見た C、...

    構造上は無制限 (実行上は 3-4 段で自然停止見込み)。
    """
    depth = 0
    current: dict = p
    while is_nested(current):
        depth += 1
        current = current["nested"]
    return depth


def perspective_tag_str(p: Perspective) -> str:
    """prompt 表示用の簡易文字列。ネストは "A←B" 形式 (A の中の B)。

    例:
      self/actual                    → "[self]"
      imagined + viewer=fear_future  → "[imagined:fear_future]"
      past_self + view_time=X        → "[past_self@X]"
      future_self + view_time=X      → "[future_self@X]"
      other + actual                 → "[ent_yuu view]"
      ネスト (self ← ent_yuu view)    → "[self]←[ent_yuu view]"
    """
    viewer = p.get("viewer", "?")
    vtype = p.get("viewer_type", "?")
    if viewer == "self" and vtype == "actual":
        base = "[self]"
    elif vtype == "imagined":
        base = f"[imagined:{viewer}]"
    elif vtype == "past_self":
        base = f"[past_self@{p.get('view_time', '?')}]"
    elif vtype == "future_self":
        base = f"[future_self@{p.get('view_time', '?')}]"
    else:
        base = f"[{viewer} view]"
    if is_nested(p):
        return base + "←" + perspective_tag_str(p["nested"])
    return base


def perspective_key_str(p: Perspective) -> str:
    """state["dispositions"] の key として使う安定文字列。

    perspective_tag_str との違い:
      ID 的 stable (view_time 等は含めない、同 viewer+type なら同 key)。

    例:
      self/actual                → "self"
      ent_yuu/actual             → "attributed:ent_yuu"
      fear_future/imagined       → "imagined:fear_future"
      past_me/past_self          → "past_self:past_me"
      future_me/future_self      → "future_self:future_me"
    """
    viewer = p.get("viewer", "?")
    vtype = p.get("viewer_type", "actual")
    if viewer == "self" and vtype == "actual":
        return "self"
    if vtype == "actual":
        return f"attributed:{viewer}"
    return f"{vtype}:{viewer}"
