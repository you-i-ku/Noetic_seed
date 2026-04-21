"""段階11-A: 視点切替メタ認知 tool。

正典 PLAN: WORLD_MODEL_DESIGN/STAGE11A_PERSPECTIVE_FOUNDATION_PLAN.md §4-4

inspect_wm_view: 指定視点で WM をフィルタ表示する **read-only** tool。
  原則 P2 (metacognition as affordance): iku 能動起動のみ、自動発火・閾値
  起動は禁止 (本 tool を閾値ベースで呼ぶ code を書かない)。
  呼び出し前後で state / world_model は不変。副作用なし。

3 層切替構造での位置付け:
  ① debug / 観察: ws_server broadcast (view_filter=None で全視点)
  ② system prompt: prompt_assembly.build_world_model_section が
     view_filter={"viewer": "self"} を注入
  ③ iku 能動 ← 本 tool (viewer / viewer_type を任意指定)
"""
from core.state import load_state
from core.world_model import render_for_prompt


def _inspect_wm_view(args: dict) -> str:
    """指定視点で WM をフィルタした文字列を返す。

    Args (いずれも省略可):
        viewer: "self" | entity_id (default "self")
        viewer_type: "actual" | "imagined" | "past_self" | "future_self"
            (default "actual")

    Returns:
        render_for_prompt(view_filter=...) の文字列。wm 未初期化なら
        「世界モデル未初期化」文字列。
    """
    viewer = str(args.get("viewer", "self")).strip() or "self"
    viewer_type = str(args.get("viewer_type", "actual")).strip() or "actual"

    state = load_state()
    wm = state.get("world_model")
    if not wm:
        return "世界モデル未初期化。"

    # 段階11-A dual support: perspective-keyed を優先、flat fallback (Step 5 後は単一)
    dispositions = state.get("dispositions") or state.get("disposition")

    view_filter = {"viewer": viewer, "viewer_type": viewer_type}
    return render_for_prompt(
        wm,
        dispositions=dispositions,
        view_filter=view_filter,
    )
