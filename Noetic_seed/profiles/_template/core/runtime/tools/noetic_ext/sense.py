"""Noetic 身体化観測 tool 群 — view_image / listen_audio / mic_record /
camera_stream / camera_stream_stop / screen_peek。

claw 文法準拠 ToolSpec。handler は legacy (tools/builtin.py, tools/device_tools.py)
を温存。同期 (view/listen) と非同期ハイブリッド (camera/screen) の混在は実装
側に隠蔽され、tool interface は統一。
"""
from core.runtime.permissions import PermissionMode
from core.runtime.registry import ToolRegistry
from core.runtime.tool_schema import ToolSpec


def _approval_props() -> dict:
    return {
        "tool_intent": {
            "type": "string",
            "description": "あなたの内部理由 (1 文、80 字目安)",
        },
        "tool_expected_outcome": {
            "type": "string",
            "description": "期待する結果 (1 文、80 字目安)",
        },
        "message": {
            "type": "string",
            "description": "端末前の協力者への一言 (承認時の依頼理由に使われる)",
        },
    }


_APPROVAL_REQUIRED = ["tool_intent", "tool_expected_outcome", "message"]


def _build_specs(tools_dict: dict) -> list:
    specs = [
        ToolSpec(
            name="view_image",
            description=(
                "画像を同期で認識し、自然言語の描写を返す。"
                "jpg/png/webp、ローカルパス or URL に対応。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "画像のパス (ローカル or URL)",
                    },
                    "intent": {
                        "type": "string",
                        "description": "描写時の着眼点 (省略可)",
                    },
                    **_approval_props(),
                },
                "required": ["path", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=tools_dict["view_image"]["func"],
        ),
        ToolSpec(
            name="listen_audio",
            description=(
                "音声ファイルを同期で聴取し、speech 書き起こしと環境音分類を返す。"
                "wav/mp3/m4a/ogg/flac/aac/webm 対応。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "音声のパス (ローカル or URL)",
                    },
                    "language": {
                        "type": "string",
                        "description": "言語 hint (ja / en 等、省略時は自動検出)",
                    },
                    **_approval_props(),
                },
                "required": ["path", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=tools_dict["listen_audio"]["func"],
        ),
        ToolSpec(
            name="mic_record",
            description=(
                "端末のマイクで同期録音し、speech 書き起こしと環境音分類を返す。"
                "承認必須 (装置作動)。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "duration_sec": {
                        "type": "number",
                        "minimum": 1.0,
                        "maximum": 30.0,
                        "description": "録音時間 (秒、1.0-30.0、default 5.0)",
                    },
                    "language": {
                        "type": "string",
                        "description": "言語 hint (省略時は自動検出)",
                    },
                    **_approval_props(),
                },
                "required": ["duration_sec", *_APPROVAL_REQUIRED],
                "additionalProperties": False,
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=tools_dict["mic_record"]["func"],
        ),
        ToolSpec(
            name="camera_stream",
            description=(
                "端末カメラで連続撮影を非同期に開始する。最初のフレームは同期で "
                "描写取得、後続は rolling buffer に蓄積。承認必須。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "facing": {
                        "type": "string",
                        "enum": ["front", "back"],
                        "description": "カメラ方向 (default back)",
                    },
                    "frames": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 30,
                        "description": "撮影枚数 (0=無制限、1-30、default 5)",
                    },
                    "interval_sec": {
                        "type": "number",
                        "minimum": 0.3,
                        "maximum": 5.0,
                        "description": "撮影間隔 (秒、0.3-5.0、default 1.0)",
                    },
                    **_approval_props(),
                },
                "required": list(_APPROVAL_REQUIRED),
                "additionalProperties": False,
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=tools_dict["camera_stream"]["func"],
        ),
        ToolSpec(
            name="camera_stream_stop",
            description=(
                "アクティブな camera_stream / screen_peek を停止する。"
            ),
            input_schema={
                "type": "object",
                "properties": _approval_props(),
                "required": list(_APPROVAL_REQUIRED),
                "additionalProperties": False,
            },
            required_permission=PermissionMode.READ_ONLY,
            handler=tools_dict["camera_stream_stop"]["func"],
        ),
        ToolSpec(
            name="screen_peek",
            description=(
                "端末のスクリーンを非同期でキャプチャする (camera_stream の画面版)。"
                "MediaProjection 許可ダイアログあり。承認必須。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 30,
                        "description": "キャプチャ枚数 (0=無制限、1-30、default 5)",
                    },
                    "interval_sec": {
                        "type": "number",
                        "minimum": 0.3,
                        "maximum": 5.0,
                        "description": "キャプチャ間隔 (秒、default 1.0)",
                    },
                    **_approval_props(),
                },
                "required": list(_APPROVAL_REQUIRED),
                "additionalProperties": False,
            },
            required_permission=PermissionMode.DANGER_FULL_ACCESS,
            handler=tools_dict["screen_peek"]["func"],
        ),
    ]
    return specs


def register(registry: ToolRegistry, tools_dict: dict) -> int:
    """sense family の 6 tool を registry に登録。"""
    specs = _build_specs(tools_dict)
    for spec in specs:
        registry.register(spec)
    return len(specs)
