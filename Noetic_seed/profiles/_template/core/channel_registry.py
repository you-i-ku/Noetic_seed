"""観察経路ごとの channel spec 生成ロジック。

各判定関数は自身の知識で spec を組み立てて返す。中央辞書は持たない (config = function)。
観察が起きた瞬間、対応する関数が呼ばれ、spec が生まれ、ensure_channel 経由で
WM に登録される。

新しい観察経路の追加は、ここに判定関数を増やす形で行う。
段階6-C では device_input / mcp_client の 2 経路のみ実装。
将来 computer use / skills / url / api 経路が追加される (段階7+)。

設計指針 (STAGE6C_IMPLEMENTATION_PLAN.md v3 準拠):
- World is observed, not given: spec はデータとして事前に存在しない、観察で生まれる
- config = function: 辞書ではなく関数が spec 生成ロジックを所有
- Noetic 主体の判別: channel spec は接続者の自己申告でなく Noetic 側で決定
"""


def channel_from_device_input() -> dict:
    """端末所有者からの入力 ([device_input]) 観察時の channel spec。"""
    return {
        "id": "device",
        "type": "direct",
        "tools_in": ["[device_input]"],
        "tools_out": ["output_display", "camera_stream", "screen_peek",
                      "view_image", "listen_audio", "mic_record"],
    }


def channel_from_mcp_client(client_name: str) -> dict:
    """MCP client 接続時の channel spec。

    既知 client 名は固定 id、未知は mcp_<safe_name> で汎用 channel を生成。
    caller 自己申告でなく Noetic 側でこの関数が spec を決定する。
    """
    name_lower = (client_name or "").lower()
    if "claude" in name_lower:
        return {
            "id": "claude",
            "type": "social",
            "tools_in": ["[claude_input]"],
            "tools_out": ["output_display"],
        }
    # 未知 client: 汎用 mcp channel を動的生成
    safe = name_lower.replace(" ", "_")[:20] or "unknown"
    return {
        "id": f"mcp_{safe}",
        "type": "social",
        "tools_in": [f"[{safe}_input]"],
        "tools_out": ["output_display"],
    }
