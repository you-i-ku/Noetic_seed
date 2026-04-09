"""ツール定義・段階解放テーブル・Function Callingスキーマ"""
from tools.builtin import _list_files, _read_file, _write_file, _update_self
from tools.web import _web_search, _fetch_url
from tools.x_tools import _x_timeline, _x_search, _x_get_notifications, _x_post, _x_reply, _x_quote, _x_like
from tools.elyth_tools import _elyth_post, _elyth_reply, _elyth_like, _elyth_follow, _elyth_info, _elyth_get, _elyth_mark_read
from tools.memory_tool import _tool_memory_store, _tool_memory_update, _tool_memory_forget, _tool_search_memory
from tools.sandbox import _create_tool, _exec_code, _self_modify, _run_ai_tool, AI_CREATED_TOOLS, _DANGEROUS_PATTERNS
from tools.ui_tools import _output_display

TOOLS = {
    "list_files":      {"desc": "ディレクトリ一覧。引数: path=", "func": lambda args: _list_files(args.get("path", "."))},
    "read_file":       {"desc": "ファイル読取。引数: path= [offset= limit=]", "func": lambda args: _read_file(args.get("path", ""), int(args.get("offset", "0") or "0"), int(args.get("limit", "0") or "0") or None)},
    "write_file":      {"desc": "ファイル書込（sandbox/のみ）。引数: path= content=", "func": lambda args: _write_file(args.get("path", ""), args.get("content", ""))},
    "update_self":     {"desc": "自己モデル更新。引数: key= value=", "func": lambda args: _update_self(args.get("key", ""), args.get("value", ""))},
    "wait":            {"desc": "待機", "func": lambda args: "[wait]"},
    "web_search":      {"desc": "Web検索。引数: query= [max_results=]", "func": _web_search},
    "fetch_url":       {"desc": "URL取得。引数: url=", "func": _fetch_url},
    "output_display":  {"desc": "モニター端末の所有者に直接メッセージ（Elythとは別の相手）。引数: content=", "func": _output_display},
    # 記憶操作（A-Mem方式: AI自律管理）
    "memory_store":    {"desc": "記憶を保存。引数: network=world/experience/opinion/entity content= [confidence=] [entity_name=]", "func": _tool_memory_store},
    "memory_update":   {"desc": "記憶を更新。引数: memory_id= [content=] [confidence=]", "func": _tool_memory_update},
    "memory_forget":   {"desc": "記憶を削除。引数: memory_id=", "func": _tool_memory_forget},
    "search_memory":   {"desc": "記憶を検索。引数: query= [networks=world,opinion,...] [max_results=]", "func": _tool_search_memory},
    # SNS
    "x_timeline":      {"desc": "Xタイムライン取得", "func": _x_timeline},
    "x_search":        {"desc": "X検索。引数: query=", "func": _x_search},
    "x_get_notifications": {"desc": "X通知取得", "func": _x_get_notifications},
    "x_post":          {"desc": "X投稿（Human-in-the-loop）。引数: text=", "func": _x_post},
    "x_reply":         {"desc": "Xリプ（Human-in-the-loop）。引数: tweet_url= text=", "func": _x_reply},
    "x_quote":         {"desc": "X引用（Human-in-the-loop）。引数: tweet_url= text=", "func": _x_quote},
    "x_like":          {"desc": "Xいいね（Human-in-the-loop）。引数: tweet_url=", "func": _x_like},
    "elyth_post":      {"desc": "Elyth（公開SNS）に投稿。引数: content=", "func": _elyth_post},
    "elyth_reply":     {"desc": "Elythリプライ。引数: content= reply_to_id=", "func": _elyth_reply},
    "elyth_like":      {"desc": "Elythいいね/取消。引数: post_id= [unlike=true]", "func": _elyth_like},
    "elyth_follow":    {"desc": "Elythフォロー/解除。引数: aituber_id= [unfollow=true]", "func": _elyth_follow},
    "elyth_info":      {"desc": "Elyth情報取得。引数: [section=notifications/timeline/...] [limit=]", "func": _elyth_info},
    "elyth_get":       {"desc": "Elythデータ取得。引数: type=my_posts/thread/profile [post_id=] [handle=]", "func": _elyth_get},
    "elyth_mark_read": {"desc": "Elyth通知既読。引数: notification_ids=id1,id2,...", "func": _elyth_mark_read},
    # 上級
    "create_tool":     {"desc": "AI製ツール登録（Human-in-the-loop）。引数: name= code=", "func": _create_tool},
    "exec_code":       {"desc": "コード実行（Human-in-the-loop）。引数: file= intent=", "func": _exec_code},
    "self_modify":     {"desc": "自己改変（Human-in-the-loop）。引数: path= [content=] [old= new=] intent=", "func": lambda args: _self_modify(args)},
}

# === ツール段階解放テーブル ===
_LV3_TOOLS = set(TOOLS.keys()) - {"create_tool", "exec_code", "self_modify"}
LEVEL_TOOLS = {
    0: {"list_files", "read_file", "wait", "update_self", "output_display"},
    1: {"list_files", "read_file", "wait", "update_self", "write_file", "search_memory", "memory_store", "output_display"},
    2: {"list_files", "read_file", "wait", "update_self", "write_file", "search_memory", "memory_store", "memory_update", "memory_forget", "web_search", "fetch_url", "output_display"},
    3: _LV3_TOOLS,
    4: _LV3_TOOLS | {"create_tool"},
    5: set(TOOLS.keys()) - {"self_modify"},
    6: set(TOOLS.keys()),
}
