"""ツール定義・段階解放テーブル"""
from tools.builtin import _list_files, _read_file, _write_file, _update_self
from tools.web import _web_search, _fetch_url
from tools.x_tools import _x_timeline, _x_search, _x_get_notifications, _x_post, _x_reply, _x_quote, _x_like
from tools.elyth_tools import _elyth_post, _elyth_reply, _elyth_timeline, _elyth_notifications, _elyth_like, _elyth_follow, _elyth_info
from tools.memory_tool import _search_memory
from tools.sandbox import _create_tool, _exec_code, _self_modify, _run_ai_tool, AI_CREATED_TOOLS, _DANGEROUS_PATTERNS

TOOLS = {
    "list_files":   {"desc": "ディレクトリの一覧を取得。引数: path=相対パス", "func": lambda args: _list_files(args.get("path", "."))},
    "read_file":    {"desc": "ファイルの内容を読み取る。引数: path=ファイルパス [offset=行番号 limit=行数]", "func": lambda args: _read_file(args.get("path", ""), int(args.get("offset", "0") or "0"), int(args.get("limit", "0") or "0") or None)},
    "write_file":   {"desc": "ファイルに書き込む（sandbox/以下のみ）。引数: path=ファイルパス content=内容", "func": lambda args: _write_file(args.get("path", ""), args.get("content", ""))},
    "update_self":  {"desc": "自己モデルを更新する。引数: key=キー名 value=値", "func": lambda args: _update_self(args.get("key", ""), args.get("value", ""))},
    "wait":         {"desc": "外部世界に変化を与えない待機", "func": lambda args: "[wait]\n待機"},
    "web_search":   {"desc": "Brave APIでWeb検索。引数: query=検索キーワード [max_results=件数]", "func": _web_search},
    "fetch_url":    {"desc": "URLの本文を取得（Jina経由）。引数: url=URL", "func": _fetch_url},
    "x_timeline":   {"desc": "Xのタイムライン取得。引数: [count=件数]", "func": _x_timeline},
    "x_search":     {"desc": "Xでキーワード検索。引数: query=キーワード [count=件数]", "func": _x_search},
    "x_get_notifications": {"desc": "Xの通知一覧取得", "func": _x_get_notifications},
    "x_post":       {"desc": "Xに投稿（Human-in-the-loop）。引数: text=投稿内容（140字以内）", "func": _x_post},
    "x_reply":      {"desc": "Xのツイートに返信（Human-in-the-loop）。引数: tweet_url= text=", "func": _x_reply},
    "x_quote":      {"desc": "Xのツイートを引用投稿（Human-in-the-loop）。引数: tweet_url= text=", "func": _x_quote},
    "x_like":       {"desc": "Xのツイートにいいね（Human-in-the-loop）。引数: tweet_url=", "func": _x_like},
    "elyth_post":   {"desc": "Elythに投稿。引数: content=（500字以内）", "func": _elyth_post},
    "elyth_reply":  {"desc": "Elythに返信。引数: content= reply_to_id=", "func": _elyth_reply},
    "elyth_timeline": {"desc": "Elythタイムライン取得。引数: [limit=件数]", "func": _elyth_timeline},
    "elyth_notifications": {"desc": "Elyth通知取得。引数: [limit=件数]", "func": _elyth_notifications},
    "elyth_like":   {"desc": "Elythにいいね。引数: post_id=", "func": _elyth_like},
    "elyth_follow": {"desc": "ElythのAITuberをフォロー。引数: aituber_id=", "func": _elyth_follow},
    "elyth_info":   {"desc": "Elythの総合情報取得", "func": _elyth_info},
    "search_memory": {"desc": "過去の記憶をベクトル/ID検索。引数: query=検索キーワード [max_results=件数]", "func": _search_memory},
    "create_tool":  {"desc": "AI製ツールを登録（Human-in-the-loop）。引数: name=ツール名 code=Pythonコード（またはfile=sandbox/tools/xxx.py）", "func": _create_tool},
    "exec_code":    {"desc": "sandbox/内のPythonファイルを実行（Human-in-the-loop）。引数: file=sandbox/xxx.py（またはcode=インラインコード）intent=実行目的", "func": _exec_code},
    "self_modify":  {"desc": "自分自身のファイルを変更する（Human-in-the-loop）。引数: path=対象ファイル(pref.json/main.py) [全文置換: content=新しい内容全文] [部分置換: old=変更前の文字列 new=変更後の文字列] intent=変更目的", "func": lambda args: _self_modify(args)},
}

# === ツール段階解放テーブル ===
_LV3_TOOLS = set(TOOLS.keys()) - {"create_tool", "exec_code", "self_modify"}
LEVEL_TOOLS = {
    0: {"list_files", "read_file", "wait", "update_self"},
    1: {"list_files", "read_file", "wait", "update_self", "write_file", "search_memory"},
    2: {"list_files", "read_file", "wait", "update_self", "write_file", "search_memory", "web_search", "fetch_url"},
    3: _LV3_TOOLS,
    4: _LV3_TOOLS | {"create_tool"},
    5: set(TOOLS.keys()) - {"self_modify"},
    6: set(TOOLS.keys()),
}
