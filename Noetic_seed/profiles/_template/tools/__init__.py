"""ツール定義・段階解放テーブル"""
from tools.builtin import _update_self, _wait_or_dismiss, _view_image, _listen_audio
from tools.x_tools import _x_timeline, _x_search, _x_get_notifications, _x_post, _x_reply, _x_quote, _x_like
from tools.elyth_tools import _elyth_post, _elyth_reply, _elyth_like, _elyth_follow, _elyth_info, _elyth_get, _elyth_mark_read
from tools.memory_tool import _search_memory, _tool_memory_store, _tool_memory_update, _tool_memory_forget, _tool_search_memory
from tools.sandbox import _create_tool, _exec_code, _self_modify, _run_ai_tool, AI_CREATED_TOOLS, _DANGEROUS_PATTERNS
from tools.ui_tools import _output_display
from tools.device_tools import _camera_stream, _camera_stream_stop, _screen_peek, _mic_record
from tools.http_tool import http_request
from tools.secret_tools import secret_read, secret_write
from tools.auth_tools import auth_profile_info
from tools.memory_graph_tool import _memory_graph

TOOLS = {
    "update_self":  {"desc": "自己モデルを更新する。引数: key=キー名 value=値", "func": lambda args: _update_self(args.get("key", ""), args.get("value", ""))},
    "wait":         {"desc": "待機。dismiss=pending_idで未対応事項を明示的に却下できる", "func": _wait_or_dismiss},
    "x_timeline":   {"desc": "Xのタイムライン取得。引数: [count=件数] [tab=following/recommend デフォルトfollowing]", "func": _x_timeline},
    "x_search":     {"desc": "Xでキーワード検索。引数: query=キーワード [count=件数]", "func": _x_search},
    "x_get_notifications": {"desc": "Xの通知一覧取得", "func": _x_get_notifications},
    "x_post":       {"desc": "Xに投稿（Human-in-the-loop）。引数: text=投稿内容（140字以内）", "func": _x_post},
    "x_reply":      {"desc": "Xのツイートに返信（Human-in-the-loop）。引数: tweet_url= text=", "func": _x_reply},
    "x_quote":      {"desc": "Xのツイートを引用投稿（Human-in-the-loop）。引数: tweet_url= text=", "func": _x_quote},
    "x_like":       {"desc": "Xのツイートにいいね（Human-in-the-loop）。引数: tweet_url=", "func": _x_like},
    "elyth_post":   {"desc": "Elyth（公開SNS）に投稿。不特定多数のAITuberに公開される。引数: content=（500字以内）", "func": _elyth_post},
    "elyth_reply":  {"desc": "Elythの投稿に返信。引数: content= reply_to_id=", "func": _elyth_reply},
    "elyth_like":   {"desc": "Elythにいいね/取消。引数: post_id= [unlike=true]", "func": _elyth_like},
    "elyth_follow": {"desc": "ElythのAITuberをフォロー/解除。引数: aituber_id= [unfollow=true]", "func": _elyth_follow},
    "elyth_info":   {"desc": "Elyth総合情報取得。引数: [section=notifications/timeline/trends/...] [limit=件数]", "func": _elyth_info},
    "elyth_get":    {"desc": "Elythデータ取得。引数: type=my_posts/thread/profile [post_id=] [handle=] [limit=]", "func": _elyth_get},
    "elyth_mark_read": {"desc": "Elyth通知を既読化。引数: notification_ids=id1,id2,...", "func": _elyth_mark_read},
    "search_memory": {"desc": "過去の記憶をベクトル/ID検索。引数: query=検索キーワード [max_results=件数]", "func": _search_memory},
    "memory_store":  {"desc": "記憶を保存。引数: network=world/experience/opinion/entity content= [confidence=] [entity_name=]", "func": _tool_memory_store},
    "memory_update": {"desc": "記憶を更新。引数: memory_id= [content=] [confidence=]", "func": _tool_memory_update},
    "memory_forget": {"desc": "記憶を削除。引数: memory_id=", "func": _tool_memory_forget},
    "reflect":       {"desc": "自己を内省し、学びや気づきを記憶に保存する（引数なし）", "func": lambda args: "[reflect] main.pyで初期化されます"},
    "create_tool":  {"desc": "AI製ツールを登録（Human-in-the-loop）。引数: name=ツール名 code=Pythonコード（またはfile=sandbox/tools/xxx.py）", "func": _create_tool},
    "exec_code":    {"desc": "sandbox/内のPythonファイルを実行（Human-in-the-loop）。引数: file=sandbox/xxx.py（またはcode=インラインコード）intent=実行目的 [message=外部への説明]", "func": _exec_code},
    "self_modify":  {"desc": "自分自身のファイルを変更する（Human-in-the-loop）。引数: path=対象ファイル(pref.json/main.py) [全文置換: content=新しい内容全文] [部分置換: old=変更前の文字列 new=変更後の文字列] intent=変更目的 [message=外部への説明]", "func": lambda args: _self_modify(args)},
    "output_display":    {"desc": "発話を channel 指定で届ける。送信先 channel は WM.channels を観察して決定、受信 channel に対応させて返す (log entry の [channel=X] header の X と同じ値を channel 引数に指定すると対応した相手に返る)。引数: content=メッセージ channel=送信先 channel id (必須)", "func": _output_display},
    "camera_stream":     {"desc": "端末のカメラ経由で連続撮影を非同期に開始する。最初のフレームは実行時に視覚入力（描写付きで返る）。後続フレームはローリング最新5枚として次サイクル以降の視覚入力に到着。観察中も他ツールを並行実行可能（read_file/reflect/memory_store等）。引数: [facing=back/front] [frames=枚数 0=無制限/1-30 default=5] [interval_sec=間隔 0.3-5.0 default=1.0] [message=外部への撮影依頼理由]。frames=0 は camera_stream_stop で明示終了（Android側絶対上限10分）、frames=1 は単発。承認必須。", "func": _camera_stream},
    "camera_stream_stop": {"desc": "アクティブな camera_stream / screen_peek を停止する。観察対象を把握した後、リソース節約のために呼ぶ。引数なし", "func": _camera_stream_stop},
    "screen_peek":       {"desc": "端末のスクリーンを非同期にキャプチャする（camera_streamの画面版）。最初のフレームは実行時に視覚入力、後続は次サイクル以降の視覚入力に到着（ローリング最新5枚）。観察中も他ツールを並行実行可能。引数: [frames=枚数 0=無制限/1-30 default=5] [interval_sec=間隔 0.3-5.0 default=1.0] [message=外部への画面キャプチャ依頼理由]。frames=0 は camera_stream_stop で明示終了（Android側絶対上限10分）。毎セッション MediaProjection 許可ダイアログが出る。承認必須。", "func": _screen_peek},
    "mic_record":        {"desc": "端末のマイク経由で短時間録音し、音声書き起こし（faster-whisper）+ 環境音分類（YAMNet 521クラス）の両方を結果として返す。完全同期実行（view_imageと同じ）。引数: [duration_sec=秒 1.0-30.0 default=5.0] [language=ja/en等 未指定なら自動検出] [message=外部への録音依頼理由]。承認必須。", "func": _mic_record},
    "view_image":   {"desc": "画像を同期で認識し、描写を結果として返す。camera_streamの結果を見直したり、任意の画像を能動的に注視する。intent=目的を指定するとその観点で描写される。引数: path=画像パス（プロファイル内相対パスまたは http(s) URL、jpg/png/webp）", "func": _view_image},
    "listen_audio": {"desc": "既存の音声ファイルまたは URL から音声を聞きに行く。view_image の音声版。同期実行で speech 書き起こし + ambient 環境音分類を返す。引数: path=音声パス（プロファイル内相対パスまたは http(s) URL、wav/mp3/m4a/ogg/flac/aac/webm 対応） [language=ja/en等 未指定なら自動]", "func": _listen_audio},
    "http_request": {"desc": "任意URLに HTTP リクエストを送る汎用ツール。GET以外（POST/PUT/DELETE/PATCH）は承認必要。引数: url=URL [method=GET/POST/...] [headers=JSON] [params=JSON] [body=JSON文字列 or 辞書] [auth=auth_profile名] [timeout=秒 default=60]", "func": http_request},
    "secret_read":  {"desc": "sandbox/secrets/ に保存された秘密情報を読む（承認不要）。引数: name=secret名【name= を使う。read_file の path= と混同しないこと】", "func": secret_read},
    "secret_write": {"desc": "sandbox/secrets/ に秘密情報を書き込む（承認必要）。引数: name=secret名 content=内容 intent=目的 [message=外部への説明]", "func": secret_write},
    "auth_profile_info": {"desc": "認証プロファイルのメタ情報を取得。name未指定で一覧、name指定で詳細（機密フィールドtoken/key/secret等は隠される）。引数: [name=プロファイル名]", "func": auth_profile_info},
    "memory_graph": {"desc": "memory entry と self を node とした graph 構造を JSON で返す。引数: [view=ego (default)] [depth=2 (default)]", "func": _memory_graph},
}

# === ツール段階解放テーブル ===
# H-2 C.1 (2026-04-18): web_search/fetch_url は claw の WebSearch/WebFetch に移行。
# H-2 C.4 Session A (2026-04-18): list_files/read_file/write_file は claw ネイティブ
# (glob_search / read_file / write_file) に移行。Noetic ガードは file_access_guard
# hook で再現。legacy handler は bridge から削除。
_CLAW_FILE_OPS = {"read_file", "write_file", "glob_search", "WebSearch", "WebFetch"}
_LV3_TOOLS = (
    set(TOOLS.keys()) - {"create_tool", "exec_code", "self_modify"}
    | _CLAW_FILE_OPS
)
LEVEL_TOOLS = {
    0: {"glob_search", "read_file", "wait", "update_self", "output_display", "view_image", "listen_audio", "bash"},
    1: {"glob_search", "read_file", "wait", "update_self", "write_file", "search_memory", "memory_store", "memory_graph", "reflect", "output_display", "view_image", "listen_audio", "bash"},
    2: {"glob_search", "read_file", "wait", "update_self", "write_file", "search_memory", "memory_store", "memory_update", "memory_forget", "memory_graph", "reflect", "WebSearch", "WebFetch", "output_display", "view_image", "listen_audio", "bash"},
    3: _LV3_TOOLS | {"bash"},
    4: _LV3_TOOLS | {"create_tool", "bash"},
    5: set(TOOLS.keys()) - {"self_modify"} | _CLAW_FILE_OPS | {"bash"},
    6: set(TOOLS.keys()) | _CLAW_FILE_OPS | {"bash"},
}
