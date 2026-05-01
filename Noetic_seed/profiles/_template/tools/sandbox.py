"""AI製ツール管理・コード実行 (旧 self_modify は段階12 Step 7 で撤廃、write_file に吸収)"""
import threading
from core.config import BASE_DIR, SANDBOX_DIR, SANDBOX_TOOLS_DIR
from core.state import load_state, save_state
from core.ws_server import request_approval

AI_CREATED_TOOLS: dict = {}  # name -> func
_AI_TOOL_TIMEOUT = 10  # 秒
_DANGEROUS_PATTERNS = ["os.system", "subprocess", "__import__", "eval(", "exec(", "open(", "__builtins__"]


def _run_ai_tool(func, args: dict) -> str:
    """AI製ツールを実行。タイムアウト・エラーを統一処理。"""
    result_box = [None]
    exc_box = [None]
    def _target():
        try:
            result_box[0] = func(args)
        except Exception as e:
            exc_box[0] = e
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(_AI_TOOL_TIMEOUT)
    if t.is_alive():
        return f"タイムアウト（{_AI_TOOL_TIMEOUT}秒）: 処理が完了しませんでした"
    if exc_box[0] is not None:
        e = exc_box[0]
        return f"{type(e).__name__}: {e}"
    return str(result_box[0]) if result_box[0] is not None else ""


def _create_tool(args: dict) -> str:
    name = args.get("name", "").strip()
    file_path = args.get("file", "").strip()
    inline_code = args.get("code", "").strip()
    desc = args.get("desc", "").strip()
    if not name:
        return "エラー: name= が必要です"
    if not file_path and not inline_code:
        return "エラー: file= または code= が必要です"
    if file_path and inline_code:
        return "エラー: file= と code= は同時に使えません"
    if inline_code:
        SANDBOX_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        file_path = f"sandbox/tools/{name}.py"
        target = BASE_DIR / file_path
        code = f'DESCRIPTION = "{desc}"\n\n{inline_code}' if desc else inline_code
    else:
        target = (BASE_DIR / file_path).resolve()
        if not str(target).startswith(str(SANDBOX_TOOLS_DIR.resolve())):
            return f"エラー: sandbox/tools/ 以下のファイルのみ登録可能です"
        if not target.exists():
            return f"エラー: {file_path} が見つかりません"
        code = target.read_text(encoding="utf-8")
    warns = [p for p in _DANGEROUS_PATTERNS if p in code]
    if warns:
        return f"エラー: 危険パターン検出 {warns}。登録できません"
    print(f"  [create_tool] {name} → {file_path}")
    target.write_text(code, encoding="utf-8")
    state = load_state()
    tc = state.setdefault("tools_created", [])
    if name not in tc:
        tc.append(name)
    save_state(state)
    return f"登録完了: {name} → {file_path}（次サイクルから使用可能）"


def _exec_code(args: dict) -> str:
    import subprocess, sys, tempfile, os
    file_path = args.get("file", "").strip()
    inline = args.get("code", "").strip()
    intent = args.get("intent", "（意図なし）")
    if not file_path and not inline:
        return "エラー: file= または code= が必要です"
    if file_path:
        target = (BASE_DIR / file_path).resolve()
        if not str(target).startswith(str(SANDBOX_DIR.resolve())):
            return "エラー: sandbox/ 以下のファイルのみ実行可能です"
        if not target.exists():
            return f"エラー: {file_path} が見つかりません"
        code = target.read_text(encoding="utf-8")
        run_target = str(target)
        tmp_path = None
    else:
        code = inline
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                          dir=str(SANDBOX_DIR), encoding="utf-8")
        tmp.write(code)
        tmp.close()
        run_target = tmp.name
        tmp_path = tmp.name
    warnings = [p for p in _DANGEROUS_PATTERNS if p in code]
    warn_str = f" ⚠{warnings}" if warnings else ""
    message = args.get("message", "").strip()
    preview_lines = [f"[exec_code] ファイル: {file_path or '(inline)'}{warn_str}"]
    if intent:
        preview_lines.append(f"意図: {intent}")
    if message:
        preview_lines.append(f"メッセージ: {message}")
    preview_lines.append(f"---\n{code[:500]}")
    preview = "\n".join(preview_lines)
    print(f"\n[exec_code 承認待ち] {file_path or '(inline)'}")
    if not request_approval("exec_code", preview):
        if tmp_path:
            os.unlink(tmp_path)
        return "キャンセル: 実行を見送りました"
    try:
        result = subprocess.run(
            [sys.executable, run_target],
            capture_output=True, text=True,
            timeout=_AI_TOOL_TIMEOUT,
            cwd=str(SANDBOX_DIR),
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        output = ""
        if out:
            output += out
        if err:
            output += ("\n" if out else "") + f"[stderr] {err}"
        return (output or "（出力なし）")[:5000]
    except subprocess.TimeoutExpired:
        return f"タイムアウト（{_AI_TOOL_TIMEOUT}秒）: 処理が完了しませんでした"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


