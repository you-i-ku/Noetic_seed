"""AI製ツール管理・コード実行・self_modify"""
import threading
from core.config import BASE_DIR, SANDBOX_DIR, SANDBOX_TOOLS_DIR
from core.state import load_state, save_state

AI_CREATED_TOOLS: dict = {}  # name -> func
_AI_TOOL_TIMEOUT = 10  # 秒
_DANGEROUS_PATTERNS = ["os.system", "subprocess", "__import__", "eval(", "exec(", "open(", "__builtins__"]
_MODIFY_ALLOWED = {"pref.json", "main.py"}


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
    warn_str = f"\n⚠ 危険パターン検出: {warns}" if warns else "\n危険パターン: なし"
    print(f"\n[create_tool 承認待ち]")
    print(f"  ツール名: {name}  説明: {desc or '（説明なし）'}")
    print(f"  ファイル: {file_path}{warn_str}")
    print(f"  --- コード ---")
    print(code[:1000] + ("..." if len(code) > 1000 else ""))
    print(f"  --------------")
    ans = input("  登録しますか？ [y/N]: ").strip().lower()
    if ans != "y":
        return "キャンセル: ツール登録を見送りました"
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
    warn_str = f"\n⚠ 危険パターン検出: {warnings}" if warnings else "\n危険パターン: なし"
    print(f"\n[exec_code 承認待ち]")
    print(f"  AIの意図: {intent}")
    print(f"  実行ファイル: {file_path or '(インラインコード)'}{warn_str}")
    print(f"  --- コード ---")
    print(code[:800] + ("..." if len(code) > 800 else ""))
    print(f"  --------------")
    ans = input("  実行しますか？ [y/N]: ").strip().lower()
    if ans != "y":
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


def _self_modify(args: dict) -> str:
    path = args.get("path", "").strip()
    content = args.get("content", "")
    old = args.get("old", "")
    new = args.get("new", "")
    intent = args.get("intent", "（意図なし）")
    if not path:
        return "エラー: path= が必要です"
    if path not in _MODIFY_ALLOWED:
        return f"エラー: 変更可能なファイルは {sorted(_MODIFY_ALLOWED)} のみです"
    if old and content:
        return "エラー: content= と old=/new= は同時に使えません"
    if not old and not content:
        return "エラー: content=（全文置換）または old=+new=（部分置換）が必要です"
    mode = "partial" if old else "full"
    target = BASE_DIR / path
    current = target.read_text(encoding="utf-8") if target.exists() else ""
    if mode == "partial":
        if old not in current:
            return f"エラー: 指定した old= の文字列がファイル内に見つかりません"
        if current.count(old) > 1:
            return f"エラー: old= の文字列がファイル内に{current.count(old)}箇所あります。より長い文字列で一意に指定してください"
        new_content = current.replace(old, new, 1)
    else:
        new_content = content
    check_target = new if mode == "partial" else content
    if path.endswith(".py"):
        warnings = [p for p in _DANGEROUS_PATTERNS if p in check_target]
        warn_str = f"\n⚠ 危険パターン検出: {warnings}" if warnings else "\n危険パターン: なし"
    else:
        warn_str = ""
    print(f"\n[self_modify 承認待ち]")
    print(f"  対象: {path}  モード: {'部分置換' if mode == 'partial' else '全文置換'}")
    print(f"  AIの意図: {intent}{warn_str}")
    if mode == "partial":
        print(f"  --- 変更前 ---")
        print(old[:400] + ("..." if len(old) > 400 else ""))
        print(f"  --- 変更後 ---")
        print(new[:400] + ("..." if len(new) > 400 else ""))
    else:
        print(f"  --- 変更後の内容（先頭400字）---")
        print(new_content[:400] + ("..." if len(new_content) > 400 else ""))
    print(f"  --------------------------------")
    ans = input("  変更を適用しますか？ [y/N]: ").strip().lower()
    if ans != "y":
        return "キャンセル: 変更を見送りました"
    if path == "main.py":
        backup = target.with_suffix(".py.bak")
        backup.write_text(current, encoding="utf-8")
        print(f"  バックアップ: {backup.name}")
    target.write_text(new_content, encoding="utf-8")
    return f"変更完了: {path}（{'部分置換' if mode == 'partial' else '全文置換'}, {len(new_content)}文字）"
