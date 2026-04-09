"""ツールパース・候補パース・計画パース"""
import re


def _extract_json_args(args_str: str) -> tuple:
    """JSON形式の値（{...}や[...]）を持つキーを抽出する。"""
    json_args = {}
    remaining = args_str
    json_key_pattern = re.compile(r'(\w+)=([{[])')

    while True:
        m = json_key_pattern.search(remaining)
        if not m:
            break

        key = m.group(1)
        opener = m.group(2)
        closer = '}' if opener == '{' else ']'
        start_pos = m.start(2)

        depth = 0
        in_str = False
        esc = False
        end_pos = -1

        for i in range(start_pos, len(remaining)):
            ch = remaining[i]
            if esc:
                esc = False
                continue
            if ch == '\\' and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break

        if end_pos == -1:
            break

        json_args[key] = remaining[start_pos:end_pos]
        remaining = remaining[:m.start()] + remaining[end_pos:]

    return json_args, remaining


def _parse_args(args_str: str) -> dict:
    """引数文字列をパースして辞書を返す。クォート付きとクォートなしの混在に対応。"""
    args = {}
    if not args_str:
        return args

    quoted = list(re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', args_str, re.DOTALL))
    # フォールバック: 閉じ引用符がないケース（LLMが閉じ忘れ）
    if not quoted:
        quoted = list(re.finditer(r'(\w+)="((?:[^"\\]|\\.)*?)(?:"|$)', args_str, re.DOTALL))
    if quoted:
        for part in quoted:
            val = part.group(2).replace('\\"', '"')
            val = val.replace('\\n', '\n').replace('\\t', '\t')
            args[part.group(1)] = val

        remaining = args_str
        for part in quoted:
            remaining = remaining.replace(part.group(0), "")
        for part in re.finditer(r'(\w+)=([^\s"]+)', remaining):
            if part.group(1) not in args:
                args[part.group(1)] = part.group(2)
    else:
        json_args, remaining = _extract_json_args(args_str)
        if json_args:
            args.update(json_args)
            key_positions = list(re.finditer(r'(?:^|[\s\[])(\w+)=', remaining))
            if len(key_positions) >= 2:
                for i, kp in enumerate(key_positions):
                    k = kp.group(1)
                    val_start = kp.end()
                    val_end = key_positions[i + 1].start() if i + 1 < len(key_positions) else len(remaining)
                    if k not in args:
                        args[k] = remaining[val_start:val_end].strip().rstrip(']')
            elif key_positions:
                single = re.match(r'\s*(\w+)=(.*)', remaining, re.DOTALL)
                if single and single.group(1) not in args:
                    args[single.group(1)] = single.group(2).strip()
        else:
            key_positions = list(re.finditer(r'(?:^|[\s\[])(\w+)=', args_str))
            if len(key_positions) >= 2:
                for i, kp in enumerate(key_positions):
                    key = kp.group(1)
                    val_start = kp.end()
                    val_end = key_positions[i + 1].start() if i + 1 < len(key_positions) else len(args_str)
                    args[key] = args_str[val_start:val_end].strip().rstrip(']')
            elif key_positions:
                single = re.match(r'(\w+)=(.*)', args_str, re.DOTALL)
                if single:
                    args[single.group(1)] = single.group(2).strip()
            else:
                if args_str.strip():
                    args["__parse_failed__"] = args_str.strip()

    return args


_parse_args_fn = _parse_args


def _extract_tool_blocks(text: str, tool_names: set) -> list[tuple[str, str]]:
    """[TOOL:name ...] をブラケット深さカウントで全件抽出。[(name, args_str), ...]
    content= 内の ] に誤反応しない。"""
    results = []
    i = 0
    while i < len(text):
        bracket_pos = text.find('[TOOL:', i)
        if bracket_pos == -1:
            break
        after = bracket_pos + len('[TOOL:')
        while after < len(text) and text[after] == ' ':
            after += 1
        name_start = after
        while after < len(text) and text[after] not in (' ', '\t', '\n', ']'):
            after += 1
        name = text[name_start:after]
        if name not in tool_names:
            i = bracket_pos + 1
            continue
        depth = 1
        j = after
        in_quote = False
        while j < len(text) and depth > 0:
            ch = text[j]
            if in_quote:
                if ch == '\\':
                    j += 1
                elif ch == '"':
                    in_quote = False
            else:
                if ch == '"':
                    in_quote = True
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
            j += 1
        # フォールバック: 閉じ ] が見つからない場合
        if depth > 0:
            j2 = after
            found = False
            while j2 < len(text):
                if text[j2] == ']':
                    depth = 0
                    j = j2 + 1
                    found = True
                    break
                j2 += 1
            if not found:
                j = len(text)
                depth = 0
        if depth == 0:
            args_str = text[after:j - 1].strip() if j > after else ""
            results.append((name, args_str))
        i = j
    return results


def parse_tool_calls(text: str, tool_names: set) -> list:
    """[TOOL:名前 引数=値 ...]を全件検出してリストで返す。[(name, args), ...]"""
    text = re.sub(
        r'"""(.*?)"""',
        lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
        text, flags=re.DOTALL
    )
    results = []
    for name, args_str in _extract_tool_blocks(text, tool_names):
        args = _parse_args_fn(args_str) if args_str else {}
        results.append((name, args))

    # フォールバック1: [TOOL:...]なしで行頭「ツール名 key=value」形式を検出
    if not results:
        names_list = sorted(tool_names, key=len, reverse=True)
        for line in text.strip().splitlines():
            line = line.strip()
            for name in names_list:
                if line.startswith(name + ' ') or line.startswith(name + '\t') or line == name:
                    args_str = line[len(name):].strip()
                    args = _parse_args_fn(args_str) if args_str else {}
                    results.append((name, args))
                    break
            if results:
                break

    # フォールバック2: 行内に「ツール名 key=value」パターン（→の後など）
    if not results:
        names_list = sorted(tool_names, key=len, reverse=True)
        for line in text.strip().splitlines():
            line = line.strip()
            for name in names_list:
                # 行内のどこかに「ツール名 key=」があれば拾う
                pattern = re.compile(rf'\b({re.escape(name)})\s+(\w+=)')
                m = pattern.search(line)
                if m:
                    args_start = m.start(2)
                    args_str = line[args_start:].strip()
                    args = _parse_args_fn(args_str) if args_str else {}
                    results.append((name, args))
                    break
            if results:
                break

    return results


def parse_candidates(text: str, allowed_tools: set) -> list:
    """LLM①のリストから候補を抽出。「1. [理由] -> ツール名」形式に対応。"""
    candidates = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        if "->" in line or "→" in line:
            parts = re.split(r'->|→', line)
            tool_part = parts[-1].strip()
            reason_part = parts[0].strip()
        else:
            cleaned = re.sub(r'^[\d]+[.:)\s]+', '', line).strip()
            cleaned = re.sub(r'^[-*]\s*', '', cleaned).strip()
            parts = cleaned.split()
            tool_part = parts[0] if parts else ""
            reason_part = cleaned

        # +で分割してから各ツール名の丸括弧以降を除去、ASCII英数字+_のみに制限
        raw_tools = []
        for t in tool_part.split('+'):
            t_clean = re.split(r'[（(]', t.strip())[0].strip()
            t_clean = re.sub(r'[^a-zA-Z0-9_]', '', t_clean)
            if t_clean:
                raw_tools.append(t_clean)
        valid_tools = [t for t in raw_tools if t in allowed_tools]

        if not valid_tools:
            for t in allowed_tools:
                if t in line:
                    valid_tools = [t]
                    break

        reason = re.sub(r'^[\d]+[.:)\s]+', '', reason_part).strip()
        reason = re.sub(r'^[-*]\s*', '', reason).strip()
        if reason.startswith('[') and reason.endswith(']'):
            reason = reason[1:-1].strip()

        chain_key = "+".join(valid_tools)
        if valid_tools and chain_key not in ["+".join(c["tools"]) for c in candidates]:
            candidates.append({"tool": valid_tools[0], "tools": valid_tools, "reason": reason})

    if not candidates:
        for t in allowed_tools:
            candidates.append({"tool": t, "reason": "（フォールバック）"})
    return candidates


def parse_plan(text: str):
    """[PLAN:goal=目標 steps=ステップ1|ステップ2]をパース"""
    m = re.search(r'\[PLAN:((?:[^\]"]|"(?:[^"\\]|\\.)*")*)\]', text, re.DOTALL)
    if not m:
        return None
    args = _parse_args_fn(m.group(1).strip())
    goal = args.get("goal", "").strip()
    steps_raw = args.get("steps", "")
    steps = [s.strip() for s in steps_raw.split("|") if s.strip()] if steps_raw else []
    if not goal:
        return None
    return {"goal": goal, "steps": steps, "current": 0}
