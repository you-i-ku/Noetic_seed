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
    """LLM①のリストから候補を抽出。「1. [理由] -> ツール名」形式に対応。

    段階9: 行末尾の「/ predicted_e2: XX」表記で chain 全体 1 組抽出。
    段階10 柱 C: 「/ predicted_ec: 0.YY」併記 (旧形式) にも対応。
    段階10.5 Fix 1: 新形式 tool 直後の「(pe2=XX, pec=0.YY)」で tool 単位の
    個別 predicted_e2/predicted_ec 抽出。旧形式は chain 全体 1 組を chain 内
    各 tool に複製 (後方互換)。

    candidate dict 構造:
        {
            "tool": 先頭 tool 名 (後方互換),
            "tools": [tool 名 list],
            "reason": ...,
            "chain": [
                {"tool": 名, "predicted_e2": int|None, "predicted_ec": float|None},
                ...
            ],
            "prediction": {..., chain[0] 由来}  # chain[0].pe2 が None なら欠落
        }
    """
    candidates = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        # --- 旧形式 (chain 全体 1 組) 抽出 + line から除去 ---
        legacy_pe2 = None
        legacy_pec = None
        legacy_pe2_match = re.search(r'predicted_e2\s*[:：]\s*(-?\d+)', line)
        if legacy_pe2_match:
            try:
                legacy_pe2 = max(0, min(100, int(legacy_pe2_match.group(1))))
            except (ValueError, TypeError):
                pass
            legacy_pec_match = re.search(
                r'predicted_ec\s*[:：]\s*(-?[0-9]*\.?[0-9]+)', line
            )
            if legacy_pec_match:
                try:
                    legacy_pec = max(0.0, min(1.0, float(legacy_pec_match.group(1))))
                except (ValueError, TypeError):
                    pass
            line = re.sub(
                r'\s*/?\s*predicted_ec\s*[:：]\s*-?[0-9]*\.?[0-9]+', '', line
            )
            line = re.sub(
                r'\s*/?\s*predicted_e2\s*[:：]\s*-?\d+', '', line
            ).strip()

        # --- tool_part / reason_part 分離 ---
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

        # --- chain の各 tool + 新形式 (pe2=X, pec=Y) 個別抽出 ---
        raw_tools = []
        raw_chain = []
        for t_frag in tool_part.split('+'):
            t_frag = t_frag.strip()
            name_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', t_frag)
            if not name_match:
                continue
            t_name = name_match.group(1)

            tool_pe2 = None
            tool_pec = None
            paren_match = re.search(r'\(([^)]*)\)', t_frag)
            if paren_match:
                inside = paren_match.group(1)
                pe2_m = re.search(r'pe2\s*=\s*(-?\d+)', inside)
                if pe2_m:
                    try:
                        tool_pe2 = max(0, min(100, int(pe2_m.group(1))))
                    except (ValueError, TypeError):
                        pass
                pec_m = re.search(r'pec\s*=\s*(-?[0-9]*\.?[0-9]+)', inside)
                if pec_m:
                    try:
                        tool_pec = max(0.0, min(1.0, float(pec_m.group(1))))
                    except (ValueError, TypeError):
                        pass
            raw_tools.append(t_name)
            raw_chain.append({
                "tool": t_name,
                "predicted_e2": tool_pe2,
                "predicted_ec": tool_pec,
            })

        # --- allowed_tools フィルタ (chain も同期除去) ---
        valid_tools = []
        valid_chain = []
        for t_name, item in zip(raw_tools, raw_chain):
            if t_name in allowed_tools:
                valid_tools.append(t_name)
                valid_chain.append(item)

        # --- 旧形式 pe2/pec を chain 内で None の tool に複製 (後方互換) ---
        for item in valid_chain:
            if item["predicted_e2"] is None and legacy_pe2 is not None:
                item["predicted_e2"] = legacy_pe2
            if item["predicted_ec"] is None and legacy_pec is not None:
                item["predicted_ec"] = legacy_pec

        # --- フォールバック (行内 tool 名検索) ---
        if not valid_tools:
            for t in allowed_tools:
                if t in line:
                    valid_tools = [t]
                    valid_chain = [{
                        "tool": t,
                        "predicted_e2": legacy_pe2,
                        "predicted_ec": legacy_pec,
                    }]
                    break

        reason = re.sub(r'^[\d]+[.:)\s]+', '', reason_part).strip()
        reason = re.sub(r'^[-*]\s*', '', reason).strip()
        if reason.startswith('[') and reason.endswith(']'):
            reason = reason[1:-1].strip()

        chain_key = "+".join(valid_tools)
        if valid_tools and chain_key not in ["+".join(c["tools"]) for c in candidates]:
            cand_dict = {
                "tool": valid_tools[0],
                "tools": valid_tools,
                "reason": reason,
                "chain": valid_chain,
            }
            head = valid_chain[0]
            if head["predicted_e2"] is not None:
                prediction = {
                    "predicted_e2": head["predicted_e2"],
                    "confidence": 0.7,
                    "source": "medium",
                }
                if head["predicted_ec"] is not None:
                    prediction["predicted_ec"] = head["predicted_ec"]
                cand_dict["prediction"] = prediction
            candidates.append(cand_dict)

    if not candidates:
        for t in allowed_tools:
            candidates.append({
                "tool": t,
                "tools": [t],
                "reason": "（フォールバック）",
                "chain": [{"tool": t, "predicted_e2": None, "predicted_ec": None}],
            })
    return candidates
