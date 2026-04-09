"""LLM呼び出し v2 — Function Calling対応 + テキストマーカーフォールバック
統一戻り値: {"text": str, "tool_calls": [{"name": str, "arguments": dict}]}
"""
import httpx
import subprocess
import json
import re
from core.config import llm_cfg

_PROVIDER_BASE_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "openai":   "https://api.openai.com/v1",
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/openai",
    "claude":   "https://api.anthropic.com",
}


def _get_base_url() -> str:
    if "base_url" in llm_cfg and llm_cfg["base_url"]:
        return llm_cfg["base_url"].rstrip("/")
    provider = llm_cfg.get("provider", "lmstudio")
    return _PROVIDER_BASE_URLS.get(provider, _PROVIDER_BASE_URLS["lmstudio"])


def _use_fc() -> bool:
    """Function Callingを使うか。claude_codeとlmstudioは非対応。"""
    provider = llm_cfg.get("provider", "lmstudio")
    if provider in ("claude_code", "lmstudio"):
        return False
    return llm_cfg.get("function_calling", False)


# === テキストマーカーパーサー（FCなし環境用）===

def _parse_text_markers(text: str, tool_names: set) -> list[dict]:
    """[TOOL:name key=value ...] をパースしてtool_callsに変換。v1パーサーの簡易版。"""
    results = []
    pattern = re.compile(r'\[TOOL:(\w+)\s*(.*?)\]', re.DOTALL)
    for m in pattern.finditer(text):
        name = m.group(1)
        if name not in tool_names:
            continue
        args_str = m.group(2).strip()
        args = {}
        # key="value" パターン
        for qm in re.finditer(r'(\w+)="((?:[^"\\]|\\.)*)"', args_str, re.DOTALL):
            args[qm.group(1)] = qm.group(2).replace('\\"', '"')
        # key=value パターン（引用符なし）
        remaining = re.sub(r'(\w+)="(?:[^"\\]|\\.)*"', '', args_str)
        key_positions = list(re.finditer(r'(?:^|[\s\[])(\w+)=', remaining))
        if len(key_positions) >= 2:
            for i, kp in enumerate(key_positions):
                key = kp.group(1)
                if key in args:
                    continue
                val_start = kp.end()
                val_end = key_positions[i + 1].start() if i + 1 < len(key_positions) else len(remaining)
                args[key] = remaining[val_start:val_end].strip().rstrip(']')
        elif key_positions and key_positions[0].group(1) not in args:
            key = key_positions[0].group(1)
            args[key] = remaining[key_positions[0].end():].strip().rstrip(']')

        results.append({"name": name, "arguments": args})
    return results


# === Provider別実装 ===

def _call_openai_compat(messages: list, tools: list = None,
                        max_tokens: int = 4096, temperature: float = 0.7) -> dict:
    """LM Studio / OpenAI / Gemini"""
    base_url = _get_base_url()
    api_key = llm_cfg.get("api_key", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    body = {
        "model": llm_cfg.get("model", "default"),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools and _use_fc():
        body["tools"] = tools
        body["tool_choice"] = "auto"

    resp = httpx.post(f"{base_url}/chat/completions", headers=headers,
                      json=body, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    msg = choice.get("message", {})

    text = msg.get("content", "") or ""
    tool_calls = []
    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append({"name": fn.get("name", ""), "arguments": arguments})

    return {"text": text, "tool_calls": tool_calls}


def _call_claude(messages: list, tools: list = None,
                 max_tokens: int = 4096) -> dict:
    """Anthropic Claude API（Function Calling = tool_use）"""
    body = {
        "model": llm_cfg.get("model", "claude-sonnet-4-6"),
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if tools and _use_fc():
        # Claude tool format変換
        claude_tools = []
        for t in tools:
            fn = t.get("function", t)
            claude_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        body["tools"] = claude_tools

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": llm_cfg.get("api_key", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body, timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    text = ""
    tool_calls = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "name": block.get("name", ""),
                "arguments": block.get("input", {}),
            })

    return {"text": text, "tool_calls": tool_calls}


def _call_claude_code(messages: list, max_tokens: int = 4096) -> dict:
    """Claude Code CLI（claude -p）。FC非対応。"""
    import tempfile
    prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    clean_dir = tempfile.mkdtemp(prefix="noetic_llm_")
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "json",
        "--system-prompt", "You are a component in an autonomous AI system. Follow the instructions exactly. Do not add explanations.",
        "--disallowedTools", "Bash,Read,Edit,Write,WebSearch,WebFetch",
        "--no-session-persistence",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                encoding="utf-8", errors="replace", cwd=clean_dir)
        if result.returncode != 0:
            raise RuntimeError(f"claude -p failed (rc={result.returncode}): {result.stderr[:200]}")
        try:
            data = json.loads(result.stdout)
            text = data.get("result", result.stdout.strip())
        except json.JSONDecodeError:
            text = result.stdout.strip()
        return {"text": text, "tool_calls": []}
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude -p timed out (300s)")


# === 統一インターフェース ===

def call_llm(messages: list, tools: list = None, tool_names: set = None,
             max_tokens: int = 4096, temperature: float = 0.7) -> dict:
    """統一LLM呼び出し。
    戻り値: {"text": str, "tool_calls": [{"name": str, "arguments": dict}]}
    FC非対応環境ではテキストマーカーを自動パース。
    """
    provider = llm_cfg.get("provider", "lmstudio")

    if provider == "claude":
        result = _call_claude(messages, tools, max_tokens)
    elif provider == "claude_code":
        result = _call_claude_code(messages, max_tokens)
    else:
        result = _call_openai_compat(messages, tools, max_tokens, temperature)

    # FC非対応でtool_callsが空の場合、テキストマーカーをパース
    if not result["tool_calls"] and result["text"] and tool_names:
        parsed = _parse_text_markers(result["text"], tool_names)
        if parsed:
            result["tool_calls"] = parsed

    return result
