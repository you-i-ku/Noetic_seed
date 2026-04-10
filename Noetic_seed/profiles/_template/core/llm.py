"""LLM呼び出し（OpenAI互換 / Claude API / Claude Code CLI）+ 画像対応"""
import httpx
import subprocess
import json
import base64
import io
from pathlib import Path
from core.config import llm_cfg


def _get_vision_max_size() -> int:
    """settings.json から画像の最大辺サイズを取得。デフォルト 896（Gemma 3/4 の SigLIP ネイティブ）。"""
    return int(llm_cfg.get("vision_max_size", 896))


def _resize_image_bytes(image_path: str, max_size: int | None = None) -> bytes | None:
    """画像を読み込み、max_size に収まるようリサイズして JPEG バイト列を返す。
    max_size 未指定なら settings.json の vision_max_size を使う。"""
    try:
        from PIL import Image
    except ImportError:
        print("  [llm] Pillow未インストール。")
        return None

    if max_size is None:
        max_size = _get_vision_max_size()

    try:
        p = Path(image_path)
        if not p.exists():
            print(f"  [llm] 画像が見つかりません: {image_path}")
            return None

        img = Image.open(p)
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        print(f"  [llm] 画像読み込みエラー: {e}")
        return None


def _load_image_base64(image_path: str, max_size: int | None = None) -> tuple[str, str] | None:
    """画像を読み込み、リサイズしてbase64にエンコード。
    戻り値: (base64_str, media_type) or None。openai-compat / claude 経路用。"""
    b = _resize_image_bytes(image_path, max_size)
    if b is None:
        return None
    return (base64.b64encode(b).decode("ascii"), "image/jpeg")

_PROVIDER_BASE_URLS = {
    "lmstudio": "http://localhost:1234/v1",
    "openai":   "https://api.openai.com/v1",
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/openai",
    "claude":   "https://api.anthropic.com",
}

def _get_base_url() -> str:
    """settings.jsonにbase_urlがあればそれを使い、なければproviderから決定"""
    if "base_url" in llm_cfg and llm_cfg["base_url"]:
        return llm_cfg["base_url"].rstrip("/")
    provider = llm_cfg.get("provider", "lmstudio")
    return _PROVIDER_BASE_URLS.get(provider, _PROVIDER_BASE_URLS["lmstudio"])

def _call_openai_compat(prompt: str, max_tokens: int, temperature: float = 0.7, image_paths: list = None) -> str:
    """LM Studio / OpenAI / Gemini（OpenAI互換エンドポイント）。複数画像対応。"""
    base_url = _get_base_url()
    api_key = llm_cfg.get("api_key", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # 画像あり → OpenAI vision形式でcontentを配列に
    if image_paths:
        content = [{"type": "text", "text": prompt}]
        loaded_count = 0
        for ip in image_paths:
            img_result = _load_image_base64(ip)
            if img_result:
                b64, media_type = img_result
                content.append({"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}})
                loaded_count += 1
        if loaded_count == 0:
            content = prompt
            print("  [llm] 画像読み込み失敗（テキストのみ送信）")
    else:
        content = prompt

    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": llm_cfg.get("model", "default"),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _call_claude(prompt: str, max_tokens: int, image_paths: list = None) -> str:
    """Anthropic Claude API。複数画像対応。"""
    if image_paths:
        content = [{"type": "text", "text": prompt}]
        loaded_count = 0
        for ip in image_paths:
            img_result = _load_image_base64(ip)
            if img_result:
                b64, media_type = img_result
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                })
                loaded_count += 1
        if loaded_count == 0:
            content = prompt
    else:
        content = prompt

    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": llm_cfg.get("api_key", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": llm_cfg.get("model", "claude-sonnet-4-6"),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

def _call_claude_code(prompt: str, max_tokens: int) -> str:
    """Claude Code CLI（claude -p）。サブスクリプション枠で動く。APIキー不要。
    --bare: hooks/MCP/memory等をスキップして高速化
    --system-prompt: Claude Codeのデフォルトプロンプトを完全置換（ikuのプロンプトのみ使用）
    --output-format json: 構造化出力でresultフィールドからテキスト抽出
    """
    import tempfile
    # CLAUDE.mdやmemoryが存在しないtempディレクトリで実行（文脈汚染防止）
    clean_dir = tempfile.mkdtemp(prefix="iku_llm_")
    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "json",
        "--system-prompt", "You are a component in an autonomous AI system. Follow the instructions in the user message exactly. Do not add explanations, greetings, or meta-commentary. Do NOT use any tools.",
        "--disallowedTools", "Bash,Read,Edit,Write,WebSearch,WebFetch",
        "--no-session-persistence",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding="utf-8",
            errors="replace",
            cwd=clean_dir,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()[:200]
            raise RuntimeError(f"claude -p failed (rc={result.returncode}): {stderr}")
        # --output-format json の場合、resultフィールドにテキストが入る
        try:
            data = json.loads(result.stdout)
            return data.get("result", result.stdout.strip())
        except json.JSONDecodeError:
            # JSONパース失敗→生テキストとして返す
            return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError("claude -p timed out (300s)")

_lmstudio_model_cache = {}

def _call_lmstudio_native(prompt: str, max_tokens: int, temperature: float = 0.7, image_paths: list = None) -> str:
    """LM Studio 公式 Python SDK 経由。複数画像対応。
    REST API /v1/chat/completions は vision が壊れている（LM Studio bug #968）ため、
    vision を使う際は必ずこちら経由で呼ぶ。
    module-level API（lms.prepare_image / lms.llm）を使う。
    Client ベース API だと prepare_image と llm.model が別クライアントになって
    画像ハンドルが紐づかない不具合がある。"""
    global _lmstudio_model_cache
    import lmstudio as lms

    model_name = llm_cfg.get("model", "default")
    if model_name == "default":
        try:
            loaded = lms.list_loaded_models()
            if not loaded:
                raise RuntimeError("No LLM loaded in LM Studio")
            model_name = loaded[0].identifier
        except Exception as e:
            raise RuntimeError(f"Could not auto-detect model: {e}")

    if model_name not in _lmstudio_model_cache:
        _lmstudio_model_cache[model_name] = lms.llm(model_name)
    model = _lmstudio_model_cache[model_name]

    chat = lms.Chat()
    if image_paths:
        from pathlib import Path
        imgs = []
        for ip in image_paths:
            p = Path(ip)
            if not p.exists():
                print(f"  [llm] 画像が見つかりません: {ip}")
                continue
            # settings.json の vision_max_size に合わせてリサイズしてから SDK に渡す
            # これにより Gemma 3/4 の pan-and-scan による token 爆発を防ぐ（896 固定で 256 tokens/image）
            resized = _resize_image_bytes(str(p))
            if resized is None:
                continue
            try:
                imgs.append(lms.prepare_image(io.BytesIO(resized), name=p.name))
            except Exception as e:
                print(f"  [llm] 画像準備失敗 {p.name}: {e}")
        if imgs:
            chat.add_user_message(prompt, images=imgs)
        else:
            chat.add_user_message(prompt)
    else:
        chat.add_user_message(prompt)

    config = {"maxTokens": max_tokens, "temperature": temperature}
    result = model.respond(chat, config=config)
    return result.content if hasattr(result, "content") else str(result)


def call_llm(prompt: str, max_tokens: int = 24000, temperature: float = 0.7,
             image_path: str = None, image_paths: list = None) -> str:
    """LLM呼び出し統一インターフェース。
    image_path (単一) / image_paths (複数) のどちらか、または両方なしで呼ぶ。
    """
    # image_paths が未指定で image_path があれば [image_path] に昇格
    if image_paths is None:
        image_paths = [image_path] if image_path else None

    provider = llm_cfg.get("provider", "lmstudio")
    if provider == "claude":
        return _call_claude(prompt, max_tokens, image_paths=image_paths)
    elif provider == "claude_code":
        return _call_claude_code(prompt, max_tokens)  # claude_codeは画像非対応
    elif provider == "lmstudio":
        # LM Studio 経由は常に公式 SDK を使う（REST API の vision バグ #968 回避）
        return _call_lmstudio_native(prompt, max_tokens, temperature, image_paths=image_paths)
    else:
        # openai, gemini 等（OpenAI互換サーバ）はREST APIが正常なのでそのまま
        return _call_openai_compat(prompt, max_tokens, temperature, image_paths=image_paths)
