"""LLM呼び出し（OpenAI互換 / Claude API / Claude Code CLI）+ 画像対応
クレデンシャルは core.auth.get_llm_credentials から動的に取得する。
settings.json の provider/model は呼び出し直前に再読み込み（次サイクルからの切替対応）。"""
import httpx
import re
import subprocess
import json
import base64
import io
from pathlib import Path
from core.config import llm_cfg, LLM_SETTINGS
from core.auth import get_llm_credentials


def _reload_active_config() -> dict:
    """settings.json から現在の provider/model/vision_max_size を再読み込み。
    mid-cycle 切替を避けるため、サイクル先頭で呼ぶ想定。"""
    try:
        with open(LLM_SETTINGS, encoding="utf-8") as f:
            fresh = json.load(f)
        llm_cfg.clear()
        llm_cfg.update(fresh)
    except Exception:
        pass
    return llm_cfg


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


def _get_active_provider_config() -> tuple[str, str, str, str]:
    """現在アクティブな provider の (provider_name, base_url, api_key, model) を取得。
    settings.json (provider/model) と secrets.json (llm_providers) を統合。
    各値は strip して末尾改行などの不正文字を除去する（httpx の InvalidHeader 回避）。"""
    provider = str(llm_cfg.get("provider", "lmstudio") or "").strip()
    model = str(llm_cfg.get("model", "") or "").strip()

    creds = get_llm_credentials(provider) or {}
    base_url = str(creds.get("base_url", "") or "").strip()
    if not base_url:
        base_url = _PROVIDER_BASE_URLS.get(provider, "")
    api_key = str(creds.get("api_key", "") or "").strip()
    if not model:
        model = str(creds.get("last_model", "default") or "").strip()

    return provider, base_url.rstrip("/"), api_key, model


def _call_openai_compat(prompt: str, max_tokens: int, temperature: float = 0.7, image_paths: list = None) -> str:
    """LM Studio / OpenAI / Gemini（OpenAI互換エンドポイント）。複数画像対応。"""
    _, base_url, api_key, model = _get_active_provider_config()
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
            "model": model,
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
    _, base_url, api_key, model = _get_active_provider_config()
    if not base_url:
        base_url = "https://api.anthropic.com"
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
        f"{base_url}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model or "claude-sonnet-4-6",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

def _call_via_claude_code_provider(prompt: str, max_tokens: int,
                                    image_paths: list = None,
                                    role: str | None = None) -> str:
    """call_llm 経路 (LLM① ③ ④) 用の shim — 新 ClaudeCodeProvider 経由。

    旧 _call_claude_code (subprocess 自前 wrap, text-only) は撤去 (2026-04-27)。
    新 provider は claude-agent-sdk を経由して JSONL stream で双方向通信、
    Step 2 で画像対応・Step 3 で tool calling 拡張予定。
    詳細: WORLD_MODEL_DESIGN/CLAUDE_CODE_UNIFIED_PROVIDER_PLAN.md

    Args:
        prompt: ユーザー prompt 文字列 (system 部分も既に埋め込まれている前提)。
        max_tokens: 上限トークン数 (claude-agent-sdk のデフォルトに従う、参考値)。
        image_paths: Step 1 では警告 + text-only fallback、Step 2 で対応予定。
        role: LLM 役割識別子 ("llm3" 等)。指定されかつ settings.json の
            model_overrides に該当 role エントリがあれば、当該呼び出しのみ
            そのモデルに切替える (claude_code 限定の動的解決)。

    Returns:
        AssistantMessage.text。
    """
    from core.providers.base import ApiRequest
    from core.providers.claude_code import ClaudeCodeProvider

    _, _, _, model = _get_active_provider_config()

    # role 別 model override (claude_code 限定)。
    # 他 provider 経路は role 引数を渡さない/無視するため、本機構は自動的に
    # 「claude_code のとき限定」が成立する。
    # 用途: LLM3 (E 値評価) を Sonnet → Haiku に切替えて refusal 削減
    # (2026-04-28 smoke v7 で LLM3 が constraint_circumvention 等の自己観察
    # ラベルに RLHF refusal 連発、reflect 滞留 22-30 分の主因と判明)。
    if role:
        overrides = llm_cfg.get("model_overrides", {}) or {}
        model_override = overrides.get(role)
        if model_override:
            model = model_override

    provider = ClaudeCodeProvider(model=model or "sonnet")
    req = ApiRequest(
        system_prompt="",
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }],
        max_tokens=max_tokens,
        image_paths=image_paths,
    )
    msg = provider.stream(req)
    return msg.text

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

    _, _, _, model_name = _get_active_provider_config()
    if not model_name or model_name == "default":
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
    sampling = llm_cfg.get("llm_sampling") or {}
    if sampling:
        # settings.json (snake_case) → LM Studio SDK (camelCase) 変換
        # SDK default 値は非公開のため、明示しない場合は legacy 挙動 (config に渡さない)
        if "top_p" in sampling:
            config["topPSampling"] = sampling["top_p"]
        if "top_k" in sampling:
            config["topKSampling"] = sampling["top_k"]
        if "min_p" in sampling:
            config["minPSampling"] = sampling["min_p"]
        if "repetition_penalty" in sampling:
            config["repeatPenalty"] = sampling["repetition_penalty"]
    result = model.respond(chat, config=config)
    return result.content if hasattr(result, "content") else str(result)


def _detect_repetition(text: str, ngram_size: int = 5, threshold: int = 4,
                       tail_chars: int = 400) -> bool:
    """末尾 tail_chars に同一 char-based n-gram が threshold 回以上出現したら True。

    日本語の「として、として、として、…」のような短い反復ループを検知する。
    通常の自然な反復 (例: 「は、」「を、」3 回) では発火しないよう、
    ngram_size=5 / threshold=4 を default にする。
    """
    if not text or len(text) < ngram_size * threshold:
        return False
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    counts: dict = {}
    for i in range(len(tail) - ngram_size + 1):
        ng = tail[i:i + ngram_size]
        counts[ng] = counts.get(ng, 0) + 1
        if counts[ng] >= threshold:
            return True
    return False


def _call_llm_inner(prompt: str, max_tokens: int = 24000, temperature: float = 0.7,
                    image_path: str = None, image_paths: list = None,
                    role: str | None = None) -> str:
    """LLM provider dispatch (内部実装)。call_llm が repetition guard で wrap する。

    role: "llm3" 等の役割識別子。claude_code provider 限定で、settings.json の
    model_overrides に該当 role があれば当該呼び出しのみモデル切替される。
    他 provider 経路 (lmstudio/openai_compat/claude) では role 引数を無視。
    """
    if image_paths is None:
        image_paths = [image_path] if image_path else None

    provider, _, _, _ = _get_active_provider_config()
    if provider == "claude":
        return _call_claude(prompt, max_tokens, image_paths=image_paths)
    elif provider == "claude_code":
        return _call_via_claude_code_provider(prompt, max_tokens,
                                               image_paths=image_paths,
                                               role=role)
    elif provider == "lmstudio":
        # LM Studio 経由は常に公式 SDK を使う（REST API の vision バグ #968 回避）
        return _call_lmstudio_native(prompt, max_tokens, temperature, image_paths=image_paths)
    else:
        # openai, gemini 等（OpenAI互換サーバ）はREST APIが正常なのでそのまま
        return _call_openai_compat(prompt, max_tokens, temperature, image_paths=image_paths)


def call_llm(prompt: str, max_tokens: int = 24000, temperature: float = 0.7,
             image_path: str = None, image_paths: list = None,
             max_retry: int = 0,
             role: str | None = None) -> str:
    """LLM呼び出し統一インターフェース (repetition guard 関数残置、default 無効)。

    出力に repetition loop を検知したら temperature を上げて再呼び出しする
    (max_retry まで)。すべての retry でループが消えなければ最後の出力を返す
    (諦め)。

    2026-04-27 default max_retry=2 → 0 (retry guard 実質デッドコード化):
      smoke 検証で全系統の retry 効果ゼロを確証 (LLM3 数値ガチャのみ /
      Reflection は連発が retry 後も同型 / memory_store 0.0 連発が完全同型 /
      正常 JSON 出力で false positive あり)。retry は時間ロスのみで gemma を
      attractor から脱出させていない事実を踏まえ、default で guard 無効化。
      _detect_repetition / retry loop は残置、必要時に呼出側で max_retry=N
      指定で復活可能 (柔軟性確保)。詳細:
      memory/project_v05_phase5_stage11d_phase8_retry_guard_deadcode.md

    image_path (単一) / image_paths (複数) のどちらか、または両方なしで呼ぶ。
    provider は llm_cfg から動的に読み出される（次サイクルから切替反映）。
    """
    out = _call_llm_inner(prompt, max_tokens, temperature, image_path, image_paths,
                          role=role)
    for retry in range(max_retry):
        if not _detect_repetition(out):
            return _post_process_response(out)
        new_temp = min(temperature + 0.2 * (retry + 1), 1.2)
        print(f"  [llm] repetition detected, retry {retry + 1}/{max_retry} (temp={new_temp:.2f})")
        out = _call_llm_inner(prompt, max_tokens, new_temp, image_path, image_paths,
                              role=role)
    return _post_process_response(out)


# ============================================================
# Reasoning model (Qwen3 / DeepSeek-R1 系) の <think> tag 対応
# ============================================================
# 2026-04-29: Qwen3 系 / DeepSeek-R1 系の reasoning model は内部思考を
# `<think>...</think>` で囲んで返す。tool 呼出 JSON / [TOOL:...] や
# 自然言語の最終応答は </think> 後に出る前提だが、parser がそのまま
# tag 入り text を受け取ると tool 検出失敗 / 自然言語結果に think 混入
# が起きる。
#
# 対処は二段構え (X 改良案):
#   ① raw (think 込み) を append_debug_log 経由で `llm_debug.log` に保存
#      → think 内容の観察手段は別 layer で完全に保たれる
#   ② <think> 区間を strip して return
#      → caller (parser / 自然言語 description 経路) は綺麗な text を受け取る
#
# autopoiesis 哲学整合: 内的状態 (think) の永続化を捨てず、parse 用 text
# だけ「邪魔な部分を取り除く」として扱う (memory `feedback_action_observation_unified`
# 系譜、内的・外的を独立 layer で持つ設計)。
# ============================================================

_THINK_PATTERN = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def _strip_think(text: str) -> str:
    """`<think>...</think>` 区間を除去 (閉じ tag のある区間のみ)。

    未閉じ <think> が残った場合はそのまま (parse 失敗で発覚 → debug)。
    None / 空 text はそのまま透過。
    """
    if not text:
        return text
    return _THINK_PATTERN.sub('', text)


def _post_process_response(raw: str) -> str:
    """call_llm の最終 return 直前の post-process。

    raw (think 込み) を `llm_debug.log` に保存しつつ <think> を strip して
    返す。append_debug_log の i/o 失敗は無害扱い (return 値には影響させない)。
    """
    try:
        from core.state import append_debug_log
        append_debug_log("call_llm (raw)", raw)
    except Exception:
        pass
    return _strip_think(raw)
