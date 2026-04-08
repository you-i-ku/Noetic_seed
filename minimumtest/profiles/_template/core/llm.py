"""LLM呼び出し（OpenAI互換 / Claude API）"""
import httpx
from core.config import llm_cfg

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

def _call_openai_compat(prompt: str, max_tokens: int, temperature: float = 0.7) -> str:
    """LM Studio / OpenAI / Gemini（OpenAI互換エンドポイント）"""
    base_url = _get_base_url()
    api_key = llm_cfg.get("api_key", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    resp = httpx.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json={
            "model": llm_cfg.get("model", "default"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

def _call_claude(prompt: str, max_tokens: int) -> str:
    """Anthropic Claude API"""
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
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]

def call_llm(prompt: str, max_tokens: int = 24000, temperature: float = 0.7) -> str:
    provider = llm_cfg.get("provider", "lmstudio")
    if provider == "claude":
        return _call_claude(prompt, max_tokens)
    else:
        return _call_openai_compat(prompt, max_tokens, temperature)
