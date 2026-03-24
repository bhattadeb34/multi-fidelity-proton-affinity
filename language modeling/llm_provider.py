"""
llm_provider.py
===============
Unified LLM provider using LiteLLM as the backend.
Supports easy switching between any provider with a single model string.

LiteLLM model string format:
  Gemini:    "gemini/gemini-3-flash-preview"
  OpenAI:    "openai/gpt-5.4-nano"
  Anthropic: "anthropic/claude-sonnet-4-6"

Usage
-----
  from llm_provider import get_provider
  llm = get_provider("gemini", model="gemini-3-flash-preview", keys_file="...")
  response = llm.chat("Hello")

Install
-------
  pip install litellm
"""

from __future__ import annotations
import os
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def load_api_keys(keys_file: str | Path | None = None) -> dict[str, str]:
    keys: dict[str, str] = {}
    if keys_file and Path(keys_file).exists():
        with open(keys_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    keys[k.strip()] = v.strip().strip("'").strip('"')

    normalised = {}
    for k, v in keys.items():
        kl = k.lower()
        if "gemini" in kl or "google" in kl:
            normalised["GEMINI_API_KEY"] = v
            normalised["GOOGLE_API_KEY"] = v
        elif "openai" in kl or "gpt" in kl:
            normalised["OPENAI_API_KEY"] = v
        elif "anthropic" in kl or "claude" in kl:
            normalised["ANTHROPIC_API_KEY"] = v
        else:
            normalised[k] = v

    # Also pull from environment
    for env_key in ("GEMINI_API_KEY", "GOOGLE_API_KEY",
                    "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if env_key not in normalised and os.getenv(env_key):
            normalised[env_key] = os.environ[env_key]

    return normalised


# ---------------------------------------------------------------------------
# Provider name → LiteLLM prefix + env key
# ---------------------------------------------------------------------------

PROVIDER_CONFIGS = {
    "gemini":    {"prefix": "gemini",    "env_key": "GEMINI_API_KEY",    "default_model": "gemini-3-flash-preview"},
    "google":    {"prefix": "gemini",    "env_key": "GEMINI_API_KEY",    "default_model": "gemini-3-flash-preview"},
    "openai":    {"prefix": "openai",    "env_key": "OPENAI_API_KEY",    "default_model": "gpt-5.4"},
    "gpt":       {"prefix": "openai",    "env_key": "OPENAI_API_KEY",    "default_model": "gpt-5.4-nano"},
    "anthropic": {"prefix": "anthropic", "env_key": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-4-6"},
    "claude":    {"prefix": "anthropic", "env_key": "ANTHROPIC_API_KEY", "default_model": "claude-sonnet-4-6"},
}


# ---------------------------------------------------------------------------
# LiteLLM-backed provider
# ---------------------------------------------------------------------------

class LiteLLMProvider:
    """
    Unified LLM provider backed by LiteLLM.
    Works with any provider LiteLLM supports — no provider-specific code.
    """

    def __init__(
        self,
        model_string: str,
        api_key: str,
        env_key: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        max_retries: int = 3,
    ):
        try:
            import litellm
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
            litellm.drop_params = True
            # Suppress all LiteLLM logging
            import logging as _logging
            _logging.getLogger("LiteLLM").setLevel(_logging.ERROR)
            _logging.getLogger("litellm").setLevel(_logging.ERROR)
        except ImportError:
            raise ImportError(
                "LiteLLM not installed. Run:\n"
                "  pip install litellm --break-system-packages")

        self._litellm   = litellm
        self.model      = model_string
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.max_retries = max_retries

        # Set the API key in environment — LiteLLM reads from env
        os.environ[env_key] = api_key
        # Also set alternate key names LiteLLM might look for
        if "GEMINI" in env_key or "GOOGLE" in env_key:
            os.environ["GEMINI_API_KEY"] = api_key
            os.environ["GOOGLE_API_KEY"] = api_key

    def chat(self, messages: list[dict] | str,
             system: str | None = None) -> str:
        """Send a chat request via LiteLLM."""
        if isinstance(messages, str):
            msgs = [{"role": "user", "content": messages}]
        else:
            msgs = list(messages)

        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        # GPT-5, o-series, and Gemini 3 models require temperature=1
        temperature = self.temperature
        if any(x in self.model for x in ("gpt-5", "o1", "o3", "o4", "gemini-3", "gpt-4.5")):
            temperature = 1

        for attempt in range(self.max_retries):
            try:
                response = self._litellm.completion(
                    model=self.model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                # Safe extraction — choices can be empty on safety block
                if not response or not response.choices:
                    raise ValueError("Empty response (no choices returned)")
                content_text = response.choices[0].message.content
                if content_text is None:
                    raise ValueError("Response content is None (likely safety block)")
                return content_text

            except Exception as e:
                err_str = str(e)
                wait = 2 ** attempt

                # Gemini safety block — rephrase instead of retry
                if "finish_reason" in err_str and "2" in err_str:
                    log.warning(f"  Safety block on attempt {attempt+1} — "
                                f"rephrasing prompt")
                    # Rephrase: strip system prompt and simplify
                    msgs = [m for m in msgs if m["role"] != "system"]
                    if msgs:
                        msgs[-1]["content"] = (
                            "Return only a number. " + msgs[-1]["content"])
                    continue

                if attempt < self.max_retries - 1:
                    log.warning(f"  API error (attempt {attempt+1}): "
                                f"{err_str[:80]} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"LiteLLM call failed after {self.max_retries} attempts: {e}")

        raise RuntimeError(f"All {self.max_retries} attempts failed")

    def __repr__(self) -> str:
        return f"LiteLLMProvider(model={self.model!r})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_provider(
    provider_name: str = "gemini",
    model: str | None = None,
    api_key: str | None = None,
    keys_file: str | Path | None = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
    max_retries: int = 3,
    **kwargs,
) -> LiteLLMProvider:
    """
    Get a LiteLLM-backed provider.

    Parameters
    ----------
    provider_name : str
        'gemini', 'openai', 'gpt', 'anthropic', 'claude'
    model : str | None
        Model name without prefix (e.g. 'gemini-3-flash-preview').
        Defaults to the best available model per provider.
    api_key : str | None
        API key. If None, loaded from keys_file or environment.
    keys_file : str | Path | None
        Path to api_keys.txt

    Examples
    --------
    >>> llm = get_provider("gemini", keys_file="language modeling/api_keys.txt")
    >>> llm = get_provider("openai", model="gpt-5.4-nano", keys_file="...")
    """
    pname = provider_name.lower().strip()
    if pname not in PROVIDER_CONFIGS:
        raise ValueError(f"Unknown provider {provider_name!r}. "
                         f"Choose from: {list(PROVIDER_CONFIGS)}")

    cfg = PROVIDER_CONFIGS[pname]

    if api_key is None:
        keys   = load_api_keys(keys_file)
        api_key = keys.get(cfg["env_key"])
        if not api_key:
            raise ValueError(
                f"No API key found for {provider_name!r}. "
                f"Expected '{cfg['env_key']}' in {keys_file} or environment.")

    if model is None:
        model = cfg["default_model"]

    # Build LiteLLM model string: "prefix/model"
    model_string = f"{cfg['prefix']}/{model}"
    log.info(f"Initialising LiteLLMProvider with model={model_string!r}")

    return LiteLLMProvider(
        model_string=model_string,
        api_key=api_key,
        env_key=cfg["env_key"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
