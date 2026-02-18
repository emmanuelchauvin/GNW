"""
Projet Ignition — MiniMax API Bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Async bridge to the MiniMax M2.5 model via its OpenAI-compatible endpoint.

Features:
  • httpx async client with connection pooling
  • Exponential back-off retry (5 attempts)
  • Forces JSON response_format for structured output
  • Strips <think> reasoning tags before JSON parsing
  • Uses the OpenAI-compatible API at https://api.minimax.io/v1
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MINIMAX_BASE_URL: str = os.getenv(
    "OPENAI_BASE_URL", "https://api.minimax.io/v1"
)
_CHAT_ENDPOINT: str = f"{_MINIMAX_BASE_URL}/chat/completions"
_MODEL_ID: str = "MiniMax-M2.5"
_MAX_RETRIES: int = 5
_BACKOFF_BASE: float = 1.0   # base delay in seconds
_BACKOFF_FACTOR: float = 2.0  # exponential multiplier
_DEFAULT_TIMEOUT: float = 60.0  # seconds

# Regex to strip MiniMax <think>…</think> reasoning blocks
_THINK_TAG_RE: re.Pattern[str] = re.compile(
    r"<think>.*?</think>", re.DOTALL
)


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract a JSON object from a model response that may contain
    ``<think>`` reasoning tags or other surrounding text.

    Strategy:
      1. Strip ``<think>…</think>`` blocks.
      2. Try ``json.loads`` on the remaining text.
      3. If that fails, locate the first ``{`` and last ``}`` and parse
         the substring between them.
    """
    cleaned: str = _THINK_TAG_RE.sub("", raw).strip()

    # Fast path: the cleaned string *is* valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the outermost { … }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise json.JSONDecodeError(
        "No JSON object found in model response",
        cleaned,
        0,
    )


class MiniMaxBridgeError(Exception):
    """Raised when the MiniMax API call fails after all retries."""


class MiniMaxBridge:
    """Asynchronous bridge to MiniMax M2.5 via the OpenAI-compatible API.

    Usage::

        async with MiniMaxBridge() as bridge:
            result = await bridge.generate_response(
                prompt="Return a JSON object with key 'status' set to 'ok'.",
                system_prompt="You are a helpful assistant.",
            )
            print(result)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = _CHAT_ENDPOINT,
        model: str = _MODEL_ID,
        max_retries: int = _MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        # Reads MINIMAX_KEY first, falls back to OPENAI_API_KEY
        self._api_key: str = (
            api_key
            or os.getenv("MINIMAX_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        if not self._api_key:
            raise ValueError(
                "API key is required. "
                "Set MINIMAX_KEY (or OPENAI_API_KEY) in the .env file "
                "or pass it explicitly."
            )
        self._endpoint = endpoint
        self._model = model
        self._max_retries = max_retries
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    # -- Context manager -----------------------------------------------------

    async def __aenter__(self) -> MiniMaxBridge:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- Public API ----------------------------------------------------------

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant that always responds in valid JSON.",
    ) -> dict[str, Any]:
        """Send a prompt to MiniMax M2.5 and return the parsed JSON response.

        Parameters
        ----------
        prompt:
            The user message to send.
        system_prompt:
            The system instruction that guides the model's behaviour.

        Returns
        -------
        dict[str, Any]
            Parsed JSON object from the model's reply.

        Raises
        ------
        MiniMaxBridgeError
            If the request fails after *max_retries* attempts.
        """
        if self._client is None:
            raise RuntimeError(
                "MiniMaxBridge must be used as an async context manager. "
                "Use `async with MiniMaxBridge() as bridge:`"
            )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        last_exception: BaseException | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response: httpx.Response = await self._client.post(
                    self._endpoint,
                    json=payload,
                )
                response.raise_for_status()

                data: dict[str, Any] = response.json()
                content: str = data["choices"][0]["message"]["content"]
                return _extract_json(content)

            except httpx.HTTPStatusError as exc:
                last_exception = exc
                status = exc.response.status_code

                # Do not retry on client errors (except 429 rate-limit)
                if 400 <= status < 500 and status != 429:
                    logger.error(
                        "Client error %s (non-retryable): %s",
                        status,
                        exc.response.text,
                    )
                    raise MiniMaxBridgeError(
                        f"MiniMax API returned {status}: {exc.response.text}"
                    ) from exc

                logger.warning(
                    "Attempt %d/%d — HTTP %s. Retrying…",
                    attempt,
                    self._max_retries,
                    status,
                )

            except (httpx.RequestError, json.JSONDecodeError, KeyError) as exc:
                last_exception = exc
                logger.warning(
                    "Attempt %d/%d — %s: %s. Retrying…",
                    attempt,
                    self._max_retries,
                    type(exc).__name__,
                    exc,
                )

            # Exponential back-off
            delay: float = _BACKOFF_BASE * (_BACKOFF_FACTOR ** (attempt - 1))
            logger.info("Waiting %.1fs before next attempt…", delay)
            await asyncio.sleep(delay)

        raise MiniMaxBridgeError(
            f"MiniMax API call failed after {self._max_retries} attempts. "
            f"Last error: {last_exception}"
        )


# ---------------------------------------------------------------------------
# OpenRouter Vision Bridge
# ---------------------------------------------------------------------------

_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
_OPENROUTER_VISION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"

class OpenRouterVisionBridge:
    """Async bridge to OpenRouter for Vision (Nemotron Nano 12B).
    
    Uses OpenAI-compatible API to send images to OpenRouter.
    """
    
    def __init__(self, api_key: str | None = None, model: str = _OPENROUTER_VISION_MODEL):
        self.api_key = api_key or os.getenv("OPENROUTER_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API Key (OPENROUTER_KEY) is required.")
        
        self.model_name = model
        self.endpoint = _OPENROUTER_ENDPOINT

    async def analyze_image(
        self, 
        image_bytes: bytes, 
        prompt: str, 
        mime_type: str = "image/jpeg"
    ) -> str:
        """Analyze an image using OpenRouter and return textual analysis."""
        
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/emmanuelchauvin/GNW", # Required by OpenRouter
            "X-Title": "GNW Engine",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        }
        
        try:
            logger.info(f"👁️ Appel OpenRouter ({self.model_name})...")
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.endpoint + "/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"OpenRouter Vision error: {e}")
            return f"Echec vision (OpenRouter) : {e}"

# ---------------------------------------------------------------------------
# Ollama Vision Bridge
# ---------------------------------------------------------------------------

try:
    import ollama
    _HAS_OLLAMA = True
except ImportError:
    _HAS_OLLAMA = False

class OllamaVisionBridge:
    """Async bridge to local Ollama instance for Vision (Gemma 3).
    
    Requires `ollama` package and a running Ollama server.
    Target Model: gemma3:27b
    """
    
    def __init__(self, model: str = "gemma3:27b"):
        if not _HAS_OLLAMA:
            raise ImportError("Package 'ollama' is required for Vision.")
        
        self.model_name = model
        # Check if model exists or pull? 
        # We assume user pulled it as per instructions.

    async def analyze_image(
        self, 
        image_bytes: bytes, 
        prompt: str, 
        mime_type: str = "image/jpeg"
    ) -> str:
        """Analyze an image and return a textual description/analysis.
        """
        
        def _call_ollama():
            # Ollama python client expects 'images' field in message
            # It accepts bytes directly (in a list)
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_bytes]
                    }
                ]
            )
            return response['message']['content']

        # Run safely in a thread
        try:
            logger.info(f"👁️ Appel Ollama ({self.model_name})...")
            result = await asyncio.to_thread(_call_ollama)
            return result
        except Exception as e:
            logger.error(f"Ollama Vision error: {e}")
            return f"Echec vision (Ollama) : {e}"
