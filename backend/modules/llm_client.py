"""
llm_client.py

Unified LLM API module for OpenAI-compatible providers.

Dependency:
    pip install openai
"""

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import sys
import json
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
)

load_dotenv()

DEBUG_LINES = os.environ.get("DEBUG_LINES", "false").lower() in ("true", "1", "yes")

LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    ".llm_call_log.log",
)

logger = logging.getLogger("llm_client")
logger.setLevel(logging.DEBUG)

_log_handler = logging.FileHandler(LOG_FILE)
_log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_log_handler)
logger.propagate = False


DEFAULT_TIMEOUT = 300.0
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_THIS_DIR)

if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
PROVIDER_CONFIG_PATH = os.path.join(_BACKEND_ROOT, ".provider_config.json")


def _log(message: str):
    """Write a timestamped line to the log file. Print to stdout if DEBUG_LINES."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    line = f"[{ts}] {message}"
    logger.info(line)

    if DEBUG_LINES:
        print(line)


class LLMClient:
    """
    Unified client for OpenAI-compatible LLM providers.

    Example:
        llm_client = LLMClient(
            "groq",
            "https://api.groq.com/openai/v1",
            "api-key",
            "openai/gpt-oss-20b",
        )
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str,
        endpoint: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string.")

        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string.")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        
        self.provider = provider
        self.endpoint = self._load_provider_config(provider).get("endpoint")
        print(f"LLMClient initialized with provider={self.provider}, endpoint={self.endpoint}, model={model}")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.endpoint,
            timeout=self.timeout,
            max_retries=0,
        )

    # -----------------------------------------------------------------
    # Helper Functions
    # -----------------------------------------------------------------
    def _load_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Load a provider's entry from .provider_config.json."""
        with open(PROVIDER_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)

        entry = config.get(provider_name)
        if not entry:
            raise ValueError(
                f"No provider config entry for '{provider_name}' found in {PROVIDER_CONFIG_PATH}"
            )
        return entry

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate(
        self,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        system: Optional[str] = None,
        developer: Optional[str] = None,
        user: Optional[str] = None,
        assistant: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send messages to the provider using the OpenAI-compatible API.

        Args:
            prompt: Backward-compatible user prompt.
            model: Optional override for the model configured on this client.
            system: System-level instructions.
            developer: Application-level instructions.
            user: User message.
            assistant: Previous assistant response / conversation history.
            tool: Output returned by a previously called tool.

        Returns:
            JSON-serializable dictionary containing the result.
        """

        model = model or self.model

        messages = []

        if system is not None:
            messages.append({
                "role": "system",
                "content": system,
            })

        if developer is not None:
            messages.append({
                "role": "developer",
                "content": developer,
            })

        if user is not None:
            messages.append({
                "role": "user",
                "content": user,
            })

        if assistant is not None:
            messages.append({
                "role": "assistant",
                "content": assistant,
            })

        if tool is not None:
            messages.append({
                "role": "tool",
                "content": tool,
            })

        # Preserve existing behavior:
        # If no explicit user message is supplied, use prompt.
        if user is None:
            messages.append({
                "role": "user",
                "content": prompt,
            })

        try:
            _log(
                f"CALL model={model} provider={self.provider} "
            )

            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
            )

            output = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            if output is None:
                _log(f"EMPTY_RESPONSE model={model}")

                return self._error(
                    model,
                    "EmptyResponseError",
                    "The API returned no output.",
                    status_code=200,
                )

            _log(f"RESPONSE model={model} output_len={len(output)} finish_reason={finish_reason}")

            if finish_reason == "length":
                _log(f"WARNING model={model} output TRUNCATED (hit token limit)")

            return self._success(model, output)

        except AuthenticationError as exc:
            _log(
                f"ERROR model={model} type=AuthenticationError msg={exc}"
            )

            return self._error(
                model,
                "AuthenticationError",
                str(exc),
                getattr(exc, "status_code", 401),
            )

        except BadRequestError as exc:
            _log(
                f"ERROR model={model} type=BadRequestError msg={exc}"
            )

            return self._error(
                model,
                "BadRequestError",
                str(exc),
                getattr(exc, "status_code", 400),
            )

        except APITimeoutError as exc:
            _log(
                f"ERROR model={model} type=TimeoutError msg={exc}"
            )

            return self._error(
                model,
                "TimeoutError",
                str(exc),
            )

        except APIConnectionError as exc:
            _log(
                f"ERROR model={model} type=APIConnectionError msg={exc}"
            )

            return self._error(
                model,
                "APIConnectionError",
                str(exc),
            )

        except APIStatusError as exc:
            _log(
                f"ERROR model={model} type=APIStatusError "
                f"status={exc.status_code} msg={exc}"
            )

            return self._error(
                model,
                "APIStatusError",
                str(exc),
                getattr(exc, "status_code", None),
            )

        except (IndexError, AttributeError) as exc:
            _log(
                f"ERROR model={model} type=ResponseParsingError msg={exc}"
            )

            return self._error(
                model,
                "ResponseParsingError",
                str(exc),
                200,
            )

        except Exception as exc:
            _log(
                f"ERROR model={model} type=UnexpectedError msg={exc}"
            )

            return self._error(
                model,
                "UnexpectedError",
                str(exc),
            )
    
    # -----------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_prompt(prompt: Any) -> Optional[str]:
        if not isinstance(prompt, str) or not prompt.strip():
            return "prompt must be a non-empty string."

        return None

    def _success(
        self,
        model: str,
        output: str,
        status_code: int = 200,
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "provider": self.provider,
            "model": model,
            "status_code": status_code,
            "output": output,
            "error": None,
        }

    def _error(
        self,
        model: str,
        error_type: str,
        message: str,
        status_code: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "provider": self.provider,
            "model": model,
            "status_code": status_code,
            "output": None,
            "error": {
                "type": error_type,
                "message": message,
            },
        }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

# OpenCode Go
#
# llm_client = LLMClient(
#     "opencode-go",
#     "...",
#     "deepseek-v4-flash-free",
# )
#
# result = llm_client.generate("Hello, how are you?")


# Override the model for a single call without affecting
# the instance default.
#
# result = llm_client.generate(
#     prompt="Fallback prompt",
#     system="You are a helpful Python tutor.",
#     developer="Always provide Python examples with type hints.",
#     user="Explain Python decorators with an example.",
#     assistant="A decorator is a function that modifies another function.",
#     tool="The weather is 28°C and partly cloudy.",
# )