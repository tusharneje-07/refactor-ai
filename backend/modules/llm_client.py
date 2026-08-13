"""
llm_client.py

Unified LLM API module for:
    - Groq
    - OpenCode Zen

Dependency:
    pip install openai
"""

from typing import Any, Dict, Optional

from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
)


DEFAULT_TIMEOUT = 60.0


class BaseLLMClient:
    """
    Base class for OpenAI-compatible LLM providers.

    Subclasses only need to set `PROVIDER_NAME`, `BASE_URL`, and
    `DEFAULT_MODEL`.
    """

    PROVIDER_NAME: str = "base"
    BASE_URL: str = ""
    DEFAULT_MODEL: str = ""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string.")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        self.api_key = api_key
        self.model = model
        self.timeout = timeout

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
            timeout=self.timeout,
            max_retries=0,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a prompt to the provider using the OpenAI-compatible API.

        Args:
            prompt: Prompt sent to the model.
            model: Optional override for the model configured on this client.

        Returns:
            JSON-serializable dictionary containing the result.
        """

        model = model or self.model

        validation_error = self._validate_prompt(prompt)
        if validation_error:
            return self._error(model, "ValidationError", validation_error)

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            output = response.choices[0].message.content

            if output is None:
                return self._error(
                    model,
                    "EmptyResponseError",
                    "The API returned no output.",
                    status_code=200,
                )

            return self._success(model, output)

        except AuthenticationError as exc:
            return self._error(
                model, "AuthenticationError", str(exc), getattr(exc, "status_code", 401)
            )

        except BadRequestError as exc:
            return self._error(
                model, "BadRequestError", str(exc), getattr(exc, "status_code", 400)
            )

        except APITimeoutError as exc:
            return self._error(model, "TimeoutError", str(exc))

        except APIConnectionError as exc:
            return self._error(model, "APIConnectionError", str(exc))

        except APIStatusError as exc:
            return self._error(
                model, "APIStatusError", str(exc), getattr(exc, "status_code", None)
            )

        except (IndexError, AttributeError) as exc:
            return self._error(model, "ResponseParsingError", str(exc), 200)

        except Exception as exc:
            return self._error(model, "UnexpectedError", str(exc))

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
            "provider": self.PROVIDER_NAME,
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
            "provider": self.PROVIDER_NAME,
            "model": model,
            "status_code": status_code,
            "output": None,
            "error": {
                "type": error_type,
                "message": message,
            },
        }


class GroqClient(BaseLLMClient):
    """Client for Groq's OpenAI-compatible API."""

    PROVIDER_NAME = "groq"
    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "openai/gpt-oss-20b"


class OpenCodeClient(BaseLLMClient):
    """Client for OpenCode Zen's OpenAI-compatible API."""

    PROVIDER_NAME = "opencode"
    BASE_URL = "https://opencode.ai/zen/v1"
    DEFAULT_MODEL = "deepseek-v4-flash-free"


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
#
# groq = GroqClient(api_key="...", model="openai/gpt-oss-20b")
# result = groq.generate("Hello, how are you?")
#
# opencode = OpenCodeClient(api_key="...", model="deepseek-v4-flash-free")
# result = opencode.generate("Hello, how are you?")
#
# # Override the model for a single call without affecting the instance default
# result = opencode.generate("Hello, how are you?", model="some-other-model")