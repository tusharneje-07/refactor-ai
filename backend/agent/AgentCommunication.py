"""
agents/AgentCommunication.py

AgentCommunication
------------------
Single shared communication layer for ALL agents. Instead of one file per
agent, every agent's configuration (system prompt, model, provider and
user_prompt_sections) lives in agent_config.json. This module fetches the
config entry for the requested agent, builds the system/user prompts and
calls the LLM through modules/llm_client.py.

Depends on:
    modules/llm_client.py   (LLMClient)
    agent_config.json       (backend root, sibling of agents/ and modules/)

Usage:
    from agent.AgentCommunication import run_agent

    result = run_agent(
        agent_name="CodingStandardsAgent",
        prompt="Review this file for coding standards issues.",
        project_context={...},
        git_context={},
        code_file={...},   # see CodeFile schema below
    )
"""

import json
import os
import sys
import hashlib
import time
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Path setup — allow running this file both as part of the `agents` package
# and standalone, and resolve modules/llm_client.py + agent_config.json
# relative to the backend root (parent of this file's directory).
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_THIS_DIR)

if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from modules.llm_client import LLMClient


AGENT_CONFIG_PATH = os.path.join(_BACKEND_ROOT, "agent_config.json")
MAX_RETRIES_ON_INVALID_JSON = 1

REQUIRED_KEYS = {
    "suggestion_title",
    "suggestion_description",
    "line_no_from",
    "line_no_to",
    "replace_by",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_agent_config(agent_name: str) -> Dict[str, Any]:
    """Load the given agent's entry from agent_config.json."""
    with open(AGENT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    entry = config.get(agent_name)
    if not entry:
        raise ValueError(
            f"No config entry for '{agent_name}' found in {AGENT_CONFIG_PATH}"
        )
    return entry


def _numbered_source(code_file: Dict[str, Any]) -> str:
    """
    Build a 1-based numbered source listing from a CodeFile dict shaped as:
        {
            "filename": str,
            "total_lines": int,
            "lines": {"1": "...", "2": "...", ...}
        }
    """
    lines_map = code_file.get("lines", {})
    try:
        ordered_line_numbers = sorted(lines_map.keys(), key=int)
    except (TypeError, ValueError):
        ordered_line_numbers = list(lines_map.keys())

    listing_lines = []
    for line_no in ordered_line_numbers:
        listing_lines.append(f"{line_no}\t{lines_map[line_no]}")
    return "\n".join(listing_lines)


def _build_user_message(
    sections: list,
    prompt: str,
    project_context: Dict[str, Any],
    git_context: Dict[str, Any],
    code_file: Dict[str, Any],
) -> str:
    """
    Build the user message from the `user_prompt_sections` list defined in
    agent_config.json. Only sections listed there are included.

    Known section tokens:
        "<instruction>"     -> the prompt/task instructions
        "<project_context>" -> project context dict
        "<git_context>"     -> git context dict
        "<code_file>"       -> the numbered source file listing
    Any other string is treated as a custom message and included verbatim.
    """
    filename = code_file.get("filename", "unknown")
    total_lines = code_file.get("total_lines", len(code_file.get("lines", {})))

    parts = []
    for section in sections:
        if section == "<instruction>":
            parts.append(f"TASK INSTRUCTIONS:\n{prompt}")
        elif section == "<project_context>":
            parts.append(f"PROJECT CONTEXT:\n{json.dumps(project_context, indent=2)}")
        elif section == "<git_context>":
            parts.append(f"GIT CONTEXT:\n{json.dumps(git_context, indent=2)}")
        elif section == "<code_file>":
            source_listing = _numbered_source(code_file)
            parts.append(
                f"Understand the project context and git context, then review the source file below.\n\n"
                f"FILE: {filename}\n"
                f"TOTAL LINES: {total_lines}\n\n"
                f"SOURCE (format is '<line_number>\\t<line_content>'):\n"
                f"{source_listing}\n"
            )
        else:
            parts.append(section)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences wrapping JSON output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        elif "```" in text:
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _repair_json(text: str) -> str:
    """Fix common JSON issues from LLM output: unescaped quotes, newlines, truncation."""
    result = []
    in_string = False
    escaped = False
    i = 0

    while i < len(text):
        ch = text[i]

        if escaped:
            result.append(ch)
            escaped = False
            i += 1
            continue

        if ch == '\\' and in_string:
            result.append(ch)
            escaped = True
            i += 1
            continue

        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
            else:
                next_idx = i + 1
                while next_idx < len(text) and text[next_idx] in ' \t\n\r':
                    next_idx += 1
                next_ch = text[next_idx] if next_idx < len(text) else ''

                if next_ch in (',', ']', '}', ':', ''):
                    in_string = False
                    result.append(ch)
                elif next_ch == '"' and next_idx + 1 < len(text) and text[next_idx + 1] == '"':
                    result.append('\\"')
                else:
                    result.append('\\"')
            i += 1
            continue

        if in_string and ch == '\n':
            result.append('\\n')
            i += 1
            continue

        if in_string and ch == '\r':
            result.append('\\r')
            i += 1
            continue

        if in_string and ch == '\t':
            result.append('\\t')
            i += 1
            continue

        result.append(ch)
        i += 1

    repaired = ''.join(result)

    if in_string:
        repaired += '"'

    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]

    while repaired and repaired[-1] not in ('"', '}', ']', ':', ',', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n', 'e'):
        repaired = repaired[:-1]

    if repaired.endswith(","):
        repaired = repaired[:-1]

    if repaired.endswith(":"):
        repaired += 'null'

    open_brackets = repaired.count("[") - repaired.count("]")
    open_braces = repaired.count("{") - repaired.count("}")

    if open_braces > 0:
        repaired += "}" * open_braces
    if open_brackets > 0:
        repaired += "]" * open_brackets

    return repaired


def run_agent(
    agent_name: str,
    prompt: str,
    project_context: Dict[str, Any],
    git_context: Dict[str, Any],
    code_file: Dict[str, Any],
    model: str = "",
    api_key: str = "",
) -> Dict[str, Any]:
    """
    Execute any configured agent on a single file.

    Args:
        agent_name: Config key in agent_config.json (e.g. "CodingStandardsAgent").
        prompt: Task instructions passed to the agent (<instruction> section).
        project_context: Project metadata dict.
        git_context: Git-related metadata dict (may be empty).
        code_file: {"filename": str, "total_lines": int, "lines": {"1": "...", ...}}
        model: Optional model name overriding agent_config.json's configured model.
        api_key: Optional API key passed directly (e.g. from KeyPool).

    Returns:
        {
            "success": bool,
            "agent": str,
            "model": str,
            "suggestions": [ {suggestion_title, suggestion_description,
                               line_no_from, line_no_to, replace_by}, ... ],
            "error": {"type": str, "message": str} | None
        }
    """
    if not isinstance(code_file, dict) or "lines" not in code_file:
        return {
            "success": False,
            "agent": agent_name,
            "model": model,
            "suggestions": None,
            "error": {"type": "ValidationError", "message": "code_file must include a 'lines' map."},
        }


    try:
        config_entry = _load_agent_config(agent_name)
        model_name = model or config_entry.get("use_model")
        provider_name = config_entry.get("provider")

        client = LLMClient(provider=provider_name, api_key=api_key, model=model_name)

        system_prompt = config_entry.get("system_prompt", "")
        sections = config_entry.get("user_prompt_sections", ["<instruction>", "<code_file>"])
        user_prompt = _build_user_message(sections, prompt, project_context, git_context, code_file)

        with open(f"{_BACKEND_ROOT}/debug/debug_{agent_name}_input.txt", "w") as f:
            f.write(f"SYSTEM PROMPT:\n{system_prompt}\n\n")
            f.write(f"USER PROMPT:\n{user_prompt}\n\n")

        client_response = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)

        if not client_response.get("success"):
            return {
                "success": False,
                "agent": agent_name,
                "model": model_name,
                "suggestions": None,
                "error": client_response.get("error", {"type": "UnknownError", "message": "Agent returned no output"}),
            }

        output = client_response.get("output", "")
        if not output or not output.strip():
            return {
                "success": False,
                "agent": agent_name,
                "model": model_name,
                "suggestions": None,
                "error": {"type": "EmptyResponseError", "message": "Model returned empty output"},
            }

        cleaned = _strip_code_fences(output)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = _repair_json(cleaned)
            parsed = json.loads(repaired)

        final_suggestions = []
        # Batch ID is used to group suggestions from the same agent run, useful for tracking and rollback.
        batch_id = hashlib.md5(str(time.time() * 1000).encode()).hexdigest()  # Unique batch ID based on current time in milliseconds
        for suggestion in parsed:
            if not isinstance(suggestion, dict):
                continue
            suggestion.setdefault("suggestion_id", hashlib.sha256(f"{agent_name}_{time.time() * 1000}_{len(final_suggestions)}".encode("utf-8")).hexdigest())
            suggestion.setdefault("suggestion_description", "")
            suggestion.setdefault("line_no_from", 0)
            suggestion.setdefault("line_no_to", 0)
            suggestion.setdefault("replace_by", "")
            suggestion.setdefault("old_lines", "")
            suggestion.setdefault("suggestion_state", "pending")
            suggestion.setdefault("batch_id", batch_id)
            lines_map = code_file["lines"]
            if suggestion["line_no_from"] > 0 and suggestion["line_no_to"] >= suggestion["line_no_from"]:
                old_lines_parts = []
                for i in range(suggestion["line_no_from"], suggestion["line_no_to"] + 1):
                    line = lines_map.get(i) or lines_map.get(str(i)) or ""
                    old_lines_parts.append(line)
                suggestion["old_lines"] = "\n".join(old_lines_parts)
            final_suggestions.append(suggestion)
        return {
            "success": True,
            "agent": agent_name,
            "model": model_name,
            "suggestions": final_suggestions,
        }
    except Exception as exc:
        return {
            "success": False,
            "agent": agent_name,
            "model": model,
            "suggestions": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


if __name__ == "__main__":
    example_code_file = {
        "filename": "game.py",
        "total_lines": 5,
        "lines": {
            "1": "import random",
            "2": "unused_variable = 'hello'",
            "3": "",
            "4": "def show_board():",
            "5": "    pass",
        },
    }

    output = run_agent(
        agent_name="CodingStandardsAgent",
        prompt="Review this file for coding standards issues.",
        project_context={"project_name": "Simple Tic Tac Toe", "technology": "Python"},
        git_context={},
        code_file=example_code_file,
    )
    print(json.dumps(output, indent=2))
