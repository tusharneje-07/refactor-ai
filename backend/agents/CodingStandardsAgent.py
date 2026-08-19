"""
agents/CodingStandardsAgent.py

CodingStandardsAgent
---------------------
Reviews a single source file (any programming language) and returns
low-risk, non-overlapping coding-standards suggestions as strict JSON.

Scope: naming conventions, dead code, structure/complexity, missing
docs, duplication, formatting/idiom consistency.
Out of scope: security, performance, logic/runtime bugs.

Depends on:
    modules/llm_client.py   (LLMClient)
    model_config.json       (backend root, sibling of agents/ and modules/)

Usage:
    from agents.CodingStandardsAgent import run_coding_standards_agent

    result = run_coding_standards_agent(
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
# and standalone, and resolve modules/llm_client.py + model_config.json
# relative to the backend root (parent of this file's directory).
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(_THIS_DIR)

if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from modules.llm_client import LLMClient


AGENT_NAME = "CodingStandardsAgent"
MODEL_CONFIG_PATH = os.path.join(_BACKEND_ROOT, "model_config.json")
MAX_RETRIES_ON_INVALID_JSON = 1

REQUIRED_KEYS = {
    "suggestion_title",
    "suggestion_description",
    "line_no_from",
    "line_no_to",
    "replace_by",
}


SYSTEM_PROMPT = """
You are CodingStandardsAgent, an expert reviewer focused ONLY on coding standards, style, and maintainability. You support ALL programming languages. Determine the language from the file name/extension and source content before reviewing.

SCOPE — review ONLY:
- Naming conventions (variables, functions, classes, constants) against the idiomatic style of the detected language
- Dead code: unused variables, unused imports, unreachable code
- Missing or inconsistent documentation (docstrings/comments) where the language convention expects them on public functions/classes
- Structure and organization: excessive function length, deep nesting, poor separation of concerns
- Avoidable duplication
- Formatting/style inconsistency within the file (mixed indentation, inconsistent spacing or quote style)
- Idiom violations (not using standard language constructs where clearly applicable)

OUT OF SCOPE — never report, even if noticed:
- Security vulnerabilities
- Performance issues
- Logic bugs or correctness issues
- Syntax or runtime errors

MANDATORY FULL-FILE SCAN PROCEDURE (perform this before writing any output):
1. Read the ENTIRE file from line 1 to the final line. Do not stop scanning after finding your first few issues of a given category.
2. Build a mental (or scratch) inventory of EVERY instance of EVERY in-scope issue type, independently, top to bottom. Treat each category separately — e.g. if you check for unused variables, check EVERY declaration in the file for usage, not just the first cluster you encounter. The same applies to naming violations, duplication, dead code, etc.
3. Do NOT stop scanning a category once you've found one or two examples. A single issue type (e.g. "unused variable") may legitimately occur in multiple, unrelated locations throughout the file (e.g. lines 10-12 AND lines 30-32 AND line 88). Each genuinely distinct occurrence is a separate finding candidate, not a duplicate to be skipped.
4. After building the full inventory across the whole file, select as much as candidates from across the ENTIRE line range — not just from the first section of the file. Do not let findings cluster only in the early lines when later lines have equally valid, equally confident issues.
5. If you find valid candidates, prioritize by: (a) confidence, (b) impact on maintainability, (c) spreading coverage across different parts of the file rather than reporting several findings from the same small region while ignoring other regions.

ANTI-HALLUCINATION RULES (mandatory):
1. The source is given with explicit 1-based line numbers. Use ONLY those exact numbers. Never invent, guess, estimate, or offset a line number.
2. Only report a finding if you can point to the exact line(s) as shown in the provided listing. If unsure a line number is correct, DROP the finding instead of guessing.
3. Never reference code, files, functions, libraries, or requirements that are not explicitly present in the provided input (prompt, project_context, git_context, code_file).
4. Do not assume intent or behavior beyond what is stated.
5. line_no_from and line_no_to must be integers within the file's actual line range, and line_no_from <= line_no_to.
6. replace_by must contain ONLY the exact replacement text for lines line_no_from..line_no_to inclusive. No surrounding unrelated lines, no ellipses, no placeholder comments (e.g. "# rest unchanged"), no stubbing-out of real code.
7. replace_by must preserve correct indentation for the language, encoded as a valid JSON string using \n between lines and \t for tab indentation.
8. Findings must be non-overlapping: no two findings may share or touch the same line number. If two issues touch the same lines, merge them into one finding or drop one.
9. Return as much as findings.
10. Never fabricate a finding just to have something to report. An empty array is a valid, good answer.

MANDATORY OUTPUT ORDERING:
After you have selected your final set of findings (per the scan procedure and anti-hallucination rules above), you MUST sort them by ascending line_no_from before writing the JSON array — the finding with the smallest line_no_from comes first, then the next smallest, and so on, strictly in top-to-bottom file order. Do not order findings by category, confidence, or discovery order. Re-check the final array before output: if any finding's line_no_from is smaller than the line_no_from of a finding listed before it, you have made an ordering error and must fix it before returning the result.

OUTPUT FORMAT — CRITICAL:
Return ONLY a raw JSON array, nothing else. No markdown code fences, no prose before or after.
Each element must be an object with EXACTLY these keys and types:
- "suggestion_title": string, short title
- "suggestion_description": string, why this change is needed
- "line_no_from": integer, 1-based inclusive start line
- "line_no_to": integer, 1-based inclusive end line
- "replace_by": string, exact replacement code for that line range, using \n for newlines and \t for tab indents

If there are no findings, return exactly: []
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_config(agent_name: str = AGENT_NAME) -> Dict[str, Any]:
    """Load this agent's entry from model_config.json."""
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    entry = config.get(agent_name)
    if not entry:
        raise ValueError(
            f"No config entry for '{agent_name}' found in {MODEL_CONFIG_PATH}"
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
    prompt: str,
    project_context: Dict[str, Any],
    git_context: Dict[str, Any],
    code_file: Dict[str, Any],
) -> str:
    filename = code_file.get("filename", "unknown")
    total_lines = code_file.get("total_lines", len(code_file.get("lines", {})))
    source_listing = _numbered_source(code_file)

    return (
        f"TASK INSTRUCTIONS:\n{prompt}\n\n"
        f"PROJECT CONTEXT:\n{json.dumps(project_context, indent=2)}\n\n"
        f"GIT CONTEXT:\n{json.dumps(git_context, indent=2)}\n\n"
        f"FILE: {filename}\n"
        f"TOTAL LINES: {total_lines}\n\n"
        f"SOURCE (format is '<line_number>\\t<line_content>'):\n"
        f"{source_listing}\n"
    )


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


def run_coding_standards_agent(
    prompt: str,
    project_context: Dict[str, Any],
    git_context: Dict[str, Any],
    code_file: Dict[str, Any],
    model: str = "",
    api_key: str = "",
) -> Dict[str, Any]:
    """
    Execute CodingStandardsAgent on a single file.

    Args:
        prompt: Task instructions passed to the agent.
        project_context: Project metadata dict.
        git_context: Git-related metadata dict (may be empty).
        code_file: {"filename": str, "total_lines": int, "lines": {"1": "...", ...}}
        model: Optional model name overriding model_config.json's configured model.
        api_key: Optional API key passed directly (e.g. from KeyPool).

    Returns:
        {
            "success": bool,
            "agent": "CodingStandardsAgent",
            "model": str,
            "suggestions": [ {suggestion_title, suggestion_description,
                               line_no_from, line_no_to, replace_by}, ... ],
            "error": {"type": str, "message": str} | None
        }
    """
    if not isinstance(code_file, dict) or "lines" not in code_file:
        return {
            "success": False,
            "agent": AGENT_NAME,
            "model": model,
            "suggestions": None,
            "error": {"type": "ValidationError", "message": "code_file must include a 'lines' map."},
        }


    try:
        config_entry = _load_model_config()
        model_name = model or config_entry.get("use_model")
        provider_name = config_entry.get("provider")
        
        client = LLMClient(provider=provider_name, api_key=api_key, model=model_name)
        system_prompt = SYSTEM_PROMPT
        user_prompt = _build_user_message(prompt, project_context, git_context, code_file)
        
        client_response = client.generate(system=system_prompt, user=user_prompt)
        print(f"Client response: {client_response}")  # Debugging line
        if not client_response.get("success"):
            return {
                "success": False,
                "agent": AGENT_NAME,
                "model": model_name,
                "suggestions": None,
                "error": client_response.get("error", {"type": "UnknownError", "message": "Agent returned no output"}),
            }

        output = client_response.get("output", "")
        if not output or not output.strip():
            return {
                "success": False,
                "agent": AGENT_NAME,
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
            suggestion.setdefault("suggestion_id", hashlib.sha256(f"CodingStandardsAgent_{time.time() * 1000}_{len(final_suggestions)}".encode("utf-8")).hexdigest())
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
            "agent": AGENT_NAME,
            "model": model_name,
            "suggestions": final_suggestions,
        }
    except Exception as exc:
        return {
            "success": False,
            "agent": AGENT_NAME,
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

    output = run_coding_standards_agent(
        prompt="Review this file for coding standards issues.",
        project_context={"project_name": "Simple Tic Tac Toe", "technology": "Python"},
        git_context={},
        code_file=example_code_file,
    )
    print(json.dumps(output, indent=2))