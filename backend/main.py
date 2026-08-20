import hashlib
import json
import os
import fnmatch
from dotenv import load_dotenv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import true

from modules.sqlite_io import SQLiteDB
from modules.file_io import read_file_content, replace_file_content
from agents.CodingStandardsAgent import run_coding_standards_agent
from open_keypool import KeyPool

load_dotenv()

app = FastAPI(title="RefactorAI")

BASE_DIR = Path(__file__).resolve().parent
CENTRAL_DB_PATH = BASE_DIR / "data" / ".airefactor_central.db"
STATE_FILE = BASE_DIR / ".state.json"

EXTENSIONS_TO_EXCLUDE = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin"}

def _ensure_central_db():
    db = SQLiteDB(str(CENTRAL_DB_PATH))
    db.runQuery(
        """CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            project_path TEXT NOT NULL,
            project_name TEXT NOT NULL,
            github_url TEXT,
            createdby TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )


def _compute_project_id(project_path: str) -> str:
    return hashlib.sha256(project_path.encode("utf-8")).hexdigest()


def _build_file_tree(folder_path: str) -> dict:
    """Recursively build a file/folder tree dict. Excludes dotfiles."""
    tree: dict = {}
    try:
        entries = sorted(os.scandir(folder_path), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return tree

    for entry in entries:
        name = entry.name
        if name.startswith("."):
            continue
        if entry.is_dir():
            tree[name] = _build_file_tree(entry.path)
        else:
            ext = os.path.splitext(name)[1]
            if ext not in EXTENSIONS_TO_EXCLUDE:
                tree[name] = None  # None marks a file
    return tree


def _parse_gitignore(project_path: str) -> list[str]:
    gitignore_path = os.path.join(project_path, ".gitignore")
    patterns = []
    if not os.path.isfile(gitignore_path):
        return patterns
    try:
        with open(gitignore_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                patterns.append(line)
    except Exception:
        pass
    return patterns


def _is_ignored(rel_path: str, is_dir: bool, gitignore_patterns: list[str]) -> bool:
    name = os.path.basename(rel_path)
    for pattern in gitignore_patterns:
        p = pattern.rstrip("/")
        is_dir_pattern = pattern.endswith("/")
        if is_dir_pattern and not is_dir:
            continue
        if fnmatch.fnmatch(name, p):
            return True
        if fnmatch.fnmatch(rel_path, p):
            return True
        if "/" not in p and fnmatch.fnmatch(name, p):
            return True
    return False


def _read_all_project_files(project_path: str, gitignore_patterns: list[str]) -> dict:
    file_contents = {}
    for root, dirs, files in os.walk(project_path):
        rel_root = os.path.relpath(root, project_path)
        if rel_root == ".":
            rel_root = ""

        dirs_to_remove = []
        for d in dirs:
            rel_dir = os.path.join(rel_root, d) if rel_root else d
            if d.startswith(".") or _is_ignored(rel_dir, True, gitignore_patterns):
                dirs_to_remove.append(d)
        for d in dirs_to_remove:
            dirs.remove(d)

        for fname in files:
            if fname.startswith(".") and fname != ".gitignore":
                continue
            ext = os.path.splitext(fname)[1]
            if ext in EXTENSIONS_TO_EXCLUDE:
                continue
            rel_file = os.path.join(rel_root, fname) if rel_root else fname
            if _is_ignored(rel_file, False, gitignore_patterns):
                continue
            full_path = os.path.join(root, fname)
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                file_contents[rel_file] = content
            except Exception:
                file_contents[rel_file] = "<UNREADABLE: binary or permission error>"
    return file_contents


def _build_project_context_for_agent(project_path: str, project_id: str) -> str:
    local_db_path = os.path.join(project_path, ".airefactor.db")
    if not os.path.isfile(local_db_path):
        raise HTTPException(status_code=404, detail="Project local DB not found")

    db = SQLiteDB(local_db_path)

    db.runQuery(
        "UPDATE {table} SET suggestion_state = ? WHERE suggestion_state = ?",
        table_name="suggestions",
        params=("discarded-[user_refresh]", "pending"),
    )

    project_context = {}
    err, rows = db.read(
        "project_info",
        "SELECT project_context FROM {table} WHERE project_id = ?",
        params=(project_id,),
    )
    if not err and rows and rows[0].get("project_context"):
        try:
            project_context = json.loads(rows[0]["project_context"])
        except (json.JSONDecodeError, TypeError):
            pass

    gitignore_patterns = _parse_gitignore(project_path)
    file_contents = _read_all_project_files(project_path, gitignore_patterns)

    err, all_suggestions = db.read(
        "suggestions",
        "SELECT suggestion_description, line_no_from, line_no_to, old_lines, "
        "replace_by, agent_id, batch_id, suggestion_state FROM {table} "
        "ORDER BY created_at ASC",
    )
    if err:
        all_suggestions = []

    output_parts = []

    output_parts.append("PROJECT_CONTEXT:")
    output_parts.append(json.dumps(project_context, indent=2, ensure_ascii=False))
    output_parts.append("")

    output_parts.append("PROJECT_FILE_CONTENT:")
    output_parts.append(json.dumps(list(file_contents.keys()), indent=2, ensure_ascii=False))
    output_parts.append("")

    for fname, content in sorted(file_contents.items()):
        output_parts.append(f"FILE_NAME: {fname}")
        output_parts.append(content)
        output_parts.append("")

    output_parts.append("PREVIOUS_SUGGESTIONS:")
    output_parts.append(json.dumps(all_suggestions, indent=2, ensure_ascii=False, default=str))

    return "\n".join(output_parts)


def _create_project_local_db(folder_path: str, project_id: str, project_name: str,
                              github_url: Optional[str], createdby: str,
                              project_context: Optional[dict] = None):
    db_path = os.path.join(folder_path, ".airefactor.db")
    db = SQLiteDB(db_path)
    now = datetime.now(timezone.utc).isoformat()
    context_json = json.dumps(project_context) if project_context is not None else None
    db.runQuery(
        """CREATE TABLE IF NOT EXISTS project_info (
            project_id TEXT PRIMARY KEY,
            project_path TEXT NOT NULL,
            project_name TEXT NOT NULL,
            github_url TEXT,
            createdby TEXT NOT NULL,
            project_context TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    db.runQuery(
        """INSERT OR REPLACE INTO project_info
           (project_id, project_path, project_name, github_url, createdby, project_context, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        params=(project_id, folder_path, project_name, github_url, createdby, context_json, now, now),
    )
    db.runQuery(
        """CREATE TABLE IF NOT EXISTS suggestions (
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            suggestion_id TEXT PRIMARY KEY,
            suggestion_description TEXT NOT NULL,
            line_no_from INTEGER NOT NULL,
            line_no_to INTEGER NOT NULL,
            old_lines TEXT NOT NULL,
            replace_by TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            suggestion_state TEXT NOT NULL
        )"""
    )


class OpenProjectRequest(BaseModel):
    is_new: bool
    project_name: Optional[str] = None
    folder_path: Optional[str] = None
    project_id: Optional[str] = None
    project_context: Optional[dict] = None


class OpenProjectResponse(BaseModel):
    project_id: str
    project_name: str
    project_path: str
    new_project: bool
    tree: dict


@app.post(
    "/open_project",
    summary="Open or register a project",
    description=(
        "Opens an existing project or registers a new one. "
        "When `is_new` is true, creates entries in both the central database and "
        "a local `.airefactor.db` inside the project folder. "
        "Always writes the current project state to `.state.json` and returns the "
        "full file/folder tree of the project (excluding dotfiles and compiled binaries)."
    ),
    response_model=OpenProjectResponse,
    status_code=200,
    responses={
        400: {"description": "Invalid folder_path or missing required parameter"},
        500: {"description": "Internal database error"},
    },
)
def open_project(req: OpenProjectRequest):
    _ensure_central_db()

    central_db = SQLiteDB(str(CENTRAL_DB_PATH))

    if not req.is_new:
        if not req.project_id:
            raise HTTPException(status_code=400, detail="project_id is required when is_new is false")

        err, rows = central_db.read(
            "projects",
            "SELECT * FROM {table} WHERE project_id = ?",
            params=(req.project_id,),
        )
        if err:
            raise HTTPException(status_code=500, detail="Database read error")
        if not rows:
            raise HTTPException(status_code=404, detail="Project not found in central DB")

        project = rows[0]
        project_id = project["project_id"]
        project_name = project["project_name"]
        folder_path = project["project_path"]

        now = datetime.now(timezone.utc).isoformat()
        central_db.runQuery(
            "UPDATE {table} SET updated_at = ? WHERE project_id = ?",
            table_name="projects",
            params=(now, project_id),
        )

        if not os.path.isdir(folder_path):
            raise HTTPException(status_code=400, detail="Project path no longer exists on disk")

        state = {
            "project_id": project_id,
            "project_path": folder_path,
            "project_name": project_name,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

        tree = _build_file_tree(folder_path)
        return {
            "project_id": project_id,
            "project_name": project_name,
            "project_path": folder_path,
            "new_project": False,
            "tree": tree,
        }

    if not req.folder_path:
        raise HTTPException(status_code=400, detail="folder_path is required when is_new is true")
    if not req.project_name:
        raise HTTPException(status_code=400, detail="project_name is required when is_new is true")
    if req.project_context is None:
        raise HTTPException(status_code=400, detail="project_context is required when is_new is true")

    folder_path = os.path.abspath(req.folder_path)
    if not os.path.isdir(folder_path):
        raise HTTPException(status_code=400, detail="folder_path is not a valid directory")

    project_id = _compute_project_id(folder_path)
    project_name = req.project_name
    github_url: Optional[str] = None
    createdby = "local"

    err, rows = central_db.read(
        "projects",
        "SELECT * FROM {table} WHERE project_id = ?",
        params=(project_id,),
    )
    if err:
        raise HTTPException(status_code=500, detail="Database read error")

    is_new_project = len(rows) == 0
    now = datetime.now(timezone.utc).isoformat()

    err, _ = central_db.runQuery(
        """INSERT OR REPLACE INTO projects
           (project_id, project_path, project_name, github_url, createdby, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        params=(project_id, folder_path, project_name, github_url, createdby, now, now),
    )
    if err:
        raise HTTPException(status_code=500, detail="Failed to insert project into central DB")

    _create_project_local_db(
        folder_path, project_id, project_name, github_url, createdby,
        project_context=req.project_context,
    )

    state = {
        "project_id": project_id,
        "project_path": folder_path,
        "project_name": project_name,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

    tree = _build_file_tree(folder_path)

    return {
        "project_id": project_id,
        "project_name": project_name,
        "project_path": folder_path,
        "new_project": is_new_project,
        "tree": tree,
    }


class ReadFileRequest(BaseModel):
    filename: str


class ReadFileResponse(BaseModel):
    project_id: str
    filename: str
    total_lines: int
    lines: dict[int, str]


@app.post(
    "/read_file",
    summary="Read a file from the active project",
    description=(
        "Returns the contents of a file inside the currently active project "
        "(stored in `.state.json` by `/open-project`) with line numbers. "
        "The `filename` is relative to the project root (e.g. `src/main.py`)."
    ),
    response_model=ReadFileResponse,
    status_code=200,
    responses={
        400: {"description": "No active project in .state.json"},
        404: {"description": "File not found"},
        500: {"description": "Failed to read file"},
    },
)
def read_file(req: ReadFileRequest):
    if not STATE_FILE.is_file():
        raise HTTPException(status_code=400, detail="No active project. Call /open-project first.")

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    project_id = state["project_id"]
    project_path = state["project_path"]
    file_path = os.path.join(project_path, req.filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        result = read_file_content(file_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read file")

    return {
        "project_id": project_id,
        "filename": req.filename,
        "total_lines": result["total_lines"],
        "lines": result["lines"],
    }


class ValidateProjectPathRequest(BaseModel):
    folder_path: str


class ValidateProjectPathResponse(BaseModel):
    exists_in_db: bool
    path_exists: bool
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    project_path: Optional[str] = None


@app.post(
    "/validate_project_path",
    summary="Check if a folder path is a registered project or exists on disk",
    description=(
        "Checks whether the given folder path is already registered in the central "
        "database. If found, returns the stored project info. If not found, checks "
        "whether the path exists on the filesystem and reports accordingly."
    ),
    response_model=ValidateProjectPathResponse,
    status_code=200,
    responses={
        400: {"description": "Invalid or empty folder_path"},
        500: {"description": "Database read error"},
    },
)
def validate_project_path(req: ValidateProjectPathRequest):
    _ensure_central_db()

    if not req.folder_path:
        raise HTTPException(status_code=400, detail="folder_path is required")

    folder_path = os.path.abspath(req.folder_path)
    path_exists = os.path.isdir(folder_path)

    project_id = _compute_project_id(folder_path)

    central_db = SQLiteDB(str(CENTRAL_DB_PATH))
    err, rows = central_db.read(
        "projects",
        "SELECT * FROM {table} WHERE project_id = ?",
        params=(project_id,),
    )
    if err:
        raise HTTPException(status_code=500, detail="Database read error")

    if rows:
        return {
            "exists_in_db": True,
            "path_exists": True,
            "project_id": rows[0]["project_id"],
            "project_name": rows[0]["project_name"],
            "project_path": rows[0]["project_path"],
        }

    return {
        "exists_in_db": False,
        "path_exists": path_exists,
        "project_id": None,
        "project_name": None,
        "project_path": None,
    }


class RefactorRequest(BaseModel):
    project_id: str
    filename: str
    agent_id: str


class RefactorResponse(BaseModel):
    success: bool
    agent: str
    suggestions: Optional[list[dict]] = None
    model: str
    error: Optional[dict] = None


def _get_agent_config_by_id(agent_id: str) -> Optional[dict]:
    model_config_path = BASE_DIR / "model_config.json"
    try:
        with open(model_config_path, "r") as f:
            config = json.load(f)
    except Exception:
        return None

    for agent_name, agent_config in config.items():
        if agent_config.get("agent_id") == agent_id:
            result = dict(agent_config)
            result["config_key"] = agent_name
            return result
    return None


def _run_coding_standards_flow(project_id: str, filename: str):
    if not STATE_FILE.is_file():
        raise HTTPException(status_code=400, detail="No active project. Call /open-project first.")

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    if state.get("project_id") != project_id:
        raise HTTPException(status_code=400, detail="project_id does not match active project")

    project_path = state["project_path"]
    file_path = os.path.join(project_path, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail="File not found in project")

    try:
        file_content = read_file_content(file_path)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read file")

    code_file = {
        "filename": filename,
        "total_lines": file_content["total_lines"],
        "lines": file_content["lines"],
    }

    project_context = {
        "project_name": state["project_name"],
    }

    local_db_path = os.path.join(project_path, ".airefactor.db")
    if os.path.isfile(local_db_path):
        local_db = SQLiteDB(local_db_path)
        err, rows = local_db.read(
            "project_info",
            "SELECT project_context FROM {table} WHERE project_id = ?",
            params=(state["project_id"],),
        )
        if not err and rows and rows[0].get("project_context"):
            try:
                project_context = json.loads(rows[0]["project_context"])
            except (json.JSONDecodeError, TypeError):
                pass

    # result = run_coding_standards_agent(
    #     prompt="Review this file for coding standards issues.",
    #     project_context=project_context,
    #     git_context={},
    #     code_file=code_file,
    #     api_key=str(os.getenv("GROQ_API_KEY")),
    # )

    # [DEV] Simulation of coding standards agent response for testing purposes
    result = {
        "success": true,
        "agent": "CodingStandardsAgent",
        "suggestions": [
            {
            "suggestion_title": "Rename variable to snake_case",
            "suggestion_description": "Variable 'randomThing' should follow snake_case naming convention.",
            "line_no_from": 8,
            "line_no_to": 8,
            "replace_by": "random_thing = random.randint(1, 100)",
            "suggestion_id": "2501c051e2bbcd24478e39b11df5cc3ba6004fd27e7ed630f31b63ad0f51cfe4",
            "old_lines": "randomThing = random.randint(1, 100)",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Remove unused variable",
            "suggestion_description": "'unused_variable' is never used.",
            "line_no_from": 9,
            "line_no_to": 9,
            "replace_by": "",
            "suggestion_id": "9a3d0920c69c9712dc8d405a38e984a8ca3c38416ff7954982699bef584d8788",
            "old_lines": "unused_variable = \"hello\"",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Remove unused variable",
            "suggestion_description": "'another_unused_variable' is never used.",
            "line_no_from": 10,
            "line_no_to": 10,
            "replace_by": "",
            "suggestion_id": "4abbd7b51ea602299f0b09508f3ff65494ffd45725a659f314aeaafc9143d641",
            "old_lines": "another_unused_variable = None",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Remove unused variable",
            "suggestion_description": "'board_size' is never used.",
            "line_no_from": 11,
            "line_no_to": 11,
            "replace_by": "",
            "suggestion_id": "4023012bd85edffb2fc3dfd054daf5dcea7e038582b5f443c0b6f9ad16c4184e",
            "old_lines": "board_size = 3",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Remove dead code",
            "suggestion_description": "Variables 'x', 'y', and 'z' are never used.",
            "line_no_from": 12,
            "line_no_to": 14,
            "replace_by": "",
            "suggestion_id": "5939cac144242287c829b12c8a10a3466207c7cc3c4b682efa95e4d0ab01f996",
            "old_lines": "x = 0\ny = 0\nz = \"nothing\"",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Add docstring to show_board",
            "suggestion_description": "Public function should have a docstring describing its purpose.",
            "line_no_from": 18,
            "line_no_to": 18,
            "replace_by": "def show_board():\n    \"\"\"Display the current game board.\"\"\"",
            "suggestion_id": "38a8d665f1d4690098ecf0bcfab79f810fa39d0c6e02456a8ba5a85e7802e511",
            "old_lines": "def show_board():",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Add docstring to check_winner",
            "suggestion_description": "Public function should have a docstring describing its purpose.",
            "line_no_from": 28,
            "line_no_to": 28,
            "replace_by": "def check_winner():\n    \"\"\"Check the board for a winner and return the winning symbol, or None.\"\"\"",
            "suggestion_id": "3f5cb5b84ed892d47f0072c07c5e1926e67319d2c039e27e217a64e55e7aae19",
            "old_lines": "def check_winner():",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Add docstring to board_full",
            "suggestion_description": "Public function should have a docstring describing its purpose.",
            "line_no_from": 56,
            "line_no_to": 56,
            "replace_by": "def board_full():\n    \"\"\"Return True if the board has no empty spaces, otherwise False.\"\"\"",
            "suggestion_id": "0cf5001668aa8ceb8a074ec5baa647f67fd3059575968807c9a95ab0ed03ac7f",
            "old_lines": "def board_full():",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Add docstring to play_game",
            "suggestion_description": "Public function should have a docstring describing its purpose.",
            "line_no_from": 63,
            "line_no_to": 63,
            "replace_by": "def play_game():\n    \"\"\"Main game loop handling player turns and game outcome.\"\"\"",
            "suggestion_id": "6d86ee452c3cf9d6d69c1dae65c9b73de77a62fdfae1c9eb78cbbed2b3215405",
            "old_lines": "def play_game():",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Use f-string for player turn message",
            "suggestion_description": "String concatenation should be replaced with an f-string for readability.",
            "line_no_from": 70,
            "line_no_to": 70,
            "replace_by": "print(f\"{player_name} turn ({current_player})\")",
            "suggestion_id": "faf089552243da8b8cc7f82170ae21cfc987e68d44117bf5d60af668f05d0660",
            "old_lines": "        print(player_name + \" turn (\" + current_player + \")\")",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Use 'is not None' comparison",
            "suggestion_description": "Comparison to None should use 'is not None' for idiomatic Python.",
            "line_no_from": 90,
            "line_no_to": 90,
            "replace_by": "if winner is not None:",
            "suggestion_id": "872a24d73db71b73f3fab682dfa92e3d9299aa790262f94678c6744988f27361",
            "old_lines": "        if winner != None:",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Simplify boolean check",
            "suggestion_description": "Comparing a boolean expression to True is unnecessary.",
            "line_no_from": 95,
            "line_no_to": 95,
            "replace_by": "if board_full():",
            "suggestion_id": "31179374f88f3c7699c8efb49b077d8239d0d1c5d613c3452a11ff3b000b21d8",
            "old_lines": "        if board_full() == True:",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Fix variable name and use f-string",
            "suggestion_description": "'gameName' does not follow naming conventions and is undefined; replace with 'game_name' using an f-string.",
            "line_no_from": 108,
            "line_no_to": 108,
            "replace_by": "print(f\"Welcome to {game_name}\")",
            "suggestion_id": "84c2b499ec24afdb080c4c79c6a8f08f2cabc960b0570a2399032316a6822c89",
            "old_lines": "print(\"Welcome to \" + gameName)",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            },
            {
            "suggestion_title": "Update variable name and use f-string",
            "suggestion_description": "'randomThing' should be renamed to 'random_thing' and printed using an f-string.",
            "line_no_from": 110,
            "line_no_to": 110,
            "replace_by": "print(f\"Random useless value: {random_thing}\")",
            "suggestion_id": "7e8e2569f850a7edbe943d631bf5527b9a82f4f69ab0e40c30960d2b6c7f3262",
            "old_lines": "print(\"Random useless value:\", randomThing)",
            "suggestion_state": "pending",
            "batch_id": "75b029c97b2a32bc620123c036489db7"
            }
        ],
        "model": "openai/gpt-oss-120b",
        "error": {
            "type": "",
            "message": ""
        }
        }

    err_type = (result.get("error") or {}).get("type", "")

    if result["success"]:
        update_suggestions(project_path, result["agent"], result.get("suggestions", []))
        return {
            "success": True,
            "agent": result["agent"],
            "suggestions": result.get("suggestions", []),
            "model": result.get("model", ""),
            "error": {
                "type": err_type,
                "message": result.get("error", {}).get("message", ""),
            }
        }
    elif err_type == "AuthenticationError":
        return {
            "success": False,
            "agent": result["agent"],
            "model": result.get("model", ""),
            "error": {
                "type": err_type,
                "message": result.get("error", {}).get("message", ""),
            }
        }
    elif err_type == "APIStatusError" and "429" in (result.get("error") or {}).get("message", ""):
        return {
            "success": False,
            "agent": result["agent"],
            "model": result.get("model", ""),
            "error": {
                "type": err_type,
                "message": result.get("error", {}).get("message", ""),
            }
        }
    else:
        return {
            "success": False,
            "error" : {
                "Internal Server Error": result.get("error", {}).get("message", "Unknown error occurred")
            }
        }

# DEV/TEST MODEL/ENDPOINTS (not for final use)
class CodingStandardsRequest(BaseModel):
    project_id: str
    filename: str

# DEV/TEST MODEL/ENDPOINTS (not for final use)
class CodingStandardsResponse(BaseModel):
    success: bool
    agent: str
    suggestions: Optional[list[dict]] = None
    model: str
    error: Optional[dict] = None

# DEV/TEST MODEL/ENDPOINTS (not for final use)
@app.post(
    "/_dev_coding_standards_agent",
    summary="Run coding standards agent on a file",
    description=(
        "Reads project info from `.state.json`, loads the specified file "
        "using `file_io.read_file_content`, and runs the CodingStandardsAgent "
        "to get style/maintainability suggestions. **For development/testing only.**"
    ),
    # response_model=CodingStandardsResponse,
    status_code=200,
    responses={
        400: {"description": "No active project or file not found"},
        500: {"description": "Agent execution error"},
    },
)
def dev_coding_standards_agent(req: CodingStandardsRequest):
    return _run_coding_standards_flow(req.project_id, req.filename)


@app.post(
    "/refactor",
    summary="Trigger a refactor on a file",
    description=(
        "Analyzes the specified file in the active project and returns suggestions "
        "from the requested agent after validating the agent_id against model_config.json."
    ),
    response_model=RefactorResponse,
    status_code=200,
    responses={
        400: {"description": "Invalid agent_id or active project mismatch"},
        404: {"description": "Project or file not found"},
        500: {"description": "Internal error during refactoring"},
    },
)
def refactor(req: RefactorRequest):
    agent_config = _get_agent_config_by_id(req.agent_id)
    if not agent_config:
        raise HTTPException(status_code=400, detail="Invalid agent_id")

    # ReTake Project Context
    
    if agent_config.get("config_key") == "CodingStandardsAgent":
        return _run_coding_standards_flow(req.project_id, req.filename)

    raise HTTPException(status_code=400, detail="Unsupported agent_id")

# (/_dev_flush-data) DEV/TEST MODEL/ENDPOINTS (not for final use)
@app.post(
    "/_dev_flush_data",
    summary="Flush all data from the central database",
    description=(
        "Drops and recreates every table in `.airefactor_central.db`, "
        "wiping all stored projects. **For development/testing only.**"
    ),
    status_code=200,
    responses={
        200: {"description": "All tables flushed successfully"},
    },
)
def dev_flush_data():
    _ensure_central_db()
    db = SQLiteDB(str(CENTRAL_DB_PATH))
    db.runQuery("DELETE FROM projects")
    return {"status": "ok", "message": "All data flushed from central DB"}

# (/_dev_flush_suggestion) DEV/TEST MODEL/ENDPOINTS (not for final use)
class FlushSuggestionsRequest(BaseModel):
    project_id: str

# (/_dev_flush_suggestion) DEV/TEST MODEL/ENDPOINTS (not for final use)
@app.post(
    "/_dev_flush_suggestion",
    summary="Flush all suggestions from a project's local database",
    description=(
        "Deletes all rows from the `suggestions` table in the project's "
        "local `.airefactor.db`. **For development/testing only.**"
    ),
    status_code=200,
    responses={
        404: {"description": "Project not found in central DB"},
        500: {"description": "Database error"},
    },
)
def dev_flush_suggestions(req: FlushSuggestionsRequest):
    _ensure_central_db()
    central_db = SQLiteDB(str(CENTRAL_DB_PATH))
    err, rows = central_db.read(
        "projects",
        "SELECT * FROM {table} WHERE project_id = ?",
        params=(req.project_id,),
    )
    if err:
        raise HTTPException(status_code=500, detail="Database read error")
    if not rows:
        raise HTTPException(status_code=404, detail="Project not found")

    project_path = rows[0]["project_path"]
    local_db_path = os.path.join(project_path, ".airefactor.db")
    if not os.path.isfile(local_db_path):
        raise HTTPException(status_code=404, detail="Project local DB not found")

    db = SQLiteDB(local_db_path)
    db.runQuery("DELETE FROM suggestions")
    return {"status": "ok", "message": "All suggestions flushed"}


def update_suggestions(project_path: str, agent_id: str, suggestions: list[dict]):
    local_db_path = os.path.join(project_path, ".airefactor.db")
    db = SQLiteDB(local_db_path)
    db.runQuery(
        """CREATE TABLE IF NOT EXISTS suggestions (
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            suggestion_id TEXT PRIMARY KEY,
            suggestion_description TEXT NOT NULL,
            line_no_from INTEGER NOT NULL,
            line_no_to INTEGER NOT NULL,
            old_lines TEXT NOT NULL,
            replace_by TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            suggestion_state TEXT NOT NULL
        )"""
    )
    for s in suggestions:
        db.runQuery(
            """INSERT OR REPLACE INTO suggestions
               (suggestion_id, suggestion_description, line_no_from, line_no_to,
                old_lines, replace_by, agent_id, batch_id, suggestion_state)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            params=(
                s.get("suggestion_id", ""),
                s.get("suggestion_description", ""),
                s.get("line_no_from", 0),
                s.get("line_no_to", 0),
                s.get("old_lines", ""),
                s.get("replace_by", ""),
                agent_id,
                s.get("batch_id", "ERBTCH"),
                s.get("suggestion_state", "pending"),
            ),
        )


class AcceptSuggestionRequest(BaseModel):
    project_id: str
    filename: str
    suggestion_id: str
    batch_id: str
    accept: bool


class AcceptSuggestionResponse(BaseModel):
    success: bool
    accepted: bool
    suggestion_id: str
    batch_id: str
    adjustment: Optional[dict] = None
    remaining_suggestions: list[dict] = []
    message: str = ""


def _shift_pending_line_numbers(
    db: SQLiteDB,
    accepted_to: int,
    extra: int,
    batch_id: str,
    exclude_suggestion_id: str,
):
    """
    Shift line_no_from/line_no_to for remaining pending suggestions in the same batch.
    Only suggestions entirely after the accepted range (line_no_from > accepted_to) are shifted.
    """
    if extra == 0:
        return

    err, rows = db.read(
        "suggestions",
        "SELECT suggestion_id, line_no_from, line_no_to FROM {table} "
        "WHERE suggestion_state = ? AND batch_id = ? AND suggestion_id != ?",
        params=("pending", batch_id, exclude_suggestion_id),
    )
    if err or not rows:
        return

    for row in rows:
        lnf = int(row["line_no_from"])
        lnt = int(row["line_no_to"])
        if lnf > accepted_to:
            new_lnf = max(1, lnf + extra)
            new_lnt = max(new_lnf, lnt + extra)
            db.runQuery(
                "UPDATE {table} SET line_no_from = ?, line_no_to = ? WHERE suggestion_id = ?",
                table_name="suggestions",
                params=(new_lnf, new_lnt, row["suggestion_id"]),
            )


def _get_pending_suggestions_by_batch(
    db: SQLiteDB,
    batch_id: str,
) -> list[dict]:
    err, rows = db.read(
        "suggestions",
        "SELECT * FROM {table} WHERE suggestion_state = ? AND batch_id = ? "
        "ORDER BY line_no_from ASC",
        params=("pending", batch_id),
    )
    if err or not rows:
        return []
    return rows


@app.post(
    "/accept_suggestion",
    summary="Accept or reject a pending suggestion",
    description=(
        "Accepts or rejects a pending suggestion from the project's local `.airefactor.db`. "
        "On accept, applies the change via `replace_file_content` and shifts line numbers "
        "of remaining pending suggestions in the same batch (matched by batch_id). "
        "Returns all remaining pending suggestions for that batch."
    ),
    response_model=AcceptSuggestionResponse,
    status_code=200,
    responses={
        400: {"description": "Invalid request or suggestion not pending"},
        404: {"description": "Project, file, or suggestion not found"},
        500: {"description": "Failed to apply change or update DB"},
    },
)
def accept_suggestion(req: AcceptSuggestionRequest):
    if not STATE_FILE.is_file():
        raise HTTPException(status_code=400, detail="No active project. Call /open-project first.")

    with open(STATE_FILE, "r") as f:
        state = json.load(f)

    if state.get("project_id") != req.project_id:
        raise HTTPException(status_code=400, detail="project_id does not match active project")

    project_path = state["project_path"]
    file_path = os.path.join(project_path, req.filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    local_db_path = os.path.join(project_path, ".airefactor.db")
    if not os.path.isfile(local_db_path):
        raise HTTPException(status_code=404, detail="Project local DB not found")

    db = SQLiteDB(local_db_path)
    err, rows = db.read(
        "suggestions",
        "SELECT * FROM {table} WHERE suggestion_id = ?",
        params=(req.suggestion_id,),
    )
    if err:
        raise HTTPException(status_code=500, detail="Database read error")
    if not rows:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    suggestion = rows[0]
    if suggestion.get("suggestion_state") != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Suggestion is not pending (state={suggestion.get('suggestion_state')})",
        )

    if suggestion.get("batch_id") != req.batch_id:
        raise HTTPException(status_code=400, detail="batch_id does not match suggestion")

    adjustment = None

    if req.accept:
        lnf = int(suggestion["line_no_from"])
        lnt = int(suggestion["line_no_to"])
        replace_by = suggestion.get("replace_by") or ""
        try:
            adjustment = replace_file_content(file_path, lnf, lnt, replace_by)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to apply file change")

        err, _ = db.runQuery(
            "UPDATE {table} SET suggestion_state = ? WHERE suggestion_id = ?",
            table_name="suggestions",
            params=("accepted", req.suggestion_id),
        )
        if err:
            raise HTTPException(status_code=500, detail="Failed to update suggestion state")

        _shift_pending_line_numbers(
            db,
            accepted_to=lnt,
            extra=adjustment["extra_added_removed"],
            batch_id=req.batch_id,
            exclude_suggestion_id=req.suggestion_id,
        )
        message = "Suggestion accepted and applied"
    else:
        err, _ = db.runQuery(
            "UPDATE {table} SET suggestion_state = ? WHERE suggestion_id = ?",
            table_name="suggestions",
            params=("rejected", req.suggestion_id),
        )
        if err:
            raise HTTPException(status_code=500, detail="Failed to update suggestion state")
        message = "Suggestion rejected"

    remaining = _get_pending_suggestions_by_batch(db, req.batch_id)

    return {
        "success": True,
        "accepted": req.accept,
        "suggestion_id": req.suggestion_id,
        "batch_id": req.batch_id,
        "adjustment": adjustment,
        "remaining_suggestions": remaining,
        "message": message,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
