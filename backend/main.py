import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from modules.sqlite_io import SQLiteDB
from modules.file_io import read_file_content

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
            suggestion_id TEXT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            lnf INTEGER NOT NULL,
            lnt INTEGER NOT NULL,
            old_lines TEXT NOT NULL,
            replaced_by TEXT NOT NULL,
            reason TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            is_accepted BOOLEAN NOT NULL DEFAULT 0
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
    "/open-project",
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
    "/read-file",
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
    "/validate-project-path",
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


class RefactorResponse(BaseModel):
    status: str
    project_id: str
    filename: str
    message: str


@app.post(
    "/refactor",
    summary="Trigger a refactor on a file",
    description=(
        "Analyzes the specified file in the project and returns refactor suggestions. "
        "This endpoint is currently a stub and always returns a pending status."
    ),
    response_model=RefactorResponse,
    status_code=200,
    responses={
        404: {"description": "Project or file not found"},
        500: {"description": "Internal error during refactoring"},
    },
)
def refactor(req: RefactorRequest):
    return {
        "status": "pending",
        "project_id": req.project_id,
        "filename": req.filename,
        "message": "Refactor endpoint is not yet implemented",
    }


@app.post(
    "/_dev_flush-data",
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
