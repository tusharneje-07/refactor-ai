# RefactorAI

A FastAPI-based backend for AI-powered code refactoring suggestions.

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# edit .env with your Doppler token
python3 main.py
# or
uvicorn main:app --reload
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/open_project` | Open a project (validates path, creates local DB) |
| POST | `/read_file` | Read a file from the active project |
| POST | `/validate_project_path` | Validate a project path exists |
| POST | `/refactor` | Trigger refactoring via an agent (validates `agent_id`) |
| POST | `/accept_suggestion` | Accept or reject a suggestion and apply changes |
| POST | `/_dev_coding_standards_agent` | Dev/test: run CodingStandardsAgent directly |
| POST | `/_dev_flush_data` | Dev/test: flush all central DB data |
| POST | `/_dev_flush_suggestion` | Dev/test: flush suggestions for a project |

## Model Config

Agent config is defined in [`backend/model_config.json`](backend/model_config.json). Supported `agent_id`:

- `coding-standards-agent-01` — CodingStandardsAgent

## Provider Config

API endpoints for LLM providers are in [`backend/.provider_config.json`](backend/.provider_config.json).

## State

Active project state is tracked in `backend/.state.json`:

```json
{
  "project_id": "<sha256 of project_path>",
  "project_path": "/absolute/path/to/project",
  "project_name": "project-folder-name"
}
```

## Databases

- **Central DB**: `backend/data/.airefactor_central.db` — stores project metadata
- **Local DB**: `<project_path>/.airefactor.db` — stores project_info and suggestions
