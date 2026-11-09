# Contributing

Thanks for your interest. Issues and pull requests are welcome.

## Set up

Requires [uv](https://docs.astral.sh/uv/) and Node 22.

```bash
cd packages/quantilo && uv sync --extra ml   # the engine alone, for its tests
cd backend && uv sync --extra agent      # add --extra deep for the PyTorch models
cd frontend && npm ci
./scripts/dev.ps1                         # backend 127.0.0.1:8000 + Vite 127.0.0.1:5173
```

## Before a pull request

Everything CI runs, locally:

```bash
cd packages/quantilo && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
cd backend && uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
cd frontend && npm run lint && npm run typecheck && npm test && npm run build
```

- **Tests first** for behaviour changes; statistics are checked against reference implementations (SciPy, statsmodels) where one exists.
- **API changes**: regenerate the typed client — `uv run python -m app.openapi_dump openapi.json` in `backend/`, then `npm run gen:api` in `frontend/`.
- **Where code goes**: see [ARCHITECTURE](docs/ARCHITECTURE.md). Every calculation lives in the engine, `packages/quantilo/` (no web code, no SQLite; CI installs it alone), with a `Dataset` method for the library. The app (`backend/`) adds state and HTTP; each module (`backend/app/modules/`) stays isolated and registers in one line.
- **Visible changes**: update the [user guide](docs/USER_GUIDE.md), which the app shows on `/help`.
- **New dependencies**: open an issue first; the Docker image has a size budget (900 MB uncompressed, checked in CI; 850 MB today). Dependabot proposes monthly updates.

## The assistant

Its engine (`backend/app/modules/agent/engine/`) is pure and tested with a scripted provider: no test makes a real API call, and every vendor key is cleared from the environment during tests. To check real answers, run the golden tasks against a running app with your own key (this makes billed or quota-limited calls):

```bash
cd backend && uv run python evals/agent_eval.py --url http://127.0.0.1:8000
```

Never commit a key: they belong in the environment or in the app's settings, which store them outside the repository.
