# Scarce Demo Backend

FastAPI service that powers the Scarce dashboard prototype with synthetic data streams.

## Getting Started

1. Create a virtual environment and install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Run the development server:

   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

3. Access the interactive API docs at <http://localhost:8000/docs>.

## Configuration

Environment variables prefixed with `SCARCE_` can be set in an `.env` file:

| Variable | Description | Default |
| --- | --- | --- |
| `SCARCE_PROJECT_NAME` | Service title | `Scarce Demo Backend` |
| `SCARCE_API_V1_PREFIX` | API version prefix | `/api/v1` |
| `SCARCE_ALLOW_ORIGINS` | Comma-separated list of CORS origins | `http://localhost:3000` |
| `SCARCE_SIMULATION_SEED` | Random seed for deterministic data | `42` |
| `SCARCE_SIMULATION_TICK_SECONDS` | Base interval for streaming updates | `1.0` |

## Project Structure

```
backend/
├── app/
│   ├── api/                # FastAPI routers & endpoints
│   ├── core/               # Configuration and dependencies
│   ├── schemas/            # Pydantic response/request models
│   └── simulation/         # Synthetic data engine
├── requirements.txt
└── README.md
```


