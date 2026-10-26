# Supervised Learning Algorithm Explorer — AdaBoost vs Random Forest

Educational full-stack app that trains and compares **Random Forest** and **AdaBoost** on scikit-learn datasets (Breast Cancer Wisconsin + Iris), with a FastAPI backend and React dashboard.

> Lives in `PROJECT2/` so the existing MNIST digit project at the repo root is untouched.

## Architecture

```text
React (Vite :5174)  --HTTP JSON-->  FastAPI (:8001)
                                      │
                                      ├─ datasets (sklearn)
                                      ├─ StandardScaler (train-only fit)
                                      ├─ RandomForestClassifier / AdaBoostClassifier
                                      └─ Joblib artifacts + JSON metadata
```

## Requirements

- Python **3.11** (project uses the repo root `.venv` with scikit-learn 1.9+)
- Node.js 20+

## Setup

```powershell
cd "PROJECT2"

# Reuse parent venv (Python 3.11)
& "..\.venv\Scripts\Activate.ps1"
# if needed: pip install -r backend\requirements.txt

cd frontend
npm install
```

## Run

**Terminal 1 — API**

```powershell
cd "PROJECT2\backend"
& "..\..\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

**Terminal 2 — UI**

```powershell
cd "PROJECT2\frontend"
npm run dev
```

Open http://localhost:5174  
API docs: http://127.0.0.1:8001/docs

## API endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health check |
| GET | `/api/datasets` | Dataset schemas & stats |
| GET | `/api/datasets/{name}/sample` | Sample rows |
| POST | `/api/train` | Train RF or AdaBoost |
| GET | `/api/models` | List saved models |
| POST | `/api/predict` | Predict from features |
| GET | `/api/models/{id}/metrics` | Metrics & diagnostics |
| POST | `/api/compare` | Train both on same split |
| GET | `/api/models/{id}/feature-importance` | Ranked importances |

## Testing

```powershell
cd "PROJECT2"
& "..\.venv\Scripts\python.exe" -m pytest backend\tests -q

cd frontend
npm test
```

## Evaluation notes

- Stratified split + `StandardScaler` fit on **train only**
- Binary metrics use positive class **index 1** (`benign` for breast cancer)
- Multiclass metrics use **macro** averaging
- Comparison uses one shared split for both algorithms
- Results are **not** hardcoded; they depend on data and hyperparameters

## Artifacts

- Models: `artifacts/models/*.joblib`
- Metadata: `artifacts/metadata/*.json`

Only artifacts written by this app are loaded (no arbitrary Joblib upload).
