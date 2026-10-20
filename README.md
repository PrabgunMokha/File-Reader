# File Reader — Track Data AI Auditor

Ask natural-language questions about a local sales-coaching CSV dataset through a FastAPI service powered by Groq.

## How it works

The API loads `TrackData_Clean.csv`, falling back to `TrackData.csv`. It selects up to three rows using question keywords, adds those rows to a prompt, and asks a Groq-hosted language model for an answer. This is a basic keyword-based retrieval-augmented generation (RAG) workflow. It does not use embeddings or a vector database.

## Features

- Natural-language questions through `POST /query`.
- Dataset summary, column metadata, previews, and numeric statistics.
- In-memory response caching by question and requested model.
- Retry and fallback model handling.
- A separate Streamlit chat prototype.

## Run the API

Use Python 3.10 or newer. Run these commands from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Supply your own `TrackData.csv` in the repository root; the dataset is not included. Optionally clean it before starting the server:

```powershell
python fix_data.py
python -m uvicorn api:app --reload --port 8000
```

Open [interactive API docs](http://127.0.0.1:8000/docs). Data is loaded at startup, so restart the server after replacing the CSV. Keep API keys and private datasets out of version control.

## Example request

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query -ContentType 'application/json' -Body '{"question":"Which trainers are listed?"}'
```

The response includes `answer`, `model_used`, and `rows_analyzed`. The default requested model is `llama-3.3-70b-versatile`; availability depends on your Groq account and provider support.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Dataset and client configuration status |
| GET | `/data/summary` | Dataset dimensions and selected categorical values |
| GET | `/data/columns` | Types and non-null counts |
| GET | `/data/preview?rows=5` | Preview, capped at 50 rows |
| GET | `/data/stats` | Numeric summaries |
| POST | `/query` | Ask a question |

## Streamlit prototype

`app.py` requires Streamlit in addition to the listed API dependencies:

```powershell
python -m pip install streamlit
python -m streamlit run app.py
```

This prototype currently uses an `API_KEY` placeholder in `app.py`, rather than the API service's environment-based key loading. Configure it locally before use and never commit a real key. It requires `TrackData_Clean.csv` and sends the first ten rows directly to Groq. Its error message names `data_fix.py`; the actual cleaning script is `fix_data.py`.

## Project files

- `api.py`: FastAPI service, retrieval, Groq calls, and cache.
- `app.py`: independent Streamlit prototype.
- `fix_data.py`: removes completely empty CSV rows using Latin-1 input encoding.
- `test.py`: standalone Gemini connectivity experiment; it requires `google-genai` and `GEMINI_API_KEY`, and is not an automated test suite.

