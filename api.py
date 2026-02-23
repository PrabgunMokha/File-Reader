from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
import os
import time

# --- Load Environment Variables ---
load_dotenv(override=True)

app = FastAPI(
    title="🏃‍♂️ Track Data AI Auditor API",
    description="REST API for querying sales coaching data using Groq LLM",
    version="1.0.0"
)

# --- Groq Client ---
# ===================================================
# INSERT YOUR GROQ API KEY IN THE .env FILE AS:
#   GROQ_API_KEY=gsk_your_key_here
# Get your free key at: https://console.groq.com/keys
# ===================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# --- Load Data ---
def load_data():
    if os.path.exists("TrackData_Clean.csv"):
        return pd.read_csv("TrackData_Clean.csv")
    elif os.path.exists("TrackData.csv"):
        return pd.read_csv("TrackData.csv", encoding="latin-1")
    return None

df = load_data()

# --- Request / Response Models ---
class QueryRequest(BaseModel):
    question: str
    model: str = "llama-3.3-70b-versatile"  # default model

class QueryResponse(BaseModel):
    answer: str
    model_used: str
    rows_analyzed: int

class DataSummary(BaseModel):
    total_rows: int
    total_columns: int
    columns: list[str]
    showrooms: list[str]
    trainers: list[str]
    clusters: list[str]

# --- Helper: Smart data filtering ---
def get_relevant_data(question: str, dataframe: pd.DataFrame, max_rows: int = 3) -> pd.DataFrame:
    """Send only relevant rows to the LLM instead of the full dataset."""
    keywords = question.lower().split()
    mask = dataframe.apply(
        lambda row: any(k in str(row).lower() for k in keywords if len(k) > 2),
        axis=1
    )
    if mask.any():
        relevant = dataframe[mask].head(max_rows)
    else:
        relevant = dataframe.head(max_rows)
    return relevant

# --- Active Groq Models (verified working) ---
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mistral-saba-24b",
]

SYSTEM_PROMPT = "You are a data analyst. Answer concisely based on the data provided."

# --- Helper: Call Groq API with retry ---
def _call_groq_once(model_name: str, prompt: str) -> str:
    """Make a single Groq API call."""
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    return response.choices[0].message.content


def call_groq(question: str, data_str: str, model_name: str) -> tuple[str, str]:
    """Call Groq with retry + fallback. Handles rate limits and decommissioned models."""
    if not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set. Add it to your .env file.")

    prompt = f"Data:\n{data_str}\n\nQuestion: {question}"

    # Try primary model with 2 retries
    for attempt in range(3):
        try:
            text = _call_groq_once(model_name, prompt)
            return text, model_name
        except Exception as e:
            error_msg = str(e)
            # Model decommissioned — skip to fallback
            if "decommissioned" in error_msg.lower() or "404" in error_msg:
                break
            # Rate limited — wait and retry
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                continue  # immediately try next model, no waiting
            # Other error — retry once
            if attempt < 2:
                time.sleep(2)
                continue
            break

    # Try fallback models
    for fallback in FALLBACK_MODELS:
        if fallback == model_name:
            continue
        try:
            text = _call_groq_once(fallback, prompt)
            return text, fallback
        except Exception as e:
            error_msg = str(e)
            if "decommissioned" in error_msg.lower():
                continue  # try next fallback
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                time.sleep(3)
                continue
            continue

    raise HTTPException(
        status_code=429,
        detail="All Groq models rate limited. Wait 30 seconds and try again."
    )


# ========================
#       ENDPOINTS
# ========================

@app.get("/", tags=["General"])
def root():
    """Welcome endpoint."""
    return {
        "message": "🏃‍♂️ Track Data AI Auditor API (Groq Powered)",
        "docs": "/docs",
        "endpoints": ["/health", "/data/summary", "/data/columns", "/data/preview", "/query"]
    }


@app.get("/health", tags=["General"])
def health_check():
    """Check if API and data are ready."""
    return {
        "status": "healthy",
        "data_loaded": df is not None,
        "total_rows": len(df) if df is not None else 0,
        "groq_connected": client is not None,
    }


@app.get("/data/summary", tags=["Data"], response_model=DataSummary)
def data_summary():
    """Get a summary of the loaded dataset."""
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded. Run fix_data.py first.")

    return DataSummary(
        total_rows=len(df),
        total_columns=len(df.columns),
        columns=df.columns.tolist(),
        showrooms=df["name_of_the_showroom"].dropna().unique().tolist() if "name_of_the_showroom" in df.columns else [],
        trainers=df["trainer"].dropna().unique().tolist() if "trainer" in df.columns else [],
        clusters=df["cluster"].dropna().unique().tolist() if "cluster" in df.columns else [],
    )


@app.get("/data/columns", tags=["Data"])
def data_columns():
    """Get all column names and their data types."""
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded.")
    return {
        "columns": [
            {"name": col, "dtype": str(df[col].dtype), "non_null_count": int(df[col].notna().sum())}
            for col in df.columns
        ]
    }


@app.get("/data/preview", tags=["Data"])
def data_preview(rows: int = 5):
    """Preview first N rows of the dataset."""
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded.")
    rows = min(rows, 50)  # cap at 50
    return {
        "total_rows": len(df),
        "showing": rows,
        "data": df.head(rows).fillna("").to_dict(orient="records")
    }


# --- Response Cache (speeds up repeated questions in Postman Runner) ---
_response_cache: dict[str, tuple[str, str, int]] = {}

@app.post("/query", tags=["AI Query"], response_model=QueryResponse)
async def query_data(req: QueryRequest):
    """Ask a question about the data using Groq LLM."""
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded. Run fix_data.py first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cache_key = f"{req.question.strip().lower()}|{req.model}"
    if cache_key in _response_cache:
        print(f"CACHE HIT: {req.question[:30]}...")
        answer, model_used, rows = _response_cache[cache_key]
        return QueryResponse(answer=answer, model_used=model_used, rows_analyzed=rows)
    print(f"CACHE MISS: {req.question[:30]}... (cache has {len(_response_cache)} entries)")

    relevant_df = get_relevant_data(req.question, df)
    data_str = relevant_df.to_string(index=False)

    answer, model_used = call_groq(req.question, data_str, req.model)

    _response_cache[cache_key] = (answer, model_used, len(relevant_df))

    return QueryResponse(
        answer=answer,
        model_used=model_used,
        rows_analyzed=len(relevant_df)
    )


@app.get("/data/stats", tags=["Data"])
def data_stats():
    """Get quick statistics about the dataset."""
    if df is None:
        raise HTTPException(status_code=404, detail="No data loaded.")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": round(df[col].mean(), 2) if df[col].notna().any() else None,
            "min": int(df[col].min()) if df[col].notna().any() else None,
            "max": int(df[col].max()) if df[col].notna().any() else None,
        }

    return {
        "numeric_columns": len(numeric_cols),
        "stats": stats
    }