"""
Pharma Forecasting API
Accepts CSV/JSON data, runs model competition, returns best forecast per territory.
"""

import io
import warnings
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

from forecast_model_selector import run_forecast_pipeline

warnings.filterwarnings("ignore")

app = FastAPI(
    title="TerraForecast API",
    description="Automatic model selection forecasting for pharma territory sales data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Pharma Forecast API is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Forecast via CSV upload
# ---------------------------------------------------------------------------

@app.post("/forecast/upload")
async def forecast_from_upload(
    file: UploadFile = File(...),
    date_col: str = "transaction_date",
    id_col: str = "territory_id",
    value_col: str = "metric_value",
    
    forecast_horizon: int = 3,
):
    """
    Upload a CSV or Excel file and get forecasts back.

    Expected columns (defaults match the sample data):
      - transaction_date  : date of transaction
      - territory_id      : territory identifier
      - metric_value      : TRx or revenue value

    Returns JSON with best model and forecast per territory.
    """
    filename = file.filename.lower()
    content = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {str(e)}")

    for col in [date_col, id_col, value_col]:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found. Available columns: {list(df.columns)}"
            )

    try:
        summary, detail, matrix = run_forecast_pipeline(
            df,
            date_col=date_col,
            id_col=id_col,
            value_col=value_col,
            
            horizon=forecast_horizon,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast pipeline error: {str(e)}")

    return {
        "territories_processed": len(summary),
        "forecast_horizon_months": forecast_horizon,
        "holdout_months_used": "adaptive",
        "summary": summary.to_dict(orient="records"),
        "model_competition": detail.to_dict(orient="records") if not detail.empty else [],
    }


# ---------------------------------------------------------------------------
# Forecast via JSON body
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    records: list               # List of dicts: [{territory_id, date, metric_value}, ...]
    date_col: str = "date"
    id_col: str = "territory_id"
    value_col: str = "metric_value"
    holdout_months: int = 3
    forecast_horizon: int = 3


@app.post("/forecast/json")
def forecast_from_json(req: ForecastRequest):
    """
    Send data as JSON array and get forecasts back.

    Example body:
    {
      "records": [
        {"territory_id": "A10101", "date": "2025-01-01", "metric_value": 5.5},
        ...
      ]
    }
    """
    try:
        df = pd.DataFrame(req.records)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse records: {str(e)}")

    for col in [req.date_col, req.id_col, req.value_col]:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found in records. Found: {list(df.columns)}"
            )

    try:
        summary, detail, matrix = run_forecast_pipeline(
            df,
            date_col=req.date_col,
            id_col=req.id_col,
            value_col=req.value_col,
            horizon=req.forecast_horizon,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast pipeline error: {str(e)}")

    return {
        "territories_processed": len(summary),
        "forecast_horizon_months": req.forecast_horizon,
        "holdout_months_used": "adaptive",
        "summary": summary.to_dict(orient="records"),
        "model_competition": detail.to_dict(orient="records") if not detail.empty else [],
    }


# ---------------------------------------------------------------------------
# Run locally
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
