from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from joblib import load

app = FastAPI(
    title="AMLA ASM2 Weather API",
    version="1.0.0",
    description="The API serves two tasks: predicting rain in exactly 7 days and predicting total rain in the next 3 days."
)

# ========= Model paths  =========
MODEL_BASE_DEFAULT = "/Users/doanvanthang/Documents/Learning/Master In Data Science and Innovation/Semster 4/AMLA/Assignments/ASM_2/FastAPI/models"

RAIN_MODEL_PATH = os.getenv(
    "RAIN_MODEL_PATH",
    str(Path(MODEL_BASE_DEFAULT) / "rain_or_not" / "XGBClassifier.joblib"),
)
PRECIP_MODEL_PATH = os.getenv(
    "PRECIP_MODEL_PATH",
    str(Path(MODEL_BASE_DEFAULT) / "precipitation_fall" / "XGBReg.joblib"),
)

GITHUB_URL_DEFAULT = "https://github.com/your-user/your-repo"  # TODO: sửa link repo của bạn
GITHUB_URL = os.getenv("GITHUB_URL", GITHUB_URL_DEFAULT)

# ========= Load models =========
def _load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return load(p)

try:
    rain_clf = _load_model(RAIN_MODEL_PATH)          # XGBClassifier.joblib
    precip_reg = _load_model(PRECIP_MODEL_PATH)      # XGBReg.joblib
except FileNotFoundError as e:
    rain_clf = None
    precip_reg = None
    load_error_msg = str(e)
else:
    load_error_msg = None


# ========= Helpers =========
def parse_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD -> datetime (00:00:00). Ném HTTP 400 nếu sai format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")

def build_features_for_date(d: datetime) -> pd.DataFrame:
    """
    Tối thiểu tạo DataFrame có cột 'date'.
    Giả định pipeline đã lưu trong joblib sẽ tự lo feature engineering từ cột 'date'.
    Nếu pipeline của bạn kỳ vọng cột khác, hãy chỉnh lại phần này cho khớp.
    """
    return pd.DataFrame({"date": [pd.to_datetime(d.date())]})


# ========= Root =========
@app.get("/", status_code=200)
def root():
    return {
        "project": "AMLA ASM2 - Weather Prediction API",
        "objectives": [
            "Predict if it will rain exactly 7 days after the given date.",
            "Predict the cumulated precipitation (mm) for the next 3 days after the given date."
        ],
        "endpoints": [
            {
                "path": "/",
                "method": "GET",
                "description": "Brief description, endpoints, inputs/outputs, repo link."
            },
            {
                "path": "/health/",
                "method": "GET",
                "description": "Health check (returns 200 and a welcome message)."
            },
            {
                "path": "/predict/rain/",
                "method": "GET",
                "query_params": {"date": "YYYY-MM-DD"},
                "output_example": {
                    "input_date": "2023-01-01",
                    "prediction": {"date": "2023-01-08", "will_rain": True}
                }
            },
            {
                "path": "/predict/precipitation/fall/",
                "method": "GET",
                "query_params": {"date": "YYYY-MM-DD"},
                "output_example": {
                    "input_date": "2023-01-01",
                    "prediction": {
                        "start_date": "2023-01-02",
                        "end_date": "2023-01-04",
                        "precipitation_fall": "28.2"
                    }
                }
            }
        ],
        "expected_input_parameters": {
            "date": "string formatted YYYY-MM-DD"
        },
        "output_format_notes": {
            "/predict/rain/": {
                "fields": ["input_date", "prediction.date", "prediction.will_rain (bool)"]
            },
            "/predict/precipitation/fall/": {
                "fields": ["input_date", "prediction.start_date", "prediction.end_date", "prediction.precipitation_fall (string/number)"]
            }
        },
        "github_repo": GITHUB_URL,
    }


# ========= Health =========
@app.get("/health/", status_code=200)
def health_check():
    if load_error_msg:
        return f"Startup warning: {load_error_msg}"
    return "API is ready. Models loaded."


# ========= Rain in exactly 7 days =========
@app.get("/predict/rain/", status_code=200)
def predict_rain(
    date: str = Query(..., description="YYYY-MM-DD (base date)"),
):
    if rain_clf is None:
        raise HTTPException(status_code=500, detail=f"Rain model not loaded. {load_error_msg or ''}")

    d0 = parse_date(date)
    d7 = d0 + timedelta(days=7)

    X = build_features_for_date(d0)
    try:
        pred = rain_clf.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error (rain): {e}")

    will_rain = bool(pred[0])

    return JSONResponse(
        {
            "input_date": d0.strftime("%Y-%m-%d"),
            "prediction": {
                "date": d7.strftime("%Y-%m-%d"),
                "will_rain": will_rain,
            },
        }
    )


# ========= Cumulated precipitation for next 3 days =========
@app.get("/predict/precipitation/fall/", status_code=200)
def predict_precipitation_fall(
    date: str = Query(..., description="YYYY-MM-DD (base date)"),
):
    if precip_reg is None:
        raise HTTPException(status_code=500, detail=f"Precipitation model not loaded. {load_error_msg or ''}")

    d0 = parse_date(date)
    start_date = d0 + timedelta(days=1)
    end_date = d0 + timedelta(days=3)

    X = build_features_for_date(d0)
    try:
        y_hat = float(precip_reg.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error (precipitation): {e}")

    precipitation_str = f"{y_hat:.1f}"

    return JSONResponse(
        {
            "input_date": d0.strftime("%Y-%m-%d"),
            "prediction": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "precipitation_fall": precipitation_str,
            },
        }
    )
