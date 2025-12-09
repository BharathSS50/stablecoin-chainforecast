#!/usr/bin/env python3
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# =========================
# Paths & config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

ETH_DATA_PATH = DATA_DIR / "eth_hourly.csv"
MODEL_PATH = MODELS_DIR / "eth_forecast_model.joblib"
SCALER_PATH = MODELS_DIR / "eth_scaler.joblib"
META_PATH = MODELS_DIR / "eth_model_meta.json"
FORECAST_CACHE_PATH = DATA_DIR / "eth_forecast_next_24h.csv"

FORECAST_HOURS = 24
RECENT_DAYS = 30  # load last 30 days for context


# =========================
# Load model, scaler, meta
# =========================
def load_model_and_meta():
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Run train_eth_model.py first.")
    if not SCALER_PATH.exists():
        raise RuntimeError("Scaler file not found. Run train_eth_model.py first.")
    if not META_PATH.exists():
        raise RuntimeError("Meta file not found. Run train_eth_model.py first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    lags: List[int] = meta["lags"]
    feature_cols: List[str] = meta["feature_cols"]
    return model, scaler, lags, feature_cols


# =========================
# Data loading
# =========================
def load_recent_eth(days: int = RECENT_DAYS) -> pd.DataFrame:
    if not ETH_DATA_PATH.exists():
        raise RuntimeError("ETH hourly data not found. Run fetch/update script first.")

    df = pd.read_csv(ETH_DATA_PATH)
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    df = df.sort_values("hour_utc").reset_index(drop=True)

    # Use avg_gas_price as the series to forecast
    df["value"] = df["avg_gas_price"]

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=days)
    df_recent = df[df["hour_utc"] >= cutoff].copy()

    if df_recent.empty:
        raise RuntimeError("No recent ETH data available to forecast.")

    return df_recent[["hour_utc", "value"]]


# =========================
# Forecast logic (multi-step with lags)
# =========================
def forecast_next_hours(
    df_recent: pd.DataFrame,
    model,
    scaler,
    lags: List[int],
    horizon: int = FORECAST_HOURS,
) -> pd.DataFrame:
    df_recent = df_recent.sort_values("hour_utc")
    history = df_recent["value"].tolist()

    if len(history) < max(lags):
        raise RuntimeError("Not enough history to build lag features.")

    last_ts = df_recent["hour_utc"].iloc[-1]

    timestamps = []
    preds = []

    for _ in range(horizon):
        feat_vals = []
        for lag in lags:
            if len(history) >= lag:
                feat_vals.append(history[-lag])
            else:
                feat_vals.append(history[0])  # fallback

        X_step = np.array(feat_vals, dtype=float).reshape(1, -1)
        X_step_scaled = scaler.transform(X_step)
        y_hat = float(model.predict(X_step_scaled)[0])

        history.append(y_hat)
        last_ts = last_ts + timedelta(hours=1)

        timestamps.append(last_ts)
        preds.append(y_hat)

    fc = pd.DataFrame(
        {
            "timestamp": timestamps,
            "forecast_value": preds,
        }
    )
    fc["generated_at"] = datetime.now(timezone.utc).isoformat()
    return fc


# =========================
# Cache helpers
# =========================
def save_forecast_cache(fc: pd.DataFrame) -> None:
    FORECAST_CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    fc.to_csv(FORECAST_CACHE_PATH, index=False)


def load_forecast_cache() -> Optional[pd.DataFrame]:
    if not FORECAST_CACHE_PATH.exists():
        return None
    df = pd.read_csv(FORECAST_CACHE_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def is_cache_valid(fc: pd.DataFrame) -> bool:
    if fc.empty or "generated_at" not in fc.columns:
        return False
    try:
        gen_at = pd.to_datetime(fc["generated_at"].iloc[0], utc=True)
    except Exception:
        return False
    now_utc = datetime.now(timezone.utc)
    return gen_at.date() == now_utc.date()


def get_or_build_forecast() -> pd.DataFrame:
    # 1) Try cache
    cached = load_forecast_cache()
    if cached is not None and is_cache_valid(cached):
        print("[FORECAST] Using cached forecast.")
        return cached

    print("[FORECAST] Cache missing/invalid. Rebuilding …")

    # 2) Load data + model
    df_recent = load_recent_eth(RECENT_DAYS)
    model, scaler, lags, _ = load_model_and_meta()

    # 3) Forecast
    fc = forecast_next_hours(df_recent, model, scaler, lags, FORECAST_HOURS)

    # 4) Cache
    save_forecast_cache(fc)
    return fc


# =========================
# FastAPI app
# =========================
app = FastAPI(
    title="ETH 24h Gas Forecast API",
    version="1.0.0",
    description="Forecasts next 24 hours of Ethereum avg gas price using saved model.",
)


class ForecastPoint(BaseModel):
    timestamp: datetime
    forecast_value: float


class ForecastResponse(BaseModel):
    generated_at: datetime
    horizon_hours: int
    points: list[ForecastPoint]


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/forecast-next-24h", response_model=ForecastResponse)
def forecast_next_24h_endpoint():
    try:
        fc = get_or_build_forecast()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    generated_at = pd.to_datetime(fc["generated_at"].iloc[0], utc=True)

    points = [
        ForecastPoint(
            timestamp=row["timestamp"].to_pydatetime(),
            forecast_value=float(row["forecast_value"]),
        )
        for _, row in fc.iterrows()
    ]

    return ForecastResponse(
        generated_at=generated_at,
        horizon_hours=FORECAST_HOURS,
        points=points,
    )


# Optional: warm up on startup
@app.on_event("startup")
async def startup_event():
    try:
        print("[STARTUP] Preparing initial forecast …")
        _ = get_or_build_forecast()
        print("[STARTUP] Initial forecast ready.")
    except Exception as e:
        print(f"[STARTUP] Warning: could not build forecast: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
