#!/usr/bin/env python3
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

# BigQuery client
from google.cloud import bigquery

# =====================================================
# CONFIG
# =====================================================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ONE_YEAR_DAYS = 365

CHAIN_TABLES = {
    "eth": "bigquery-public-data.crypto_ethereum.transactions",
    "polygon": "public-data-finance.crypto_polygon.transactions",
    "tron": "public-data-finance.crypto_tron.transactions",
    "solana": "public-data-finance.crypto_solana.transactions",
}

FILE_MAP = {
    "eth": DATA_DIR / "eth_hourly.csv",
    "polygon": DATA_DIR / "polygon_hourly.csv",
    "tron": DATA_DIR / "tron_hourly.csv",
    "solana": DATA_DIR / "solana_hourly.csv",
}

# =====================================================
# Helper: Fetch hourly aggregated data from BigQuery
# =====================================================
def fetch_hourly(chain: str, start_ts: datetime, end_ts: datetime):
    client = bigquery.Client()

    start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S%z")
    end_str = end_ts.strftime("%Y-%m-%d %H:%M:%S%z")

    table = CHAIN_TABLES[chain]

    # Hourly aggregation: timestamp → avg fee + transaction count
    query = f"""
        SELECT
            TIMESTAMP_TRUNC(block_timestamp, HOUR) AS hour_utc,
            COUNT(*) AS tx_count,
            AVG(CAST(gas_price AS FLOAT64)) AS avg_gas_price
        FROM `{table}`
        WHERE block_timestamp >= TIMESTAMP('{start_str}')
          AND block_timestamp <  TIMESTAMP('{end_str}')
        GROUP BY hour_utc
        ORDER BY hour_utc
    """

    df = client.query(query).to_dataframe()

    if df.empty:
        return pd.DataFrame(columns=["hour_utc", "tx_count", "avg_gas_price"])

    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    return df


# =====================================================
# Helper: Load, update, and save CSV
# =====================================================
def update_chain_data(chain: str):
    file_path = FILE_MAP[chain]

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Default: pull last 1 year
    start_ts_default = now - timedelta(days=ONE_YEAR_DAYS)

    # ---- Case 1: No file → full one-year download ----
    if not file_path.exists():
        print(f"[{chain.upper()}] File not found → downloading last 1 year")
        df_new = fetch_hourly(chain, start_ts_default, now)
        df_new.to_csv(file_path, index=False)
        print(f"[{chain.upper()}] Saved {len(df_new)} rows → {file_path}")
        return df_new

    # ---- Case 2: File exists → only fetch missing hours ----
    df_old = pd.read_csv(file_path)
    df_old["hour_utc"] = pd.to_datetime(df_old["hour_utc"], utc=True)

    last_ts = df_old["hour_utc"].max()

    # If already up-to-date
    if last_ts >= now - timedelta(hours=1):
        print(f"[{chain.upper()}] Already up to date (last: {last_ts})")
        return df_old

    # Fetch only missing hours
    start_ts = last_ts + timedelta(hours=1)
    print(f"[{chain.upper()}] Updating from {start_ts} to {now}")

    df_new = fetch_hourly(chain, start_ts, now)

    # Append & dedupe
    df_full = pd.concat([df_old, df_new], ignore_index=True)
    df_full = df_full.drop_duplicates(subset=["hour_utc"]).sort_values("hour_utc")

    df_full.to_csv(file_path, index=False)
    print(f"[{chain.upper()}] Updated. Total rows: {len(df_full)}")

    return df_full


# =====================================================
# MAIN (Run for all chains)
# =====================================================
if __name__ == "__main__":
    print("\n=== Updating 1-year hourly data for all chains ===\n")

    for chain in ["eth", "polygon", "tron", "solana"]:
        try:
            update_chain_data(chain)
        except Exception as e:
            print(f"[ERROR] {chain}: {e}")

    print("\n=== DONE ===\n")
