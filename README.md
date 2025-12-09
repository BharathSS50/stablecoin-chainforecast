# stablecoin-chainforecast

```markdown
# 🚀 Blockchain Hourly Forecast API
Predict the next 24 hours of blockchain activity using historical hourly data and a trained machine learning model.

This project includes:
- Automated hourly Ethereum data ingestion (1 year)
- Machine Learning model training pipeline (Random Forest)
- FastAPI endpoint to serve 24-hour forecasts
- Docker support for easy deployment

---

# 📁 Project Structure

```

brackt/
├── app.py                   # FastAPI service
├── fetch_chain_data.py      # Downloads 1-year hourly ETH data
├── train_eth_model.py       # Trains and saves the ML model
├── Dockerfile
├── docker-compose.yml
├── data/                    # Stores eth_hourly.csv
├── models/                  # Stores ML model + scaler + metadata
└── README.md

````

---

# 🛠️ 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
````

---

# 🧰 2. Run Locally (Without Docker)

## 2.1 Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# .\venv\Scripts\activate      # Windows
```

## 2.2 Install Dependencies

```bash
pip install -r requirements.txt
```

If missing:

```bash
pip install fastapi "uvicorn[standard]" pandas numpy scikit-learn joblib google-cloud-bigquery
```

---

# 📥 3. Fetch ETH Historical Data

(First-time setup or when updating data)

Authenticate:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
```

Run:

```bash
python fetch_chain_data.py
```

Creates:

```
data/eth_hourly.csv
```

---

# 🧠 4. Train the Forecast Model

```bash
python train_eth_model.py
```

Creates:

```
models/eth_forecast_model.joblib
models/eth_scaler.joblib
models/eth_model_meta.json
```

---

# 🚀 5. Run FastAPI API Locally

```bash
python app.py
```

Open:

* Swagger Docs → [http://localhost:8000/docs](http://localhost:8000/docs)
* Forecast → [http://localhost:8000/forecast-next-24h](http://localhost:8000/forecast-next-24h)

---

# 🐳 6. Run with Docker

## 6.1 Build Image

```bash
docker build -t eth-forecast-api .
```

## 6.2 Start Container

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  eth-forecast-api
```

API available at:

* [http://localhost:8000](http://localhost:8000)
* [http://localhost:8000/docs](http://localhost:8000/docs)
* [http://localhost:8000/forecast-next-24h](http://localhost:8000/forecast-next-24h)

---

# 🔄 Update Data + Retrain Later

```bash
python fetch_chain_data.py
python train_eth_model.py
```

Restart the API afterward.

---

# 📝 Notes

* `data/` and `models/` are stored outside Docker for persistence
* Model loads only once for fast inference
* System is modular and ready for more blockchains

---
