# Smart-Cargo-Monitoring-and-Decision-Support-System

A Streamlit decision-support application that estimates shipment-delay risk from historical SCMS delivery data. It provides an interactive shipment form, a calibrated delay-risk score, model-driver explanations, and one-variable-at-a-time what-if analysis.

> **Important:** The app reports learned associations from historical data. It does not establish that changing a shipment attribute will cause delay risk to change.

## Features

- Estimates a shipment's probability of delay and assigns a Low, Moderate, Elevated, or High risk band.
- Explains the largest factors influencing the model score relative to typical historical shipments.
- Compares a selected shipment input with alternate values using what-if scoring.
- Shows historical support for category alternatives and flags thin data.
- Generates practical planner guidance using a local or cloud Ollama model, with a built-in rule-based fallback if the LLM is unavailable.
- Includes notebooks for preprocessing, exploratory/vendor analysis, and model development.

## Project layout

```text
.
├── data/
│   ├── raw/SCMS_Delivery_History_Dataset.csv   # source delivery-history data
│   └── processed_data/                         # notebook-generated data outputs
├── Notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_EDA_and_Vendor_Analysis.ipynb
│   └── 03_Modeling_clean.ipynb
├── model_artifacts/
│   └── delay_model.joblib                      # model used by the Streamlit app
├── streamlit/
│   └── app.py                                  # application entry point
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or later
- A trained model artifact at `model_artifacts/delay_model.joblib`
- Optional: [Ollama](https://ollama.com/) for AI-generated planner advice

## Run the application

From the project root in PowerShell:

```powershell
# Optional: activate the repository's existing virtual environment
.\logi\Scripts\Activate.ps1

# Or create and activate your own environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the dashboard
streamlit run streamlit/app.py
```

Open the local URL displayed by Streamlit, usually `http://localhost:8501`.

If PowerShell prevents environment activation, run this once for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Using the dashboard

1. Enter shipment details, or select **Start from a past shipment**.
2. Review the estimated delay risk, risk band, and alert status.
3. Inspect the driver chart to understand which inputs most influence the model score.
4. Use **What-if** to change one field and compare the resulting model estimate.
5. Select **Generate advice** for operational guidance. If Ollama cannot be reached, the app automatically shows rule-based advice instead.

What-if results keep all other fields fixed. They are useful for exploring model behavior, but they are not causal recommendations.

## Ollama configuration (optional)

By default, the app uses local Ollama at `http://localhost:11434` with `qwen2.5-coder:latest`. Install and start Ollama, then download the default local model:

```powershell
ollama pull qwen2.5-coder:latest
```

For Ollama Cloud, enter an API key in the app sidebar for the current browser session, or set `OLLAMA_API_KEY`. The app does not save sidebar keys to disk.

The following optional environment variables override the defaults:

| Variable | Purpose |
|---|---|
| `OLLAMA_API_KEY` | Ollama Cloud API key |
| `OLLAMA_HOST` | Custom Ollama server URL |
| `OLLAMA_MODEL` | Model name to use |
| `DELAY_MODEL_PATH` | Path to an alternate `delay_model.joblib` artifact |

Example for the current PowerShell session:

```powershell
$env:OLLAMA_MODEL = "qwen2.5-coder:latest"
streamlit run streamlit/app.py
```

## Rebuilding the model and data products

Run the notebooks in order from the `Notebooks` directory or in an environment where their relative paths resolve correctly:

1. `01_Preprocessing.ipynb` — cleans the raw SCMS delivery-history data.
2. `02_EDA_and_Vendor_Analysis.ipynb` — explores delay patterns and vendor performance.
3. `03_Modeling_clean.ipynb` — trains, evaluates, calibrates, and saves the delay-risk model artifact.

After retraining, ensure the application model is available at `model_artifacts/delay_model.joblib`, or set `DELAY_MODEL_PATH` before launching Streamlit.

## Outputs

| Location | Description |
|---|---|
| `data/processed_data/scms_clean.pkl` | Cleaned shipment data |
| `data/processed_data/vendor_scores.pkl` | Vendor performance scores |
| `data/processed_data/feature_ranking.csv` | Feature-ranking export |
| `data/processed_data/selected_features.json` | Selected model features |
| `data/processed_data/powerbi_delay_risk.csv` | Power BI-compatible risk export |
| `model_artifacts/delay_model.joblib` | Application model, calibration data, and historical reference data |

## Technology

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, Streamlit, Altair, joblib, and requests.

## Model limitations

- Predictions depend on the coverage and quality of the historical training data.
- Low-frequency categories and unfamiliar shipment profiles may produce unreliable estimates.
- The explanatory chart describes the model's behavior, not real-world causes.
- Use the score to prioritize review and follow-up; combine it with operational knowledge before making decisions.
