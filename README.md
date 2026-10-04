# Smart-Cargo-Monitoring-and-Decision-Support-System

An end-to-end supply-chain analytics project for understanding shipment delays,
comparing vendor performance, scoring delivery-risk levels, and presenting
operational insights through a Streamlit application and Power BI-ready data.

## Project overview

This project uses the SCMS Delivery History dataset to:

- clean and prepare shipment records;
- explore delivery performance and operational patterns;
- analyze vendor and shipment-mode performance;
- train a machine-learning model for shipment delay risk;
- generate model scores for dashboard analysis;
- provide an interactive what-if and risk-analysis interface.

The Google Gemini API is included in the dependencies for optional
LLM-generated explanations and insights in the application.

## Architecture

```text
intelligent-cargo-logistics/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── SCMS_Delivery_History_Dataset.csv
│   └── processed_data/
│       ├── scms_clean.pkl
│       ├── vendor_scores.pkl
│       ├── feature_ranking.csv
│       ├── selected_features.json
│       └── powerbi_delay_risk.csv
│
├── Notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_EDA_and_Vendor_Analysis.ipynb
│   └── 03_Modeling_clean.ipynb
│
├── models/
│   └── delay_model.joblib
│
├── model_artifacts/
│   └── delay_model.joblib
│
├── streamlit/
│   └── app.py
│
├── processed_data/
│   ├── scms_clean.pkl
│   └── vendor_scores.pkl
│
└── logi/
    └── Python virtual environment
```

## Notebook workflow

Run the notebooks in this order:

### 1. Data preprocessing

`Notebooks/01_Preprocessing.ipynb`

- loads the raw CSV from `data/raw/`;
- cleans column names and values;
- handles missing values and data types;
- creates shipment-level features;
- saves the cleaned dataset to `data/processed_data/scms_clean.pkl`.

### 2. Exploratory and vendor analysis

`Notebooks/02_EDA_and_Vendor_Analysis.ipynb`

- loads the cleaned dataset;
- analyzes delay distributions and shipment patterns;
- compares shipment modes and vendors;
- creates vendor performance scores;
- saves vendor analysis outputs to `data/processed_data/`.

### 3. Model development

`Notebooks/03_Modeling_clean.ipynb`

- prepares model features;
- trains and evaluates classification models;
- selects a decision threshold;
- saves the trained model to `models/delay_model.joblib`;
- exports scored records to `data/processed_data/powerbi_delay_risk.csv`.

## Running the project

### 1. Open the project directory

```powershell
cd "E:\NLP- predictive cargo latency engine"
```

### 2. Activate the existing environment

```powershell
.\logi\Scripts\Activate.ps1
```

If PowerShell blocks script execution for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\logi\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run the notebooks

Run the notebooks in the following order:

```text
01_Preprocessing.ipynb
02_EDA_and_Vendor_Analysis.ipynb
03_Modeling_clean.ipynb
```

The notebooks should be opened from the project workspace so paths such as
`../data/processed_data/` resolve correctly.

### 5. Launch the Streamlit application

From the project root:

```powershell
streamlit run streamlit/app.py
```

The trained model must exist before launching the application:

```text
models/delay_model.joblib
```

## Data outputs

| Output | Purpose |
|---|---|
| `data/processed_data/scms_clean.pkl` | Cleaned shipment records |
| `data/processed_data/vendor_scores.pkl` | Vendor-level performance analysis |
| `data/processed_data/feature_ranking.csv` | Feature importance or ranking results |
| `data/processed_data/selected_features.json` | Selected model features |
| `data/processed_data/powerbi_delay_risk.csv` | Model-scored records for Power BI |
| `models/delay_model.joblib` | Serialized trained model and related artifacts |

## Configuration and API keys

Do not commit API keys to the repository. If the Streamlit application uses
Google Gemini, provide the key through an environment variable or a local
`.env` file that is excluded from version control.

Example:

```text
GOOGLE_API_KEY=your_api_key_here
```

Never place a real key directly in a notebook, Python file, README, or
dashboard export.

## Technology stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- imbalanced-learn
- joblib
- Streamlit
- Altair
- Power BI-compatible CSV exports
- Google Gemini API for optional natural-language insights

## Important modeling note

The model output represents learned risk associations in historical shipment
data. It should support operational review and decision-making, not be treated
as proof that a particular shipment factor directly causes a delay.

## Future improvements

- consolidate duplicate generated-output folders;
- move reusable notebook logic into a `src/` package;
- add automated tests for preprocessing and inference;
- add model versioning and experiment tracking;
- add a `.env.example` file and `.gitignore`;
- add application screenshots and evaluation metrics.
