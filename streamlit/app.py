"""
Shipment delay: risk & what-if simulator  (Streamlit app)

HOW TO RUN
    1. Run the notebook first (Section 13 creates model_artifacts/delay_model.joblib)
    2. In a terminal:   streamlit run app.py

HOW THIS FILE IS ORGANISED (read it top to bottom)
    PART 1  Settings and the "not causal" warning text
    PART 2  Load the saved model
    PART 3  Helper functions (turn inputs into a row, score it, explain it, what-if, advice)
    PART 4  The web page itself (sections 1 to 5)
"""

import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


# =====================================================================================
# PART 1: SETTINGS
# =====================================================================================

# Where the notebook saved the model.
MODEL_FILE = Path(os.environ.get("DELAY_MODEL_PATH",
                                 Path(__file__).parent / "model_artifacts" / "delay_model.joblib"))

# Google Gemini configuration
GEMINI_API_KEY = "your_api_key_here"  # Replace with your actual API key
LLM_MODEL = "gemini-2.5-flash-lite"

# Column names that need special treatment.
WEIGHT = "Weight (Kilograms)"
FREIGHT = "Freight Cost (USD)"
DATE_FIELD = "Scheduled delivery date"      # a form field only; the model uses its month / weekday

# Features the app calculates by itself from the form (the user does not type them).
CALCULATED_FEATURES = {"cost_per_kg", "scheduled_month", "scheduled_weekday",
                       "weight_missing", "freight_missing"}

RISK_BAND_NAMES = ["Low", "Moderate", "Elevated", "High"]
RISK_BAND_ICON = {"Low": "🟢", "Moderate": "🟡", "Elevated": "🟠", "High": "🔴"}

# The warning shown at the top of the page.
DISCLAIMER = (
    "**Model-based estimate, not a causal claim.** Every number on this page is what a statistical model "
    "learned from past shipments. The what-if only re-scores the model with one input changed and everything "
    "else held fixed. It shows how the *model's score* responds, **not** what would actually happen if you made "
    "that change: in real life, switching mode also changes freight cost, lead time, route and more, and the "
    "model cannot tell cause from correlation."
)

st.set_page_config(page_title="Shipment delay what-if", page_icon="🚚", layout="wide")


# =====================================================================================
# PART 2: LOAD THE SAVED MODEL
# =====================================================================================

@st.cache_resource(show_spinner="Loading model...")
def load_model_file(path):
    """Read the joblib file once and prepare a few lookup tables the page needs."""
    art = joblib.load(path)                       # 'art' = everything the notebook saved
    history = art["history"]

    # For every category column: how many past shipments had each value, and the list of values
    # sorted from most common to least common.
    art["counts"] = {}
    art["options"] = {}
    for col in art["cat_features"]:
        counts = history[col].value_counts()
        art["counts"][col] = counts.to_dict()
        art["options"][col] = list(counts.index)

    # A "typical shipment": most common category, median number.
    typical = {}
    for col in art["cat_features"]:
        typical[col] = art["options"][col][0]
    for col in art["num_features"]:
        typical[col] = art["numeric_stats"][col]["median"]
    art["typical"] = typical
    return art


# =====================================================================================
# PART 3: HELPER FUNCTIONS
# =====================================================================================

# ---------- 3a. Inputs -> one row the model understands ---------------------------------

def get_form_fields(art):
    """Names of the fields the user fills in (model features minus the ones we calculate)."""
    fields = [f for f in art["features"] if f not in CALCULATED_FEATURES]
    if "scheduled_month" in art["features"] or "scheduled_weekday" in art["features"]:
        fields.append(DATE_FIELD)                 # one date box replaces month + weekday
    return fields


def build_row(inputs, art):
    """Turn the form answers (a dict) into a one-row table with exactly the model's columns."""
    row = {}
    for feature in art["features"]:
        if feature in art["cat_features"]:
            row[feature] = inputs[feature]
        elif feature == "scheduled_month":
            row[feature] = inputs[DATE_FIELD].month
        elif feature == "scheduled_weekday":
            row[feature] = inputs[DATE_FIELD].weekday()
        elif feature == "cost_per_kg":
            freight, weight = inputs.get(FREIGHT), inputs.get(WEIGHT)
            if freight is not None and weight:    # both known and weight is not zero
                row[feature] = freight / weight
            else:
                row[feature] = np.nan
        elif feature == "weight_missing":
            row[feature] = int(inputs.get(WEIGHT) is None)
        elif feature == "freight_missing":
            row[feature] = int(inputs.get(FREIGHT) is None)
        else:                                      # an ordinary number
            value = inputs[feature]
            row[feature] = np.nan if value is None else float(value)
    return pd.DataFrame([row])[art["features"]]


# ---------- 3b. Score a row --------------------------------------------------------------

def calibrate(logit, art):
    """Platt scaling: turns the model's raw score (as a logit) into a real probability.
    This is just  1 / (1 + exp(-(a * logit + b)))  with two numbers a and b saved by the notebook.
    We compute it by hand instead of calling scikit-learn's LogisticRegression, so the app does not
    break when scikit-learn versions differ between the notebook and the app."""
    if "calibrator_params" in art:                       # saved by the current notebook
        a, b = art["calibrator_params"]
    else:                                                # older model file: read the numbers from the object
        a = float(art["calibrator"].coef_[0][0])
        b = float(art["calibrator"].intercept_[0])
    return 1 / (1 + np.exp(-(a * logit + b)))


def score(rows, art):
    """Delay probability (0 to 1) for each row. Same two steps as the notebook:
    1) the model gives a raw score, 2) the calibrator turns it into a real probability."""
    raw = art["pipeline"].predict_proba(rows)[:, 1]
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(raw / (1 - raw))
    return calibrate(logit, art)


def get_risk_band(probability, art):
    """Low / Moderate / Elevated / High, using the cut-offs saved by the notebook."""
    band = pd.cut([probability], bins=art["risk_bins"], labels=RISK_BAND_NAMES, include_lowest=True)
    return str(band[0])


# ---------- 3c. "Why this risk?" ---------------------------------------------------------

def group_features(features):
    """Features that belong together are explained as one factor
    (e.g. weight, freight cost and cost per kg are one 'Weight & freight cost' factor)."""
    groups = {}
    for feature in features:
        if feature in (WEIGHT, FREIGHT, "cost_per_kg", "weight_missing", "freight_missing"):
            name = "Weight & freight cost"
        elif feature in ("scheduled_month", "scheduled_weekday"):
            name = "Scheduled delivery date"
        else:
            name = feature
        groups.setdefault(name, []).append(feature)
    return groups


def describe_value(group_name, columns, row):
    """Short text for the factor's current value, e.g. 'Ocean' or 'weight 785, freight cost 5,151'."""
    r = row.iloc[0]
    if group_name == "Scheduled delivery date" and "scheduled_month" in columns:
        return datetime.date(2000, int(r["scheduled_month"]), 1).strftime("%B")
    if group_name == "Weight & freight cost":
        parts = []
        for col in (WEIGHT, FREIGHT):
            if col in columns:
                short_name = col.split(" (")[0].lower()
                if pd.isna(r[col]):
                    parts.append(f"{short_name} unknown")
                else:
                    parts.append(f"{short_name} {r[col]:,.4g}")
        return ", ".join(parts) if parts else "n/a"
    value = r[columns[0]]
    if isinstance(value, str):
        return value
    return "unknown" if pd.isna(value) else f"{value:,.4g}"


def compute_drivers(row, art):
    """For each factor: effect = (risk now) - (average risk if this factor took the values of
    typical past shipments). Positive = this shipment's value pushes the risk UP."""
    background = art["background"]                # a sample of past shipments
    n = len(background)
    groups = group_features(art["features"])
    current_risk = float(score(row, art)[0])

    # Build one big table: for each factor, n copies of our row where that factor's columns
    # are replaced by the values from the background shipments. Then score everything at once.
    blocks = []
    for name, columns in groups.items():
        copies = pd.concat([row] * n, ignore_index=True)
        for col in columns:
            copies[col] = background[col].to_numpy()
        blocks.append(copies)
    all_risks = score(pd.concat(blocks, ignore_index=True), art)
    average_risk_per_factor = all_risks.reshape(len(groups), n).mean(axis=1)

    table = pd.DataFrame({
        "factor": list(groups),
        "value": [describe_value(name, cols, row) for name, cols in groups.items()],
        "effect_pts": (current_risk - average_risk_per_factor) * 100,    # percentage points
    })
    table["label"] = table["factor"] + ": " + table["value"].astype(str)
    # biggest effects first (ignoring the + / - sign)
    return table.sort_values("effect_pts", key=np.abs, ascending=False).reset_index(drop=True)


# ---------- 3d. What-if ------------------------------------------------------------------

def risk_after_change(inputs, field, new_value, art):
    """Copy the inputs, change ONE field, and score again."""
    changed = dict(inputs)
    changed[field] = new_value
    return float(score(build_row(changed, art), art)[0])


def past_data_support(inputs, field, value, art):
    """How many past shipments had this value? (Few = the model is guessing.)"""
    history = art["history"]
    matching = history[history[field] == value]
    info = {
        "n_total": int(len(matching)),
        "observed_delay_rate": float(matching["is_delayed"].mean()) if len(matching) else None,
        "n_same_country": None,
    }
    if "Country" in art["cat_features"] and field != "Country":
        info["n_same_country"] = int((matching["Country"] == inputs["Country"]).sum())
    return info


def category_sweep(inputs, field, art, max_options=12):
    """Score the same shipment once for every value of a category field (e.g. every shipment mode)."""
    values = art["options"][field][:max_options]
    if inputs[field] not in values:
        values = values + [inputs[field]]
    rows = []
    for value in values:
        changed = dict(inputs)
        changed[field] = value
        rows.append(build_row(changed, art))
    risks = score(pd.concat(rows, ignore_index=True), art)

    history = art["history"]
    return pd.DataFrame({
        field: values,
        "Model risk": risks,
        "Past shipments": [int((history[field] == v).sum()) for v in values],
        "Observed delay rate (raw)": [float(history.loc[history[field] == v, "is_delayed"].mean())
                                      for v in values],
    })


def number_sweep(inputs, field, art, points=25):
    """Score the same shipment for 25 values of a number field, from the 1st to the 99th percentile."""
    stats = art["numeric_stats"][field]
    values = np.linspace(stats["p01"], stats["p99"], points)
    rows = []
    for value in values:
        changed = dict(inputs)
        changed[field] = float(value)
        rows.append(build_row(changed, art))
    return pd.DataFrame({field: values, "Model risk": score(pd.concat(rows, ignore_index=True), art)})


# ---------- 3e. Advice -------------------------------------------------------------------

ADVICE_INSTRUCTIONS = (
    "You are a logistics risk analyst assistant inside a shipment-delay what-if tool. You receive outputs of a "
    "statistical model: a calibrated delay probability, factor effects and one what-if. Write short, practical "
    "advice for the shipment planner.\n"
    "Rules:\n"
    "- Use only the numbers and facts provided. Do not invent carriers, costs, lead times, routes or statistics.\n"
    "- Never state or imply that making the what-if change will cause the risk to change. The what-if shows how the "
    "model's score responds; it is not causal. Say 'the model estimates', not 'this will'.\n"
    "- If the facts flag thin historical support for the alternative, say the estimate is unreliable.\n"
    "- Give 3 to 5 bullet points, then one short caveat line. Maximum 160 words."
)


def collect_facts(inputs, probability, band, drivers, whatif, art):
    """Gather everything the advice is based on into one dictionary."""
    raising = drivers[drivers["effect_pts"] > 0].head(3)
    lowering = drivers[drivers["effect_pts"] < 0].head(3)
    return {
        "shipment": {k: ("unknown" if v is None else str(v)) for k, v in inputs.items()},
        "estimated_delay_risk": f"{probability:.1%}",
        "risk_band": band,
        "alert_threshold": f"{art['threshold']:.0%}",
        "flagged_as_high_risk": bool(probability >= art["threshold"]),
        "historical_average_delay_rate": f"{art['base_rate']:.1%}",
        "factors_raising_risk_vs_typical_shipment": [f"{r.label} ({r.effect_pts:+.1f} pts)" for r in raising.itertuples()],
        "factors_lowering_risk_vs_typical_shipment": [f"{r.label} ({r.effect_pts:+.1f} pts)" for r in lowering.itertuples()],
        "what_if": whatif,
        "model": f"{art['model_name']}, test ROC-AUC {art['metrics']['test']['ROC-AUC']:.2f}, "
                 f"PR-AUC {art['metrics']['test']['PR-AUC']:.2f} (modest skill)",
    }


def simple_advice(facts):
    """Plain rule-based advice, used when there is no API key (or the LLM call fails)."""
    lines = []
    if facts["flagged_as_high_risk"]:
        lines.append(f"- The model puts this shipment at {facts['estimated_delay_risk']} "
                     f"({facts['risk_band']} band), above its alert threshold of {facts['alert_threshold']}. "
                     "Treat it as a candidate for closer follow-up.")
    else:
        lines.append(f"- The model puts this shipment at {facts['estimated_delay_risk']} "
                     f"({facts['risk_band']} band), below its alert threshold of {facts['alert_threshold']}.")

    if facts["factors_raising_risk_vs_typical_shipment"]:
        lines.append("- Biggest factors pushing the score up: " +
                     "; ".join(facts["factors_raising_risk_vs_typical_shipment"]) +
                     ". Check whether these can be confirmed or de-risked with the vendor.")

    whatif = facts["what_if"]
    if whatif and whatif.get("delta_pts") is not None:
        if whatif["delta_pts"] < -1:
            lines.append(f"- The model scores the alternative ({whatif['field']} = {whatif['to']}) lower by "
                         f"{abs(whatif['delta_pts']):.0f} points. That is an association in past data, so check "
                         "cost and feasibility before acting on it.")
        elif whatif["delta_pts"] > 1:
            lines.append(f"- The model scores the alternative ({whatif['field']} = {whatif['to']}) higher by "
                         f"{whatif['delta_pts']:.0f} points, so it does not look attractive on risk.")
        else:
            lines.append(f"- Changing {whatif['field']} to {whatif['to']} barely moves the model's score.")
        if whatif.get("support_warning"):
            lines.append(f"- Caution: {whatif['support_warning']}")

    lines.append("\n*Caveat: model-based estimate from past data, not a causal claim.*")
    return "\n".join(lines)


def find_api_key():
    key = os.environ.get("GEMINI_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def get_advice(facts):

    key = find_api_key()

    if not key:
        return simple_advice(facts), "Rule-based advice."

    try:
        from google import genai

        client = genai.Client(api_key=key)

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=(
                ADVICE_INSTRUCTIONS
                + "\n\nModel outputs (JSON):\n"
                + json.dumps(facts, indent=2)
            )
        )

        if not response.text:
            raise ValueError("Empty response")

        return response.text, f"Gemini AI advice ({LLM_MODEL})"

    except Exception as error:
        return (
            simple_advice(facts),
            f"LLM call failed ({type(error).__name__}); "
            "showing rule-based advice instead."
        )
# =====================================================================================
# PART 4: THE PAGE
# =====================================================================================

st.title("🚚 Shipment delay: risk & what-if simulator")
st.warning(DISCLAIMER, icon="⚠️")

# ---- Load the model (stop with a clear message if the file is missing) -------------------
if not MODEL_FILE.exists():
    st.error(f"Model file not found: `{MODEL_FILE}`. Run Section 13 of `04_Modeling_clean.ipynb` first.")
    st.stop()

art = load_model_file(str(MODEL_FILE))

saved_version = art.get("versions", {}).get("scikit-learn")
if saved_version not in (None, sklearn.__version__):
    st.warning(f"The model was saved with scikit-learn {saved_version}, but this app is running "
               f"scikit-learn {sklearn.__version__}. To avoid errors, install the same version: "
               f"`pip install scikit-learn=={saved_version}`")

form_fields = get_form_fields(art)
category_fields = [f for f in form_fields if f in art["cat_features"]]
number_fields = [f for f in form_fields if f in art["num_features"]]


# ---- Form defaults and the two helper buttons ---------------------------------------------
# Streamlit remembers each form box in st.session_state under the box's "key".

def reset_form():
    """Put every box back to the typical shipment."""
    for f in category_fields:
        st.session_state[f"in_{f}"] = art["typical"][f]
    for f in number_fields:
        st.session_state[f"in_{f}"] = float(art["typical"][f])
        st.session_state[f"unk_{f}"] = False
    st.session_state["in_date"] = datetime.date.today()


def load_random_example():
    """Fill the form with one random past shipment."""
    example = art["background"].sample(1, random_state=int(time.time()) % 10_000).iloc[0]
    for f in category_fields:
        st.session_state[f"in_{f}"] = example[f]
    for f in number_fields:
        is_missing = pd.isna(example[f])
        st.session_state[f"in_{f}"] = float(art["typical"][f] if is_missing else example[f])
        st.session_state[f"unk_{f}"] = bool(is_missing)


if "in_date" not in st.session_state:             # first visit: start with the typical shipment
    reset_form()


# ---- SECTION 1: shipment details ----------------------------------------------------------
st.header("1. Shipment details")

button_a, button_b, _ = st.columns([1, 1, 4])
button_a.button("Start from a past shipment", on_click=load_random_example,
                help="Fills the form with a random past shipment.")
button_b.button("Reset to typical", on_click=reset_form,
                help="Most common category, median number.")

inputs = {}                                       # the user's answers: field name -> value

# Category fields (drop-down lists), three per row.
columns = st.columns(3)
for i, field in enumerate(category_fields):
    with columns[i % 3]:
        counts = art["counts"][field]
        inputs[field] = st.selectbox(
            field, art["options"][field], key=f"in_{field}",
            format_func=lambda value, c=counts: f"{value}  ({c.get(value, 0):,})",
        )
st.caption("Numbers in brackets = past shipments with that value.")

# Number fields, three per row. Some have an "Unknown" tick-box (past data had gaps there).
columns = st.columns(3)
for i, field in enumerate(number_fields):
    stats = art["numeric_stats"][field]
    with columns[i % 3]:
        step = 1.0 if stats["median"] >= 100 else 0.01
        value = st.number_input(field, min_value=min(0.0, stats["min"]), step=step,
                                format="%g", key=f"in_{field}")
        is_unknown = stats["has_missing"] and st.checkbox("Unknown", key=f"unk_{field}")
        inputs[field] = None if is_unknown else float(value)

if DATE_FIELD in form_fields:
    inputs[DATE_FIELD] = st.date_input(DATE_FIELD, key="in_date")

# Score the shipment the user described.
row = build_row(inputs, art)
risk = float(score(row, art)[0])
band = get_risk_band(risk, art)


# ---- SECTION 2: estimated risk ------------------------------------------------------------
st.header("2. Estimated delay risk")

col1, col2, col3 = st.columns(3)
col1.metric("Estimated delay risk", f"{risk:.0%}",
            delta=f"{(risk - art['base_rate']) * 100:+.0f} pts vs average shipment",
            delta_color="inverse")                # red when above average
col2.metric("Risk band", f"{RISK_BAND_ICON[band]} {band}")
col3.metric("Alert threshold", f"{art['threshold']:.0%}",
            help="Chosen in the notebook to maximise F1 on out-of-fold training data.")

if risk >= art["threshold"]:
    flag_text = "🚩 **Flagged**: at or above the alert threshold."
else:
    flag_text = "✅ Not flagged: below the alert threshold."
st.write(f"{flag_text} The historical delay rate is {art['base_rate']:.0%}.")


# ---- SECTION 3: why this risk? ------------------------------------------------------------
st.header("3. Why this risk?")

drivers = compute_drivers(row, art)
top_drivers = drivers.head(8).copy()              # show the 8 biggest factors
top_drivers["Direction"] = np.where(top_drivers["effect_pts"] >= 0, "Raises risk", "Lowers risk")

drivers_chart = alt.Chart(top_drivers).mark_bar().encode(
    x=alt.X("effect_pts:Q", title="Effect on risk (percentage points)"),
    y=alt.Y("label:N", sort=list(top_drivers["label"]), title=None),
    color=alt.Color("Direction:N", legend=None,
                    scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=["#d62728", "#2ca02c"])),
    tooltip=[alt.Tooltip("label:N", title="Factor"),
             alt.Tooltip("effect_pts:Q", title="Effect (pts)", format="+.1f")],
).properties(height=28 * len(top_drivers) + 30)
st.altair_chart(drivers_chart, width="stretch")
st.caption("Each bar compares this shipment's value with the values of typical past shipments, with all other "
           "inputs unchanged. Bars are approximate and do not add up exactly to the total (the model has "
           "interactions). They describe the model, not causes.")


# ---- SECTION 4: what-if -------------------------------------------------------------------
st.header("4. What-if")
st.info("Model-based estimate, **not a causal claim**. Only the selected input changes; freight cost, weight, "
        "lead time and everything else stay as entered above.", icon="ℹ️")

# 4a. Which input to change? (Shipment Mode is the default.)
what_if_choices = category_fields + number_fields + ([DATE_FIELD] if DATE_FIELD in form_fields else [])
default_index = what_if_choices.index("Shipment Mode") if "Shipment Mode" in what_if_choices else 0
what_if_field = st.selectbox("Change this input", what_if_choices, index=default_index, key="wi_field")
current_value = inputs[what_if_field]

# 4b. What should it change to?
if what_if_field in category_fields:
    other_values = [v for v in art["options"][what_if_field] if v != current_value]
    new_value = st.selectbox(
        f"New {what_if_field}", other_values, key=f"wi_new_{what_if_field}",
        format_func=lambda v: f"{v}  ({art['counts'][what_if_field].get(v, 0):,})")
elif what_if_field == DATE_FIELD:
    new_value = st.date_input("New scheduled delivery date", value=current_value + datetime.timedelta(days=30))
else:
    stats = art["numeric_stats"][what_if_field]
    if current_value is None:
        suggested = stats["median"]
    elif current_value > 0:
        suggested = current_value * 1.25          # suggest +25%
    else:
        suggested = stats["median"]
    new_value = st.number_input(f"New {what_if_field}", min_value=min(0.0, stats["min"]),
                                value=float(suggested), step=1.0 if stats["median"] >= 100 else 0.01,
                                format="%g")

# 4c. The result.
new_risk = risk_after_change(inputs, what_if_field, new_value, art)
change_pts = (new_risk - risk) * 100

current_text = "unknown" if current_value is None else (
    f"{current_value:,.4g}" if isinstance(current_value, float) else str(current_value))
new_text = f"{new_value:,.4g}" if isinstance(new_value, float) else str(new_value)

if what_if_field == "Shipment Mode":
    sentence_start = f"If this ships by **{new_text}** instead of **{current_text}**"
else:
    sentence_start = f"If **{what_if_field}** is **{new_text}** instead of **{current_text}**"

st.markdown(f"### {sentence_start}, the model's estimated risk goes from **{risk:.0%}** to "
            f"**{new_risk:.0%}** ({change_pts:+.0f} pts)")
st.caption("Model-based estimate, not a causal claim.")

# This dictionary is also what the advice section uses.
what_if_facts = {"field": what_if_field, "from": current_text, "to": new_text,
                 "risk_from": f"{risk:.1%}", "risk_to": f"{new_risk:.1%}",
                 "delta_pts": round(change_pts, 1), "support_warning": None}

# 4d. Extra detail: a chart of every option, plus how much past data backs the alternative.
if what_if_field in category_fields:
    support = past_data_support(inputs, what_if_field, new_value, art)
    observed = "n/a" if support["observed_delay_rate"] is None else f"{support['observed_delay_rate']:.0%}"
    message = (f"Past data for {what_if_field} = {new_text}: {support['n_total']:,} shipments, "
               f"raw delay rate {observed}.")
    if support["n_same_country"] is not None:
        message += f" Of these, {support['n_same_country']:,} went to {inputs['Country']}."
    st.write(message)

    # Warn when there is little past data (the model is then guessing).
    little_data = support["n_total"] < 50 or (
        support["n_same_country"] is not None and support["n_same_country"] < 20)
    if little_data:
        warning_text = (f"little past data for {what_if_field} = {new_text} with this shipment's profile "
                        f"({support['n_total']} shipments overall")
        if support["n_same_country"] is not None:
            warning_text += f", {support['n_same_country']} to {inputs['Country']}"
        warning_text += "), so the model is extrapolating and the estimate is unreliable."
        st.warning("Limited history: " + warning_text)
        what_if_facts["support_warning"] = warning_text

    sweep = category_sweep(inputs, what_if_field, art)
    sweep["is_current"] = sweep[what_if_field] == current_value
    sweep_chart = alt.Chart(sweep).mark_bar().encode(
        x=alt.X("Model risk:Q", axis=alt.Axis(format="%"), title="Model's estimated delay risk"),
        y=alt.Y(f"{what_if_field}:N", sort="-x", title=None),
        color=alt.condition("datum.is_current", alt.value("#1f77b4"), alt.value("#9ecae1")),
        tooltip=[what_if_field, alt.Tooltip("Model risk:Q", format=".1%"), "Past shipments",
                 alt.Tooltip("Observed delay rate (raw):Q", format=".1%")],
    ).properties(height=28 * len(sweep) + 30)
    st.altair_chart(sweep_chart, width="stretch")
    st.caption(f"Dark bar = your current {what_if_field}. Same shipment, only {what_if_field} swapped. "
               "'Observed delay rate (raw)' in the table is not adjusted for anything else, so it can differ "
               "a lot from the model's number.")
    st.dataframe(
        sweep.drop(columns="is_current").style.format(
            {"Model risk": "{:.1%}", "Observed delay rate (raw)": "{:.1%}", "Past shipments": "{:,}"}),
        width="stretch", hide_index=True)

elif what_if_field in number_fields:
    sweep = number_sweep(inputs, what_if_field, art)
    line = alt.Chart(sweep).mark_line().encode(
        x=alt.X(f"{what_if_field}:Q"),
        y=alt.Y("Model risk:Q", axis=alt.Axis(format="%"), title="Model's estimated risk"))

    # Two dots on the line: the what-if point and (if the value is known) the current point.
    dots = [("What-if", new_value, new_risk)]
    if current_value is not None:
        dots.append(("Current", current_value, risk))
    dots_table = pd.DataFrame(dots, columns=["Point", what_if_field, "Model risk"])
    dots_chart = alt.Chart(dots_table).mark_point(size=120, filled=True).encode(
        x=f"{what_if_field}:Q", y="Model risk:Q", color=alt.Color("Point:N", legend=alt.Legend(title=None)))

    st.altair_chart(line + dots_chart, width="stretch")
    st.caption(f"Model's score as {what_if_field} varies (1st to 99th percentile of past shipments), everything "
               "else fixed. A curve shows how the model reacts, not what would happen.")


# ---- SECTION 5: advice --------------------------------------------------------------------
st.header("5. Advice")

facts = collect_facts(inputs, risk, band, drivers, what_if_facts, art)
# A fingerprint of the current situation: if the inputs change, the old advice is out of date.
facts_fingerprint = hashlib.md5(json.dumps(facts, sort_keys=True, default=str).encode()).hexdigest()

if st.button("Generate advice", type="primary"):
    with st.spinner("Writing advice..."):
        advice_text, advice_note = get_advice(facts)
        st.session_state["advice"] = (facts_fingerprint, advice_text, advice_note)

saved_advice = st.session_state.get("advice")
if saved_advice and saved_advice[0] == facts_fingerprint:
    st.markdown(saved_advice[1])
    st.caption(saved_advice[2])
elif saved_advice:
    st.info("Inputs changed since the last advice. Click **Generate advice** to refresh.")
elif find_api_key():
    st.caption("LLM advice is written only from the numbers above.")
else:
    st.caption("No ANTHROPIC_API_KEY found: you will get simple rule-based advice. Set the key for LLM advice.")


# ---- About ---------------------------------------------------------------------------------
with st.expander("About this model and its limits"):
    test = art["metrics"]["test"]
    cutoffs = ", ".join(f"{b:.0%}" for b in art["risk_bins"][1:-1])
    st.markdown(
        f"- **Model:** {art['model_name']} with probability calibration, trained on {art['n_train']:,} shipments "
        f"scheduled {art['train_period'][0]} to {art['train_period'][1]}.\n"
        f"- **Held-out test (newest 20%):** ROC-AUC {test['ROC-AUC']:.2f}, PR-AUC {test['PR-AUC']:.2f}, "
        f"RMSE {test['RMSE (calibrated)']:.3f} vs {art['metrics']['baseline_rmse_test']:.3f} for always "
        f"predicting the base rate. The signal is real but modest.\n"
        f"- **Risk bands** (Low/Moderate/Elevated/High) are cut at the 40/70/90th percentiles of out-of-fold "
        f"scores: {cutoffs}.\n"
        "- **What-if** re-scores the same row with one input changed. Associations in past data are not causes, "
        "and inputs that really move together (mode, freight cost, lead time) are not updated for you.\n"
        "- **Reasons** compare each input with typical past values; they explain the model, not the world.")
