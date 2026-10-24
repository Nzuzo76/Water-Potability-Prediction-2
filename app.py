# =============================================================================
# app.py  |  Water Potability Analysis  (redesigned)
# -----------------------------------------------------------------------------
# Run with:   streamlit run app.py
#
# Files expected next to this script:
#   - random_forest_potability_model.pkl   (joblib dict: {"model": ..., "features": [...]})
#   - water_potability.csv                 (Kaggle water potability dataset)
#   - brisbane_water_quality.csv           (optional, only used by the Forecasting page)
#
# Design notes
#   * Palette: deep lagoon teal + cool mist background. Green / amber / red are
#     reserved for meaning only (pass / borderline / fail), never decoration.
#   * Type: "Bricolage Grotesque" for headings, "IBM Plex Sans" for body text.
#   * The signature element is the verdict panel on the Predict page: it shows the
#     ML estimate AND a parameter-by-parameter check against reference limits,
#     so users can see *why* a sample looks good or bad.
# =============================================================================

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
MODEL_PATH = "random_forest_potability_model.pkl"
DATA_PATH = "water_potability.csv"
BRISBANE_PATH = "brisbane_water_quality.csv"

# Colour tokens (kept in one place so charts and CSS always match)
INK = "#12313D"        # main text
LAGOON = "#0E7C86"     # primary accent
DEEP = "#0B4F5C"       # headers, hero band
MIST = "#F3F7F8"       # page background
LINE = "#D5E2E6"       # borders and gridlines
MUTED = "#5B7480"      # secondary text
PASS = "#2F8F5B"       # semantic: within limit
WARN = "#D99A29"       # semantic: borderline
FAIL = "#C8553D"       # semantic: outside limit

# Metadata for every model input.
#   key      -> column name in the dataset / model
#   label    -> text shown to the user
#   unit     -> unit shown next to the label
#   min,max  -> slider range
#   default  -> slider starting value
#   step     -> slider step
#   ref      -> (low, high, source note) indicative drinking-water guideline.
# These limits are INDICATIVE (WHO / US EPA style). Edit them to match your local
# standard, e.g. SANS 241 in South Africa.
FEATURES = [
    dict(key="ph", label="pH", unit="", min=0.0, max=14.0, default=7.0, step=0.01,
         ref=(6.5, 8.5, "WHO guideline range")),
    dict(key="Hardness", label="Hardness", unit="mg/L", min=0.0, max=350.0, default=196.0, step=0.1,
         ref=(0, 300, "Aesthetic guidance")),
    dict(key="Solids", label="Total dissolved solids", unit="ppm", min=0.0, max=65000.0, default=20900.0, step=100.0,
         ref=(0, 1000, "WHO palatability limit")),
    dict(key="Chloramines", label="Chloramines", unit="ppm", min=0.0, max=14.0, default=7.1, step=0.01,
         ref=(0, 4, "US EPA maximum residual")),
    dict(key="Sulfate", label="Sulfate", unit="mg/L", min=0.0, max=500.0, default=333.0, step=0.1,
         ref=(0, 250, "Taste threshold")),
    dict(key="Trihalomethanes", label="Trihalomethanes", unit="µg/L", min=0.0, max=130.0, default=66.4, step=0.1,
         ref=(0, 80, "US EPA maximum")),
    dict(key="Conductivity", label="Conductivity", unit="µS/cm", min=0.0, max=800.0, default=421.0, step=0.1,
         ref=(0, 400, "Indicative limit")),
    dict(key="Organic_carbon", label="Organic carbon", unit="mg/L", min=0.0, max=30.0, default=14.3, step=0.1,
         ref=(0, 4, "Indicative limit")),
    dict(key="Turbidity", label="Turbidity", unit="NTU", min=0.0, max=7.0, default=3.97, step=0.01,
         ref=(0, 5, "WHO aesthetic limit")),
]
FEATURE_BY_KEY = {f["key"]: f for f in FEATURES}

# How the sliders are grouped on the Predict page
INPUT_GROUPS = {
    "Acidity and minerals": ["ph", "Hardness", "Solids"],
    "Treatment chemicals": ["Chloramines", "Sulfate", "Trihalomethanes"],
    "Clarity and organics": ["Conductivity", "Organic_carbon", "Turbidity"],
}

NUMERIC_COLS = [f["key"] for f in FEATURES]

# -----------------------------------------------------------------------------
# 2. PAGE SETUP AND THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Water Potability Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS. NOTE: markup passed to st.markdown must not be indented by 4+
# spaces, otherwise Markdown renders it as a code block. The `html()` helper
# below strips indentation to avoid that.
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'IBM Plex Sans', sans-serif;
    color: {INK};
}}
.stApp {{ background: {MIST}; }}
h1, h2, h3, h4, .display {{
    font-family: 'Bricolage Grotesque', sans-serif !important;
    color: {DEEP};
    letter-spacing: -0.01em;
}}

/* Hide Streamlit chrome we do not need (keep the header: it holds the sidebar toggle) */
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px; }}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {{
    background: {DEEP};
}}
section[data-testid="stSidebar"] * {{ color: #E6F2F4 !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 0.45rem 0.6rem; border-radius: 6px; width: 100%;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{ background: rgba(255,255,255,0.08); }}
.brand {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 700; line-height: 1.15; }}
.brand-sub {{ font-size: 0.82rem; opacity: 0.75; margin-top: 0.25rem; }}

/* ---------- Page header ---------- */
.page-head {{ border-bottom: 1px solid {LINE}; padding-bottom: 1rem; margin-bottom: 1.5rem; }}
.page-head h1 {{ font-size: 2.1rem; margin: 0; }}
.page-head p {{ color: {MUTED}; margin: 0.35rem 0 0 0; max-width: 62ch; font-size: 1.02rem; }}

/* ---------- Hero band (Home) ---------- */
.hero {{
    background: {DEEP}; color: #fff; border-radius: 14px;
    padding: 2.2rem 2.4rem; margin-bottom: 1.4rem;
}}
.hero h1 {{ color: #fff !important; font-size: 2.5rem; margin: 0 0 0.6rem 0; max-width: 20ch; }}
.hero p {{ color: #CFE5E9; font-size: 1.05rem; max-width: 60ch; margin: 0; line-height: 1.55; }}

/* ---------- Stat strip: one panel, divided, not three identical cards ---------- */
.stat-strip {{
    display: flex; background: #fff; border: 1px solid {LINE}; border-radius: 12px;
    margin-bottom: 1.6rem; overflow: hidden;
}}
.stat {{ flex: 1; padding: 1.1rem 1.4rem; border-right: 1px solid {LINE}; }}
.stat:last-child {{ border-right: none; }}
.stat .num {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 2rem; font-weight: 700; color: {DEEP}; line-height: 1.1; }}
.stat .lbl {{ color: {MUTED}; font-size: 0.9rem; margin-top: 0.2rem; }}

/* ---------- Panels ---------- */
.panel {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; padding: 1.3rem 1.5rem; }}
.panel h3 {{ margin-top: 0; font-size: 1.15rem; }}
.note {{ color: {MUTED}; font-size: 0.9rem; line-height: 1.5; }}

/* ---------- Verdict panel (signature element) ---------- */
.verdict {{ border-radius: 12px; padding: 1.5rem 1.7rem; border: 1px solid; }}
.verdict.ok  {{ background: #EAF6EF; border-color: {PASS}; }}
.verdict.bad {{ background: #FBEDEA; border-color: {FAIL}; }}
.verdict .title {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.9rem; font-weight: 700; margin: 0; }}
.verdict.ok .title  {{ color: {PASS}; }}
.verdict.bad .title {{ color: {FAIL}; }}
.verdict .sub {{ margin: 0.3rem 0 1rem 0; color: {INK}; }}
.probbar {{ height: 12px; background: #fff; border-radius: 6px; overflow: hidden; border: 1px solid {LINE}; }}
.probbar > div {{ height: 100%; background: {LAGOON}; }}
.probrow {{ display: flex; justify-content: space-between; font-size: 0.88rem; margin-top: 0.35rem; color: {MUTED}; }}

/* ---------- Guideline check table ---------- */
table.check {{ width: 100%; border-collapse: collapse; font-size: 0.93rem; }}
table.check th {{ text-align: left; color: {MUTED}; font-weight: 500; padding: 0.4rem 0.5rem; border-bottom: 1px solid {LINE}; }}
table.check td {{ padding: 0.5rem 0.5rem; border-bottom: 1px solid {LINE}; font-variant-numeric: tabular-nums; }}
.pill {{ display: inline-block; padding: 0.1rem 0.6rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }}
.pill.pass {{ background: #DDF0E5; color: {PASS}; }}
.pill.warn {{ background: #FBF0D6; color: #9A6B0E; }}
.pill.fail {{ background: #F8DDD7; color: {FAIL}; }}

/* ---------- Streamlit widgets ---------- */
div[data-testid="stMetric"] {{
    background: #fff; border: 1px solid {LINE}; border-left: 4px solid {LAGOON};
    border-radius: 10px; padding: 0.8rem 1rem;
}}
.stButton > button, .stDownloadButton > button {{
    background: {LAGOON}; color: #fff; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.6rem 1.2rem;
}}
.stButton > button:hover {{ background: {DEEP}; color: #fff; }}
.stButton > button:focus-visible {{ outline: 3px solid {WARN}; outline-offset: 2px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}

@media (max-width: 760px) {{
    .stat-strip {{ flex-direction: column; }}
    .stat {{ border-right: none; border-bottom: 1px solid {LINE}; }}
    .hero h1 {{ font-size: 1.8rem; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render raw HTML safely: strip leading indentation so Markdown does not
    mistake it for a code block."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """Consistent title block used at the top of every page except Home."""
    html(f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p></div>')


def apply_chart_theme() -> None:
    """Make every matplotlib chart match the app's look."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": LINE,
        "axes.labelcolor": INK,
        "axes.titlecolor": DEEP,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": LINE,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


apply_chart_theme()


def show_fig(fig) -> None:
    """Display a matplotlib figure and free its memory."""
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. DATA AND MODEL LOADING (cached so they run once, not on every click)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained Random Forest and its feature order."""
    payload = joblib.load(MODEL_PATH)
    return payload["model"], payload["features"]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the potability dataset."""
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_brisbane_daily() -> pd.DataFrame:
    """Load Brisbane readings and turn them into one value per day.

    Steps: drop quality-flag columns, average duplicate timestamps, resample to
    daily means and fill short gaps by linear interpolation.
    """
    raw = pd.read_csv(BRISBANE_PATH)
    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    quality_cols = [c for c in raw.columns if "quality" in c.lower()]
    clean = raw.drop(columns=quality_cols + ["Record number"], errors="ignore")
    clean = clean.groupby("Timestamp", as_index=False).mean().sort_values("Timestamp")
    daily = clean.set_index("Timestamp")[["pH", "Turbidity"]].resample("D").mean()
    return daily.interpolate(method="linear")


@st.cache_data(show_spinner="Evaluating model on held-out data...")
def evaluate_model(data: pd.DataFrame) -> dict:
    """Re-create the training recipe and score it on a held-out 20% split.

    The recipe: stratified split, median imputation, up-sample the minority
    class in the TRAINING set only, then fit a depth-limited Random Forest.
    Cached, so the (slow) fit only runs once per session.
    """
    X = data[NUMERIC_COLS]
    y = data["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fill missing values using TRAINING medians only (avoids test-set leakage)
    medians = X_train.median()
    X_train, X_test = X_train.fillna(medians), X_test.fillna(medians)

    # Balance the classes by up-sampling each to the size of the larger one
    train = pd.concat([X_train, y_train], axis=1)
    majority = max(train["Potability"].value_counts())
    parts = [
        resample(train[train["Potability"] == c], replace=True, n_samples=majority, random_state=42)
        for c in (0, 1)
    ]
    train_bal = pd.concat(parts).sample(frac=1, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=20,
        min_samples_leaf=10, random_state=42, n_jobs=-1,
    )
    rf.fit(train_bal[NUMERIC_COLS], train_bal["Potability"])

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    report = classification_report(
        y_test, y_pred, target_names=["Not potable", "Potable"], output_dict=True
    )
    return dict(
        cm=confusion_matrix(y_test, y_pred),
        report=pd.DataFrame(report).transpose(),
        fpr=fpr, tpr=tpr,
        auc=roc_auc_score(y_test, y_prob),
        accuracy=report["accuracy"],
        precision=report["Potable"]["precision"],
        recall=report["Potable"]["recall"],
    )


@st.cache_data(show_spinner=False)
def fit_forecast(series: pd.Series, steps: int):
    """Fit ARIMA(2,1,2) and forecast `steps` days ahead with an 80% interval."""
    from statsmodels.tsa.arima.model import ARIMA  # imported here: only needed on this page

    fitted = ARIMA(series, order=(2, 1, 2)).fit()
    fc = fitted.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int(alpha=0.2)


# Load data and model separately so one missing file does not break everything
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found. Place `{DATA_PATH}` next to app.py and reload.")
    st.stop()

try:
    model, feature_names = load_model()
    model_loaded = True
except FileNotFoundError:
    model, feature_names, model_loaded = None, NUMERIC_COLS, False

# -----------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    html("""
        <div class="brand">Water Potability<br>Analysis</div>
        <div class="brand-sub">Screening and forecasting from water quality readings</div>
    """)
    st.write("")
    page = st.radio(
        "Navigate",
        ["Home", "Explore data", "Predict", "Forecast", "Model performance"],
        label_visibility="collapsed",
    )
    st.write("")
    if model_loaded:
        st.caption("Model loaded")
    else:
        st.warning("Model file not found. Prediction pages are disabled.")


# -----------------------------------------------------------------------------
# 5. HOME
# -----------------------------------------------------------------------------
if page == "Home":
    html("""
        <div class="hero">
            <h1>Check whether a water sample looks drinkable.</h1>
            <p>Enter nine lab readings and get a machine-learning estimate, checked against
            drinking-water guideline limits so you can see which readings cause concern.</p>
        </div>
    """)

    potable_pct = df["Potability"].mean() * 100
    html(f"""
        <div class="stat-strip">
            <div class="stat"><div class="num">{len(df):,}</div><div class="lbl">Samples in the dataset</div></div>
            <div class="stat"><div class="num">{potable_pct:.1f}%</div><div class="lbl">Labelled potable</div></div>
            <div class="stat"><div class="num">{len(NUMERIC_COLS)}</div><div class="lbl">Readings per sample</div></div>
            <div class="stat"><div class="num">{df[NUMERIC_COLS].isnull().sum().sum():,}</div><div class="lbl">Missing values</div></div>
        </div>
    """)

    left, right = st.columns([3, 2], gap="large")
    with left:
        html("""
            <div class="panel">
                <h3>What you can do here</h3>
                <p><b>Predict</b> — enter readings for one sample and see the estimate, the probability and a
                reading-by-reading guideline check.</p>
                <p><b>Explore data</b> — compare potable and non-potable samples, spot correlations and missing data.</p>
                <p><b>Forecast</b> — project pH and turbidity for the coming days from Brisbane monitoring data.</p>
                <p><b>Model performance</b> — accuracy, errors, ROC curve and which readings matter most.</p>
            </div>
        """)
    with right:
        html("""
            <div class="panel">
                <h3>Read this before relying on a result</h3>
                <p class="note">The estimate is a screening aid trained on a public dataset. It does not replace
                laboratory testing or your local drinking-water standard. Always confirm with an accredited lab
                before treating water as safe.</p>
            </div>
        """)

# -----------------------------------------------------------------------------
# 6. EXPLORE DATA
# -----------------------------------------------------------------------------
elif page == "Explore data":
    page_header("Explore data", "See how each reading differs between potable and non-potable samples.")

    tab_dist, tab_corr, tab_missing, tab_raw = st.tabs(
        ["Distributions", "Correlations", "Missing values", "Raw data"]
    )

    # --- Distributions: one feature at a time, split by class (clearer than 9 small charts)
    with tab_dist:
        feature = st.selectbox(
            "Reading",
            NUMERIC_COLS,
            format_func=lambda k: f"{FEATURE_BY_KEY[k]['label']} ({FEATURE_BY_KEY[k]['unit']})"
            if FEATURE_BY_KEY[k]["unit"] else FEATURE_BY_KEY[k]["label"],
        )
        fig, ax = plt.subplots(figsize=(9, 4))
        for cls, name, colour in [(0, "Not potable", FAIL), (1, "Potable", LAGOON)]:
            ax.hist(df.loc[df["Potability"] == cls, feature].dropna(), bins=30,
                    alpha=0.6, color=colour, label=name, edgecolor="white")
        lo, hi, _ = FEATURE_BY_KEY[feature]["ref"]
        ax.axvline(hi, color=INK, linestyle="--", linewidth=1.2, label=f"Guideline upper limit ({hi:g})")
        ax.set_xlabel(FEATURE_BY_KEY[feature]["label"])
        ax.set_ylabel("Samples")
        ax.legend(frameon=False)
        show_fig(fig)

        summary = df.groupby("Potability")[feature].describe()[["mean", "50%", "min", "max"]].round(2)
        summary.index = ["Not potable", "Potable"]
        summary.columns = ["Mean", "Median", "Min", "Max"]
        st.dataframe(summary, use_container_width=True)

    # --- Correlations
    with tab_corr:
        corr = df[NUMERIC_COLS + ["Potability"]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))  # hide the duplicate upper half
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="BrBG", center=0,
                    linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax)
        ax.grid(False)
        show_fig(fig)
        st.caption("Correlations between readings and with potability are all weak, which is why "
                   "accuracy on this dataset tops out near 77%.")

    # --- Missing values
    with tab_missing:
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values()
        if missing.empty:
            st.success("No missing values.")
        else:
            fig, ax = plt.subplots(figsize=(8, 3.2))
            pct = missing / len(df) * 100
            ax.barh(pct.index, pct.values, color=WARN)
            for y_pos, v in enumerate(pct.values):
                ax.text(v + 0.2, y_pos, f"{v:.1f}%", va="center", color=INK)
            ax.set_xlabel("Share of samples missing (%)")
            ax.grid(False)
            show_fig(fig)
            st.caption("Missing values are filled with the median when the model is evaluated.")

    # --- Raw data
    with tab_raw:
        st.dataframe(df.head(200), use_container_width=True, height=360)
        st.download_button("Download full dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="water_potability.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# 7. PREDICT
# -----------------------------------------------------------------------------
elif page == "Predict":
    page_header("Predict potability",
                "Enter the nine readings for one sample. Use an example to see how the result changes.")

    if not model_loaded:
        st.error(f"Model file `{MODEL_PATH}` was not found.")
        st.stop()

    # ----- Example presets: fill sliders with the median potable / non-potable sample
    def load_example(potability: int) -> None:
        """Callback: write dataset medians for one class into the slider state."""
        medians = df[df["Potability"] == potability][NUMERIC_COLS].median()
        for f in FEATURES:
            st.session_state[f"in_{f['key']}"] = float(
                np.clip(medians[f["key"]], f["min"], f["max"])
            )

    def reset_inputs() -> None:
        """Callback: return every slider to its default."""
        for f in FEATURES:
            st.session_state[f"in_{f['key']}"] = float(f["default"])

    # Initialise slider state once
    for f in FEATURES:
        st.session_state.setdefault(f"in_{f['key']}", float(f["default"]))

    b1, b2, b3, _ = st.columns([1.3, 1.5, 1, 2.2])
    b1.button("Example: potable", on_click=load_example, args=(1,), use_container_width=True)
    b2.button("Example: not potable", on_click=load_example, args=(0,), use_container_width=True)
    b3.button("Reset", on_click=reset_inputs, use_container_width=True)

    st.write("")

    # ----- Sliders, grouped by theme in three columns
    values = {}
    columns = st.columns(3, gap="large")
    for col, (group, keys) in zip(columns, INPUT_GROUPS.items()):
        with col:
            st.markdown(f"**{group}**")
            for key in keys:
                f = FEATURE_BY_KEY[key]
                unit = f" ({f['unit']})" if f["unit"] else ""
                values[key] = st.slider(
                    f"{f['label']}{unit}", f["min"], f["max"],
                    step=f["step"], key=f"in_{key}",
                )

    st.write("")
    run = st.button("Predict potability", type="primary")

    # Remember that a prediction was requested so the result stays visible
    # when the user nudges a slider afterwards.
    if run:
        st.session_state["predicted"] = True

    if st.session_state.get("predicted"):
        # Build the input row in the exact column order the model was trained on
        row = pd.DataFrame([values])[feature_names]
        pred = int(model.predict(row)[0])
        prob = model.predict_proba(row)[0]
        p_potable = prob[1] * 100

        st.markdown("---")
        left, right = st.columns([2, 3], gap="large")

        # ----- Left: the verdict
        with left:
            if pred == 1:
                css, title, sub = "ok", "Likely potable", "The model's estimate for this sample."
            else:
                css, title, sub = "bad", "Likely not potable", "The model's estimate for this sample."
            html(f"""
                <div class="verdict {css}">
                    <p class="title">{title}</p>
                    <p class="sub">{sub}</p>
                    <div class="probbar"><div style="width:{p_potable:.1f}%"></div></div>
                    <div class="probrow"><span>Not potable {prob[0]*100:.1f}%</span><span>Potable {p_potable:.1f}%</span></div>
                </div>
            """)
            st.caption("A screening estimate, not a safety certificate. Confirm with an accredited lab.")

        # ----- Right: reading-by-reading guideline check
        with right:
            rows, fails = "", 0
            for f in FEATURES:
                v = values[f["key"]]
                lo, hi, source = f["ref"]
                if lo <= v <= hi:
                    status, pill = "Within limit", "pass"
                elif (hi == 0) or (lo - 0.1 * abs(lo) <= v <= hi * 1.1):
                    status, pill = "Borderline", "warn"   # within 10% outside the limit
                else:
                    status, pill = "Outside limit", "fail"
                fails += pill == "fail"
                unit = f" {f['unit']}" if f["unit"] else ""
                limit_txt = f"{lo:g} to {hi:g}" if lo > 0 else f"up to {hi:g}"
                rows += (f"<tr><td>{f['label']}</td><td>{v:,.2f}{unit}</td>"
                         f"<td>{limit_txt}</td><td><span class='pill {pill}'>{status}</span></td></tr>")
            html(f"""
                <div class="panel">
                    <h3>Guideline check</h3>
                    <table class="check">
                        <tr><th>Reading</th><th>Your value</th><th>Reference</th><th>Status</th></tr>
                        {rows}
                    </table>
                </div>
            """)
            st.caption(f"{fails} of {len(FEATURES)} readings are outside the reference limit. "
                       "The dataset itself contains many samples above these limits, so the model "
                       "verdict and this check can disagree. Limits are editable in FEATURES.")

# -----------------------------------------------------------------------------
# 8. FORECAST
# -----------------------------------------------------------------------------
elif page == "Forecast":
    page_header("Forecast water quality",
                "ARIMA time-series projection of pH and turbidity from Brisbane monitoring data.")

    try:
        daily = load_brisbane_daily()
    except FileNotFoundError:
        st.warning(f"Add `{BRISBANE_PATH}` next to app.py to enable forecasting.")
        st.stop()

    st.caption(f"{len(daily):,} days of readings, {daily.index.min():%d %b %Y} to {daily.index.max():%d %b %Y}.")

    horizon = st.slider("Days to forecast", 1, 14, 7)

    if st.button("Generate forecast", type="primary"):
        with st.spinner("Fitting models..."):
            try:
                ph_mean, ph_ci = fit_forecast(daily["pH"], horizon)
                tu_mean, tu_ci = fit_forecast(daily["Turbidity"], horizon)
            except Exception as exc:  # statsmodels missing or model failed to converge
                st.error(f"Forecast failed: {exc}")
                st.info("Install statsmodels (`pip install statsmodels`) and check the data file.")
                st.stop()

        # ----- Headline numbers: tomorrow vs latest reading
        c1, c2 = st.columns(2)
        c1.metric("pH tomorrow", f"{ph_mean.iloc[0]:.2f}", f"{ph_mean.iloc[0] - daily['pH'].iloc[-1]:+.2f} vs latest")
        c2.metric("Turbidity tomorrow (NTU)", f"{tu_mean.iloc[0]:.2f}",
                  f"{tu_mean.iloc[0] - daily['Turbidity'].iloc[-1]:+.2f} vs latest", delta_color="inverse")

        # ----- Charts: last 60 days of history + forecast with 80% interval
        for name, series, mean, ci, colour in [
            ("pH", daily["pH"], ph_mean, ph_ci, LAGOON),
            ("Turbidity (NTU)", daily["Turbidity"], tu_mean, tu_ci, WARN),
        ]:
            fig, ax = plt.subplots(figsize=(9, 3.4))
            history = series.iloc[-60:]
            ax.plot(history.index, history.values, color=colour, linewidth=1.6, label="Observed")
            ax.plot(mean.index, mean.values, color=INK, linewidth=1.6, linestyle="--", label="Forecast")
            ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color=INK, alpha=0.12, label="80% interval")
            ax.set_title(name)
            ax.legend(frameon=False, ncol=3, loc="upper left")
            show_fig(fig)

        # ----- Feed the forecast into the classifier (other readings held at the dataset average)
        if model_loaded:
            avg = df[NUMERIC_COLS].mean()
            sample = avg.copy()
            sample["ph"], sample["Turbidity"] = ph_mean.iloc[0], tu_mean.iloc[0]
            row = pd.DataFrame([sample])[feature_names]
            p = model.predict_proba(row)[0][1] * 100
            label = "likely potable" if p >= 50 else "likely not potable"
            html(f"""
                <div class="panel"><h3>Potability outlook for tomorrow</h3>
                <p>With forecast pH and turbidity and all other readings at the dataset average,
                the model rates the water <b>{label}</b> ({p:.1f}% potable).</p>
                <p class="note">Only two readings are forecast; the rest are assumed. Treat this as a rough
                indication, not a prediction for Brisbane's actual water supply.</p></div>
            """)

# -----------------------------------------------------------------------------
# 9. MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "Model performance":
    page_header("Model performance",
                "How well a Random Forest separates potable from non-potable samples on data it has not seen.")

    ev = evaluate_model(df)

    # ----- Headline metrics, computed (not hard-coded) so they always match the charts
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{ev['accuracy']*100:.1f}%")
    m2.metric("Precision (potable)", f"{ev['precision']*100:.0f}%")
    m3.metric("Recall (potable)", f"{ev['recall']*100:.0f}%")
    m4.metric("ROC AUC", f"{ev['auc']:.3f}")
    st.caption("Precision: of samples called potable, how many truly are. "
               "Recall: of truly potable samples, how many were found. "
               "Scores come from a 20% held-out split using the same settings as the saved model.")

    st.write("")
    left, right = st.columns(2, gap="large")

    # ----- Feature importance (from the saved model)
    with left:
        st.markdown("**Which readings matter most**")
        if model_loaded:
            imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.barh(imp.index, imp.values, color=LAGOON)
            for y_pos, v in enumerate(imp.values):
                ax.text(v + 0.002, y_pos, f"{v:.3f}", va="center", fontsize=9, color=INK)
            ax.grid(False)
            ax.set_xlabel("Importance")
            show_fig(fig)
        else:
            st.info("Load the model file to see feature importance.")

    # ----- Confusion matrix
    with right:
        st.markdown("**Correct and incorrect calls**")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.heatmap(ev["cm"], annot=True, fmt="d", cmap="GnBu", cbar=False, linewidths=1,
                    xticklabels=["Not potable", "Potable"], yticklabels=["Not potable", "Potable"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        show_fig(fig)

    left, right = st.columns(2, gap="large")

    # ----- ROC curve
    with left:
        st.markdown("**ROC curve**")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(ev["fpr"], ev["tpr"], color=LAGOON, linewidth=2.2, label=f"Random Forest (AUC {ev['auc']:.3f})")
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Random guessing")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(frameon=False, loc="lower right")
        show_fig(fig)

    # ----- Full classification report
    with right:
        st.markdown("**Classification report**")
        st.dataframe(ev["report"].round(3), use_container_width=True)

# -----------------------------------------------------------------------------
# 10. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("Water Potability Analysis. Built with Streamlit and scikit-learn. "
           "For screening only; not a substitute for laboratory testing.")
