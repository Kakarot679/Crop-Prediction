from pathlib import Path
from html import escape
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset_with_ndvi_15groups.csv"
MODELS_DIR = PROJECT_ROOT / "models"
SEASON_MODEL_PATH = MODELS_DIR / "cb_season_predictor.cbm"
VALID_SEASONS = ["Kharif", "Rabi", "Zaid", "Annual"]


def inject_styles() -> None:
    st.html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --bg: #f4efe6;
            --panel: rgba(255, 250, 241, 0.84);
            --panel-strong: rgba(35, 62, 42, 0.9);
            --ink: #1d2b20;
            --muted: #5f705f;
            --line: rgba(42, 74, 53, 0.12);
            --accent: #2f6b45;
            --accent-2: #b98038;
            --soft: #dde6d6;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(185, 128, 56, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(47, 107, 69, 0.18), transparent 24%),
                linear-gradient(180deg, #f7f2e8 0%, #efe7d9 100%);
            color: var(--ink);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            font-family: "DM Serif Display", "Palatino Linotype", serif;
            color: var(--ink);
            letter-spacing: 0.02em;
        }

        p, label, [data-testid="stMarkdownContainer"], [data-testid="stMetricLabel"], .stCaption {
            font-family: "Manrope", "Trebuchet MS", sans-serif;
        }

        [data-testid="stForm"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.2rem 1.1rem 1rem 1.1rem;
            box-shadow: 0 18px 50px rgba(46, 58, 41, 0.09);
            backdrop-filter: blur(10px);
        }

        [data-testid="stExpander"] {
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.35);
        }

        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid var(--line);
            padding: 1rem;
            border-radius: 20px;
        }

        label, .stSelectbox label, .stNumberInput label, .stTextInput label {
            color: var(--ink) !important;
            font-weight: 600;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            background: rgba(255, 252, 246, 0.92) !important;
            color: var(--ink) !important;
            border: 1px solid rgba(42, 74, 53, 0.18) !important;
            border-radius: 16px !important;
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="base-input"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] svg,
        div[data-testid="stNumberInput"] button svg {
            fill: var(--accent) !important;
            color: var(--accent) !important;
        }

        div[data-baseweb="select"]:focus-within,
        div[data-baseweb="base-input"]:focus-within {
            box-shadow: 0 0 0 3px rgba(47, 107, 69, 0.12) !important;
            border-radius: 16px !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #7d8b7e !important;
            opacity: 1 !important;
        }

        [data-testid="stExpander"] details summary p,
        [data-testid="stExpander"] details summary span {
            color: var(--ink) !important;
        }

        .hero-shell {
            background:
                linear-gradient(135deg, rgba(33, 61, 42, 0.97), rgba(56, 97, 63, 0.92));
            border-radius: 32px;
            padding: 1.7rem 1.8rem;
            color: #f9f3e7;
            box-shadow: 0 24px 60px rgba(35, 62, 42, 0.22);
            overflow: hidden;
            position: relative;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -50px -60px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(185, 128, 56, 0.24), transparent 68%);
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.76rem;
            opacity: 0.76;
            margin-bottom: 0.55rem;
        }

        .hero-title {
            font-family: "DM Serif Display", "Palatino Linotype", serif;
            font-size: 3rem;
            line-height: 0.94;
            margin: 0;
            max-width: 10ch;
        }

        .hero-copy {
            max-width: 54ch;
            margin-top: 0.9rem;
            color: rgba(249, 243, 231, 0.84);
            line-height: 1.6;
            font-size: 0.98rem;
        }

        .glass-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.1rem 1.15rem;
            box-shadow: 0 18px 45px rgba(46, 58, 41, 0.08);
            backdrop-filter: blur(8px);
        }

        .mini-stat {
            background: rgba(255, 255, 255, 0.56);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            min-height: 120px;
        }

        .mini-stat-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
        }

        .mini-stat-value {
            font-family: "DM Serif Display", "Palatino Linotype", serif;
            font-size: 2rem;
            margin-top: 0.4rem;
            color: var(--ink);
        }

        .mini-stat-copy {
            color: var(--muted);
            line-height: 1.5;
            margin-top: 0.4rem;
            font-size: 0.92rem;
        }

        .section-title {
            font-family: "DM Serif Display", "Palatino Linotype", serif;
            font-size: 1.55rem;
            margin-bottom: 0.2rem;
            color: var(--ink);
        }

        .section-copy {
            color: var(--muted);
            line-height: 1.55;
            margin-bottom: 0.9rem;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.9rem;
        }

        .pill {
            border: 1px solid rgba(255, 255, 255, 0.16);
            color: rgba(249, 243, 231, 0.88);
            border-radius: 999px;
            padding: 0.42rem 0.78rem;
            font-size: 0.8rem;
            background: rgba(255, 255, 255, 0.08);
        }

        .result-card {
            background: linear-gradient(180deg, rgba(255, 250, 241, 0.96), rgba(240, 234, 223, 0.92));
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.2rem;
            box-shadow: 0 18px 45px rgba(46, 58, 41, 0.09);
        }

        .result-label {
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
        }

        .result-value {
            font-family: "DM Serif Display", "Palatino Linotype", serif;
            font-size: 2.2rem;
            margin-top: 0.3rem;
            color: var(--ink);
        }

        .probability-card {
            background: rgba(255, 252, 246, 0.82);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.1rem;
            margin-top: 1rem;
        }

        .prob-row {
            margin-bottom: 0.95rem;
        }

        .prob-meta {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.28rem;
            font-size: 0.95rem;
            color: var(--ink);
        }

        .prob-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(47, 107, 69, 0.12);
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button {
            background: linear-gradient(90deg, #2f6b45, #4c815a);
            color: #f8f4ea;
            border: none;
            border-radius: 999px;
            padding: 0.7rem 1.4rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            box-shadow: 0 14px 30px rgba(47, 107, 69, 0.18);
        }

        div[data-testid="stButton"] > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: linear-gradient(90deg, #27583a, #436f4f);
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 2.35rem;
            }
        }
        </style>
        """
    )


def detect_header_offset(csv_path: Path) -> int:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
    except OSError:
        return 0
    if first_line.lower().startswith("year,"):
        return 0
    return 1


@st.cache_data(show_spinner=False)
def load_dataset(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    skiprows = detect_header_offset(path)
    return pd.read_csv(path, skiprows=skiprows, low_memory=False)


@st.cache_data(show_spinner=False)
def build_input_metadata(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    numeric_defaults = df.median(numeric_only=True).to_dict()
    categorical_defaults: Dict[str, str] = {}
    categorical_options: Dict[str, List[str]] = {}

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            cleaned = df[col].dropna().astype(str)
            if cleaned.empty:
                categorical_defaults[col] = ""
                categorical_options[col] = []
                continue
            categorical_defaults[col] = cleaned.mode().iloc[0]
            categorical_options[col] = sorted(cleaned.unique().tolist())

    state_to_districts: Dict[str, List[str]] = {}
    if "state" in df.columns and "district" in df.columns:
        pairs = (
            df[["state", "district"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values(["state", "district"])
        )
        for state, block in pairs.groupby("state"):
            state_to_districts[state] = block["district"].tolist()

    return numeric_defaults, categorical_defaults, categorical_options, state_to_districts


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[object, Dict[str, object]]:
    if CatBoostClassifier is None:
        raise RuntimeError("catboost is not installed")

    season_model = CatBoostClassifier()
    season_model.load_model(str(SEASON_MODEL_PATH))

    crop_models: Dict[str, object] = {}
    for season in VALID_SEASONS:
        model_path = MODELS_DIR / f"cb_crop_{season.lower()}_15groups.cbm"
        if model_path.exists():
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            crop_models[season] = model

    return season_model, crop_models


def get_default_value(
    feature_name: str,
    numeric_defaults: Dict[str, float],
    categorical_defaults: Dict[str, str],
) -> object:
    if feature_name in numeric_defaults:
        value = numeric_defaults[feature_name]
        if pd.isna(value):
            return 0.0
        return float(value)
    return categorical_defaults.get(feature_name, "")


def is_categorical_feature(feature_name: str, categorical_defaults: Dict[str, str]) -> bool:
    known = {"state", "district", "season", "Season", "State Name", "Dist Name"}
    return feature_name in categorical_defaults or feature_name in known


def format_auto_filled_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return str(value)


def validate_inputs(input_values: Dict[str, object]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    def as_float(name: str) -> float | None:
        value = input_values.get(name)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    state = str(input_values.get("state", "")).strip()
    district = str(input_values.get("district", "")).strip()
    if not state:
        errors.append("State is required.")
    if not district:
        errors.append("District is required.")

    bounds = {
        "temp_avg": (-10.0, 50.0, "Average Temperature"),
        "temp_min": (-20.0, 40.0, "Minimum Temperature"),
        "temp_max": (-10.0, 60.0, "Maximum Temperature"),
        "rain_kharif": (0.0, 5000.0, "Kharif Rainfall"),
        "rain_rabi": (0.0, 2000.0, "Rabi Rainfall"),
        "rain_zaid": (0.0, 2000.0, "Zaid Rainfall"),
        "n_kg_per_ha": (0.0, 1000.0, "Nitrogen (kg/ha)"),
        "p_kg_per_ha": (0.0, 500.0, "Phosphorus (kg/ha)"),
        "k_kg_per_ha": (0.0, 500.0, "Potassium (kg/ha)"),
        "ndvi_lag1": (0.0, 1.0, "NDVI Lag"),
        "evi_lag1": (0.0, 1.0, "EVI Lag"),
        "ndvi_roll3": (0.0, 1.0, "NDVI Rolling Avg"),
        "evi_roll3": (0.0, 1.0, "EVI Rolling Avg"),
    }

    for feature, (lower, upper, label) in bounds.items():
        if feature not in input_values:
            continue
        value = as_float(feature)
        if value is None:
            errors.append(f"{label} must be a number.")
            continue
        if value < lower or value > upper:
            errors.append(f"{label} must be between {lower:g} and {upper:g}.")

    temp_min = as_float("temp_min")
    temp_avg = as_float("temp_avg")
    temp_max = as_float("temp_max")
    if None not in (temp_min, temp_avg, temp_max):
        if not (temp_min <= temp_avg <= temp_max):
            errors.append("Temperature values must follow: Minimum Temperature <= Average Temperature <= Maximum Temperature.")

    unusual_ranges = {
        "temp_avg": (5.0, 40.0, "Average Temperature"),
        "temp_min": (-5.0, 30.0, "Minimum Temperature"),
        "temp_max": (10.0, 45.0, "Maximum Temperature"),
        "rain_kharif": (20.0, 4500.0, "Kharif Rainfall"),
        "rain_rabi": (0.0, 1200.0, "Rabi Rainfall"),
        "rain_zaid": (0.0, 1000.0, "Zaid Rainfall"),
    }
    for feature, (lower, upper, label) in unusual_ranges.items():
        value = as_float(feature)
        if value is not None and (value < lower or value > upper):
            warnings.append(f"{label} is outside the usual range seen in the training data.")

    return errors, warnings


def prettify_label(feature_name: str) -> str:
    label_map = {
        "year": "Reference Year",
        "temp_avg": "Average Temperature",
        "temp_max": "Maximum Temperature",
        "temp_min": "Minimum Temperature",
        "rain_kharif": "Kharif Rainfall",
        "rain_rabi": "Rabi Rainfall",
        "rain_zaid": "Zaid Rainfall",
        "n_kg_per_ha": "Nitrogen (kg/ha)",
        "p_kg_per_ha": "Phosphorus (kg/ha)",
        "k_kg_per_ha": "Potassium (kg/ha)",
        "ndvi_lag1": "NDVI Lag",
        "evi_lag1": "EVI Lag",
        "ndvi_roll3": "NDVI Rolling Avg",
        "evi_roll3": "EVI Rolling Avg",
    }
    if feature_name in label_map:
        return label_map[feature_name]
    return feature_name.replace("_", " ").title()


def render_probability_panel(title: str, rows: List[Tuple[str, float]]) -> None:
    blocks = []
    for name, value in rows:
        width = max(2.0, min(100.0, value * 100))
        blocks.append(
            f"""
            <div class="prob-row">
                <div class="prob-meta">
                    <span>{escape(str(name))}</span>
                    <span>{value * 100:.1f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width: {width:.1f}%;"></div>
                </div>
            </div>
            """
        )

    st.html(
        f"""
        <div class="probability-card">
            <div class="section-title" style="font-size:1.2rem; margin-bottom:0.9rem;">{escape(title)}</div>
            {''.join(blocks)}
        </div>
        """
    )


def main() -> None:
    st.set_page_config(page_title="Crop Prediction", layout="wide")
    inject_styles()

    if CatBoostClassifier is None:
        st.error("Missing dependency: catboost")
        st.code("pip install -r requirements.txt", language="bash")
        st.stop()

    if not SEASON_MODEL_PATH.exists():
        st.error(f"Season model not found at: {SEASON_MODEL_PATH.name}")
        st.stop()

    if not DATASET_PATH.exists():
        st.error(f"Dataset not found at: {DATASET_PATH.name}")
        st.stop()

    try:
        df = load_dataset(str(DATASET_PATH))
    except Exception as exc:
        st.error(f"Unable to read dataset: {exc}")
        st.stop()

    try:
        season_model, crop_models = load_models()
    except Exception as exc:
        st.error(f"Unable to load models: {exc}")
        st.stop()

    (
        numeric_defaults,
        categorical_defaults,
        categorical_options,
        state_to_districts,
    ) = build_input_metadata(df)

    season_features = list(season_model.feature_names_)
    input_values: Dict[str, object] = {
        feature: get_default_value(feature, numeric_defaults, categorical_defaults)
        for feature in season_features
    }

    state_count = len(state_to_districts)
    district_count = int(df["district"].nunique()) if "district" in df.columns else 0
    crop_group_count = int(df["Crop_Group_15"].nunique()) if "Crop_Group_15" in df.columns else 0

    visible_features = [
        "state",
        "district",
        "temp_avg",
        "temp_max",
        "temp_min",
        "rain_kharif",
        "rain_rabi",
        "rain_zaid",
        "n_kg_per_ha",
        "p_kg_per_ha",
        "k_kg_per_ha",
        "ndvi_lag1",
        "evi_lag1",
        "ndvi_roll3",
        "evi_roll3",
    ]

    st.html(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Decision Support</div>
            <h1 class="hero-title">Crop prediction with a cleaner field dashboard.</h1>
            <p class="hero-copy">
                Enter a district profile, rainfall pattern, soil nutrients, and vegetation indicators.
                The app predicts the likely season first, then recommends the strongest crop group for that context.
            </p>
            <div class="pill-row">
                <span class="pill">Season classifier</span>
                <span class="pill">15 crop groups</span>
                <span class="pill">Weather + NDVI + soil</span>
            </div>
        </div>
        """
    )

    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.html(
            f"""
            <div class="mini-stat">
                <div class="mini-stat-label">States Covered</div>
                <div class="mini-stat-value">{state_count}</div>
                <div class="mini-stat-copy">Available from the current training dataset for quick district selection.</div>
            </div>
            """
        )
    with stat_col2:
        st.html(
            f"""
            <div class="mini-stat">
                <div class="mini-stat-label">District Records</div>
                <div class="mini-stat-value">{district_count}</div>
                <div class="mini-stat-copy">Location options are pulled directly from the dataset used by the trained models.</div>
            </div>
            """
        )
    with stat_col3:
        st.html(
            f"""
            <div class="mini-stat">
                <div class="mini-stat-label">Crop Groups</div>
                <div class="mini-stat-value">{crop_group_count}</div>
                <div class="mini-stat-copy">Season-specific crop models choose from the available 15-group recommendation space.</div>
            </div>
            """
        )

    input_col, result_col = st.columns([1.08, 0.92], gap="large")
    submitted = False
    season_table = pd.DataFrame()
    crop_table = pd.DataFrame()
    season_prediction = ""
    season_auto_filled: Dict[str, object] = {}
    crop_auto_filled: Dict[str, object] = {}
    validation_errors: List[str] = []
    validation_warnings: List[str] = []

    with input_col:
        st.html('<div class="section-title">Field Profile</div>')
        st.html(
            '<div class="section-copy">Keep the core values visible, and use advanced inputs only when you want finer control.</div>'
        )
        st.caption("Predictions are approximate. Some model fields may still be auto-filled from the training dataset.")

        location_cols = st.columns(2)
        state_options = sorted(state_to_districts.keys()) or categorical_options.get("state", [])
        state_default = str(input_values.get("state", ""))
        if state_options:
            if state_default not in state_options:
                state_default = state_options[0]
            state_index = state_options.index(state_default)
            selected_state = location_cols[0].selectbox(
                prettify_label("state"),
                options=state_options,
                index=state_index,
                key="selected_state",
            )
        else:
            selected_state = location_cols[0].text_input(
                prettify_label("state"),
                value=state_default,
                key="selected_state",
            )
        input_values["state"] = selected_state

        district_options = state_to_districts.get(str(selected_state), categorical_options.get("district", []))
        district_default = str(input_values.get("district", ""))
        if district_options:
            if district_default not in district_options:
                district_default = district_options[0]
            district_index = district_options.index(district_default)
            selected_district = location_cols[1].selectbox(
                prettify_label("district"),
                options=district_options,
                index=district_index,
                key="selected_district",
            )
        else:
            selected_district = location_cols[1].text_input(
                prettify_label("district"),
                value=district_default,
                key="selected_district",
            )
        input_values["district"] = selected_district

        with st.form("prediction_form"):
            location_cols = st.columns(2)
            climate_cols = st.columns(2)
            soil_cols = st.columns(2)

            feature_groups = {
                "location": ["state", "district", "year"],
                "climate": ["temp_avg", "temp_max", "temp_min", "rain_kharif", "rain_rabi", "rain_zaid"],
                "soil": ["n_kg_per_ha", "p_kg_per_ha", "k_kg_per_ha", "ndvi_lag1", "evi_lag1", "ndvi_roll3", "evi_roll3"],
            }

            st.markdown("**Location**")
            for idx, feature in enumerate(feature_groups["location"]):
                if feature not in season_features:
                    continue
                if feature in {"state", "district", "year"}:
                    continue
                target_col = location_cols[idx % 2]
                label = prettify_label(feature)
                default_number = float(input_values[feature]) if str(input_values[feature]) != "" else 0.0
                input_values[feature] = target_col.number_input(label, value=default_number, step=1.0)

            st.markdown("**Climate**")
            for idx, feature in enumerate(feature_groups["climate"]):
                if feature not in season_features:
                    continue
                target_col = climate_cols[idx % 2]
                label = prettify_label(feature)
                default_number = float(input_values[feature]) if str(input_values[feature]) != "" else 0.0
                input_values[feature] = target_col.number_input(label, value=default_number, step=0.1)

            st.markdown("**Soil and Vegetation**")
            for idx, feature in enumerate(feature_groups["soil"]):
                if feature not in season_features:
                    continue
                target_col = soil_cols[idx % 2]
                label = prettify_label(feature)
                default_number = float(input_values[feature]) if str(input_values[feature]) != "" else 0.0
                input_values[feature] = target_col.number_input(label, value=default_number, step=0.1)

            with st.expander("Advanced inputs"):
                for feature in season_features:
                    if feature in visible_features:
                        continue
                    label = prettify_label(feature)
                    if is_categorical_feature(feature, categorical_defaults):
                        options = categorical_options.get(feature, [])
                        default = str(input_values[feature])
                        if options:
                            default_index = options.index(default) if default in options else 0
                            input_values[feature] = st.selectbox(label, options=options, index=default_index)
                        else:
                            input_values[feature] = st.text_input(label, value=default)
                    else:
                        default_number = float(input_values[feature]) if str(input_values[feature]) != "" else 0.0
                        input_values[feature] = st.number_input(label, value=default_number, step=0.1)

            submitted = st.form_submit_button("Predict Crop Outlook")

        if submitted:
            validation_errors, validation_warnings = validate_inputs(input_values)
            for warning in validation_warnings:
                st.warning(warning, icon=":material/warning:")
            if validation_errors:
                for error in validation_errors:
                    st.error(error, icon=":material/error:")
                submitted = False

    if submitted:
        season_auto_filled = {
            feature: input_values[feature]
            for feature in season_features
            if feature not in visible_features
        }
        season_input = pd.DataFrame([{f: input_values[f] for f in season_features}], columns=season_features)

        season_prediction = str(season_model.predict(season_input).ravel()[0])
        season_probabilities = np.array(season_model.predict_proba(season_input))[0]
        season_classes = np.array(season_model.classes_)
        season_table = (
            pd.DataFrame({"Season": season_classes, "Probability": season_probabilities})
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )

        model_for_season = crop_models.get(season_prediction)
        if model_for_season is not None:
            crop_features = list(model_for_season.feature_names_)
            crop_input_values = {}
            for feature in crop_features:
                if feature in input_values:
                    crop_input_values[feature] = input_values[feature]
                elif feature.lower() == "season":
                    crop_input_values[feature] = season_prediction
                else:
                    default_value = get_default_value(feature, numeric_defaults, categorical_defaults)
                    crop_input_values[feature] = default_value
                    crop_auto_filled[feature] = default_value

            crop_input = pd.DataFrame([crop_input_values], columns=crop_features)
            crop_probabilities = np.array(model_for_season.predict_proba(crop_input))[0]
            crop_classes = np.array(model_for_season.classes_)
            crop_table = (
                pd.DataFrame({"Crop Group": crop_classes, "Probability": crop_probabilities})
                .sort_values("Probability", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )

    with result_col:
        st.html('<div class="section-title">Prediction</div>')
        st.html(
            '<div class="section-copy">Results appear here as early model suggestions based on your entered values and any remaining defaults.</div>'
        )

        if not submitted:
            st.html(
                """
                <div class="glass-card">
                    <div class="result-label">Ready</div>
                    <div class="result-value">Awaiting Input</div>
                    <p class="section-copy" style="margin-top:0.7rem;">
                        Use the field profile panel to the left and submit the form to see season confidence and crop-group ranking.
                    </p>
                </div>
                """
            )
            if validation_errors:
                st.info("Update the highlighted values in the form, then run the prediction again.")
            return

        top_crop = str(crop_table.iloc[0]["Crop Group"]) if not crop_table.empty else "Unavailable"
        confidence = float(season_table.iloc[0]["Probability"]) if not season_table.empty else 0.0

        top_result_cols = st.columns(2)
        with top_result_cols[0]:
            st.html(
                f"""
                <div class="result-card">
                    <div class="result-label">Suggested Season</div>
                    <div class="result-value">{escape(season_prediction)}</div>
                    <p class="section-copy" style="margin-top:0.6rem;">Estimated model confidence: {confidence * 100:.1f}%</p>
                </div>
                """
            )
        with top_result_cols[1]:
            st.html(
                f"""
                <div class="result-card">
                    <div class="result-label">Top Suggested Crop Group</div>
                    <div class="result-value">{escape(top_crop)}</div>
                    <p class="section-copy" style="margin-top:0.6rem;">This is the highest-ranked suggestion from the current model output.</p>
                </div>
                """
            )

        if season_auto_filled or crop_auto_filled:
            combined_auto_filled = season_auto_filled.copy()
            for feature, value in crop_auto_filled.items():
                combined_auto_filled.setdefault(feature, value)

            with st.expander("Auto-filled model fields"):
                st.caption("These values were not entered in the main form and were filled from training-data defaults.")
                auto_filled_table = pd.DataFrame(
                    [
                        {"Field": prettify_label(feature), "Value": format_auto_filled_value(value)}
                        for feature, value in combined_auto_filled.items()
                    ]
                )
                if not auto_filled_table.empty:
                    st.dataframe(auto_filled_table, use_container_width=True, hide_index=True)

        render_probability_panel(
            "Season confidence",
            list(zip(season_table["Season"].tolist(), season_table["Probability"].tolist())),
        )

        if crop_table.empty:
            st.warning(f"No crop model file found for season: {season_prediction}")
            return

        render_probability_panel(
            "Top suggested crop groups",
            list(zip(crop_table["Crop Group"].tolist(), crop_table["Probability"].tolist())),
        )


if __name__ == "__main__":
    main()
