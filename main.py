from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "final_dataset_with_ndvi_15groups.csv"
SEASON_MODEL_PATH = BASE_DIR / "cb_season_predictor.cbm"
VALID_SEASONS = ["Kharif", "Rabi", "Zaid", "Annual"]


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
        model_path = BASE_DIR / f"cb_crop_{season.lower()}_15groups.cbm"
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
    }
    if feature_name in label_map:
        return label_map[feature_name]
    return feature_name.replace("_", " ").title()


def main() -> None:
    st.set_page_config(page_title="Crop Prediction", layout="wide")
    st.title("Crop Prediction")
    st.caption("Minimal Streamlit UI for Season -> Season-specific Crop Group prediction")

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

    st.subheader("Input")
    st.caption("Predictions are approximate. Some model fields may still be auto-filled from the training dataset.")

    location_cols = st.columns(2)
    state_options = sorted(state_to_districts.keys()) or categorical_options.get("state", [])
    state_default = str(input_values.get("state", ""))
    if state_options:
        if state_default not in state_options:
            state_default = state_options[0]
        state_index = state_options.index(state_default)
        selected_state = location_cols[0].selectbox("State", options=state_options, index=state_index, key="selected_state")
    else:
        selected_state = location_cols[0].text_input("State", value=state_default, key="selected_state")
    input_values["state"] = selected_state

    district_options = state_to_districts.get(str(selected_state), categorical_options.get("district", []))
    district_default = str(input_values.get("district", ""))
    if district_options:
        if district_default not in district_options:
            district_default = district_options[0]
        district_index = district_options.index(district_default)
        selected_district = location_cols[1].selectbox("District", options=district_options, index=district_index, key="selected_district")
    else:
        selected_district = location_cols[1].text_input("District", value=district_default, key="selected_district")
    input_values["district"] = selected_district

    with st.form("prediction_form"):
        col_left, col_right = st.columns(2)

        for idx, feature in enumerate(visible_features):
            if feature not in season_features:
                continue
            if feature in {"state", "district", "year"}:
                continue
            target_col = col_left if idx % 2 == 0 else col_right

            if is_categorical_feature(feature, categorical_defaults):
                options = categorical_options.get(feature, [])
                default = str(input_values[feature])
                if options:
                    default_index = options.index(default) if default in options else 0
                    input_values[feature] = target_col.selectbox(prettify_label(feature), options=options, index=default_index)
                else:
                    input_values[feature] = target_col.text_input(prettify_label(feature), value=default)
            else:
                default_number = float(input_values[feature]) if str(input_values[feature]) != "" else 0.0
                input_values[feature] = target_col.number_input(prettify_label(feature), value=default_number, step=0.1)

        with st.expander("Advanced (optional)"):
            for feature in season_features:
                if feature in visible_features:
                    continue
                if is_categorical_feature(feature, categorical_defaults):
                    options = categorical_options.get(feature, [])
                    default = str(input_values[feature])
                    if options:
                        default_index = options.index(default) if default in options else 0
                        input_values[feature] = st.selectbox(prettify_label(feature), options=options, index=default_index)
                    else:
                        input_values[feature] = st.text_input(prettify_label(feature), value=default)
                else:
                    default_number = float(input_values[feature]) if str(input_values[feature]) != "" else 0.0
                    input_values[feature] = st.number_input(prettify_label(feature), value=default_number, step=0.1)

        submitted = st.form_submit_button("Predict")

    validation_errors: List[str] = []
    validation_warnings: List[str] = []
    if submitted:
        validation_errors, validation_warnings = validate_inputs(input_values)
        for warning in validation_warnings:
            st.warning(warning)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return

    if not submitted:
        st.info("Set inputs and click Predict.")
        return

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

    st.subheader("Result")
    st.metric("Suggested Season", season_prediction)
    st.dataframe(season_table, use_container_width=True, hide_index=True)

    model_for_season = crop_models.get(season_prediction)
    if model_for_season is None:
        st.warning(f"No crop model file found for season: {season_prediction}")
        return

    crop_features = list(model_for_season.feature_names_)
    crop_input_values = {}
    crop_auto_filled: Dict[str, object] = {}
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

    st.metric("Top Suggested Crop Group", str(crop_table.loc[0, "Crop Group"]))
    st.dataframe(crop_table, use_container_width=True, hide_index=True)

    combined_auto_filled = season_auto_filled.copy()
    for feature, value in crop_auto_filled.items():
        combined_auto_filled.setdefault(feature, value)
    if combined_auto_filled:
        with st.expander("Auto-filled model fields"):
            st.caption("These values were not entered in the main form and were filled from training-data defaults.")
            auto_filled_table = pd.DataFrame(
                [
                    {"Field": prettify_label(feature), "Value": format_auto_filled_value(value)}
                    for feature, value in combined_auto_filled.items()
                ]
            )
            st.dataframe(auto_filled_table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
