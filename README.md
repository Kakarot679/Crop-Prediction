# Crop Recommendation System

Machine learning project for hierarchical crop recommendation. The app predicts the agricultural season first, then uses the matching CatBoost crop model to recommend a crop group.

## Project Structure

```text
finalyr-project/
├── code/                 # Streamlit application code
│   ├── app.py            # Main polished Streamlit app
│   └── main.py           # Simpler Streamlit app version
├── data/
│   ├── processed/        # Final dataset used by the app
│   └── raw/              # Original/source datasets
├── models/               # Trained CatBoost model files
├── notebooks/            # Training and experimentation notebooks
├── reports/
│   └── figures/          # Generated plots and result images
├── docs/                 # Project documentation PDFs
├── requirements.txt      # Python dependencies
└── README.md
```

## Runtime Files

The Streamlit app expects:

- `data/processed/final_dataset_with_ndvi_15groups.csv`
- `models/cb_season_predictor.cbm`
- `models/cb_crop_kharif_15groups.cbm`
- `models/cb_crop_rabi_15groups.cbm`
- `models/cb_crop_annual_15groups.cbm`

## Run The App

From the project root:

```bash
pip install -r requirements.txt
streamlit run code/app.py
```

For the simpler interface:

```bash
streamlit run code/main.py
```

## Model Flow

1. Load the processed crop dataset.
2. Predict the likely season with `cb_season_predictor.cbm`.
3. Load the season-specific crop model.
4. Return ranked crop-group suggestions with confidence scores.
