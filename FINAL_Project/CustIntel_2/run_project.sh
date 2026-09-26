#!/usr/bin/env bash
# Run the whole CustIntel pipeline, then launch the dashboard.
# Put the 9 Olist CSV files in data/raw/ before running this.
set -e

python src/ingest.py
python src/features.py
python src/train_ml.py
python src/train_dl.py
streamlit run app/streamlit_app.py
