@echo off
REM Run the whole CustIntel pipeline, then launch the dashboard.
REM Put the 9 Olist CSV files in data\raw\ before running this.

python src\ingest.py
if errorlevel 1 goto :error

python src\features.py
if errorlevel 1 goto :error

python src\train_ml.py
if errorlevel 1 goto :error

python src\train_dl.py
if errorlevel 1 goto :error

streamlit run app\streamlit_app.py
goto :eof

:error
echo.
echo Pipeline step failed -- see the message above.
pause
