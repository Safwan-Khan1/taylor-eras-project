@echo off
cd /d "%~dp0"
echo Starting Taylor Swift Eras Web Portal...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
