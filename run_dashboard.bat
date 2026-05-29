@echo off
REM Launches the MeansRev Streamlit dashboard detached (no console window).
REM Uses pythonw so it survives terminal/session close. Started by the
REM "MeansRev Dashboard" scheduled task at logon. Logs -> dashboard.log.
cd /d "C:\Users\Cody\Claude\MeansRev"
start "" /B "C:\Python314\pythonw.exe" -m streamlit run dashboard.py --server.port 8501 --server.headless true >> "C:\Users\Cody\Claude\MeansRev\dashboard.log" 2>&1
