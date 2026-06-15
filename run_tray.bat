@echo off
REM Launches the MeansRev system-tray app (orange "MR" icon) with no console
REM window. Uses Python 3.14's pythonw so it inherits the shared quantcore
REM package. Double-click this file, or add a shortcut to it in your Startup
REM folder to launch the tray at logon.
cd /d "%~dp0"
start "" /B "C:\Python314\pythonw.exe" "%~dp0tray.py"
