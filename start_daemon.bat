@echo off
TITLE Ultragent Daemon (The Watcher)
ECHO ===================================================
ECHO   ULTRAGENT LEVEL 5: DAEMON MODE ACTIVATED
ECHO ===================================================
ECHO.
ECHO [1/3] Checking environment...
IF EXIST "venv\Scripts\activate.bat" (
    CALL venv\Scripts\activate.bat
    ECHO    -> Virtual environment activated.
) ELSE (
    ECHO    -> Using system Python (venv not found).
)

ECHO [2/3] Installing dependencies...
pip install watchdog > NUL 2>&1
ECHO    -> Watchdog ready.

ECHO [3/3] Starting The Watcher...
ECHO.
ECHO    [INFO] Monitoring directory for changes...
ECHO    [INFO] Quality Metrics will trigger on SAVE.
ECHO.
python daemon.py
PAUSE
