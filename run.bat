@echo off
REM Windows batch file to activate .venv and run main.py with uv

REM Set execution policy to bypass for this session to allow script execution
powershell -ExecutionPolicy Bypass -Command "& { .\.venv\Scripts\activate; uv run main.py }"

REM Alternative approach using cmd activation (uncomment if the above doesn't work)
REM call .venv\Scripts\activate.bat
REM uv run main.py

pause
