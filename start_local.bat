@echo off
title OS Tutor Launcher
color 0A

echo ============================================================
echo    OS Tutor - Powered by TinyLlama + rohit21789/OS-tutor
echo ============================================================
echo.

REM Check if Qdrant Docker container is running
echo [1/3] Checking Qdrant vector database...
docker ps --filter "name=qdrant" --filter "status=running" | find "qdrant" >nul 2>&1
if %errorlevel% NEQ 0 (
    echo     Qdrant not running. Starting Qdrant Docker container...
    docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
    echo     Waiting for Qdrant to start...
    timeout /t 5 /nobreak >nul
) else (
    echo     Qdrant is already running.
)

echo.
echo [2/3] Starting OS Tutor API backend (loads model + ingests textbook)...
echo     NOTE: First run downloads ~2.2 GB from HuggingFace. Please wait.
start "OS Tutor API" cmd /k "python api.py"

echo.
echo [3/3] Waiting 15 seconds for backend to initialize...
timeout /t 15 /nobreak >nul

echo.
echo Starting Streamlit UI...
start "OS Tutor UI" cmd /k "streamlit run app.py"

echo.
echo ============================================================
echo  OS Tutor is starting up!
echo  Open your browser at: http://localhost:8501
echo  API available at:     http://localhost:8000
echo ============================================================
echo.
pause
