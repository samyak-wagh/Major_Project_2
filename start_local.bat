@echo off
title OS Tutor Launcher
color 0A

echo ============================================================
echo    OS Tutor - Powered by Qwen2.5-1.5B + qwen-os-tutor-lora
echo ============================================================
echo.

set "PYTHON_EXE=D:\New folder\Samyak\venv\genai\Scripts\python.exe"
if not exist "%PYTHON_EXE%" (
    if exist "venv\Scripts\python.exe" (
        set "PYTHON_EXE=venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

REM Check if Qdrant Docker container is running
echo [1/3] Checking Qdrant vector database...
docker ps --filter "name=qdrant" --filter "status=running" | find "qdrant" >nul 2>&1
if %errorlevel% NEQ 0 (
    echo     Checking if Docker is available to start Qdrant...
    docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant >nul 2>&1
    if %errorlevel% NEQ 0 (
        echo     [INFO] Docker not active; assuming external or cloud Qdrant.
    ) else (
        echo     Waiting for Qdrant to start...
        timeout /t 5 /nobreak >nul
    )
) else (
    echo     Qdrant is already running.
)

echo.
echo [2/3] Starting OS Tutor API backend (loads model + ingests textbook)...
echo     NOTE: Loading Qwen2.5-1.5B + LoRA. Please wait.
start "OS Tutor API" cmd /k ""%PYTHON_EXE%" api.py"

echo.
echo [3/3] Waiting 15 seconds for backend to initialize...
timeout /t 15 /nobreak >nul

echo.
echo Starting Streamlit UI...
start "OS Tutor UI" cmd /k ""%PYTHON_EXE%" -m streamlit run app.py"

echo.
echo ============================================================
echo  OS Tutor is starting up!
echo  Open your browser at: http://localhost:8501
echo  API available at:     http://localhost:8001
echo ============================================================
echo.
pause
