@echo off
setlocal enabledelayedexpansion

echo ====================================
echo Starting SuperBizAgent Services
echo ====================================
echo.

REM Check for uv package manager
echo [1/6] Checking package manager...
where uv >nul 2>&1
if errorlevel 1 (
    echo [INFO] uv not found, will use pip
    echo [TIP] Install uv for faster setup: pip install uv
    set USE_UV=0
) else (
    echo [SUCCESS] uv package manager detected
    set USE_UV=1
)
echo.

REM Ensure Python version
echo [2/6] Configuring Python version...
if exist .python-version (
    set /p PYTHON_VERSION=<.python-version
    echo [INFO] Current configured version: !PYTHON_VERSION!
    
    REM Check if 3.10
    echo !PYTHON_VERSION! | findstr /C:"3.10" >nul
    if not errorlevel 1 (
        echo [WARNING] Python 3.13 is recommended, updating config...
        echo 3.13> .python-version
        echo [SUCCESS] Updated to Python 3.13
    )
) else (
    echo [INFO] Creating .python-version file...
    echo 3.13> .python-version
)
echo.

REM Create or sync virtual environment
echo [3/6] Creating or Syncing virtual environment...
if exist .venv\Scripts\python.exe (
    echo [INFO] Virtual environment exists, checking updates...
    
    if "!USE_UV!"=="1" (
        uv sync 2>nul
        if errorlevel 1 (
            echo [WARNING] uv sync failed, using pip...
            .venv\Scripts\python.exe -m pip install -e . -q
        ) else (
            echo [SUCCESS] uv sync complete
        )
    ) else (
        echo [INFO] Updating dependencies via pip...
        .venv\Scripts\python.exe -m pip install -e . -q
    )
) else (
    echo [INFO] Creating new virtual environment...
    
    if "!USE_UV!"=="1" (
        echo [INFO] Trying uv sync...
        uv sync 2>nul
        if not errorlevel 1 (
            echo [SUCCESS] Environment created with uv
            goto :venv_created
        )
        echo [WARNING] uv sync failed, falling back to venv...
    )
    
    echo [INFO] Creating via python -m venv...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    
    echo [INFO] Installing dependencies - this may take a few minutes...
    .venv\Scripts\python.exe -m pip install --upgrade pip -q
    .venv\Scripts\python.exe -m pip install -e . -q
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment ready
)

:venv_created
echo [SUCCESS] Virtual environment ready
echo.

set PYTHON_CMD=.venv\Scripts\python.exe

REM Start Milvus
echo [4/6] Starting Milvus database...
docker ps --format "{{.Names}}" | findstr "milvus-standalone" >nul 2>&1
if not errorlevel 1 (
    echo [INFO] Milvus container already running
) else (
    docker compose -f vector-database.yml up -d
    if errorlevel 1 (
        echo [ERROR] Docker failed to start. Ensure Docker Desktop is running.
        pause
        exit /b 1
    )
    echo [INFO] Waiting for Milvus database to be ready - 10s...
    timeout /t 10 /nobreak >nul
)
echo [SUCCESS] Milvus ready
echo.

REM Start MCP Servers
echo [5/6] Starting CLS MCP Server...
start "CLS MCP Server" /min %PYTHON_CMD% mcp_servers/cls_server.py
timeout /t 2 /nobreak >nul

echo [6/6] Starting Monitor MCP Server...
start "Monitor MCP Server" /min %PYTHON_CMD% mcp_servers/monitor_server.py
timeout /t 2 /nobreak >nul

REM Start FastAPI
echo [7/8] Starting FastAPI Service...
start "SuperBizAgent API" %PYTHON_CMD% -m uvicorn app.main:app --host 0.0.0.0 --port 9900
echo [INFO] Waiting for service to start - 15s...
timeout /t 15 /nobreak >nul

REM Health check and file upload
echo.
echo [INFO] Checking service health...
curl -s http://localhost:9900/api/health >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Service might still be starting, please wait a moment
) else (
    echo [SUCCESS] FastAPI service running
    echo.
    
    echo [8/8] Uploading documents to vector database...
    for %%f in (aiops-docs\*.md) do (
        echo   Uploading: %%~nxf
        curl -s -X POST http://localhost:9900/api/upload -F "file=@%%f" >nul 2>&1
    )
    echo [SUCCESS] Document upload complete
)

echo.
echo ====================================
echo Startup Complete!
echo ====================================
echo Web UI: http://localhost:9900
echo API Docs: http://localhost:9900/docs
echo.
echo To stop the services, run stop-windows.bat
echo ====================================
pause
