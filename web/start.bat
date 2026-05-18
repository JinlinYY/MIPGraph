@echo off
chcp 65001 >nul
title MIPGraph — Ionic Liquid Property Predictor
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d "%~dp0"

echo.
echo  ============================================
echo   MIPGraph  Ionic Liquid Property Predictor
echo  ============================================
echo.

:: ── Step 1: Find Python ──────────────────────────────────────────────────────
set PYTHON=

python --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=python & goto :check_deps )

for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "%USERPROFILE%\Anaconda3\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
  "%USERPROFILE%\Miniconda3\python.exe"
  "E:\Anaconda3\python.exe"
  "C:\ProgramData\Anaconda3\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
) do (
  if exist %%P ( set PYTHON=%%P & goto :check_deps )
)

:: Python not found at all
echo  [ERROR] Python 3.9+ was not found on this computer.
echo.
echo  Please install Python or Anaconda first:
echo.
echo    Python:   https://www.python.org/downloads/
echo    Anaconda: https://www.anaconda.com/download
echo.
echo  During installation, check the box:
echo    [x] Add Python to PATH
echo.
pause
exit /b 1

:: ── Step 2: Check / install dependencies ────────────────────────────────────
:check_deps
echo  Python found: %PYTHON%
echo.

%PYTHON% -c "import fastapi, uvicorn, torch, torch_geometric, rdkit, numpy" >nul 2>&1
if not errorlevel 1 goto :launch

echo  [INFO] Some dependencies are missing. Installing now...
echo  (This only runs once and may take a few minutes.)
echo.
%PYTHON% -m pip install -r requirements_app.txt
if errorlevel 1 (
  echo.
  echo  [ERROR] Dependency installation failed.
  echo  Please run the following command manually in your terminal:
  echo.
  echo    pip install -r requirements_app.txt
  echo.
  pause
  exit /b 1
)
echo.
echo  [OK] Dependencies installed successfully.
echo.

:: ── Step 3: Launch ──────────────────────────────────────────────────────────
:launch
echo  Starting MIPGraph server...
echo  Browser will open automatically in 4 seconds.
echo  URL: http://127.0.0.1:8765
echo.
echo  Press Ctrl+C to stop the server.
echo.

start /b cmd /c "timeout /t 4 /nobreak >nul && start http://127.0.0.1:8765"
%PYTHON% scripts/serve_screening_ui.py

pause
