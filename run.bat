@echo off
REM run.bat - Windows batch script for KnoCLIP-XAI Pipeline
REM Usage: run.bat --dataset-dir data\MIMIC-CXR-RRG_small --mrconso data\umls\META\MRCONSO.RRF

setlocal enabledelayedexpansion

REM Default values
set DATASET_DIR=
set MRCONSO=
set SPLIT=test
set SEED=42
set MODEL_TYPE=modern-radgraph-xl
set SKIP_INSTALL=false
set VENV_DIR=.venv
set OUTPUT_DIR=

REM Parse arguments
:parse
if "%~1"=="" goto endparse
if "%~1"=="-h" goto help
if "%~1"=="--help" goto help
if "%~1"=="-d" set DATASET_DIR=%~2& shift & shift & goto parse
if "%~1"=="--dataset-dir" set DATASET_DIR=%~2& shift & shift & goto parse
if "%~1"=="-m" set MRCONSO=%~2& shift & shift & goto parse
if "%~1"=="--mrconso" set MRCONSO=%~2& shift & shift & goto parse
if "%~1"=="-s" set SPLIT=%~2& shift & shift & goto parse
if "%~1"=="--split" set SPLIT=%~2& shift & shift & goto parse
if "%~1"=="--seed" set SEED=%~2& shift & shift & goto parse
if "%~1"=="--model-type" set MODEL_TYPE=%~2& shift & shift & goto parse
if "%~1"=="-o" set OUTPUT_DIR=%~2& shift & shift & goto parse
if "%~1"=="--output-dir" set OUTPUT_DIR=%~2& shift & shift & goto parse
if "%~1"=="--skip-install" set SKIP_INSTALL=true& shift & goto parse
echo Unknown option: %~1
goto help

:endparse

REM Display banner
echo.
echo ============================================================
echo          KnoCLIP-XAI Hybrid Knowledge Graph Pipeline       
echo ============================================================
echo.

REM Check Python installation
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3 is not installed. Please install Python 3.8 or higher.
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% found

REM Create virtual environment
if "%SKIP_INSTALL%"=="false" (
    echo [INFO] Setting up virtual environment in '%VENV_DIR%'...
    if not exist "%VENV_DIR%" (
        python -m venv %VENV_DIR%
        echo [SUCCESS] Virtual environment created
    ) else (
        echo [WARNING] Virtual environment already exists
    )
    
    REM Activate virtual environment
    echo [INFO] Activating virtual environment...
    call %VENV_DIR%\Scripts\activate.bat
    if errorlevel 1 (
        echo [ERROR] Failed to activate virtual environment
        exit /b 1
    )
    echo [SUCCESS] Virtual environment activated
    
    REM Install dependencies
    echo [INFO] Installing dependencies from requirements.txt...
    python -m pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    echo [SUCCESS] Dependencies installed
) else (
    echo [WARNING] Skipping installation (--skip-install flag set)
)

REM Validate arguments
echo [INFO] Validating configuration...

if "%DATASET_DIR%"=="" (
    echo [ERROR] Dataset directory is required
    echo [ERROR] Use --dataset-dir or download first with: python download_dataset.py --preset small
    goto help
)

if not exist "%DATASET_DIR%" (
    echo [ERROR] Dataset directory does not exist: %DATASET_DIR%
    echo [ERROR] Download first with: python download_dataset.py --preset small --output-dir %DATASET_DIR%
    exit /b 1
)

if "%MRCONSO%"=="" (
    echo [ERROR] MRCONSO path is required
    echo [ERROR] Download MRCONSO.RRF manually from UMLS website
    echo [ERROR] Or use: python src\downloads\UMLS_ontology_download.py --api-key KEY --version 2021AB
    goto help
)

if not exist "%MRCONSO%" (
    echo [ERROR] MRCONSO file does not exist: %MRCONSO%
    exit /b 1
)

echo [SUCCESS] Configuration validated

REM Build command
echo [INFO] Preparing to run pipeline...

set CMD=python main.py --dataset-dir "%DATASET_DIR%" --mrconso "%MRCONSO%" --split "%SPLIT%" --seed %SEED% --model-type "%MODEL_TYPE%"

if not "%OUTPUT_DIR%"=="" set CMD=%CMD% --output-dir "%OUTPUT_DIR%"

REM Run pipeline
echo.
echo [INFO] Starting pipeline with the following configuration:
echo   Dataset: %DATASET_DIR%
echo   MRCONSO: %MRCONSO%
echo   Split: %SPLIT%
echo   Model: %MODEL_TYPE%
echo   Seed: %SEED%
if not "%OUTPUT_DIR%"=="" echo   Output: %OUTPUT_DIR%
echo.

%CMD%

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed with errors
    exit /b 1
) else (
    echo.
    echo [SUCCESS] Pipeline completed successfully!
    echo.
)

goto end

:help
echo Usage: run.bat [OPTIONS]
echo.
echo Run the KnoCLIP-XAI Hybrid Knowledge Graph Pipeline.
echo.
echo PREREQUISITES:
echo     1. Download dataset first:    python download_dataset.py --preset small
echo     2. Download MRCONSO.RRF:      Manual download from UMLS website recommended
echo                                   (or use UMLS_ontology_download.py for 2021AB only)
echo.
echo OPTIONS:
echo     -h, --help              Display this help message
echo     -d, --dataset-dir DIR   Path to dataset directory (required)
echo     -m, --mrconso FILE      Path to MRCONSO.RRF file (required)
echo     -s, --split SPLIT       Dataset split to use (default: test)
echo     --seed SEED             Random seed (default: 42)
echo     --model-type TYPE       RadGraph model type (default: modern-radgraph-xl)
echo     -o, --output-dir DIR    Output directory for results
echo     --skip-install          Skip dependency installation
echo.
echo EXAMPLES:
echo     REM Run with downloaded dataset and MRCONSO
echo     run.bat --dataset-dir data\MIMIC-CXR-RRG_small --mrconso data\umls\META\MRCONSO.RRF
echo.
echo     REM Specify output directory
echo     run.bat -d data\MIMIC-CXR-RRG_small -m data\umls\META\MRCONSO.RRF -o outputs\my_run
echo.
exit /b 0

:end
endlocal
