@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul
:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Setting up environment...

:: Check NVIDIA GPU driver
echo Checking NVIDIA GPU driver...
nvidia-smi >nul 2>nul
if !errorlevel! neq 0 (
    echo.
    echo Warning: NVIDIA GPU not found or driver not installed
    echo For best performance, please install NVIDIA driver from:
    echo https://www.nvidia.com/download/index.aspx
    echo.
    echo Press any key to continue anyway...
    pause
) else (
    :: Check driver version
    for /f "tokens=1" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do set "driver_version=%%i"
    if defined driver_version (
        echo NVIDIA GPU found, driver version: !driver_version!
        echo Driver check passed
        timeout /t 2 >nul
    ) else (
        echo Error: Could not determine driver version
        pause
    )
)


:: Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment
        pause
        exit /b 1
    )
    
    :: Activate environment and install dependencies
    call venv\Scripts\activate.bat
    if %errorlevel% neq 0 (
        echo Failed to activate virtual environment
        pause
        exit /b 1
    )
    
    :: Upgrade pip
    python -m pip install --upgrade pip
    if %errorlevel% neq 0 (
        echo Failed to upgrade pip
        pause
        exit /b 1
    )
    
    :: Set pip mirror
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    
    echo Installing requirements...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install requirements
        pause
        exit /b 1
    )
    
    echo Setup complete!
) else (
    call venv\Scripts\activate.bat
    if %errorlevel% neq 0 (
        echo Failed to activate virtual environment
        pause
        exit /b 1
    )
)

:: Launch GUI and catch errors
python launcher.py
if %errorlevel% neq 0 (
    echo.
    echo Error occurred while running launcher.py
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: Pause if any errors occurred
if %errorlevel% neq 0 (
    echo.
    echo Script ended with errors
    pause
    exit /b 1
) 