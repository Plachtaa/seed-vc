@echo off
chcp 65001 >nul
:: Get script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Checking environment...

:: Check required files
if not exist "launcher.py" (
    echo Error: launcher.py not found!
    echo Please make sure you are running this script from the correct directory.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo Error: requirements.txt not found!
    echo Please make sure you are running this script from the correct directory.
    pause
    exit /b 1
)

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found!
    echo Please download and install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    echo Press any key to open download page...
    pause >nul
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Python version must be 3.10 or higher
    echo Current installed version:
    python --version
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    call setup.bat
    if %errorlevel% neq 0 (
        :: If setup failed, remove the venv directory if it exists
        if exist "venv" (
            echo Setup failed. Cleaning up virtual environment...
            rmdir /s /q "venv"
        )
        exit /b 1
    )
) else (
    :: Activate virtual environment
    call venv\Scripts\activate.bat
    
    :: Run main program and catch errors
    python launcher.py
    if %errorlevel% neq 0 (
        echo.
        echo Error occurred while running launcher.py
        echo Press any key to exit...
        pause >nul
        exit /b 1
    )
) 

:: Pause if any errors occurred
if %errorlevel% neq 0 (
    echo.
    echo Script ended with errors
    pause
    exit /b 1
) 