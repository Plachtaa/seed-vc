@echo off
echo Checking environment...

:: 检查Python是否已安装
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
    exit
)

:: 检查Python版本
python -c "import sys; assert sys.version_info >= (3,10)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Python version must be 3.10 or higher
    echo Current installed version:
    python --version
    pause
    exit
)

:: 检查是否存在项目内的虚拟环境
if not exist "venv" (
    call setup.bat
) else (
    :: 激活项目内的虚拟环境
    call venv\Scripts\activate.bat
    python launcher.py
) 