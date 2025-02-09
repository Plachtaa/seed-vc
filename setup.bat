@echo off
echo Setting up Seed-VC environment...

:: 检查NVIDIA驱动
echo Checking NVIDIA GPU driver...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: NVIDIA GPU not found or driver not installed
    echo For best performance, please install NVIDIA driver from:
    echo https://www.nvidia.com/download/index.aspx
    echo.
    echo Press any key to continue anyway...
    pause >nul
) else (
    :: 检查驱动版本
    for /f "tokens=1" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv,noheader') do set "driver_version=%%i"
    echo Current NVIDIA driver version: %driver_version%
    
    :: 比较版本号
    if %driver_version% LSS 525.60.13 (
        echo Warning: Your NVIDIA driver version is below 525.60.13
        echo For optimal performance with PyTorch 2.4.0, please update your driver
        echo Visit: https://www.nvidia.com/download/index.aspx
        echo.
        echo Press any key to continue anyway...
        pause >nul
    ) else (
        echo NVIDIA driver version is compatible
        timeout /t 2 >nul
    )
)

:: 创建虚拟环境
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    
    :: 激活环境并安装依赖
    call venv\Scripts\activate.bat
    
    :: 升级pip
    python -m pip install --upgrade pip
    
    :: 设置pip镜像源
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    
    echo Installing requirements...
    pip install -r requirements.txt
    
    echo Setup complete!
) else (
    call venv\Scripts\activate.bat
)

:: 启动GUI
python launcher.py 