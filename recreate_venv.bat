@echo off
echo Checking Python environment...

:: 检查seed_vc环境是否存在
if not exist "C:\Users\SeungHee\miniconda3\envs\seed_vc\python.exe" (
    echo seed_vc environment not found!
    echo Please check your conda environment
    pause
    exit
)

echo Recreating virtual environment...

:: 删除旧的虚拟环境
if exist "venv" rd /s /q "venv"

:: 使用seed_vc环境的Python创建新环境
"C:\Users\SeungHee\miniconda3\envs\seed_vc\python.exe" -m venv venv

:: 激活并安装依赖
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Virtual environment recreated!
pause 