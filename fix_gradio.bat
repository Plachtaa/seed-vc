@echo off
echo Installing required packages...

:: 激活虚拟环境
call venv\Scripts\activate.bat

:: 安装typing包
pip install typing==3.7.4.3

:: 安装gradio
pip install gradio==4.44.0

echo Installation complete!
pause 