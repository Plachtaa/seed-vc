import sys
import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QFileDialog, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seed-VC Launcher")
        self.setFixedSize(600, 400)
        
        # 主窗口部件
        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 标题
        title = QLabel("Seed-VC 启动器")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; margin: 20px;")
        layout.addWidget(title)
        
        # 模式选择
        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "语音转换 (app_vc.py)", 
            "歌声转换 (app_svc.py)", 
            "实时转换 (real-time-gui.py)"
        ])
        layout.addWidget(QLabel("选择转换模式:"))
        layout.addWidget(self.mode_selector)
        
        # 模型选择
        layout.addWidget(QLabel("模型文件 (可选):"))
        self.model_path = ""
        self.model_label = QLabel("未选择")
        model_button = QPushButton("选择模型文件")
        model_button.clicked.connect(self.select_model)
        layout.addWidget(self.model_label)
        layout.addWidget(model_button)
        
        # 配置选择
        layout.addWidget(QLabel("配置文件 (可选):"))
        self.config_path = ""
        self.config_label = QLabel("未选择")
        config_button = QPushButton("选择配置文件")
        config_button.clicked.connect(self.select_config)
        layout.addWidget(self.config_label)
        layout.addWidget(config_button)
        
        # 启动按钮
        launch_button = QPushButton("启动")
        launch_button.setStyleSheet("font-size: 18px; padding: 10px;")
        launch_button.clicked.connect(self.launch)
        layout.addWidget(launch_button)
        
        # 添加说明文本
        note = QLabel("注意：如果不选择模型和配置文件，将使用默认设置")
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)

    def select_model(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Model Files (*.pth)")
        if file_name:
            self.model_path = file_name
            self.model_label.setText(os.path.basename(file_name))

    def select_config(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择配置文件", "", "Config Files (*.yml *.yaml)")
        if file_name:
            self.config_path = file_name
            self.config_label.setText(os.path.basename(file_name))

    def launch(self):
        mode = self.mode_selector.currentText()
        script = mode.split("(")[1].strip(")").strip()
        
        # 构建路径
        project_root = os.getcwd()
        venv_path = os.path.join(project_root, "venv")
        
        # 构建激活命令和运行命令
        if os.name == 'nt':  # Windows
            activate_cmd = os.path.join(venv_path, "Scripts", "activate.bat")
            cmd = [
                "cmd.exe", "/c",
                f"{activate_cmd} && python {script}"
            ]
        else:  # Linux/Mac
            activate_cmd = os.path.join(venv_path, "bin", "activate")
            cmd = [
                "bash", "-c",
                f"source {activate_cmd} && python {script}"
            ]
            
        # 添加参数
        if self.model_path:
            cmd[-1] += f" --checkpoint-path {self.model_path}"
        if self.config_path:
            cmd[-1] += f" --config-path {self.config_path}"
        if "svc" in script or "vc" in script:
            cmd[-1] += " --fp16 True"
            # 添加 GPU 设备选择
            cmd[-1] += " --gpu 0"  # 默认使用第一个 GPU
            
        try:
            # 使用 shell 命令运行，不重定向输出
            process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=os.environ.copy(),
                shell=True
            )
            
            # 等待一小段时间确保进程启动
            import time
            time.sleep(2)
            
            # 检查进程是否还在运行
            if process.poll() is None:
                QMessageBox.information(self, "启动成功", f"已启动 {script}")
            else:
                QMessageBox.critical(self, "错误", f"启动失败")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec_()) 