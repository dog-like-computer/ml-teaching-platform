import os
import shutil
import zipfile
import json
from supplement.encryption import Other

class SDKCreator:
    def __init__(self):
        self.other = Other()
        self.json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'json')
        self.temp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'temp')
        
        # 确保temp目录存在
        os.makedirs(self.temp_path, exist_ok=True)
    
    def create_sdk(self, model_id, model_token, model_file, vectorization=None, normalization=None, user_id=None):
        """
        生成SDK并压缩为zip文件
        
        Args:
            model_id: 模型ID
            model_token: 模型令牌
            model_file: 模型文件路径
            vectorization: 向量化文件名称
            normalization: 归一化文件名称
            user_id: 用户ID
            
        Returns:
            dict: 包含SDK文件信息的字典
        """
        try:
            # 生成SDK ID
            sdk_id = self.other.get_random_id(user_id or 'default', 'sdk')['content']
            sdk_dir = os.path.join(self.temp_path, sdk_id)
            os.makedirs(sdk_dir, exist_ok=True)
            
            # 生成main.py
            main_py_content = f"""import json
import pickle
import numpy as np
import requests
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PyQt5.QtCore import Qt

# 模型信息
MODEL_ID = '{model_id}'
MODEL_TOKEN = '{model_token}'
# 校验token的网址
TOKEN_VALIDATION_URL = 'http://p90wpcbruk8x.guyubao.com/api/validate_token'  # 实际使用的网址
# 本地测试用的网址
LOCAL_VALIDATION_URL = 'http://127.0.0.1:8080/api/validate_token'  # 本地测试用

# 加载模型
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f'Error loading model: {e}')
        return None

# 加载向量化配置
def load_vectorization():
    try:
        with open('vectorization.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'Error loading vectorization: {e}')
        return None

# 加载归一化配置
def load_normalization():
    try:
        with open('normalization.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'Error loading normalization: {e}')
        return None

# 验证token
def validate_token(token):
    try:
        # 尝试使用本地测试地址
        response = requests.post(LOCAL_VALIDATION_URL, json={'model_id': MODEL_ID, 'token': token})
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 1:
                return True
        
        # 如果本地测试失败，尝试使用实际地址
        response = requests.post(TOKEN_VALIDATION_URL, json={'model_id': MODEL_ID, 'token': token})
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 1:
                return True
        return False
    except Exception as e:
        print(f'Error validating token: {e}')
        return False

# 预测函数
def predict(input_data):
    model = load_model()
    if model is None:
        return None
    
    try:
        # 转换输入数据为numpy数组
        input_array = np.array(input_data).reshape(1, -1)
        result = model.predict(input_array)
        return result['content']
    except Exception as e:
        print(f'Error predicting: {e}')
        return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('模型预测工具')
        self.setGeometry(100, 100, 400, 300)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # 添加标题
        title_label = QLabel('模型预测工具')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('font-size: 18px; font-weight: bold;')
        layout.addWidget(title_label)
        
        # 添加输入框
        self.input_label = QLabel('输入特征值（以逗号分隔）:')
        layout.addWidget(self.input_label)
        
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText('例如: 1.2,3.4,5.6,7.8')
        layout.addWidget(self.input_edit)
        
        # 添加预测按钮
        self.predict_button = QPushButton('预测')
        self.predict_button.clicked.connect(self.on_predict)
        layout.addWidget(self.predict_button)
        
        # 添加结果标签
        self.result_label = QLabel('预测结果:')
        layout.addWidget(self.result_label)
        
        self.result_value = QLabel('')
        self.result_value.setStyleSheet('font-size: 16px; color: blue;')
        layout.addWidget(self.result_value)
        
        # 添加token验证
        self.token_label = QLabel('输入Token:')
        layout.addWidget(self.token_label)
        
        self.token_edit = QLineEdit()
        self.token_edit.setText(MODEL_TOKEN)
        layout.addWidget(self.token_edit)
        
        self.validate_button = QPushButton('验证Token')
        self.validate_button.clicked.connect(self.on_validate)
        layout.addWidget(self.validate_button)
        
        self.token_status = QLabel('')
        layout.addWidget(self.token_status)
    
    def on_predict(self):
        # 获取输入
        input_text = self.input_edit.text()
        if not input_text:
            QMessageBox.warning(self, '警告', '请输入特征值')
            return
        
        # 解析输入
        try:
            input_data = [float(x.strip()) for x in input_text.split(',')]
        except ValueError:
            QMessageBox.warning(self, '错误', '输入格式错误，请输入数字，以逗号分隔')
            return
        
        # 验证token
        token = self.token_edit.text()
        if not validate_token(token):
            QMessageBox.warning(self, '错误', 'Token验证失败')
            return
        
        # 预测
        result = predict(input_data)
        if result is not None:
            self.result_value.setText(str(result))
        else:
            QMessageBox.warning(self, '错误', '预测失败')
    
    def on_validate(self):
        token = self.token_edit.text()
        if validate_token(token):
            self.token_status.setText('Token验证成功')
            self.token_status.setStyleSheet('color: green;')
        else:
            self.token_status.setText('Token验证失败')
            self.token_status.setStyleSheet('color: red;')

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
"""
            
            # 写入main.py
            with open(os.path.join(sdk_dir, 'main.py'), 'w', encoding='utf-8') as f:
                f.write(main_py_content)
            
            # 生成setup.py
            setup_py_content = f"""from cx_Freeze import setup, Executable
import sys

# 基础配置
base = None
if sys.platform == 'win32':
    base = 'Win32GUI'  # 使用GUI基础，不显示控制台

# 构建选项
build_exe_options = {
    'packages': ['numpy', 'requests', 'PyQt5'],
    'includes': ['pickle', 'json'],
    'include_files': ['model.pkl', 'vectorization.json', 'normalization.json'],
    'excludes': ['tkinter'],
    'optimize': 2
}

# 配置setup
setup(
    name='ModelPredictor',
    version='1.0',
    description='基于训练模型的预测工具',
    options={'build_exe': build_exe_options},
    executables=[Executable('main.py', base=base, icon=None)]
)
"""
            
            # 写入setup.py
            with open(os.path.join(sdk_dir, 'setup.py'), 'w', encoding='utf-8') as f:
                f.write(setup_py_content)
            
            # 生成打包脚本（bat文件）
            build_bat_content = f"""@echo off

REM 安装依赖
pip install cx-Freeze numpy requests PyQt5

REM 打包成exe
python setup.py build

echo 打包完成！
echo 可执行文件位于 build/exe.win-amd64-3.x/ 目录中
pause
"""
            
            # 写入build.bat
            with open(os.path.join(sdk_dir, 'build.bat'), 'w', encoding='utf-8') as f:
                f.write(build_bat_content)
            
            # 复制模型文件
            shutil.copy(model_file, os.path.join(sdk_dir, 'model.pkl'))
            print(f'Model file copied: {model_file}')
            
            # 复制vectorization.json（如果存在）
            if vectorization:
                vectorization_path = os.path.join(self.json_path, vectorization)
                if os.path.exists(vectorization_path):
                    shutil.copy(vectorization_path, os.path.join(sdk_dir, 'vectorization.json'))
                    print(f'Vectorization file copied: {vectorization_path}')
            
            # 复制normalization.json（如果存在）
            if normalization:
                normalization_path = os.path.join(self.json_path, normalization)
                if os.path.exists(normalization_path):
                    shutil.copy(normalization_path, os.path.join(sdk_dir, 'normalization.json'))
                    print(f'Normalization file copied: {normalization_path}')
            
            # 压缩成zip文件
            zip_file = os.path.join(self.temp_path, f'{sdk_id}.zip')
            try:
                with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(sdk_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, sdk_dir)
                            zf.write(file_path, arcname)
                print(f'SDK compressed successfully: {zip_file}')
            except Exception as e:
                print(f'Error compressing SDK: {str(e)}')
                # 清理临时目录
                shutil.rmtree(sdk_dir)
                return {'status': 0, 'msg': f'Error compressing SDK: {str(e)}', 'sdk_file': None}
            
            # 清理临时目录
            shutil.rmtree(sdk_dir)
            print(f'Temporary directory cleaned: {sdk_dir}')
            
            return {
                'status': 1,
                'msg': 'SDK generated successfully',
                'sdk_file': f'{sdk_id}.zip',
                'sdk_path': zip_file
            }
            
        except Exception as e:
            print(f'Error generating SDK: {str(e)}')
            return {'status': 0, 'msg': f'Error generating SDK: {str(e)}', 'sdk_file': None}
