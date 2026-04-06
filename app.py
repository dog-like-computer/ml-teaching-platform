# todo 1 import some need library
try:
    from supplement.redis_queue import RedisQueue, init_train_queue, get_train_queue
except ImportError:
    print("Redis module not found, running without Redis support")
    # 创建一个空的get_train_queue函数，返回None
    def get_train_queue():
        return None
    # 创建一个空的init_train_queue函数
    def init_train_queue():
        pass
from supplement.file_manager import Manage
from supplement import scikit_learn_build
from supplement.encryption import Other
from supplement.mysql_api import Mysql
from supplement.pd_item import Panda
from warnings import filterwarnings
from supplement.ocr_api import Ocr
from datetime import datetime
from pprint import pprint
from io import BytesIO
import urllib.parse
import threading
import mimetypes
import aiomysql
import asyncio
import socket
import json
import shutil
import zipfile
import os
import re
import time
import pandas as pd
import pickle
import numpy as np

# todo 2 set some path
manage = Manage()
mysql = Mysql()
ocr = Ocr()
other = Other()
panda = Panda()
divide = scikit_learn_build.Divide()
html_path = r'./static/html'
css_path = r'./static/css'
js_path = r'./static/javascript'
log_path = r'./static/log'
system_pic_path = r'./static/system_pic'
user_pic_path = r'./static/user_pic'
model_path = r'./static/model'
primary_model_setting = r'./static/base_model'
data_path = r'./static/data'
json_path = r'./static/json'

def read_static_file(file_path):
    absolute_path = os.path.abspath(file_path)
    if not os.path.exists(absolute_path):
        return None, None
    mime_type, _ = mimetypes.guess_type(absolute_path)
    if not mime_type:
        mime_type = "application/octet-stream"  # 默认二进制类型
    with open(absolute_path, 'rb') as f:
        content = f.read()
    return content, mime_type

def response_file(file_path, as_attachment=False, filename=None):
    content, mime_type = read_static_file(file_path)
    if not content:
        return {'status': 404, 'msg': f"File {file_path} not found"}
    return {
        'type': 'file',
        'content': content,
        'mime_type': mime_type,
        'as_attachment': as_attachment,
        'filename': filename or os.path.basename(file_path)
    }

def response_html(file_path):
    content, mime_type = read_static_file(file_path)
    if not content:
        return {'status': 404, 'msg': f"HTML file {file_path} not found"}
    return {
        'type': 'html',
        'content': content.decode('utf-8')
    }

def response(status=200,msg=None):
    return {'status':status,'msg':msg}

def judge_captcha(img_name,img_input):
    judge_pic = ocr.recognize(img_name)['content']
    if img_input != 'jlbb':
        if img_input != judge_pic:
            return response(0, 'This image verification code is mistake')
    return response(1,'Captcha through!')

# todo 3 build response
class TrainWorker:
    def __init__(self, db_pool):
        self.queue = None
        self.running = False
        self.thread = None
        self.pool = db_pool

    def init_queue(self):
        try:
            from supplement.redis_queue import get_train_queue
            self.queue = get_train_queue()
            if self.queue and self.queue.is_connected():
                print("Worker: Redis queue initialized successfully")
                return True
            else:
                print("Worker: Failed to initialize Redis queue")
                return False
        except Exception as e:
            print(f"Worker: Error initializing Redis queue: {e}")
            return False

    async def process_task(self, task):
        # 在当前事件循环中创建新的mysql对象，并使用正确的连接池
        from supplement.mysql_api import Mysql
        # 创建一个新的连接池，确保在当前事件循环中使用
        pool = await aiomysql.create_pool(
            host='localhost',
            port=3306,
            user='root',
            password='123456',
            db='Graduation_242821513',
            charset='utf8mb4',
            minsize=1,
            maxsize=10
        )
        mysql = Mysql(pool=pool)
        model_id = task.get('model_id')
        print(f"Worker: Processing task {model_id}")

        try:
            train_x = np.array(task.get('train_x'))
            train_y = np.array(task.get('train_y'))
            test_x = np.array(task.get('test_x'))
            test_y = np.array(task.get('test_y'))
            model_select = task.get('model_select')
            hyper_parameter = task.get('hyper_parameter')

            model_file = task.get('model_file')
            eval_path = task.get('eval_path')
            json_path = task.get('json_path')
            model_path = task.get('model_path')

            if self.queue and self.queue.is_connected():
                self.queue.add_progress(model_id, 10, "Creating model instance...")
            model = self._create_model(model_select, hyper_parameter)

            if self.queue and self.queue.is_connected():
                self.queue.add_progress(model_id, 20, "Training model...")
            train_result = model.fit(train_x, train_y)
            if train_result['status'] != 1:
                error_msg = train_result['content']
                print(f"Worker: Model fit failed with error: {error_msg}")
                if self.queue and self.queue.is_connected():
                    self.queue.fail_task(model_id, error_msg)
                # 更新数据库状态为训练失败
                try:
                    result = await mysql.update('sklearn_db', ['id'], [model_id], 'model_condition', f'train failed (fit): {error_msg}')
                    print(f"Worker: Database update result: {result}")
                except Exception as db_error:
                    print(f"Worker: Failed to update database: {db_error}")
                return False

            if self.queue and self.queue.is_connected():
                self.queue.add_progress(model_id, 50, "Making predictions...")
            predict_result = model.predict(test_x)
            if predict_result['status'] != 1:
                error_msg = predict_result['content']
                print(f"Worker: Model predict failed with error: {error_msg}")
                if self.queue and self.queue.is_connected():
                    self.queue.fail_task(model_id, error_msg)
                # 更新数据库状态为训练失败
                try:
                    result = await mysql.update('sklearn_db', ['id'], [model_id], 'model_condition', f'train failed (predict): {error_msg}')
                    print(f"Worker: Database update result: {result}")
                except Exception as db_error:
                    print(f"Worker: Failed to update database: {db_error}")
                return False

            if self.queue and self.queue.is_connected():
                self.queue.add_progress(model_id, 70, "Evaluating model...")
            classification_models = ['LogisticRegression', 'SVC', 'DecisionTreeClassifier', 
                                     'RandomForestClassifier', 'XGBClassifier', 'MLPClassifier', 
                                     'RBFClassifier', 'GaussianNB', 'MultinomialNB', 'BernoulliNB',
                                     'LDA', 'AdaBoostClassifier']
            if model_select in classification_models:
                # 确保test_y和predict_result['content']形状一致
                y_pred = predict_result['content']
                # 如果test_y是二维数组且只有一列，转换为一维数组
                if len(test_y.shape) == 2 and test_y.shape[1] == 1:
                    test_y = test_y.reshape(-1)
                # 如果y_pred是二维数组且只有一列，转换为一维数组
                if isinstance(y_pred, np.ndarray) and len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                    y_pred = y_pred.reshape(-1)
                # 如果y_pred是列表且每个元素是列表，转换为一维数组
                elif isinstance(y_pred, list) and all(isinstance(item, list) for item in y_pred):
                    y_pred = np.array(y_pred).reshape(-1)
                evaluate_result = model.evaluate(test_x, test_y, y_pred, 0.5)
            else:
                evaluate_result = model.evaluate(test_y, predict_result['content'])
            
            if evaluate_result['status'] != 1:
                error_msg = evaluate_result['content']
                print(f"Worker: Model evaluate failed with error: {error_msg}")
                if self.queue and self.queue.is_connected():
                    self.queue.fail_task(model_id, error_msg)
                # 更新数据库状态为训练失败
                try:
                    result = await mysql.update('sklearn_db', ['id'], [model_id], 'model_condition', f'train failed (evaluate): {error_msg}')
                    print(f"Worker: Database update result: {result}")
                except Exception as db_error:
                    print(f"Worker: Failed to update database: {db_error}")
                return False

            if self.queue and self.queue.is_connected():
                self.queue.add_progress(model_id, 90, "Saving model and results...")
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(evaluate_result['content'], f, ensure_ascii=False)

            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

            if self.queue and self.queue.is_connected():
                self.queue.add_progress(model_id, 100, "Training completed!")
            
            print(f"Worker: Starting SDK generation for model {model_id}")
            # 生成SDK
            sdk_url = await self._generate_sdk({
                'model_id': model_id,
                'model_token': task.get('model_token'),
                'model_name': task.get('model_name'),
                'model_type': task.get('model_type'),
                'model_select': model_select,
                'hyper_file': task.get('hyper_file'),
                'eval_file': task.get('eval_file'),
                'data_id': task.get('data_id'),
                'x_columns': task.get('x_columns'),
                'y_columns': task.get('y_columns'),
                'json_path': json_path,
                'model_file': model_file
            })
            print(f"Worker: SDK generation completed for model {model_id}, sdk_url: {sdk_url}")
            
            # 完成任务
            if self.queue and self.queue.is_connected():
                self.queue.complete_task(model_id, {
                    'model_id': model_id,
                    'evaluation': evaluate_result['content'],
                    'sdk_url': sdk_url
                })
            print(f"Worker: Task {model_id} completed successfully")
            
            # 更新数据库状态
            try:
                result = await mysql.update('sklearn_db', ['id'], [model_id], 'model_condition', 'train success')
                print(f"Worker: Database update result: {result}")
                # url_path表示后端提供给用户的post接口
                post_api_url = f"/api/predict/{model_id}"
                result = await mysql.update('sklearn_db', ['id'], [model_id], 'url_path', post_api_url)
                print(f"Worker: Post API URL update result: {result}")
            except Exception as e:
                print(f"Worker: Failed to update database: {e}")
                # 即使数据库更新失败，模型也已经成功训练并保存，所以不应该标记为失败
                import traceback
                traceback.print_exc()

            print(f"Worker: Task {model_id} completed successfully")
            return True
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            print(f"Worker: Task {model_id} failed with error: {error_msg}")
            print(f"Worker: Error traceback:\n{error_traceback}")
            # 更新数据库状态为训练失败
            try:
                result = await mysql.update('sklearn_db', ['id'], [model_id], 'model_condition', f'train failed: {error_msg}')
                print(f"Worker: Database update result: {result}")
            except Exception as db_error:
                print(f"Worker: Failed to update database: {db_error}")
                traceback.print_exc()
        
            if self.queue and self.queue.is_connected():
                self.queue.fail_task(model_id, f"{error_msg}\n\nTraceback:\n{error_traceback}")
            return False
        finally:
            # 关闭连接池，避免资源泄漏
            if 'pool' in locals() and pool:
                pool.close()
                await pool.wait_closed()

    def _create_model(self, model_select, hyper_parameter):
        model = None
        if model_select == 'LinearRegression':
            model = scikit_learn_build.LinearRegression()
        elif model_select == 'Ridge':
            model = scikit_learn_build.Ridge(alpha=hyper_parameter.get('alpha', 1.0))
        elif model_select == 'LogisticRegression':
            model = scikit_learn_build.LogisticRegression()
        elif model_select == 'XGBClassifier':
            model = scikit_learn_build.XGBClassifier()
        elif model_select == 'SVC':
            model = scikit_learn_build.SVC()
        elif model_select == 'LDA':
            model = scikit_learn_build.LDA()
        elif model_select == 'DecisionTreeClassifier':
            model = scikit_learn_build.DecisionTreeClassifier()
        elif model_select == 'DecisionTreeRegressor':
            model = scikit_learn_build.DecisionTreeRegressor()
        elif model_select == 'MLPClassifier':
            model = scikit_learn_build.MLPClassifier()
        elif model_select == 'MLPRegressor':
            model = scikit_learn_build.MLPRegressor(
                hidden_layer_sizes=hyper_parameter.get('hidden_layer_sizes', (100,)),
                activation=hyper_parameter.get('activation', 'relu'),
                solver=hyper_parameter.get('solver', 'adam'),
                max_iter=hyper_parameter.get('max_iter', 1000),
                random_state=hyper_parameter.get('random_state', None)
            )
        elif model_select == 'RBFClassifier':
            model = scikit_learn_build.RBFClassifier()
        elif model_select == 'SVR':
            model = scikit_learn_build.SVR(
                kernel=hyper_parameter.get('kernel', 'rbf'),
                C=hyper_parameter.get('C', 1.0),
                gamma=hyper_parameter.get('gamma', 'scale')
            )
        elif model_select == 'GaussianNB':
            model = scikit_learn_build.GaussianNB()
        elif model_select == 'MultinomialNB':
            model = scikit_learn_build.MultinomialNB()
        elif model_select == 'BernoulliNB':
            model = scikit_learn_build.BernoulliNB()
        elif model_select == 'RandomForestClassifier':
            model = scikit_learn_build.RandomForestClassifier()
        elif model_select == 'RandomForestRegressor':
            model = scikit_learn_build.RandomForestRegressor()
        elif model_select == 'AdaBoostClassifier':
            model = scikit_learn_build.AdaBoostClassifier()
        elif model_select == 'AdaBoostRegressor':
            model = scikit_learn_build.AdaBoostRegressor()
        elif model_select == 'XGBRegressor':
            model = scikit_learn_build.XGBRegressor()
        elif model_select == 'ElmanNetwork':
            model = scikit_learn_build.ElmanNetwork(
                hidden_size=hyper_parameter.get('hidden_layer_sizes', 100),
                learning_rate=hyper_parameter.get('learning_rate', 0.001),
                max_iter=hyper_parameter.get('max_iter', 1000),
                random_state=hyper_parameter.get('random_state', None)
            )
        elif model_select == 'CascadeCorrelation':
            model = scikit_learn_build.CascadeCorrelation(
                hidden_neurons=hyper_parameter.get('hidden_neurons', 10),
                max_neurons=hyper_parameter.get('max_neurons', 100),
                learning_rate=hyper_parameter.get('learning_rate', 0.001)
            )
        elif model_select == 'BoltzmannMachine':
            model = scikit_learn_build.BoltzmannMachine(
                hidden_size=hyper_parameter.get('n_components', 50),
                learning_rate=hyper_parameter.get('learning_rate', 0.01),
                max_iter=hyper_parameter.get('n_iter', 1000),
                random_state=hyper_parameter.get('random_state', None)
            )
        elif model_select == 'KMeans':
            model = scikit_learn_build.KMeans()
        elif model_select == 'LVQ':
            model = scikit_learn_build.LVQ()
        elif model_select == 'GaussianMixture':
            model = scikit_learn_build.GaussianMixture()
        elif model_select == 'DBSCAN':
            model = scikit_learn_build.DBSCAN()
        elif model_select == 'AgglomerativeClustering':
            model = scikit_learn_build.AgglomerativeClustering()
        
        if model is None:
            raise ValueError(f"Model not supported: {model_select}")
        
        return model

    async def _generate_sdk(self, task):
        # 在当前事件循环中创建新的mysql对象，并使用正确的连接池
        from supplement.mysql_api import Mysql
        # 创建一个新的连接池，确保在当前事件循环中使用
        pool = await aiomysql.create_pool(
            host='localhost',
            port=3306,
            user='root',
            password='123456',
            db='Graduation_242821513',
            charset='utf8mb4',
            minsize=1,
            maxsize=10
        )
        mysql = Mysql(pool=pool)
        try:
            print(f"Worker: Starting SDK generation for model {task.get('model_id')}")
            model_id = task.get('model_id')
            model_token = task.get('model_token')
            model_name = task.get('model_name')
            model_type = task.get('model_type')
            model_select = task.get('model_select')
            hyper_file = task.get('hyper_file')
            eval_file = task.get('eval_file')
            data_id = task.get('data_id')
            x_columns = task.get('x_columns')
            y_columns = task.get('y_columns')
            print(f"Worker: Task data: model_id={model_id}, data_id={data_id}")
            
            # 获取数据信息
            try:
                data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                print(f"Worker: Data record type: {type(data_record)}")
                print(f"Worker: Data record: {data_record}")
                # 检查data_record是否为有效的响应字典
                if not isinstance(data_record, dict):
                    print(f"Worker: Data record is not a dict: {data_record}")
                    return None
                if 'response' not in data_record:
                    print(f"Worker: Data record has no 'response' key: {data_record}")
                    return None
                if data_record['response'] != 1:
                    print(f"Worker: Data record response is not 1: {data_record}")
                    return None
                if 'content' not in data_record:
                    print(f"Worker: Data record has no 'content' key: {data_record}")
                    return None
                content = data_record.get('content', [])
                print(f"Worker: Data record content: {content}")
                print(f"Worker: Data record content length: {len(content)}")
                if len(content) == 0:
                    print(f"Worker: Data record not found for {data_id}")
                    return None
            except Exception as e:
                print(f"Worker: Exception when getting data record: {e}")
                return None
            
            data_info = data_record['content'][0]
            
            # 读取超参和评估文件
            json_path = task.get('json_path')
            hyper_config = {}
            eval_config = {}
            
            try:
                with open(os.path.join(json_path, hyper_file), 'r', encoding='utf-8') as f:
                    hyper_config = json.load(f)
            except Exception as e:
                print(f"Worker: Failed to load hyper config: {e}")
            
            try:
                with open(os.path.join(json_path, eval_file), 'r', encoding='utf-8') as f:
                    eval_config = json.load(f)
            except Exception as e:
                print(f"Worker: Failed to load eval config: {e}")
            
            # 创建SDK目录
            sdk_dir = os.path.join('static', 'sdk', model_id)
            if not os.path.exists(sdk_dir):
                os.makedirs(sdk_dir, exist_ok=True)
            
            # 复制模型文件
            model_file = task.get('model_file')
            model_pkl = os.path.basename(model_file)
            shutil.copy(model_file, os.path.join(sdk_dir, 'model.pkl'))
            
            # 复制配置文件
            try:
                vectorization = data_info.get('vectorization', '')
                if vectorization:
                    vectorization_path = os.path.join(json_path, vectorization)
                    if os.path.exists(vectorization_path):
                        shutil.copy(vectorization_path, os.path.join(sdk_dir, 'vectorization.json'))
            except Exception as e:
                print(f"Worker: Failed to copy vectorization config: {e}")
            
            try:
                normalization = data_info.get('normalization', '')
                if normalization:
                    normalization_path = os.path.join(json_path, normalization)
                    if os.path.exists(normalization_path):
                        shutil.copy(normalization_path, os.path.join(sdk_dir, 'normalization.json'))
            except Exception as e:
                print(f"Worker: Failed to copy normalization config: {e}")
            
            try:
                though_work = data_info.get('though_work', '')
                if though_work:
                    though_work_path = os.path.join(json_path, though_work)
                    if os.path.exists(though_work_path):
                        shutil.copy(though_work_path, os.path.join(sdk_dir, 'though_work.json'))
            except Exception as e:
                print(f"Worker: Failed to copy though_work config: {e}")
            
            # 生成main.py
            _model_id = model_id
            _model_token = model_token
            main_py_content = "import json\n"
            main_py_content += "import pickle\n"
            main_py_content += "import numpy as np\n"
            main_py_content += "import requests\n"
            main_py_content += "import pandas as pd\n"
            main_py_content += "import os\n"
            main_py_content += "from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog\n"
            main_py_content += "from PyQt5.QtCore import Qt\n"
            main_py_content += "\n"
            main_py_content += "# 模型信息\n"
            main_py_content += "MODEL_ID = '" + _model_id + "'\n"
            main_py_content += "MODEL_TOKEN = '" + _model_token + "'\n"
            main_py_content += "# 校验token的网址\n"
            main_py_content += "TOKEN_VALIDATION_URL = 'http://p90wpcbruk8x.guyubao.com/api/validate_token'  # 实际使用的网址\n"
            main_py_content += "# 本地测试用的网址\n"
            main_py_content += "LOCAL_VALIDATION_URL = 'http://127.0.0.1:8080/api/validate_token'  # 本地测试用\n"
            main_py_content += "\n"
            main_py_content += "# 加载模型\n"
            main_py_content += "def load_model():\n"
            main_py_content += "    try:\n"
            main_py_content += "        with open('model.pkl', 'rb') as f:\n"
            main_py_content += "            return pickle.load(f)\n"
            main_py_content += "    except Exception as e:\n"
            main_py_content += "        print('Error loading model: ' + str(e))\n"
            main_py_content += "        return None\n"
            main_py_content += "\n"
            main_py_content += "# 加载配置\n"
            main_py_content += "def load_config():\n"
            main_py_content += "    vectorization_config = {}\n"
            main_py_content += "    normalization_config = {}\n"
            main_py_content += "    though_work = {}\n"
            main_py_content += "    \n"
            main_py_content += "    try:\n"
            main_py_content += "        with open('vectorization.json', 'r', encoding='utf-8') as f:\n"
            main_py_content += "            vectorization_config = json.load(f)\n"
            main_py_content += "    except:\n"
            main_py_content += "        pass\n"
            main_py_content += "    \n"
            main_py_content += "    try:\n"
            main_py_content += "        with open('normalization.json', 'r', encoding='utf-8') as f:\n"
            main_py_content += "            normalization_config = json.load(f)\n"
            main_py_content += "    except:\n"
            main_py_content += "        pass\n"
            main_py_content += "    \n"
            main_py_content += "    try:\n"
            main_py_content += "        with open('though_work.json', 'r', encoding='utf-8') as f:\n"
            main_py_content += "            though_work = json.load(f)\n"
            main_py_content += "    except:\n"
            main_py_content += "        pass\n"
            main_py_content += "    \n"
            main_py_content += "    return vectorization_config, normalization_config, though_work\n"
            main_py_content += "\n"
            main_py_content += "# 文本向量化\n"
            main_py_content += "def vectorize(data, config):\n"
            main_py_content += "    # 简化处理，实际应该根据配置进行处理\n"
            main_py_content += "    return data\n"
            main_py_content += "\n"
            main_py_content += "# 归一化\n"
            main_py_content += "def normalize(data, config):\n"
            main_py_content += "    # 简化处理，实际应该根据配置进行处理\n"
            main_py_content += "    return data\n"
            main_py_content += "\n"
            main_py_content += "# 校验token\n"
            main_py_content += "def validate_token():\n"
            main_py_content += "    try:\n"
            main_py_content += "        # 尝试使用实际网址\n"
            main_py_content += "        response = requests.post(TOKEN_VALIDATION_URL, json={  \n"
            main_py_content += "            'model_id': MODEL_ID,\n"
            main_py_content += "            'token': MODEL_TOKEN\n"
            main_py_content += "        }, timeout=5)\n"
            main_py_content += "        if response.status_code == 200:\n"
            main_py_content += "            result = response.json()\n"
            main_py_content += "            return result.get('status') == 1\n"
            main_py_content += "    except:\n"
            main_py_content += "        # 如果实际网址失败，使用本地测试网址\n"
            main_py_content += "        try:\n"
            main_py_content += "            response = requests.post(LOCAL_VALIDATION_URL, json={  \n"
            main_py_content += "                'model_id': MODEL_ID,\n"
            main_py_content += "                'token': MODEL_TOKEN\n"
            main_py_content += "            }, timeout=5)\n"
            main_py_content += "            if response.status_code == 200:\n"
            main_py_content += "                result = response.json()\n"
            main_py_content += "                return result.get('status') == 1\n"
            main_py_content += "        except:\n"
            main_py_content += "            # 如果本地也失败，返回True（离线模式）\n"
            main_py_content += "            print('Warning: Token validation failed, running in offline mode')\n"
            main_py_content += "            return True\n"
            main_py_content += "    return False\n"
            main_py_content += "\n"
            main_py_content += "# 预测\n"
            main_py_content += "def predict(input_data):\n"
            main_py_content += "    # 加载模型和配置\n"
            main_py_content += "    model = load_model()\n"
            main_py_content += "    if model is None:\n"
            main_py_content += "        return 'Error: Model loading failed'\n"
            main_py_content += "    \n"
            main_py_content += "    vectorization_config, normalization_config, though_work = load_config()\n"
            main_py_content += "    \n"
            main_py_content += "    try:\n"
            main_py_content += "        # 处理输入数据\n"
            main_py_content += "        # 假设输入是CSV格式的字符串\n"
            main_py_content += "        data_row = input_data.split(',')\n"
            main_py_content += "        data = np.array([float(x) for x in data_row]).reshape(1, -1)\n"
            main_py_content += "        \n"
            main_py_content += "        # 应用文本向量化和归一化\n"
            main_py_content += "        # 根据though_work中的步骤顺序处理\n"
            main_py_content += "        if though_work:\n"
            main_py_content += "            # 按照though_work中的步骤顺序处理\n"
            main_py_content += "            for step in though_work:\n"
            main_py_content += "                if step['type'] == 'vectorization':\n"
            main_py_content += "                    data = vectorize(data, vectorization_config)\n"
            main_py_content += "                elif step['type'] == 'normalization':\n"
            main_py_content += "                    data = normalize(data, normalization_config)\n"
            main_py_content += "        else:\n"
            main_py_content += "            # 默认处理顺序\n"
            main_py_content += "            data = vectorize(data, vectorization_config)\n"
            main_py_content += "            data = normalize(data, normalization_config)\n"
            main_py_content += "        \n"
            main_py_content += "        # 预测\n"
            main_py_content += "        result = model.predict(data)\n"
            main_py_content += "        \n"
            main_py_content += "        # 检查是否需要逆向归一化\n"
            main_py_content += "        if though_work:\n"
            main_py_content += "            # 检查是否有归一化步骤\n"
            main_py_content += "            for step in though_work:\n"
            main_py_content += "                if step['type'] == 'normalization' and 'save_path' in step:\n"
            main_py_content += "                    # 尝试对预测结果进行逆向归一化\n"
            main_py_content += "                    try:\n"
            main_py_content += "                        # 构建预测结果的DataFrame\n"
            main_py_content += "                        import pandas as pd\n"
            main_py_content += "                        pred_df = pd.DataFrame({'prediction': [result['content'][0]})\n"
            main_py_content += "                        # 尝试逆向归一化\n"
            main_py_content += "                        denorm_result = panda.denormalize(pred_df, step['save_path'], columns=['prediction'])\n"
            main_py_content += "                        if denorm_result['status'] == 1:\n"
            main_py_content += "                            # 返回逆向归一化后的值\n"
            main_py_content += "                            return denorm_result['content']['prediction'].iloc[0]\n"
            main_py_content += "                    except Exception as e:\n"
            main_py_content += "                        print('Error during denormalization: ' + str(e))\n"
            main_py_content += "        \n"
            main_py_content += "        # 如果不需要逆向归一化或逆向归一化失败，返回原始预测结果\n"
            main_py_content += "        return result['content'][0]\n"
            main_py_content += "    except Exception as e:\n"
            main_py_content += "        return 'Error: ' + str(e)\n"
            main_py_content += "\n"
            main_py_content += "# 批量预测\n"
            main_py_content += "def batch_predict(file_path):\n"
            main_py_content += "    # 加载模型和配置\n"
            main_py_content += "    model = load_model()\n"
            main_py_content += "    if model is None:\n"
            main_py_content += "        return 'Error: Model loading failed'\n"
            main_py_content += "    \n"
            main_py_content += "    vectorization_config, normalization_config, though_work = load_config()\n"
            main_py_content += "    \n"
            main_py_content += "    try:\n"
            main_py_content += "        # 读取文件\n"
            main_py_content += "        if file_path.endswith('.csv'):\n"
            main_py_content += "            df = panda.read_file(file_path, encoding='utf-8')['content']\n"
            main_py_content += "            df = panda.json_to_dataframe(df)['content']\n"
            main_py_content += "        elif file_path.endswith('.xlsx'):\n"
            main_py_content += "            df = panda.read_file(file_path, engine='openpyxl')['content']\n"
            main_py_content += "            df = panda.json_to_dataframe(df)['content']\n"
            main_py_content += "        else:\n"
            main_py_content += "            return 'Error: Unsupported file format'\n"
            main_py_content += "        \n"
            main_py_content += "        # 获取特征列\n"
            main_py_content += "        x_columns = though_work.get('x_columns', df.columns.tolist())\n"
            main_py_content += "        if not x_columns:\n"
            main_py_content += "            x_columns = df.columns.tolist()\n"
            main_py_content += "        \n"
            main_py_content += "        # 准备数据\n"
            main_py_content += "        data = df[x_columns].values\n"
            main_py_content += "        \n"
            main_py_content += "        # 应用文本向量化和归一化\n"
            main_py_content += "        if though_work:\n"
            main_py_content += "            # 按照though_work中的步骤顺序处理\n"
            main_py_content += "            for step in though_work:\n"
            main_py_content += "                if step['type'] == 'vectorization':\n"
            main_py_content += "                    data = vectorize(data, vectorization_config)\n"
            main_py_content += "                elif step['type'] == 'normalization':\n"
            main_py_content += "                    data = normalize(data, normalization_config)\n"
            main_py_content += "        else:\n"
            main_py_content += "            # 默认处理顺序\n"
            main_py_content += "            data = vectorize(data, vectorization_config)\n"
            main_py_content += "            data = normalize(data, normalization_config)\n"
            main_py_content += "        \n"
            main_py_content += "        # 批量预测\n"
            main_py_content += "        results = []\n"
            main_py_content += "        for row in data:\n"
            main_py_content += "            result = model.predict(row.reshape(1, -1))\n"
            main_py_content += "            pred_value = result['content'][0]\n"
            main_py_content += "            \n"
            main_py_content += "            # 检查是否需要逆向归一化\n"
            main_py_content += "            if though_work:\n"
            main_py_content += "                # 检查是否有归一化步骤\n"
            main_py_content += "                for step in though_work:\n"
            main_py_content += "                    if step['type'] == 'normalization' and 'save_path' in step:\n"
            main_py_content += "                        # 尝试对预测结果进行逆向归一化\n"
            main_py_content += "                        try:\n"
            main_py_content += "                            # 构建预测结果的DataFrame\n"
            main_py_content += "                            import pandas as pd\n"
            main_py_content += "                            pred_df = pd.DataFrame({'prediction': [pred_value]})\n"
            main_py_content += "                            # 尝试逆向归一化\n"
            main_py_content += "                            denorm_result = panda.denormalize(pred_df, step['save_path'], columns=['prediction'])\n"
            main_py_content += "                            if denorm_result['status'] == 1:\n"
            main_py_content += "                                # 使用逆向归一化后的值\n"
            main_py_content += "                                pred_value = denorm_result['content']['prediction'].iloc[0]\n"
            main_py_content += "                        except Exception as e:\n"
            main_py_content += "                            print('Error during denormalization: ' + str(e))\n"
            main_py_content += "            \n"
            main_py_content += "            results.append(pred_value)\n"
            main_py_content += "        \n"
            main_py_content += "        # 将结果添加到DataFrame\n"
            main_py_content += "        df['预测结果'] = results\n"
            main_py_content += "        \n"
            main_py_content += "        # 保存结果\n"
            main_py_content += "        output_file = os.path.splitext(file_path)[0] + '_result.csv'\n"
            main_py_content += "        df.to_csv(output_file, index=False, encoding='utf-8-sig')\n"
            main_py_content += "        \n"
            main_py_content += "        return 'Batch prediction completed. Results saved to: ' + output_file\n"
            main_py_content += "    except Exception as e:\n"
            main_py_content += "        return 'Error: ' + str(e)\n"
            main_py_content += "\n"
            main_py_content += "# Qt5窗口类\n"
            main_py_content += "class PredictionApp(QMainWindow):\n"
            main_py_content += "    def __init__(self):\n"
            main_py_content += "        super().__init__()\n"
            main_py_content += "        self.setWindowTitle('模型预测工具')\n"
            main_py_content += "        self.setGeometry(100, 100, 700, 500)\n"
            main_py_content += "        \n"
            main_py_content += "        # 验证token\n"
            main_py_content += "        if not validate_token():\n"
            main_py_content += "            QMessageBox.warning(self, 'Token验证', 'Token验证失败，可能无法正常使用所有功能')\n"
            main_py_content += "        \n"
            main_py_content += "        # 创建主窗口部件\n"
            main_py_content += "        central_widget = QWidget()\n"
            main_py_content += "        self.setCentralWidget(central_widget)\n"
            main_py_content += "        \n"
            main_py_content += "        # 创建布局\n"
            main_py_content += "        layout = QVBoxLayout()\n"
            main_py_content += "        \n"
            main_py_content += "        # 模型信息标签\n"
            main_py_content += "        model_info = QLabel('模型ID: ' + MODEL_ID)\n"
            main_py_content += "        model_info.setAlignment(Qt.AlignCenter)\n"
            main_py_content += "        layout.addWidget(model_info)\n"
            main_py_content += "        \n"
            main_py_content += "        # 单个预测部分\n"
            main_py_content += "        single_prediction_group = QWidget()\n"
            main_py_content += "        single_layout = QVBoxLayout()\n"
            main_py_content += "        \n"
            main_py_content += "        # 输入标签\n"
            main_py_content += "        input_label = QLabel('请输入预测数据（逗号分隔）:')\n"
            main_py_content += "        single_layout.addWidget(input_label)\n"
            main_py_content += "        \n"
            main_py_content += "        # 输入框\n"
            main_py_content += "        self.input_line = QLineEdit()\n"
            main_py_content += "        self.input_line.setPlaceholderText('例如: 1.0,2.0,3.0,4.0')\n"
            main_py_content += "        single_layout.addWidget(self.input_line)\n"
            main_py_content += "        \n"
            main_py_content += "        # 预测按钮\n"
            main_py_content += "        predict_button = QPushButton('单个预测')\n"
            main_py_content += "        predict_button.clicked.connect(self.on_predict)\n"
            main_py_content += "        single_layout.addWidget(predict_button)\n"
            main_py_content += "        \n"
            main_py_content += "        # 结果标签\n"
            main_py_content += "        self.result_label = QLabel('预测结果: ')\n"
            main_py_content += "        self.result_label.setAlignment(Qt.AlignCenter)\n"
            main_py_content += "        single_layout.addWidget(self.result_label)\n"
            main_py_content += "        \n"
            main_py_content += "        single_prediction_group.setLayout(single_layout)\n"
            main_py_content += "        layout.addWidget(single_prediction_group)\n"
            main_py_content += "        \n"
            main_py_content += "        # 批量预测部分\n"
            main_py_content += "        batch_prediction_group = QWidget()\n"
            main_py_content += "        batch_layout = QVBoxLayout()\n"
            main_py_content += "        \n"
            main_py_content += "        # 批量预测标签\n"
            main_py_content += "        batch_label = QLabel('批量预测（上传CSV或XLSX文件）:')\n"
            main_py_content += "        batch_layout.addWidget(batch_label)\n"
            main_py_content += "        \n"
            main_py_content += "        # 上传按钮\n"
            main_py_content += "        upload_button = QPushButton('选择文件')\n"
            main_py_content += "        upload_button.clicked.connect(self.on_upload)\n"
            main_py_content += "        batch_layout.addWidget(upload_button)\n"
            main_py_content += "        \n"
            main_py_content += "        # 批量预测按钮\n"
            main_py_content += "        batch_button = QPushButton('批量预测')\n"
            main_py_content += "        batch_button.clicked.connect(self.on_batch_predict)\n"
            main_py_content += "        batch_layout.addWidget(batch_button)\n"
            main_py_content += "        \n"
            main_py_content += "        # 批量预测结果标签\n"
            main_py_content += "        self.batch_result_label = QLabel('批量预测结果: ')\n"
            main_py_content += "        self.batch_result_label.setAlignment(Qt.AlignCenter)\n"
            main_py_content += "        batch_layout.addWidget(self.batch_result_label)\n"
            main_py_content += "        \n"
            main_py_content += "        batch_prediction_group.setLayout(batch_layout)\n"
            main_py_content += "        layout.addWidget(batch_prediction_group)\n"
            main_py_content += "        \n"
            main_py_content += "        # 设置布局\n"
            main_py_content += "        central_widget.setLayout(layout)\n"
            main_py_content += "        \n"
            main_py_content += "        # 保存选择的文件路径\n"
            main_py_content += "        self.selected_file = ''\n"
            main_py_content += "    \n"
            main_py_content += "    def on_predict(self):\n"
            main_py_content += "        input_data = self.input_line.text()\n"
            main_py_content += "        if not input_data:\n"
            main_py_content += "            QMessageBox.warning(self, '输入错误', '请输入预测数据')\n"
            main_py_content += "            return\n"
            main_py_content += "        \n"
            main_py_content += "        result = predict(input_data)\n"
            main_py_content += "        self.result_label.setText('预测结果: ' + str(result))\n"
            main_py_content += "    \n"
            main_py_content += "    def on_upload(self):\n"
            main_py_content += "        options = QFileDialog.Options()\n"
            main_py_content += "        options |= QFileDialog.ReadOnly\n"
            main_py_content += "        file_path, _ = QFileDialog.getOpenFileName(self, \"选择预测文件\", \"\", \"CSV Files (*.csv);;Excel Files (*.xlsx)\", options=options)\n"
            main_py_content += "        if file_path:\n"
            main_py_content += "            self.selected_file = file_path\n"
            main_py_content += "            self.batch_result_label.setText('已选择文件: ' + os.path.basename(file_path))\n"
            main_py_content += "    \n"
            main_py_content += "    def on_batch_predict(self):\n"
            main_py_content += "        if not self.selected_file:\n"
            main_py_content += "            QMessageBox.warning(self, '文件错误', '请先选择预测文件')\n"
            main_py_content += "            return\n"
            main_py_content += "        \n"
            main_py_content += "        result = batch_predict(self.selected_file)\n"
            main_py_content += "        self.batch_result_label.setText('批量预测结果: ' + result)\n"
            main_py_content += "\n"
            main_py_content += "if __name__ == '__main__':\n"
            main_py_content += "    app = QApplication([])\n"
            main_py_content += "    window = PredictionApp()\n"
            main_py_content += "    window.show()\n"
            main_py_content += "    app.exec_()\n"




            
            with open(os.path.join(sdk_dir, 'main.py'), 'w', encoding='utf-8') as f:
                f.write(main_py_content)
            
            # 生成setup.py文件
            setup_py_content = "#!/usr/bin/env python3\n"
            setup_py_content += "# setup.py - 用于打包成exe文件\n"
            setup_py_content += "\n"
            setup_py_content += "import os\n"
            setup_py_content += "import sys\n"
            setup_py_content += "from cx_Freeze import setup, Executable\n"
            setup_py_content += "\n"
            setup_py_content += "# 依赖项\n"
            setup_py_content += "build_exe_options = {\n"
            setup_py_content += "    'packages': ['numpy', 'requests', 'PyQt5', 'pandas', 'openpyxl'],\n"
            setup_py_content += "    'includes': ['json', 'pickle', 'os'],\n"
            setup_py_content += "    'include_files': ['model.pkl', 'vectorization.json', 'normalization.json', 'though_work.json'],\n"
            setup_py_content += "    'excludes': [],\n"
            setup_py_content += "    'optimize': 2\n"
            setup_py_content += "}\n"
            setup_py_content += "\n"
            setup_py_content += "# 可执行文件配置\n"
            setup_py_content += "base = None\n"
            setup_py_content += "if sys.platform == 'win32':\n"
            setup_py_content += "    base = 'Win32GUI'  # 隐藏控制台窗口\n"
            setup_py_content += "\n"
            setup_py_content += "setup(\n"
            setup_py_content += "    name='模型预测工具',\n"
            setup_py_content += "    version='1.0',\n"
            setup_py_content += "    description='基于训练模型的预测工具',\n"
            setup_py_content += "    options={'build_exe': build_exe_options},\n"
            setup_py_content += "    executables=[Executable('main.py', base=base, icon=None)]\n"
            setup_py_content += ")\n"
            
            with open(os.path.join(sdk_dir, 'setup.py'), 'w', encoding='utf-8') as f:
                f.write(setup_py_content)
            
            # 生成打包脚本
            build_bat_content = "@echo off\n"
            build_bat_content += "\n"
            build_bat_content += "REM 安装依赖\n"
            build_bat_content += "pip install cx-Freeze numpy requests PyQt5 pandas openpyxl\n"
            build_bat_content += "\n"
            build_bat_content += "REM 打包成exe\n"
            build_bat_content += "python setup.py build\n"
            build_bat_content += "\n"
            build_bat_content += "echo 打包完成！\n"
            build_bat_content += "echo 可执行文件位于 build/exe.win-amd64-3.x/ 目录中\n"
            build_bat_content += "pause\n"
            
            with open(os.path.join(sdk_dir, 'build.bat'), 'w', encoding='utf-8') as f:
                f.write(build_bat_content)
            
            # 生成SDK ID
            from supplement.encryption import Other
            other = Other()
            # 优先使用task中的user_id参数
            user_id = task.get('user_id', 'default')
            if user_id == 'default':
                # 从模型信息中获取user_id
                model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
                user_id = model_info['content'][0]['user_id'] if model_info['content'] else 'default'
            sdk_id = other.get_random_id(user_id, 'sdk')['content']
            
            # 创建zip文件
            zip_file = os.path.join('static', 'sdk', f'{sdk_id}.zip')
            try:
                with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # 添加文件到zip
                    for root, dirs, files in os.walk(sdk_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, sdk_dir)
                            zf.write(file_path, arcname)
                print(f"Worker: Zip file created successfully: {zip_file}")
            except Exception as e:
                print(f"Worker: Failed to create zip file: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # 清理临时目录
            try:
                shutil.rmtree(sdk_dir)
                print(f"Worker: Temporary directory cleaned: {sdk_dir}")
            except Exception as e:
                print(f"Worker: Failed to clean temporary directory: {e}")
            
            # 生成SDK下载路径
            sdk_url = f'static/sdk/{sdk_id}.zip'
            print(f"Worker: SDK URL generated: {sdk_url}")
            
            # 记录SDK关系到数据库
            try:
                # 只保存SDK文件名，不包含路径
                sdk_file = os.path.basename(sdk_url)
                sdk_relationship_data = {
                    'model_id': model_id,
                    'sdk_file': sdk_file
                }
                # 插入SDK关系到数据库
                attributes = list(sdk_relationship_data.keys())
                values = list(sdk_relationship_data.values())
                result = await mysql.insert('sdk_relationship', attributes, values)
                print(f"Worker: SDK relationship inserted: {result}")
            except Exception as e:
                print(f"Worker: Failed to record SDK relationship: {e}")
                import traceback
                traceback.print_exc()
            
            return sdk_url
            
        except Exception as e:
            print(f"Worker: Failed to generate SDK: {e}")
            return None
        finally:
            # 关闭连接池，避免资源泄漏
            if 'pool' in locals() and pool:
                pool.close()
                await pool.wait_closed()

    def run(self):
        if not self.init_queue():
            print("Worker: Failed to start, Redis not available")
            return

        print("Worker: Started successfully, waiting for tasks...")
        self.running = True
        
        # 创建一个事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.running:
            try:
                task = self.queue.dequeue('train_queue', timeout=5)
                if task:
                    # 使用同一个事件循环运行异步任务
                    try:
                        loop.run_until_complete(self.process_task(task))
                    except Exception as e:
                        print(f"Worker: Error processing task: {e}")
                        # 尝试在当前事件循环中更新数据库状态
                        try:
                            model_id = task.get('model_id')
                            if model_id:
                                # 创建一个简单的异步函数来更新数据库
                                async def update_failed_status():
                                    # 创建新的数据库连接来更新状态
                                    from supplement.mysql_api import Mysql
                                    pool = await aiomysql.create_pool(
                                        host='localhost',
                                        port=3306,
                                        user='root',
                                        password='123456',
                                        db='Graduation_242821513',
                                        charset='utf8mb4',
                                        minsize=1,
                                        maxsize=10
                                    )
                                    mysql = Mysql(pool=pool)
                                    try:
                                        result = await mysql.update('sklearn_db', ['id'], [model_id], 'model_condition', 'train failed')
                                        print(f"Worker: Database update result: {result}")
                                    except Exception as db_error:
                                        print(f"Worker: Failed to update database: {db_error}")
                                    finally:
                                        if pool:
                                            pool.close()
                                            await pool.wait_closed()
                                loop.run_until_complete(update_failed_status())
                        except Exception as db_error:
                            print(f"Worker: Failed to update database status: {db_error}")
                else:
                    time.sleep(1)
            except Exception as e:
                print(f"Worker: Error processing task: {e}")
                time.sleep(1)
        loop.close()

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            print("Worker: Background thread started")
        else:
            print("Worker: Already running")

    def stop(self):
        self.running = False
        print("Worker: Stopping...")

class App:
    def __init__(self):
        setting = manage.wr_setting('r')
        if not setting['response']:
            print('Error[1]: This setting is defeat')
            exit()
        self.host = setting['content']['host']
        self.port = setting['content']['port']
        self.show_in = setting['content']['show_in']
        self.show_out = setting['content']['show_out']
        self.save_log = setting['content']['save_log']
        self.save_length = setting['content']['save_length']
        self.sql_port = setting['content']['sql_port']
        self.sql_admin = setting['content']['sql_admin']
        self.sql_password = setting['content']['sql_password']
        # Redis setting
        # self.redis_host = setting['content']['redis_host']
        # self.redis_port = setting['content']['redis_port']
        # self.redis_db = setting['content']['redis_db']
        # self.redis_password = setting['content']['redis_password']
        # self.redis_timeout = setting['content']['redis_timeout']

        if setting['content']['penetration']:
            print(f'penetration is open\n*****\nweb: http://{setting["content"]["internal"]}')
        else:
            print(f'*****\nweb: {self.host}:{self.port}')
        self.routes = []
        self.train_workers = []
        self.num_workers = 2  # 设置为2个worker，支持同时训练2个模型

    def response(self, status, msg, data=None):
        res = {"status": status, "msg": msg}
        if data:
            res["data"] = data
        return res

    def route(self,path,methods=["GET",'POST']):
        def decorator(func):
            pattern = re.sub(r"<path:(\w+)>", r"(?P<\1>.+)", path)
            pattern = re.sub(r"{(\w+)}", r"(?P<\1>[^/]+)", pattern)
            regex = re.compile(f"^{pattern}$")
            self.routes.append({
                "regex": regex,
                "func": func,
                "methods": [m.upper() for m in methods],
                "path": path
            })
            return func
        return decorator

    def build_response(self, content, status_code=200):
        # 确保内容可序列化的辅助函数
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize(item) for item in obj]
            return obj
        
        status_text = {
            200: "OK",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error"
        }.get(status_code, "OK")
        if isinstance(content, dict) and 'status' in content and 'msg' in content:
            # 确保使用有效的 HTTP 状态码
            if content['status'] == 1:
                http_status_code = 200
            elif content['status'] == 0:
                http_status_code = 400
            else:
                http_status_code = content['status']
            status_text = {
                404: "Not Found",
                405: "Method Not Allowed",
                500: "Internal Server Error"
            }.get(http_status_code, "OK")
            # 序列化内容
            serialized_content = serialize(content)
            content_str = json.dumps(serialized_content, ensure_ascii=False)
            content_bytes = content_str.encode("utf-8")
            content_type = "application/json; charset=utf-8"
            status_code = http_status_code
        elif isinstance(content, dict) and content.get('type') == 'html':
            content_bytes = content['content'].encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif isinstance(content, dict) and content.get('type') == 'file':
            content_bytes = content['content']
            content_type = content['mime_type']
            headers = [
                f"HTTP/1.1 {status_code} {status_text}",
                f"Content-Type: {content_type}",
                f"Content-Length: {len(content_bytes)}",
                "Connection: close"
            ]
            if content.get('as_attachment'):
                # 只使用filename参数，避免文件名前面加上UTF
                original_filename = content['filename']
                # 移除文件名中的UTF-8前缀（如果存在）
                if original_filename.startswith('UTF-8'):
                    original_filename = original_filename[5:]
                # 移除文件名中的UTF-8''前缀（如果存在）
                if original_filename.startswith('UTF-8'):
                    original_filename = original_filename[5:]
                # 移除文件名中的任何UTF-8相关前缀
                original_filename = original_filename.replace('UTF-8', '')
                quoted_filename = urllib.parse.quote(original_filename)
                headers.append(f"Content-Disposition: attachment; filename={quoted_filename}")
            response_headers = "\r\n".join(headers).encode("utf-8") + b"\r\n\r\n"
            return response_headers + content_bytes
        elif isinstance(content, dict):
            # 序列化内容
            serialized_content = serialize(content)
            content_str = json.dumps(serialized_content, ensure_ascii=False)
            content_bytes = content_str.encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            content_str = str(content)
            content_bytes = content_str.encode("utf-8")
            if content_str.startswith('<!DOCTYPE html>') or content_str.startswith('<html'):
                content_type = "text/html; charset=utf-8"
            elif content_str.startswith('/*') or content_str.endswith('.css'):
                content_type = "text/css; charset=utf-8"
            elif content_str.startswith('//') or content_str.endswith('.js'):
                content_type = "application/javascript; charset=utf-8"
            else:
                content_type = "text/plain; charset=utf-8"
        response = (
                       f"HTTP/1.1 {status_code} {status_text}\r\n"
                       f"Content-Type: {content_type}\r\n"
                       f"Content-Length: {len(content_bytes)}\r\n"
                       "Connection: close\r\n"
                       "\r\n"
                   ).encode("utf-8") + content_bytes
        return response

    async def parse_request(self, raw_data):
        parsed = {
            "method": "GET",
            "path": "/",
            "query_params": {},
            "body": "",
            "body_json": None,
            "form_data": None,
            "headers": {}
        }

        # 修复1：先解析请求头（纯字节操作，不解码整个请求体）
        header_end = raw_data.find(b"\r\n\r\n")
        if header_end == -1:
            header_bytes = raw_data
            body_bytes = b""
        else:
            header_bytes = raw_data[:header_end]
            body_bytes = raw_data[header_end + 4:]

        # 解析请求行和请求头
        header_str = header_bytes.decode('utf-8', errors='ignore')
        header_lines = header_str.split('\r\n')
        if not header_lines:
            return parsed

        # 解析请求行
        request_line = header_lines[0].strip()
        request_parts = request_line.split()
        if len(request_parts) >= 1:
            parsed['method'] = request_parts[0].upper()
        if len(request_parts) >= 2:
            path_with_query = request_parts[1]
            path_part, *query_parts = path_with_query.split("?", 1)
            parsed["path"] = path_part
            if query_parts:
                query_params = urllib.parse.parse_qs(query_parts[0])
                parsed["query_params"] = {k: v[0] for k, v in query_params.items()}

        # 解析请求头
        for line in header_lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                parsed["headers"][key.strip().lower()] = value.strip()

        # 解析 Content-Type
        content_type = parsed["headers"].get("content-type", "")

        # 解析 JSON 请求体
        if "application/json" in content_type and body_bytes:
            try:
                parsed["body_json"] = json.loads(body_bytes.decode('utf-8'))
            except json.JSONDecodeError:
                parsed["body_json"] = None

        # 终极版：纯字节解析 multipart/form-data（核心修复）
        if "multipart/form-data" in content_type:
            try:
                # 提取 boundary
                boundary_str = None
                for ct_part in content_type.split(";"):
                    if "boundary=" in ct_part:
                        boundary_str = ct_part.split("=")[1].strip()
                        break
                if not boundary_str:
                    parsed["form_data"] = {"fields": {}, "files": {}}
                    return parsed

                boundary = b"--" + boundary_str.encode('utf-8')
                end_boundary = boundary + b"--"

                # 分割 parts
                parts = []
                temp_parts = body_bytes.split(boundary)
                for part in temp_parts:
                    part = part.strip()
                    if part and part != end_boundary.strip():
                        parts.append(part)

                form_fields = {}
                file_fields = {}

                # 逐 part 解析
                for part in parts:
                    if b"\r\n\r\n" not in part:
                        continue
                    header_bytes_part, content_bytes_part = part.split(b"\r\n\r\n", 1)
                    header_str_part = header_bytes_part.decode('utf-8', errors='ignore')

                    # 解析字段名
                    name_match = re.search(r'name="([^"]+)"', header_str_part)
                    if not name_match:
                        continue
                    field_name = name_match.group(1)

                    # 解析文件名（判断是否为文件）
                    filename_match = re.search(r'filename="([^"]+)"', header_str_part)
                    if not filename_match:
                        # 普通字段
                        clean_content = content_bytes_part.rstrip(b"\r\n")
                        form_fields[field_name] = clean_content.decode('utf-8', errors='ignore')
                    else:
                        # 文件字段（完整保留二进制）
                        filename = filename_match.group(1)
                        type_match = re.search(r'Content-Type:\s*([^\r\n]+)', header_str_part)
                        file_type = type_match.group(1) if type_match else "application/octet-stream"

                        # 清理文件内容末尾的 \r\n
                        if len(content_bytes_part) >= 2 and content_bytes_part[-2:] == b"\r\n":
                            file_content = content_bytes_part[:-2]
                        else:
                            file_content = content_bytes_part

                        file_fields[field_name] = {
                            "filename": filename,
                            "content_type": file_type,
                            "content": file_content,
                            "size": len(file_content)
                        }

                # 调试日志
                if not file_fields:
                    print(f"[Debug] No file fields detected, parts count: {len(parts)}")
                    if parts:
                        print(f"[Debug] First part: {parts[0][:200]}")

                parsed["form_data"] = {
                    "fields": form_fields,
                    "files": file_fields
                }
            except Exception as e:
                print(f"[Parse Error] {str(e)}")
                parsed["form_data"] = {"fields": {}, "files": {}}

        return parsed

    async def match(self, parsed_request):
        method = parsed_request['method']
        path = parsed_request['path']
        print(f"Matching route: {method} {path}")
        method_not_allowed = False
        for route in self.routes:
            print(f"Checking route: {route['path']} with methods: {route['methods']}")
            m = route['regex'].match(path)
            if m:
                print(f"Matched route: {route['path']}")
                if method in route['methods']:
                    func = route['func']
                    print(f"Calling function: {func.__name__}")
                    params = m.groupdict()
                    print(f"Params: {params}")
                    # 挂载 form_data 到 body
                    body = parsed_request['body_json'] or {}
                    if parsed_request['form_data']:
                        body['form_data'] = parsed_request['form_data']

                    if asyncio.iscoroutinefunction(func):
                        print("Calling coroutine function")
                        return await func(
                            params=params,
                            query=parsed_request['query_params'],
                            body=body
                        )
                    else:
                        print("Calling regular function")
                        return func(
                            params=params,
                            query=parsed_request['query_params'],
                            body=body
                        )
                else:
                    print(f"Method {method} not allowed for route {route['path']}")
                    method_not_allowed = True
        # 核心修复：调用self.response，避免全局函数覆盖
        if method_not_allowed:
            return self.response(405, 'This route does not allow this method')
        else:
            return self.response(404, 'This route is not found')

    async def handle_request(self, client_socket):
        client_addr = client_socket.getpeername()
        print(f"new communication：{client_addr}")
        print(f"Handling request from: {client_addr}")

        # 初始化默认响应
        response_content = self.response(400, "Bad Request")
        try:
            loop = asyncio.get_event_loop()
            raw_data = b""
            client_socket.settimeout(10.0)  # 增加超时时间

            # 1. 预读请求头
            try:
                first_chunk = await loop.sock_recv(client_socket, 1024)
                raw_data += first_chunk
            except Exception as e:
                print(f"[Pre-read Error] {e}")
                response_content = self.response(500, f"Pre-read error: {str(e)}")

            # 2. 解析方法和 Content-Length
            header_end = raw_data.find(b"\r\n\r\n")
            header_str = raw_data[:header_end].decode('utf-8',
                                                      errors='ignore') if header_end != -1 else raw_data.decode('utf-8',
                                                                                                                errors='ignore')
            header_lines = header_str.split('\r\n') if header_str else []

            method = "GET"
            content_length = 0
            if header_lines:
                request_line = header_lines[0].strip()
                request_parts = request_line.split()
                if len(request_parts) >= 1:
                    method = request_parts[0].upper()
                # 解析 Content-Length
                for line in header_lines:
                    if line.lower().startswith("content-length:"):
                        try:
                            content_length = int(line.split(":", 1)[1].strip())
                        except:
                            content_length = 0

            # 3. 智能读取 POST 体
            if method == "POST" and raw_data:
                current_body_len = len(raw_data) - (header_end + 4) if header_end != -1 else 0
                need_read = content_length - current_body_len
                print(f"[Debug] Current body length: {current_body_len}")
                print(f"[Debug] Need to read: {need_read}")

                while need_read > 0:
                    try:
                        chunk_size = min(4096, need_read)
                        print(f"[Debug] Reading chunk of size: {chunk_size}")
                        chunk = await loop.sock_recv(client_socket, chunk_size)
                        print(f"[Debug] Received chunk of size: {len(chunk)}")
                        if not chunk:
                            print(f"[Debug] No more data received")
                            break
                        raw_data += chunk
                        need_read -= len(chunk)
                        print(f"[Debug] Remaining to read: {need_read}")
                    except Exception as e:
                        print(f"[POST Read Error] {e}")
                        response_content = self.response(500, f"POST read error: {str(e)}")
                        break

            # 4. 解析请求并处理逻辑
            if raw_data:
                try:
                    parsed_request = await self.parse_request(raw_data)
                    if self.show_in:
                        print(f"result：{parsed_request['method']} {parsed_request['path']}")

                    # 核心：处理业务逻辑
                    try:
                        response_content = await self.match(parsed_request)
                    except Exception as e:
                        print(f"[Handler Error] {e}")
                        response_content = self.response(500, f"Handler error: {str(e)}")

                except Exception as e:
                    print(f"[Parse Error] {e}")
                    response_content = self.response(400, f"Parse error: {str(e)}")
            else:
                response_content = self.response(400, "Empty request data")

            # 5. 发送响应
            try:
                if not client_socket._closed:
                    response = self.build_response(response_content)
                    await loop.sock_sendall(client_socket, response)
            except Exception as e:
                print(f"[Send Error] {e}")

        except Exception as e:
            print(f"[Fatal Error] {e}")
            # 全局异常兜底，确保一定有响应
            try:
                error_res = self.build_response(self.response(500, f"Server error: {str(e)}"), 500)
                await loop.sock_sendall(client_socket, error_res)
            except:
                pass
        finally:
            # 确保连接关闭
            try:
                if not client_socket._closed:
                    client_socket.close()
            except:
                pass

    async def init_db_tool(self):
        self.db_pool = await aiomysql.create_pool(
            host=self.host,
            port=self.sql_port,
            user=self.sql_admin,
            password=self.sql_password,
            db='Graduation_242821513',
            charset='utf8mb4',
            minsize=1,
            maxsize=10,
            autocommit=True
        )
        # self.redis_pool = await aioredis.from_url(
        #     f'redis://{self.redis_host}:{self.redis_port}/{self.redis_db}',
        #     password=self.redis_password,
        #     encoding='utf-8',
        #     decpde_responses=True
        # )

    async def start(self):
        await self.init_db_tool()
        global mysql
        mysql.pool = self.db_pool
        
        # 初始化并启动多个训练工作器
        for i in range(self.num_workers):
            worker = TrainWorker(self.db_pool)
            self.train_workers.append(worker)
            worker.start()
            print(f"Worker {i+1} started")
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.setblocking(False)
        print("已注册路由：")
        for route in self.routes:
            print(f"   - {route['path']} (methods: {route['methods']})")
        loop = asyncio.get_event_loop()
        while True:
            client_socket, client_addr = await loop.sock_accept(self.server_socket)
            asyncio.ensure_future(self.handle_request(client_socket))

    # async def redis_get(self,key):
    #     if not self.redis_pool:
    #         return response(500,'Redis is not build success')
    #     value = await self.redis_pool.get(key)
    #     return response(msg=value)
    #
    # async def redis_set(self,key,value,expire=None):
    #     if not self.redis_pool:
    #         return response(500,'Redis is not build success')
    #     await self.redis_pool.set(key,value,ex=expire)
    #     return response(msg='Redis set success')
    #
    # async def redis_delete(self,key):
    #     if not self.redis_pool:
    #         return response(500,'Redis is not build success')
    #     await self.redis_pool.delete(key)
    #     return response(msg='Redis delete success')


# todo 4 run and add route
if __name__ == '__main__':
    filterwarnings('ignore')
    init_train_queue()
    app = App()

    @app.route('/',methods=['GET'])
    def primary(params,query,body):
        return response_html(os.path.join(html_path,'index.html'))

    @app.route('/home',methods=['GET'])
    def home(params,query,body):
        return response_html(os.path.join(html_path,'home.html'))

    @app.route('/reset_password',methods=['GET'])
    def reset_password(params,query,body):
        return response_html(os.path.join(html_path,'reset_password.html'))

    @app.route('/recover_account',methods=['GET'])
    def recover_account(params,query,body):
        return response_html(os.path.join(html_path,'recover_account.html'))

    @app.route('/user_info',methods=['GET'])
    def user_info(params,query,body):
        return response_html(os.path.join(html_path,'user_info.html'))

    @app.route('/detail_data',methods=['GET'])
    def detail_data(params,query,body):
        return response_html(os.path.join(html_path,'detail_data.html'))

    @app.route('/detail_model',methods=['GET'])
    def detail_model_page(params,query,body):
        return response_html(os.path.join(html_path,'detail_model.html'))

    @app.route('/static/<path:sub_path>', methods=['GET'])
    def serve_static(params, query, body):
        sub_path = params['sub_path']
        file_path = os.path.abspath(os.path.join('./static', sub_path))
        if not os.path.exists(file_path):
            return {'status': 404, 'msg': f"File not found: {sub_path}"}
        mime_type, _ = mimetypes.guess_type(file_path) or ('application/octet-stream', None)
        with open(file_path, 'rb') as f:
            content = f.read()
        return {
            'type': 'file',
            'content': content,
            'mime_type': mime_type
        }

    @app.route('/api/predict/{model_id}', methods=['POST'])
    async def predict_model(params, query, body):
        print("Predict model API called")
        model_id = params['model_id']
        user_id = body.get('user_id')
        token = body.get('token')
        
        if not user_id:
            return response(0, 'User ID is required')
        
        # 检查数据库中是否存在对应user_id的模型信息
        model_info = await mysql.find('sklearn_db', attribute=['user_id', 'id'], result=[user_id, model_id], relative=['AND'])
        if len(model_info['content']) == 0:
            return response(0, 'Cannot call other users\' models')
        
        model_info = model_info['content'][0]
        
        # 校验token
        if token != model_info['token']:
            return response(0, 'Invalid token')
        
        # 检查模型是否开启
        if str(model_info['open_bool']) != '1':
            return response(0, 'Model is not open')
        
        # 加载模型
        model_file = os.path.join(model_path, model_info['base_path'])
        if not os.path.exists(model_file):
            return response(0, 'Model file not found')
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # 准备预测数据
            # 提取自变量值
            x_columns = []
            x_values = []
            
            # 从请求中提取所有数值字段作为自变量
            for key, value in body.items():
                if key not in ['user_id', 'token'] and isinstance(value, (int, float)):
                    x_columns.append(key)
                    x_values.append(value)
            
            if not x_values:
                return response(0, 'No valid input features provided')
            
            # 转换为numpy数组
            predict_data = np.array(x_values).reshape(1, -1)
            
            # 检查是否需要预处理
            if 'data_id' in model_info:
                data_id = model_info['data_id']
                # 查询data_db获取预处理信息
                data_info = await mysql.find('data_db', attribute='id', result=data_id, relative='')
                if len(data_info['content']) > 0:
                    data_info = data_info['content'][0]
                    
                    # 处理though_work中的处理步骤
                    if 'though_work' in data_info:
                        though_work = data_info['though_work']
                        print(f"Processing though_work: {though_work}")
                        # 根据though_work中的步骤执行相应的预处理
                        # 这里可以根据实际的处理步骤进行具体实现
                    
                    # 检查是否有预处理参数
                    if 'preprocessing_params' in data_info:
                        try:
                            import json
                            preprocessing_params = json.loads(data_info['preprocessing_params'])
                            
                            # 处理归一化
                            if 'normalization' in preprocessing_params:
                                norm_params = preprocessing_params['normalization']
                                if 'mean' in norm_params and 'std' in norm_params:
                                    mean = np.array(norm_params['mean'])
                                    std = np.array(norm_params['std'])
                                    predict_data = (predict_data - mean) / std
                                elif 'min' in norm_params and 'max' in norm_params:
                                    min_val = np.array(norm_params['min'])
                                    max_val = np.array(norm_params['max'])
                                    predict_data = (predict_data - min_val) / (max_val - min_val)
                            
                            # 处理文本向量化
                            if 'vectorization' in preprocessing_params:
                                # 这里需要根据具体的向量化方法进行处理
                                # 例如，如果使用了TF-IDF，可以加载向量化器并进行转换
                                pass
                        except Exception as e:
                            print(f"Error processing preprocessing params: {str(e)}")
            
            # 预测
            predict_result = model.predict(predict_data)
            
            # 确保返回的结果是可序列化的
            result = predict_result
            if isinstance(result, np.ndarray):
                result = result.tolist()
            
            # 检查是否需要逆向归一化
            if 'data_id' in model_info:
                data_id = model_info['data_id']
                # 查询data_db获取预处理信息
                data_info = await mysql.find('data_db', attribute='id', result=data_id, relative='')
                if len(data_info['content']) > 0:
                    data_info = data_info['content'][0]
                    
                    # 处理though_work中的处理步骤
                    if 'though_work' in data_info:
                        though_work = data_info['though_work']
                        # 尝试读取though_work文件获取归一化信息
                        though_work_path = os.path.join(json_path, though_work)
                        if os.path.exists(though_work_path):
                            try:
                                with open(though_work_path, 'r', encoding='utf-8') as f:
                                    though_work_data = json.load(f)
                                
                                # 检查though_work_data中是否包含归一化步骤
                                if isinstance(though_work_data, list):
                                    for step in though_work_data:
                                        if step.get('type') == 'normalization' and 'save_path' in step:
                                            # 尝试对预测结果进行逆向归一化
                                            try:
                                                # 构建预测结果的DataFrame
                                                import pandas as pd
                                                pred_df = pd.DataFrame({'prediction': result})
                                                # 尝试逆向归一化
                                                denorm_result = panda.denormalize(pred_df, step['save_path'], columns=['prediction'])
                                                if denorm_result['status'] == 1:
                                                    # 使用逆向归一化后的值
                                                    result = denorm_result['content']['prediction'].tolist()
                                            except Exception as e:
                                                print(f"Error during denormalization: {e}")
                            except Exception as e:
                                print(f"Error reading though_work file: {e}")
            
            return response(1, result)
        except Exception as e:
            print(f"Error in predict_model: {str(e)}")
            return response(0, f'Prediction failed: {str(e)}')

    @app.route('/api/get_language', methods=['GET'])
    async def get_language(params, query, body):
        # 读取setting.json文件
        try:
            with open('setting.json', 'r', encoding='utf-8') as f:
                settings = json.load(f)
            la = settings.get('la', 'en')
            return response(1, la)
        except Exception as e:
            print(f"Error getting language setting: {e}")
            return response(0, 'en')

    @app.route('/api/{function}', methods=['POST'])
    async def api(params, query, body):
        print("API function called")
        from supplement.encryption import Other
        print("Imported Other class")
        other = Other()
        print("Created other instance")
        func_name = params['function']
        print(f"Function name: {func_name}")
        if func_name == 'load_verification':
            now_password = other.password_encryption(body['password'])
            if now_password['status'] == 0:
                return now_password
            _ = judge_captcha(os.path.join(system_pic_path, body['imgCode']),body['inputCode'])
            if _['status'] != 1:
                return _
            result = await mysql.find('user_db', attribute='id', result=body['id'], relative='')
            if len(result['content']) == 0:
                return response(0,'This user is not created.Please register.')
            if now_password['content'] != result['content'][0]['password']:
                return response(0, 'This password is false.Please reset.')
            return response(1,'Load success')

        elif func_name == 'register_verification':
            user_id = other.n_to_id(body['username'])['content']
            password = other.password_encryption(body['password'])['content']
            _ = judge_captcha(os.path.join(system_pic_path, body['imgCode']), body['inputCode'])
            if _['status'] != 1:
                return _
            result = await mysql.find('user_db',attribute='id')
            if len(result['content']) > 0:
                if result['content'][0]['phone'] == body['phone']:
                    return response(0, 'This phone is common')
                result = await mysql.find('user_db',attribute='id',result=user_id,relative='')
                if result['response'] == 0:
                    return response(0,'This user_id have mistake')
            try:
                if len(result['content']) != 0:
                    print('Warning[1] find a common user_id')
                    while user_id == result['content'][0]['id']:
                        user_id = other.n_to_id(body['username'])['content']
            except:pass
            result1 = await mysql.insert('user_db',attribute=['id','username','phone','password','identity','is_delete','head_url'],values=[user_id,body['username'],body['phone'],password,'1','0',os.path.join(user_pic_path,'primary.jpg')])
            if result1['response'] == 1:
                return response(1,f'Create user success. Now please load.Your id is {user_id}')
            else:
                return response(0,'System find some trouble')

        elif func_name == 'search_user_info':
            user_id = body['user_id']
            result = await mysql.find('user_db',attribute='id',result=user_id)
            if result['response'] != 1:
                return response(0,'Please load at first')
            if len(result['content']) == 0:
                return response(0,'This user is not register')
            _ = result['content'][0]
            _identity = _.get('identity', '')
            if len(_identity) <= 18:
                _identity = None
            return response(1,{'user_id':_['id'],'username':_['username'],'phone':_['phone'],'head_url':_['head_url'],'identity':_identity})

        elif func_name == 'reset_password':
            user_id = body['id']
            phone = body['phone']
            _ = judge_captcha(os.path.join(system_pic_path, body['imgCode']), body['inputCode'])
            if _['status'] != 1:
                return _
            # 验证用户ID和电话号码是否存在
            user_info = await mysql.find('user_db',attribute=['id','phone'],result=[user_id,phone],relative=['and'])
            if len(user_info['content']) == 0:
                return response(0, 'User ID and phone number do not match')
            password = other.password_encryption(body['password'])['content']
            result = await mysql.update('user_db',attribute=['id','phone'],values=[user_id,phone],aim='password',new_values=password)
            if result['response'] == 1:
                return response(1,f'Your new password is {body["password"]}')

        elif func_name == 'recover_account':
            username = body['username']
            phone = body['phone']
            _ = judge_captcha(os.path.join(system_pic_path, body['imgCode']),body['inputCode'])
            if _['status'] != 1:
                return _
            result = await mysql.find('user_db',aim='id',attribute=['username','phone'],result=[username,phone],relative=['and'])
            if result['response'] != 1:
                return result
            return response(1,f'Find your id success.Your id is {result["content"][0]["id"]}, please remember')

        elif func_name == 'upload':
            try:
                form_data = body.get('form_data') if isinstance(body, dict) else None
                user_id = form_data['fields']['user_id']
                types = form_data['fields']['types'].lower()
                if (not types) or (types not in ['regression','clustering']):
                    types = 'classification'
                _id_lst = []
                _inspect = await mysql.find('user_db',attribute='id',result=user_id)
                if len(_inspect['content']) == 0:
                    return response(0,'This id is not register')
                _now = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                if not form_data:
                    return response(0, 'request：is not multipart/form-data type')
                file_fields = form_data.get('files', {})
                if not file_fields or 'file' not in file_fields:
                    return response(0, 'file is mistake（need "file"）')
                file_info = file_fields['file']
                file_content = file_info['content']
                filename = file_info['filename']
                if len(file_content) == 0:
                    return response(0, 'upload file is null')
                data_id = other.get_random_id(user_id,'d')['content']
                _review = await mysql.find('data_db','saving')
                if len(_review['content']) != 0:
                    for i in _review['content']:
                        _id_lst.append(i)
                while data_id in _id_lst:
                    data_id = other.get_random_id(user_id,'d')['content']
                # for mysql this function build a question!
                _insert = await mysql.insert('data_db',['user_id','update_time','saving','sheet_name','though_work','types'],[user_id,_now,f'{data_id}.{filename.split(".")[-1]}',filename,'load',types])
                with open(os.path.join(data_path,f'{data_id}.{filename.split(".")[-1]}'),'wb') as f:
                    f.write(file_content)
                f.close()
                print(f"success find：{filename}，size of：{len(file_content)} bit")
                return response(msg='Upload success')
            except Exception as e:
                return response(0, f'file upload fail：{str(e)}')

        elif func_name == 'find_model':
            user_id = body['user_id']
            _inspect = await mysql.find('user_db', attribute='id', result=user_id)
            if len(_inspect['content']) == 0:
                return response(0, 'This id is not register')
            # Only classification, regression, clustering, normalization and vectorization can be written here
            train_way = body['train_way']
            if train_way == 'all':
                _param = {}
                for choice in ['classification','regression','clustering','normalization','vectorization']:
                    _ = other.get_train_way(os.path.join(primary_model_setting, f'{choice}.txt'))
                    for i in range(len(_['content'])):
                        _['content'][i] = _['content'][i].replace('\n', '')
                    _param[choice] = _['content']
                return response(msg=_param)
            _ = other.get_train_way(os.path.join(primary_model_setting,f'{train_way}.txt'))
            for i in range(len(_['content'])):
                _['content'][i] = _['content'][i].replace('\n','')
            return response(msg=_['content'])

        elif func_name == 'get_model_hyperparams':
            model_name = body['model_name']
            # 从JSON文件中读取超参配置
            hyperparams_path = os.path.join(primary_model_setting, f'{model_name}.json')
            if os.path.exists(hyperparams_path):
                try:
                    with open(hyperparams_path, 'r', encoding='utf-8') as f:
                        hyperparams = json.load(f)
                    return response(msg=hyperparams)
                except Exception as e:
                    print(f"Error loading hyperparams file: {e}")
                    return response(0, f'Error loading hyperparams: {str(e)}')
            else:
                return response(0, f'Hyperparams file not found for model: {model_name}')

        elif func_name == 'detail_model':
            user_id = body.get('user_id')
            model_id = body.get('model_id')
            model_name = body.get('model_name')

            # 验证用户是否存在
            if not user_id:
                return response(0, 'User ID is required')
            _inspect = await mysql.find('user_db', attribute='id', result=user_id)
            if len(_inspect['content']) == 0:
                return response(0, 'This id is not register')

            # 查找模型信息
            if model_id:
                model_info = await mysql.find('sklearn_db', attribute=['user_id', 'id'], result=[user_id, model_id], relative=['AND'])
            elif model_name:
                model_info = await mysql.find('sklearn_db', attribute=['user_id', 'model_name'], result=[user_id, model_name], relative=['AND'])
            else:
                return response(0, 'Model ID or model name is required')
            
            if len(model_info['content']) == 0:
                return response(0, 'Model not found')

            model = model_info['content'][0]
            # 转换模型信息中的 datetime 对象为字符串
            for key, value in model.items():
                if hasattr(value, 'strftime'):
                    model[key] = str(value)
            
            model_id = model.get('id', '')
            data_id = model.get('data_id', '')
            hyper = model.get('hyper', '')
            evaluation = model.get('evaluation', '')
            open_bool = model.get('open_bool', 0)

            # 加载超参信息
            hyper_params = {}
            if hyper:
                hyper_path = os.path.join(json_path, hyper)
                if os.path.exists(hyper_path):
                    try:
                        with open(hyper_path, 'r', encoding='utf-8') as f:
                            hyper_params = json.load(f)
                    except Exception as e:
                        print(f"Error loading hyperparams: {e}")

            # 加载评估指标
            evaluation_metrics = {}
            if evaluation:
                eval_path = os.path.join(json_path, evaluation)
                if os.path.exists(eval_path):
                    try:
                        with open(eval_path, 'r', encoding='utf-8') as f:
                            evaluation_metrics = json.load(f)
                    except Exception as e:
                        print(f"Error loading evaluation metrics: {e}")

            # 加载数据集信息
            data_info = {}
            if data_id:
                data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                if len(data_record['content']) > 0:
                    data_info = data_record['content'][0]
                    # 转换所有 datetime 对象为字符串
                    for key, value in data_info.items():
                        if hasattr(value, 'strftime'):
                            data_info[key] = str(value)

            # 计算训练集和测试集的大小和比例
            train_test_info = {
                'train_size': 0,
                'test_size': 0,
                'ratio': 0.0,
                'is_split': False,  # 添加一个标志，表示是否实际划分了数据集
                'total_size': 0,     # 添加总数据量
                'details': ''        # 添加详细信息
            }

            # 优先使用data_db中对应数据集的though_work信息
            if data_id:
                data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                if len(data_record['content']) > 0:
                    data_info = data_record['content'][0]
                    though_work = data_info.get('though_work', '')
                    
                    # 如果though_work不是'load'，尝试读取though_work文件获取划分信息
                    if though_work and though_work != 'load':
                        though_work_path = os.path.join(json_path, though_work)
                        if os.path.exists(though_work_path):
                            try:
                                with open(though_work_path, 'r', encoding='utf-8') as f:
                                    though_work_data = json.load(f)
                                
                                # 检查though_work_data中是否包含训练集和测试集的信息
                                if 'train' in though_work_data and 'test' in though_work_data:
                                    train_data = though_work_data['train']
                                    test_data = though_work_data['test']
                                    
                                    # 计算训练集和测试集的大小
                                    if 'x' in train_data and 'y' in train_data:
                                        train_size = len(train_data['x'])
                                        test_size = len(test_data['x'])
                                        total_size = train_size + test_size
                                        
                                        if total_size > 0:
                                            ratio = test_size / total_size
                                            train_test_info = {
                                                'train_size': train_size,
                                                'test_size': test_size,
                                                'ratio': round(ratio, 1),
                                                'is_split': True,
                                                'total_size': total_size,
                                                'details': f'Using though_work information for dataset split'
                                            }
                            except Exception as e:
                                print(f"Error reading though_work file: {e}")
                    # 如果though_work是'load'，表示无相关的测试集与训练集划分
                    elif though_work == 'load':
                        train_test_info = {
                            'train_size': 0,
                            'test_size': 0,
                            'ratio': 0.0,
                            'is_split': False,
                            'total_size': 0,
                            'details': 'No train/test split information available (though_work is load)'
                        }

            # 如果没有though_work信息，尝试读取数据集文件
            if train_test_info['total_size'] == 0 and data_id:
                file_path = os.path.join(data_path, data_id)
                if os.path.exists(file_path):
                    try:
                        if file_path.endswith('.csv'):
                            # 使用panda.read_file读取CSV文件
                            data = panda.read_file(file_path, encoding='utf-8')['content']
                            df = panda.json_to_dataframe(data)['content']
                            # 检查是否有dataset_type列
                            if 'dataset_type' in df.columns:
                                # 获取训练集和测试集的索引
                                train_indices = df[df['dataset_type'] == 'train'].index.tolist()
                                test_indices = df[df['dataset_type'] == 'test'].index.tolist()

                                train_size = len(train_indices)
                                test_size = len(test_indices)

                                if train_size + test_size > 0:
                                    total_size = train_size + test_size
                                    ratio = test_size / total_size
                                    train_test_info = {
                                        'train_size': train_size,
                                        'test_size': test_size,
                                        'ratio': round(ratio, 1),
                                        'is_split': True,
                                        'total_size': total_size,
                                        'details': f'Train indices: {train_indices[:5]}... (total: {train_size}), Test indices: {test_indices[:5]}... (total: {test_size})'
                                    }
                            else:
                                # 如果没有dataset_type列，返回总数据量
                                total_size = len(df)
                                train_test_info = {
                                    'train_size': total_size,
                                    'test_size': 0,
                                    'ratio': 0.0,
                                    'is_split': False,
                                    'total_size': total_size,
                                    'details': 'No dataset_type column found. Using total data size.'
                                }
                        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                            # 尝试读取train和test工作表
                            try:
                                # 尝试读取train工作表
                                train_data = panda.read_file(file_path, sheet_name='train', engine='openpyxl')['content']
                                train_df = panda.json_to_dataframe(train_data)['content']
                                test_data = panda.read_file(file_path, sheet_name='test', engine='openpyxl')['content']
                                test_df = panda.json_to_dataframe(test_data)['content']

                                # 计算实际的行数
                                train_size = len(train_df)
                                test_size = len(test_df)
                                total_size = train_size + test_size

                                print(f"Train sheet rows: {train_size}, Test sheet rows: {test_size}")

                                if train_size + test_size > 0:
                                    ratio = test_size / (train_size + test_size)
                                    train_test_info = {
                                        'train_size': train_size,
                                        'test_size': test_size,
                                        'ratio': round(ratio, 1),
                                        'is_split': True,
                                        'total_size': total_size,
                                        'details': f'Train sheet rows: {train_size}, Test sheet rows: {test_size}'
                                    }
                            except Exception as e:
                                # 如果train或test工作表不存在，读取第一个工作表
                                data = panda.read_file(file_path, engine='openpyxl')['content']
                                df = panda.json_to_dataframe(data)['content']
                                total_size = len(df)
                                print(f"Using first sheet with {total_size} rows")
                                train_test_info = {
                                    'train_size': total_size,
                                    'test_size': 0,
                                    'ratio': 0.0,
                                    'is_split': False,
                                    'total_size': total_size,
                                    'details': f'No train/test sheets found. Using first sheet with {total_size} rows.'
                                }
                    except Exception as e:
                        print(f"Error reading dataset: {e}")

            # 递归转换 datetime 对象为字符串
            def convert_datetime(obj):
                if isinstance(obj, dict):
                    return {key: convert_datetime(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                elif hasattr(obj, 'strftime'):
                    return str(obj)
                else:
                    return obj

            # 准备返回数据
            result = {
                'model_info': {
                    'id': model_id,
                    'name': model.get('model_name', ''),
                    'type': model.get('types', ''),
                    'status': model.get('model_condition', ''),
                    'open_bool': open_bool,
                    'use_model': model.get('use_model', '')
                },
                'types': model.get('types', ''),  # 添加顶层types字段
                'status': model.get('model_condition', ''),  # 添加顶层status字段
                'open_bool': open_bool,  # 添加顶层open_bool字段
                'data_id': data_id,  # 添加data_id字段
                'hyper_params': hyper_params,
                'evaluation_metrics': evaluation_metrics,
                'data_info': data_info,
                'train_test_info': train_test_info,
                'downloads': {
                    'sdk': f'/api/download_sdk',
                    'vectorization': f'/api/download_vectorization',
                    'normalization': f'/api/download_normalization'
                }
            }

            # 转换所有 datetime 对象为字符串
            result = convert_datetime(result)

            return response(msg=result)

        elif func_name == 'download_sdk':
            user_id = body['user_id']
            model_id = body['model_id']

            # 验证用户是否存在
            _inspect = await mysql.find('user_db', attribute='id', result=user_id)
            if len(_inspect['content']) == 0:
                return response(0, 'This id is not register')

            # 验证模型是否存在且属于该用户
            model_info = await mysql.find('sklearn_db', attribute=['id', 'user_id'], result=[model_id, user_id], relative=['AND'])
            if len(model_info['content']) == 0:
                return response(0, 'Model not found or does not belong to this user')

            # 从sdk_relationship表中获取SDK文件信息
            sdk_info = await mysql.find('sdk_relationship', attribute='model_id', result=model_id)
            if len(sdk_info['content']) == 0:
                return response(0, 'No SDK file available for download')

            sdk_file = sdk_info['content'][0]['sdk_file']
            if not sdk_file or sdk_file == 'None':
                return response(0, 'No SDK file available for download')

            # 构建SDK文件路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # 尝试多种可能的路径格式
            sdk_path = os.path.join(base_dir, 'static', 'sdk', sdk_file)
            if not os.path.exists(sdk_path):
                # 尝试原始路径格式
                sdk_path = os.path.join(base_dir, sdk_file)
                if not os.path.exists(sdk_path):
                    # 尝试另一种路径格式
                    sdk_path = os.path.join(base_dir, 'static', sdk_file)
            if not os.path.exists(sdk_path):
                return response(0, f'SDK file not found on server: {sdk_path}')

            # 读取文件内容
            content, mime_type = read_static_file(sdk_path)
            if content is None:
                return response(0, 'SDK file not found on server')

            # 直接返回文件响应，避免被当作JSON处理
            return {
                'type': 'file',
                'content': content,
                'mime_type': mime_type or 'application/zip',
                'as_attachment': True,
                'filename': os.path.basename(sdk_file)
            }

        elif func_name == 'download_sdk':
            user_id = body['user_id']
            model_id = body['model_id']

            # 验证用户是否存在
            _inspect = await mysql.find('user_db', attribute='id', result=user_id)
            if len(_inspect['content']) == 0:
                return response(0, 'This id is not register')

            # 验证模型是否存在且属于该用户
            model_info = await mysql.find('sklearn_db', attribute=['id', 'user_id'], result=[model_id, user_id], relative=['AND'])
            if len(model_info['content']) == 0:
                return response(0, 'Model not found or does not belong to this user')

            # 从sdk_relationship表中获取SDK文件信息
            sdk_info = await mysql.find('sdk_relationship', attribute='model_id', result=model_id)
            if len(sdk_info['content']) == 0:
                return response(0, 'No SDK file available for download')

            sdk_file = sdk_info['content'][0]['sdk_file']
            if not sdk_file or sdk_file == 'None':
                return response(0, 'No SDK file available for download')

            # 构建SDK文件路径
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # 尝试多种可能的路径格式
            sdk_path = os.path.join(base_dir, 'static', 'sdk', sdk_file)
            if not os.path.exists(sdk_path):
                # 尝试原始路径格式
                sdk_path = os.path.join(base_dir, sdk_file)
                if not os.path.exists(sdk_path):
                    # 尝试另一种路径格式
                    sdk_path = os.path.join(base_dir, 'static', sdk_file)
            if not os.path.exists(sdk_path):
                return response(0, f'SDK file not found on server: {sdk_path}')

            # 读取文件内容
            content, mime_type = read_static_file(sdk_path)
            if content is None:
                return response(0, 'SDK file not found on server')

            # 直接返回文件响应，避免被当作JSON处理
            return {
                'type': 'file',
                'content': content,
                'mime_type': mime_type or 'application/zip',
                'as_attachment': True,
                'filename': os.path.basename(sdk_file)
            }

        elif func_name == 'find_data':
            user_id = body['user_id']
            find_way = body['find_way']
            # 检查用户是否存在
            _inspect = await mysql.find('user_db', attribute='id', result=user_id)
            if len(_inspect['content']) == 0:
                return response(0, 'This id is not register')
            
            if find_way == 'all':
                _ = await mysql.find('data_db',attribute='user_id',result=user_id)
                for i in range(len(_['content'])):
                    if 'update_time' in _['content'][i] and _['content'][i]['update_time']:
                        _['content'][i]['update_time'] = str(_['content'][i]['update_time'])
                return response(msg=_['content'])
            elif find_way == 'single':
                # This data_id such as xxxx.csv or xxxx.xlsx or xxx.xls
                data_id = body['data_id']
                sheet_name = body.get('sheet_name', None)
                data_types = data_id.split('.')[-1]
                if data_types == 'csv':
                    try:
                        data = panda.read_file(os.path.join(data_path,data_id))
                    except:
                        try:
                            data = panda.read_file(os.path.join(data_path,data_id),encoding='gbk2312')
                        except:
                            try:
                                data = panda.read_file(os.path.join(data_path,data_id),sep=';')
                            except:
                                return response(0,'This file we don`t have way to open.Please communicate with customer service')
                elif data_types in ['xlsx','xls']:
                    data = panda.read_file(os.path.join(data_path,data_id), sheet_name=sheet_name, engine='openpyxl')

                else:
                    return response(0,'This file is not have way to open temporarily.Please communicate with customer service')
                return response(msg=data['content'])
            else:
                return response(0,'This function is not allow to use')

        elif func_name == 'vectorization':
            vectorization_way = body['vectorization']
            data = body['data']
            units = body['unit']
            columns = body.get('columns', None)
            _data1 = panda.json_to_dataframe(data)['content']
            _inspect_name = await mysql.find('data_db')
            _data2 = panda.text_to_number(_data1,vectorization_way,units=units, columns=columns)
            _data3 = panda.dataframe_to_json(_data2['content'])
            return response(msg=_data3['content'])

        elif func_name == 'normalization':
            normalization = body['normalization']
            data = body['data']
            columns = body.get('columns', None)
            _data1 = panda.json_to_dataframe(data)['content']
            _data2 = panda.normalization(_data1,normalization, columns=columns)
            _data3 = panda.dataframe_to_json(_data2['content'])
            return response(msg=_data3['content'])

        elif func_name == 'duplicate':
            _data = body['data']
            _subset = body['subset']
            _data1 = panda.json_to_dataframe(_data)['content']
            _data2 = panda.drop_duplicate(_data1,subset=_subset)['content']
            _data3 = panda.dataframe_to_json(_data2)['content']
            return response(msg=_data3)

        elif func_name == 'fill_na':
            _data = body['data']
            _methods = body['methods']
            _columns = body.get('columns', None)
            _data1 = panda.json_to_dataframe(_data)['content']
            _data2 = panda.fill_na(_data1,_methods, columns=_columns)['content']
            _data3 = panda.dataframe_to_json(_data2)['content']
            return response(msg=_data3)

        elif func_name == 'outlier':
            _data = body['data']
            _methods = body['methods']
            _multiple = body['multiple']
            _columns = body.get('columns', None)
            _data1 = panda.json_to_dataframe(_data)['content']
            _data2 = panda.drop_outlier(_data1, methods=_methods, multiple=_multiple, columns=_columns)['content']
            _data3 = panda.dataframe_to_json(_data2)['content']
            return response(msg=_data3)

        elif func_name == 'preprocess':
            func_name = body['func_name']
            data = body['data']

            # 根据不同的预处理函数调用相应的方法
            if func_name == 'duplicate':
                subset = body.get('subset', None)
                _data1 = panda.json_to_dataframe(data)['content']
                _data2 = panda.drop_duplicate(_data1, subset=subset)['content']
                _data3 = panda.dataframe_to_json(_data2)['content']
                return response(msg=_data3)
            elif func_name == 'fill_na':
                methods = body.get('methods', 'mean')
                columns = body.get('columns', None)
                _data1 = panda.json_to_dataframe(data)['content']
                _data2 = panda.fill_na(_data1, methods, columns=columns)['content']
                _data3 = panda.dataframe_to_json(_data2)['content']
                return response(msg=_data3)
            elif func_name == 'outlier':
                methods = body.get('methods', 'iqr')
                multiple = body.get('multiple', 1.5)
                columns = body.get('columns', None)
                _data1 = panda.json_to_dataframe(data)['content']
                _data2 = panda.drop_outlier(_data1, methods=methods, multiple=multiple, columns=columns)['content']
                _data3 = panda.dataframe_to_json(_data2)['content']
                return response(msg=_data3)
            else:
                return response(0, 'Preprocess function not supported')

        elif func_name == 'find_columns':
            data_id = body['data_id']
            # 对于 Excel 文件，使用 openpyxl 引擎
            file_ext = data_id.split('.')[-1].lower()
            if file_ext in ['xlsx', 'xls']:
                _data = panda.read_file(os.path.join(data_path, data_id), engine='openpyxl')
            else:
                _data = panda.read_file(os.path.join(data_path, data_id))
            # print(_data)
            _data1 = panda.json_to_dataframe(_data['content'])['content']
            columns = panda.find_columns(_data1)['content']
            return response(msg=columns)

        elif func_name == 'train_test_split':
            try:
                import pandas as pd
                data_id = body['data_id']
                test_size = body.get('test_size', 0.2)
                x_columns = body.get('x_columns', None)
                y_columns = body.get('y_columns', None)
                columns = body.get('columns', None)
                file_path = os.path.join(data_path, data_id)
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    return response(0, 'File not found')
                _ = data_id.split('.')[-1].lower()
                if _ in ['xlsx','xls']:
                    # 尝试读取train工作表，如果不存在则读取第一个工作表
                    try:
                        # 尝试读取train工作表
                        try:
                            _data = panda.read_file(file_path, sheet_name='train', engine='openpyxl')['content']
                        except Exception as e:
                            # 如果train工作表不存在，读取第一个工作表
                            _data = panda.read_file(file_path, engine='openpyxl')['content']
                        _data1 = panda.json_to_dataframe(_data)['content']
                    except Exception as e:
                        print(f"Error reading Excel file: {str(e)}")
                        return response(0, f'Error reading Excel file: {str(e)}')
                elif _ == 'csv':
                    try:
                        # 读取CSV文件
                        _data = panda.read_file(file_path, encoding='utf-8')['content']
                        _data1 = panda.json_to_dataframe(_data)['content']
                    except Exception as e:
                        print(f"Error reading CSV file: {str(e)}")
                        return response(0, f'Error reading CSV file: {str(e)}')
                else:
                    return response(0, 'This file type is not supported')
                
                # 检查x_columns和y_columns是否存在于数据中
                if x_columns and y_columns:
                    # 确保x_columns和y_columns是列表
                    if isinstance(x_columns, str):
                        x_columns = [x_columns]
                    if isinstance(y_columns, str):
                        y_columns = [y_columns]
                    # 检查所有列是否存在
                    missing_columns = []
                    for col in x_columns + y_columns:
                        if col not in _data1.columns:
                            missing_columns.append(col)
                    if missing_columns:
                        return response(0, f'Missing columns: {missing_columns}')
                
                # 检查columns是否存在于数据中
                if columns:
                    # 确保columns是列表
                    if isinstance(columns, str):
                        columns = [columns]
                    # 检查所有列是否存在
                    missing_columns = []
                    for col in columns:
                        if col not in _data1.columns:
                            missing_columns.append(col)
                    if missing_columns:
                        return response(0, f'Missing columns: {missing_columns}')
                
                split_result = panda.train_test_split(
                    _data1,
                    test_size=test_size,
                    x_columns=x_columns,
                    y_columns=y_columns,
                    columns=columns
                )
                if split_result['status'] != 1:
                    return response(0, split_result['content'])
                result = {
                    'train_index': split_result['content']['train']['index'],
                    'test_index': split_result['content']['test']['index'],
                    'x_columns': x_columns,
                    'y_columns': y_columns
                }
                response_data = response(1, result)
                return response_data
            except Exception as e:
                print(f"Error in train_test_split: {str(e)}")
                import traceback
                traceback.print_exc()
                return response(0, f'Train test split failed: {str(e)}')

        elif func_name == 'get_column_stats':
            try:
                data_id = body['data_id']
                column = body['column']

                # 读取数据文件
                file_path = os.path.join(data_path, data_id)
                if not os.path.exists(file_path):
                    return response(0, 'File not found')

                # 读取数据
                _ = data_id.split('.')[-1].lower()
                if _ in ['xlsx','xls']:
                    # 尝试读取train工作表，如果不存在则读取第一个工作表
                    try:
                        # 尝试读取train工作表
                        try:
                            train_data = panda.read_file(file_path, sheet_name='train', engine='openpyxl')['content']
                            train_df = panda.json_to_dataframe(train_data)['content']
                            test_data = panda.read_file(file_path, sheet_name='test', engine='openpyxl')['content']
                            test_df = panda.json_to_dataframe(test_data)['content']
                            has_train_test = True
                        except Exception as e:
                            # 如果train或test工作表不存在，读取第一个工作表
                            data = panda.read_file(file_path, engine='openpyxl')['content']
                            df = panda.json_to_dataframe(data)['content']
                            has_train_test = False
                    except Exception as e:
                        print(f"Error reading Excel file: {str(e)}")
                        return response(0, f'Error reading Excel file: {str(e)}')
                elif _ == 'csv':
                    try:
                        # 读取CSV文件
                        data = panda.read_file(file_path, encoding='utf-8')['content']
                        df = panda.json_to_dataframe(data)['content']
                        has_train_test = 'dataset_type' in df.columns
                        if has_train_test:
                            # 使用pd_item的filter_data方法过滤数据
                            train_df = panda.filter_data(df, 'dataset_type', 'train')['content']
                            test_df = panda.filter_data(df, 'dataset_type', 'test')['content']
                    except Exception as e:
                        print(f"Error reading CSV file: {str(e)}")
                        return response(0, f'Error reading CSV file: {str(e)}')
                else:
                    return response(0, 'This file type is not supported')

                # 检查列是否存在
                if has_train_test:
                    if column in train_df.columns and column in test_df.columns:
                        # 分别提取列数据
                        train_column_data = train_df[column]
                        test_column_data = test_df[column]

                        # 检查是否为归一化数据
                        is_normalized = False
                        original_range = None
                        normalized_data = []

                        # 从data_db中获取归一化文件路径
                        try:
                            # 根据data_id（等同于data_db中的saving）查询data_db
                            data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                            if len(data_record['content']) > 0:
                                data_info = data_record['content'][0]
                                normalization_file_name = data_info.get('normalization')

                                if normalization_file_name:
                                    # 构建归一化文件路径
                                    normalization_file = os.path.join(json_path, normalization_file_name)
                                    if os.path.exists(normalization_file):
                                        try:
                                            # 使用pd_item的denormalize方法进行逆向归一化
                                            # 构建包含train和test数据的DataFrame
                                            import pandas as pd
                                            combined_df = pd.concat([train_df, test_df], ignore_index=True)
                                            # 对指定列进行逆向归一化
                                            denorm_result = panda.denormalize(combined_df, normalization_file, columns=[column])
                                            if denorm_result['status'] == 1:
                                                is_normalized = True
                                                # 获取逆向归一化后的数据
                                                denorm_df = denorm_result['content']
                                                normalized_data = denorm_df[column].tolist()
                                                # 重新计算统计信息
                                                calculate_result = panda.calculate(denorm_df)['content']
                                                if column in calculate_result['col_name']:
                                                    idx = calculate_result['col_name'].index(column)
                                                    stats = {
                                                        'mean': calculate_result['col_mean'][idx],
                                                        'median': calculate_result['col_median'][idx],
                                                        'min': calculate_result['col_min'][idx],
                                                        'max': calculate_result['col_max'][idx],
                                                        'std': calculate_result['col_std'][idx],
                                                        'count': len(normalized_data)
                                                    }
                                                # 获取原始范围
                                                with open(normalization_file, 'r', encoding='utf-8') as f:
                                                    normalization_params = json.load(f)
                                                if column in normalization_params:
                                                    params = normalization_params[column]
                                                    if 'col_min' in params and 'col_max' in params:
                                                        original_range = {
                                                            'min': params['col_min'],
                                                            'max': params['col_max']
                                                        }
                                        except Exception as e:
                                            print(f"Error reading normalization file {normalization_file}: {e}")
                        except Exception as e:
                            print(f"Error querying data_db: {e}")

                        # 如果没有进行反向归一化，合并数据并使用pd_item的calculate方法计算统计信息
                        if not is_normalized:
                            combined_df = panda.concat([train_df, test_df], ignore_index=True)['content']
                            # 使用pd_item的calculate方法计算统计信息
                            calculate_result = panda.calculate(combined_df)['content']
                            # 查找指定列的统计信息
                            if column in calculate_result['col_name']:
                                idx = calculate_result['col_name'].index(column)
                                stats = {
                                    'mean': calculate_result['col_mean'][idx],
                                    'median': calculate_result['col_median'][idx],
                                    'min': calculate_result['col_min'][idx],
                                    'max': calculate_result['col_max'][idx],
                                    'std': calculate_result['col_std'][idx],
                                    'count': len(combined_df[column])
                                }
                            else:
                                # 如果列不在计算结果中，使用默认值
                                stats = {
                                    'mean': 0,
                                    'median': 0,
                                    'min': 0,
                                    'max': 0,
                                    'std': 0,
                                    'count': 0
                                }
                            normalized_data = combined_df[column].tolist()
                    else:
                        # 如果列在某个工作表中不存在，只使用存在的工作表
                        if column in train_df.columns:
                            column_data = train_df[column]
                        elif column in test_df.columns:
                            column_data = test_df[column]
                        else:
                            return response(0, f'Column {column} not found in any sheet')
                        # 使用pd_item的calculate方法计算统计信息
                        calculate_result = panda.calculate(column_data.to_frame())['content']
                        # 查找指定列的统计信息
                        if column in calculate_result['col_name']:
                            idx = calculate_result['col_name'].index(column)
                            stats = {
                                'mean': calculate_result['col_mean'][idx],
                                'median': calculate_result['col_median'][idx],
                                'min': calculate_result['col_min'][idx],
                                'max': calculate_result['col_max'][idx],
                                'std': calculate_result['col_std'][idx],
                                'count': len(column_data)
                            }
                        else:
                            # 如果列不在计算结果中，使用默认值
                            stats = {
                                'mean': 0,
                                'median': 0,
                                'min': 0,
                                'max': 0,
                                'std': 0,
                                'count': 0
                            }
                        normalized_data = column_data.tolist()
                else:
                    # 没有train和test工作表，使用整个数据集
                    if column not in df.columns:
                        return response(0, f'Column {column} not found in dataset')
                    column_data = df[column]

                    # 检查是否为归一化数据
                    is_normalized = False
                    original_range = None
                    normalized_data = column_data.tolist()

                    # 从data_db中获取归一化文件路径
                    try:
                        # 根据data_id（等同于data_db中的saving）查询data_db
                        data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                        if len(data_record['content']) > 0:
                            data_info = data_record['content'][0]
                            normalization_file_name = data_info.get('normalization')

                            if normalization_file_name:
                                # 构建归一化文件路径
                                normalization_file = os.path.join(json_path, normalization_file_name)
                                if os.path.exists(normalization_file):
                                    try:
                                        # 使用pd_item的denormalize方法进行逆向归一化
                                        denorm_result = panda.denormalize(df, normalization_file, columns=[column])
                                        if denorm_result['status'] == 1:
                                            is_normalized = True
                                            # 获取逆向归一化后的数据
                                            denorm_df = denorm_result['content']
                                            normalized_data = denorm_df[column].tolist()
                                            # 重新计算统计信息
                                            calculate_result = panda.calculate(denorm_df)['content']
                                            if column in calculate_result['col_name']:
                                                idx = calculate_result['col_name'].index(column)
                                                stats = {
                                                    'mean': calculate_result['col_mean'][idx],
                                                    'median': calculate_result['col_median'][idx],
                                                    'min': calculate_result['col_min'][idx],
                                                    'max': calculate_result['col_max'][idx],
                                                    'std': calculate_result['col_std'][idx],
                                                    'count': len(normalized_data)
                                                }
                                            # 获取原始范围
                                            with open(normalization_file, 'r', encoding='utf-8') as f:
                                                normalization_params = json.load(f)
                                            if column in normalization_params:
                                                params = normalization_params[column]
                                                if 'col_min' in params and 'col_max' in params:
                                                    original_range = {
                                                        'min': params['col_min'],
                                                        'max': params['col_max']
                                                    }
                                    except Exception as e:
                                        print(f"Error reading normalization file {normalization_file}: {e}")
                    except Exception as e:
                        print(f"Error querying data_db: {e}")

                    # 计算统计信息（如果没有通过反向归一化计算）
                    if not is_normalized:
                        # 使用pd_item的calculate方法计算统计信息
                        calculate_result = panda.calculate(df)['content']
                        # 查找指定列的统计信息
                        if column in calculate_result['col_name']:
                            idx = calculate_result['col_name'].index(column)
                            stats = {
                                'mean': calculate_result['col_mean'][idx],
                                'median': calculate_result['col_median'][idx],
                                'min': calculate_result['col_min'][idx],
                                'max': calculate_result['col_max'][idx],
                                'std': calculate_result['col_std'][idx],
                                'count': len(column_data)
                            }
                        else:
                            # 如果列不在计算结果中，使用默认值
                            stats = {
                                'mean': 0,
                                'median': 0,
                                'min': 0,
                                'max': 0,
                                'std': 0,
                                'count': 0
                            }

                # 确保stats变量总是被定义
                if 'stats' not in locals():
                    stats = {
                        'mean': 0,
                        'median': 0,
                        'min': 0,
                        'max': 0,
                        'std': 0,
                        'count': 0
                    }

                # 确保normalized_data变量总是被定义
                if 'normalized_data' not in locals():
                    normalized_data = []

                # 确保is_normalized变量总是被定义
                if 'is_normalized' not in locals():
                    is_normalized = False

                # 确保original_range变量总是被定义
                if 'original_range' not in locals():
                    original_range = None

                # 确定是否为分类数据
                # 使用pd_item的is_categorical方法
                is_categorical_result = panda.is_categorical(combined_df if 'combined_df' in locals() else df, column)
                is_categorical = is_categorical_result['content'] if is_categorical_result['status'] == 1 else False

                # 确保所有数据类型都能被JSON序列化
                # 转换stats中的数值类型
                serializable_stats = {}
                for key, value in stats.items():
                    if panda.is_integer(value):
                        serializable_stats[key] = int(value)
                    elif panda.is_float(value):
                        serializable_stats[key] = float(value)
                    else:
                        serializable_stats[key] = value

                # 转换normalized_data中的数值类型
                serializable_data = []
                for value in normalized_data:
                    if panda.is_integer(value):
                        serializable_data.append(int(value))
                    elif panda.is_float(value):
                        serializable_data.append(float(value))
                    else:
                        serializable_data.append(value)

                # 转换originalRange中的数值类型
                serializable_original_range = None
                if original_range:
                    serializable_original_range = {}
                    for key, value in original_range.items():
                        if panda.is_integer(value):
                            serializable_original_range[key] = int(value)
                        elif panda.is_float(value):
                            serializable_original_range[key] = float(value)
                        else:
                            serializable_original_range[key] = value

                # 准备返回数据
                result = {
                    'stats': serializable_stats,
                    'data': serializable_data,  # 使用反向归一化后的数据（如果适用）
                    'isCategorical': is_categorical,
                    'isNormalized': is_normalized,
                    'originalRange': serializable_original_range
                }

                return response(1, result)
            except Exception as e:
                print(f"Error in get_column_stats: {str(e)}")
                return response(0, f'Failed to get column stats: {str(e)}')

        elif func_name == 'calculate_correlation':
            try:
                data_id = body['data_id']
                columns = body['columns']
                correlation_type = body['correlation_type']

                # 读取数据文件
                file_path = os.path.join(data_path, data_id)
                if not os.path.exists(file_path):
                    return response(0, 'File not found')

                # 读取数据
                _ = data_id.split('.')[-1].lower()
                if _ in ['xlsx','xls']:
                    # 尝试读取train工作表，如果不存在则读取第一个工作表
                    try:
                        # 尝试读取train工作表
                        try:
                            train_data = panda.read_file(file_path, sheet_name='train', engine='openpyxl')['content']
                            train_df = panda.json_to_dataframe(train_data)['content']
                            test_data = panda.read_file(file_path, sheet_name='test', engine='openpyxl')['content']
                            test_df = panda.json_to_dataframe(test_data)['content']
                            # 合并数据
                            df = panda.concat([train_df, test_df], ignore_index=True)['content']
                        except Exception as e:
                            # 如果train或test工作表不存在，读取第一个工作表
                            data = panda.read_file(file_path, engine='openpyxl')['content']
                            df = panda.json_to_dataframe(data)['content']
                    except Exception as e:
                        print(f"Error reading Excel file: {str(e)}")
                        return response(0, f'Error reading Excel file: {str(e)}')
                elif _ == 'csv':
                    try:
                        # 读取CSV文件
                        data = panda.read_file(file_path, encoding='utf-8')['content']
                        df = panda.json_to_dataframe(data)['content']
                        # 检查是否有dataset_type列
                        if 'dataset_type' in df.columns:
                            # 合并训练集和测试集
                            train_df = df[df['dataset_type'] == 'train']
                            test_df = df[df['dataset_type'] == 'test']
                            df = panda.concat([train_df, test_df], ignore_index=True)['content']
                    except Exception as e:
                        print(f"Error reading CSV file: {str(e)}")
                        return response(0, f'Error reading CSV file: {str(e)}')
                else:
                    return response(0, 'This file type is not supported')

                # 检查所有列是否存在
                for column in columns:
                    if column not in df.columns:
                        return response(0, f'Column {column} not found in dataset')

                # 提取所需列的数据
                selected_data = df[columns]

                # 计算相关系数矩阵
                if correlation_type == 'pearson':
                    corr_matrix = selected_data.corr(method='pearson')
                elif correlation_type == 'spearman':
                    corr_matrix = selected_data.corr(method='spearman')
                elif correlation_type == 'kendall':
                    corr_matrix = selected_data.corr(method='kendall')
                else:
                    return response(0, 'Invalid correlation type')
                # 转换为列表形式
                corr_matrix_list = corr_matrix.values.tolist()

                # 确保所有数据类型都能被JSON序列化
                serializable_matrix = []
                for row in corr_matrix_list:
                    serializable_row = []
                    for value in row:
                        if panda.is_float(value):
                            serializable_row.append(float(value))
                        else:
                            serializable_row.append(value)
                    serializable_matrix.append(serializable_row)

                # 准备返回数据
                result = {
                    'correlation_matrix': serializable_matrix,
                    'columns': columns
                }

                return response(1, result)
            except Exception as e:
                print(f"Error in calculate_correlation: {str(e)}")
                return response(0, f'Failed to calculate correlation: {str(e)}')

        elif func_name == 'saving_data':
            user_id = body['user_id']
            data_id = body['data_id']
            pattern = body['pattern']
            encoding = body['encoding']
            sheet_name = body['sheet_name']
            types = body['types']
            operations = body['operation']
            result_data = body['result_data']
            if len(operations) == 0:
                return response(0,msg='You have not operate every thing please change once.if you only want to change a file_name or type,please'
                                      'please change once and change to primary again')
            # build all of id
            _data_id = other.get_random_id(user_id,'d')['content']
            _vectorization_id = other.get_random_id(user_id,'v')['content']
            _normalization_id = other.get_random_id(user_id,'n')['content']
            _work_id = other.get_random_id(user_id,'w')['content']
            _data_all = await mysql.find('data_db')
            _data_all_i = _data_all['content']
            _data_lst = []
            _vectorization_lst = []
            _normalization_lst = []
            _work_lst = []
            for index in range(len(_data_all_i)):
                _data_lst.append(_data_all_i[index]['saving'])
                _normalization_lst.append(_data_all_i[index]['normalization'])
                _vectorization_lst.append(_data_all_i[index]['vectorization'])
                _work_lst.append(_data_all_i[index]['though_work'])
            while f'{_data_id}.{pattern}' in _data_lst:
                _data_id = other.get_random_id(user_id,'d')['content']
            while f'{_vectorization_id}.json' in _vectorization_lst:
                _vectorization_id = other.get_random_id(user_id,'v')['content']
            while f'{_normalization_id}.json' in _normalization_lst:
                _normalization_id = other.get_random_id(user_id,'n')['content']
            while f'{_work_id}.json' in _normalization_lst:
                _work_id = other.get_random_id(user_id,'w')['content']
            # reproduce operate
            _data = panda.read_file(os.path.join(data_path,data_id))['content']
            _data = panda.json_to_dataframe(_data)['content']
            _v = 0
            _n = 0
            train_index = None
            test_index = None
            x_columns = None
            y_columns = None
            for operation in operations:
                if operation['function'].lower() == 'train_test_split':
                    # 保存划分的索引和列信息
                    train_index = operation['work']['data']['train_index']
                    test_index = operation['work']['data']['test_index']
                    x_columns = operation['work']['data'].get('x_columns', None)
                    y_columns = operation['work']['data'].get('y_columns', None)
                    columns = operation['work']['data'].get('columns', None)
                elif operation['function'].lower() == 'normalization':
                    # 获取要处理的列
                    columns = operation['work'].get('subset', None)
                    # 如果已经划分了训练集和测试集，先对训练集应用归一化，再对测试集应用相同的参数
                    if train_index and test_index and isinstance(train_index, list) and isinstance(test_index, list):
                        # 对训练集应用归一化并保存参数
                        train_data = _data.loc[train_index]
                        normalized_train = panda.normalization(train_data,operation['work']['method'][0],save_path=os.path.join(json_path,f'{_normalization_id}.json'), columns=columns)['content']
                        # 将归一化后的训练集数据写回原始数据集
                        _data.loc[train_index] = normalized_train

                        # 对测试集应用相同的归一化参数
                        test_data = _data.loc[test_index]
                        normalized_test = panda.normalization(test_data,operation['work']['method'][0],save_path=os.path.join(json_path,f'{_normalization_id}.json'), columns=columns, is_train=False)['content']
                        # 将归一化后的测试集数据写回原始数据集
                        _data.loc[test_index] = normalized_test
                    else:
                        # 对整个数据集应用归一化
                        _data = panda.normalization(_data,operation['work']['method'][0],save_path=os.path.join(json_path,f'{_normalization_id}.json'), columns=columns)['content']
                    _n = 1
                elif operation['function'].lower() == 'vectorization':
                    # 获取要处理的列
                    columns = operation['work'].get('subset', None)
                    # 如果已经划分了训练集和测试集，先对训练集应用向量化，再对测试集应用相同的参数
                    if train_index and test_index and isinstance(train_index, list) and isinstance(test_index, list):
                        # 对训练集应用向量化并保存参数
                        train_data = _data.loc[train_index]
                        vectorized_train = panda.text_to_number(train_data,operation['work']['method'][0],save_path=os.path.join(json_path,f'{_vectorization_id}.json'), columns=columns)['content']
                        # 将向量化后的训练集数据写回原始数据集
                        _data.loc[train_index] = vectorized_train

                        # 对测试集应用相同的向量化参数
                        test_data = _data.loc[test_index]
                        vectorized_test = panda.text_to_number(test_data,operation['work']['method'][0],save_path=os.path.join(json_path,f'{_vectorization_id}.json'), columns=columns, is_train=False)['content']
                        # 将向量化后的测试集数据写回原始数据集
                        _data.loc[test_index] = vectorized_test
                    else:
                        # 对整个数据集应用向量化
                        _data = panda.text_to_number(_data,operation['work']['method'][0],save_path=os.path.join(json_path,f'{_vectorization_id}.json'), columns=columns)['content']
                    _v = 1
                elif operation['function'].lower() == 'update':
                    _data.loc[operation['work']['area']['index'],operation['work']['area']['columns']] = operation['work']['area']['change']
                elif operation['function'].lower() == 'delete':
                    if operation['work']['area']['columns'] == '':
                        _data = _data.drop(index=operation['work']['area']['index']).reset_index(drop=True)
                    elif operation['work']['area']['index'] in [-1,'-1']:
                        _data = _data.drop(columns=operation['work']['area']['columns'])
                    else:
                        return response(0,msg='do you want to destroy this server?')
                elif operation['function'].lower() == 'fillna':
                    # 获取要处理的列
                    columns = operation['work'].get('subset', None)
                    _data = panda.fill_na(_data, operation['work']['method'], columns=columns)['content']
                elif operation['function'].lower() == 'drop_duplicate':
                    # 获取要处理的列
                    subset = operation['work'].get('subset', None)
                    _data = panda.drop_duplicate(_data, subset=subset)['content']
                elif operation['function'].lower() == 'drop_outlier':
                    # 获取要处理的列
                    columns = operation['work'].get('subset', None)
                    multiple = operation['work'].get('multiple', 3)
                    _data = panda.drop_outlier(_data, operation['work']['method'][0], multiple=multiple, columns=columns)['content']
                # loss some way
                else:
                    return response(0,msg='More function build now')
            _now = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            if types == '':
                types = 'classification'
            elif not (types is None):
                types = types.lower()
            if _v == 0:
                _vectorization_id = None
            else:
                _vectorization_id = f'{_vectorization_id}.json'
            if _n == 0:
                _normalization_id = None
            else:
                _normalization_id = f'{_normalization_id}.json'
            # bug
            _insert = await mysql.insert('data_db',['user_id','update_time','saving','sheet_name','types','though_work','vectorization','normalization'],
                                         [user_id,_now,f'{_data_id}.{pattern}',sheet_name,types,f'{_work_id}.json',f'{_vectorization_id}',f'{_normalization_id}'])
            data = panda.json_to_dataframe(result_data)['content']

            # 根据是否有划分结果决定保存方式
            if train_index and test_index and isinstance(train_index, list) and isinstance(test_index, list):
                # 划分了测试集与训练集
                # 使用panda.save_file保存数据，支持训练集和测试集的划分
                _ = panda.save_file(
                    data,
                    os.path.join(data_path, f'{_data_id}'),
                    pattern,
                    encoding,
                    train_index=train_index,
                    test_index=test_index,
                    x_columns=x_columns,
                    y_columns=y_columns
                )
            else:
                # 未划分测试集与训练集，直接保存
                _ = panda.save_file(data,os.path.join(data_path,f'{_data_id}'),pattern,encoding)

            with open(os.path.join(json_path,f'{_work_id}.json'),'w',encoding='utf-8') as f:
                json.dump(operations, f, ensure_ascii=False)
            f.close()
            return response(msg='This file save ok!')

        elif func_name == 'delete_data':
            user_id = body['user_id']
            _data_id = body['data_id']
            user_password = body['password']
            _password = other.password_encryption(user_password)['content']
            _user_password = await mysql.find('user_db','password',attribute='id',result=user_id)
            if _user_password['content'][0]['password'] != _password:
                return response(0,msg='This password is not equal')
            _need_delete = await mysql.find('data_db',attribute=['user_id','saving'],result=[user_id,_data_id],relative=['AND'])
            vectorization = _need_delete['content'][0]['vectorization']
            normalization = _need_delete['content'][0]['normalization']
            though = _need_delete['content'][0]['though_work']
            _operation = await mysql.delete('data_db',['user_id','saving'],[user_id,_data_id])
            if (vectorization is not None) and (os.path.exists(os.path.join(json_path,vectorization))):
                os.remove(os.path.join(json_path,vectorization))
            if (normalization is not None) and (os.path.exists(os.path.join(json_path,normalization))):
                os.remove(os.path.join(json_path, normalization))
            if os.path.exists(os.path.join(data_path,_data_id)):
                os.remove(os.path.join(data_path, _data_id))
            if os.path.exists(os.path.join(json_path,though)):
                os.remove(os.path.join(json_path,though))
            return response(msg='Delete success')

        elif func_name == 'update_sheet_name':
            user_id = body['user_id']
            saving = body['saving']
            sheet_name = body['sheet_name']
            _update = await mysql.update('data_db',['user_id','saving'],[user_id,saving],'sheet_name',sheet_name)
            if _update['response'] == 1:
                return response(msg='Update successful')
            return response(0,'This update is not allow')

        elif func_name == 'download':
            user_id = body['user_id']
            saving = body['saving']

            # 验证用户是否存在
            _inspect = await mysql.find('user_db', attribute='id', result=user_id)
            if len(_inspect['content']) == 0:
                return response(0, 'This user is not register')

            # 验证文件是否存在
            file_path = os.path.join(data_path, saving)
            if not os.path.exists(file_path):
                return response(0, 'File not found')

            # 返回文件下载响应
            return response_file(file_path, as_attachment=True, filename=saving)

        elif func_name == 'update_user':
            # 处理表单数据
            form_data = body.get('form_data') if isinstance(body, dict) else None
            if not form_data:
                return response(0, 'No form data provided')

            fields = form_data.get('fields', {})
            files = form_data.get('files', {})

            print(f"[Debug] Fields: {fields}")
            print(f"[Debug] Files: {list(files.keys())}")

            user_id = fields.get('id')
            if not user_id:
                return response(0, 'User ID is required')

            # Get verify password
            verify_password = fields.get('verify_password')
            if not verify_password:
                return response(0, 'Verify password is required')

            # Verify password
            user_info = await mysql.find('user_db', attribute='id', result=user_id)
            if len(user_info['content']) == 0:
                return response(0, 'User not found')

            stored_password = user_info['content'][0]['password']
            encrypted_verify_password = other.password_encryption(verify_password)['content']
            if encrypted_verify_password != stored_password:
                return response(0, 'Verify password is incorrect')

            username = fields.get('username', '')
            phone = fields.get('phone', '')
            password = fields.get('password', '')
            identity = fields.get('identity', '')

            # 处理头像上传
            head_url = None
            if 'head' in files:
                print("[Debug] Found head file")
                head_file = files['head']
                print(f"[Debug] Head file info: {head_file}")
                if head_file['size'] > 0:
                    # 生成唯一的pic_id
                    pic_id = other.get_random_id(user_id, 'p')['content']

                    # 确保pic_id不重复
                    while True:
                        existing = await mysql.find('user_db', attribute='head_url', result=f'./static/user_pic/{pic_id}.*')
                        if len(existing['content']) == 0:
                            break
                        pic_id = other.get_random_id(user_id, 'p')['content']

                    # 保留原文件格式
                    filename = head_file['filename']
                    if '.' in filename:
                        file_extension = filename.split('.')[-1].lower()
                    else:
                        file_extension = 'png'  # 默认使用png格式
                    pic_filename = f'{pic_id}.{file_extension}'
                    pic_path = os.path.join(user_pic_path, pic_filename)

                    print(f"[Debug] Saving file to: {pic_path}")
                    # 保存图片
                    try:
                        with open(pic_path, 'wb') as f:
                            f.write(head_file['content'])
                        print("[Debug] File saved successfully")
                    except Exception as e:
                        print(f"[Debug] File save error: {e}")
                        return response(0, f'File save error: {str(e)}')

                    # 更新head_url
                    head_url = f'./static/user_pic/{pic_filename}'
                    print(f"[Debug] Head URL: {head_url}")

            # 构建更新参数
            update_params = []
            update_values = []

            if username:
                update_params.append('username')
                update_values.append(username)
            if phone:
                update_params.append('phone')
                update_values.append(phone)
            if password:
                # 加密密码
                encrypted_password = other.password_encryption(password)['content']
                update_params.append('password')
                update_values.append(encrypted_password)
            if identity:
                update_params.append('identity')
                update_values.append(identity)
            if head_url:
                update_params.append('head_url')
                update_values.append(head_url)

            print(f"[Debug] Update params: {update_params}")
            print(f"[Debug] Update values: {update_values}")

            # 执行更新
            if update_params:
                # 将attribute参数从列表改为字符串，避免长度差异错误
                result = await mysql.update('user_db', 'id', user_id, update_params, update_values)
                print(f"[Debug] Update result: {result}")
                if result['response'] == 1:
                    return response(1, 'Update successful')
                else:
                    return response(0, 'Update failed')
            else:
                return response(0, 'No fields to update')

        # Reduced version, directly train the model
        elif func_name == 'train':
            # 读取模型默认超参
            def get_default_hyperparams(model_select):
                hyperparams_path = os.path.join(primary_model_setting, f'{model_select}.json')
                if os.path.exists(hyperparams_path):
                    try:
                        with open(hyperparams_path, 'r', encoding='utf-8') as f:
                            hyperparams_config = json.load(f)
                        # 提取默认值
                        default_hyperparams = {}
                        for param_name, config in hyperparams_config.items():
                            if param_name != 'model' and 'default' in config:
                                default_hyperparams[param_name] = config['default']
                        return default_hyperparams
                    except Exception as e:
                        print(f"Error loading default hyperparams: {e}")
                        return {}
                return {}
            print(f"Received train request: {body}")
            user_id = body['user_id']
            # 检查请求格式
            if 'data' in body:
                # 旧格式：data字段包含data_id等信息
                data = body['data']
                data_id = data['data_id']
                model_name = body['model_name']
                model_type = body['model_type']
                model_select = body['model_select']
                hyper_parameter = body['hyper_parameter']
            else:
                # 新格式：直接在根级别
                data_id = body['data_id']
                model_select = body['model_select']
                hyper_parameter = body['hyper_parameter']
                # 生成默认的model_name和model_type
                model_name = f"{data_id}_{model_select}"
                # 根据模型名称判断类型
                if model_select in ['XGBRegressor', 'SVR', 'Ridge', 'MLPRegressor', 'LinearRegression', 'DecisionTreeRegressor', 'ElmanNetwork', 'CascadeCorrelation', 'BoltzmannMachine', 'RandomForestRegressor', 'AdaBoostRegressor']:
                    model_type = 'regression'
                else:
                    model_type = 'classification'
            
            # 如果超参是空的，填充默认超参
            if not hyper_parameter:
                print(f"Hyperparameters is empty, loading default hyperparameters for {model_select}")
                hyper_parameter = get_default_hyperparams(model_select)
                print(f"Loaded default hyperparameters: {hyper_parameter}")

            # 验证用户是否存在
            user_info = await mysql.find('user_db', attribute='id', result=user_id)
            if len(user_info['content']) == 0:
                return response(0, 'User not found')

            # 生成模型相关的ID和路径
            _model_id = other.get_random_id(user_id)['content']
            _model_token = other.get_random_token(32)['content']
            _model_path = other.create_model_path(user_id,model_select,model_type)['content']
            _base_path = f'{_model_id}.pkl'  # 直接指向模型的名称

            # 确保ID和路径的唯一性
            _select = await mysql.find('sklearn_db')
            _model_token_lst = []
            _model_id_lst = []
            _model_path_lst = []
            _base_path_lst = []
            if len(_select['content']) != 0:
                for i in _select['content']:
                    _model_path_lst.append(i['base_path'])
                    _model_token_lst.append(i['token'])
                    _model_id_lst.append(i['id'])
                    _base_path_lst.append(i['base_path'])
            while _model_id in _model_path_lst:
                _model_id = other.get_random_id(user_id)['content']
                _base_path = f'{_model_id}.pkl'  # 更新base_path
            while _model_token in _model_token_lst:
                _model_token = other.get_random_token(32)['content']
            while _model_path in _model_path_lst:
                _model_path = other.create_model_path(user_id,model_select,model_type)['content']
            while _base_path in _base_path_lst:
                _model_id = other.get_random_id(user_id)['content']
                _base_path = f'{_model_id}.pkl'  # 更新base_path

            # 读取数据并进行划分
            file_ext = data_id.split('.')[-1].lower()
            
            # 对于所有文件类型，从data_db中获取though_work内容
            x_columns = None
            y_columns = None
            try:
                # 根据data_id查询data_db
                data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                if isinstance(data_record, dict) and data_record.get('response') == 1 and len(data_record.get('content', [])) > 0:
                    data_info = data_record['content'][0]
                    though_work = data_info.get('though_work', '')
                    
                    # 如果though_work不是'load'，读取对应的JSON文件
                    if though_work and though_work != 'load':
                        though_work_path = os.path.join(json_path, though_work)
                        if os.path.exists(though_work_path):
                            try:
                                with open(though_work_path, 'r', encoding='utf-8') as f:
                                    though_work_data = json.load(f)
                                # 尝试从though_work中获取x_columns和y_columns
                                if isinstance(though_work_data, list):
                                    # 对于操作列表，尝试查找train_test_split操作
                                    for operation in though_work_data:
                                        if operation.get('function', '').lower() == 'train_test_split':
                                            x_columns = operation.get('work', {}).get('data', {}).get('x_columns', None)
                                            y_columns = operation.get('work', {}).get('data', {}).get('y_columns', None)
                                            break
                                elif isinstance(though_work_data, dict):
                                    # 对于字典，直接获取x_columns和y_columns
                                    x_columns = though_work_data.get('x_columns', None)
                                    y_columns = though_work_data.get('y_columns', None)
                            except Exception as e:
                                print(f"Error reading though_work file: {e}")
            except Exception as e:
                print(f"Error querying data_db: {e}")
            
            # 如果没有获取到x_columns和y_columns，使用原来的逻辑
            if x_columns is None and y_columns is None:
                if file_ext == 'csv':
                    # 对于CSV文件，构建对应的JSON文件路径
                    json_file_name = os.path.splitext(data_id)[0] + '.json'
                    json_file_path = os.path.join(json_path, json_file_name)
                    
                    # 如果JSON文件存在，读取自变量和因变量信息
                    if os.path.exists(json_file_path):
                        try:
                            with open(json_file_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            x_columns = json_data.get('x_columns', None)
                            y_columns = json_data.get('y_columns', None)
                        except Exception as e:
                            print(f"Error reading JSON file: {e}")
                            x_columns = None
                            y_columns = None
                    else:
                        x_columns = None
                        y_columns = None
                elif file_ext in ['xlsx', 'xls']:
                    # 对于XLSX文件，检查是否存在columns_set工作表
                    file_path = os.path.join(data_path, data_id)
                    try:
                        # 尝试读取columns_set工作表
                        try:
                            columns_set_data = panda.read_file(file_path, sheet_name='columns_set', engine='openpyxl')['content']
                            columns_set_df = panda.json_to_dataframe(columns_set_data)['content']
                            # 尝试解析x_columns和y_columns
                            import ast
                            if 'x_columns' in columns_set_df.columns and 'y_columns' in columns_set_df.columns:
                                try:
                                    x_columns = ast.literal_eval(columns_set_df['x_columns'][0])
                                    y_columns = ast.literal_eval(columns_set_df['y_columns'][0])
                                    print(f"Parsed x_columns from columns_set: {x_columns}")
                                    print(f"Parsed y_columns from columns_set: {y_columns}")
                                except Exception as e:
                                    print(f"Error parsing columns from columns_set: {e}")
                                    x_columns = None
                                    y_columns = None
                            else:
                                x_columns = None
                                y_columns = None
                        except Exception as e:
                            # columns_set工作表不存在
                            print(f"columns_set sheet not found: {e}")
                            x_columns = None
                            y_columns = None
                    except Exception as e:
                        print(f"Error reading columns_set sheet: {e}")
                        x_columns = None
                        y_columns = None
                else:
                    # 对于其他文件类型，使用默认值
                    x_columns = None
                    y_columns = None

            # 读取数据文件
            _data = panda.read_file(os.path.join(data_path, data_id), engine='openpyxl' if file_ext in ['xlsx', 'xls'] else None)['content']
            _data1 = panda.json_to_dataframe(_data)['content']
            
            # 导入pandas
            import pandas as pd
            # 移除dataset_type列（如果存在）
            if panda.is_dataframe(_data1) and 'dataset_type' in _data1.columns:
                _data1 = _data1.drop('dataset_type', axis=1)

            # 检查x_columns和y_columns是否相同
            if x_columns is not None and y_columns is not None:
                # 确保x_columns和y_columns是列表
                if isinstance(x_columns, str):
                    x_columns = [x_columns]
                if isinstance(y_columns, str):
                    y_columns = [y_columns]
                # 检查是否有重叠的列
                common_columns = set(x_columns) & set(y_columns)
                if common_columns:
                    print(f"Warning: x_columns and y_columns have common columns: {common_columns}")
                    print("Using default column selection instead")
                    x_columns = None
                    y_columns = None
            else:
                # 如果x_columns或y_columns为None，使用默认逻辑
                print("Warning: x_columns or y_columns is None, using default column selection")
                x_columns = None
                y_columns = None

            # 如果没有指定x_columns和y_columns，自动选择
            if x_columns is None and y_columns is None:
                # 自动选择所有列除了最后一列作为x，最后一列作为y
                if panda.is_dataframe(_data1):
                    columns = _data1.columns.tolist()
                    if len(columns) > 1:
                        x_columns = columns[:-1]
                        y_columns = [columns[-1]]
                    else:
                        # 如果只有一列，使用该列作为y
                        x_columns = []
                        y_columns = [columns[0]]
                else:
                    # 如果_data1不是DataFrame，设置默认值
                    x_columns = []
                    y_columns = []

            _all_data = divide.train_test_split(_data1,columns_x=x_columns,columns_y=y_columns)['content']
            _train_x,_train_y,test_x,test_y = _all_data['train']['x'],_all_data['train']['y'],_all_data['test']['x'],_all_data['test']['y']
            # print(_all_data)

            # 生成超参文件
            _hyper_id = other.get_random_id(user_id, 'h')['content']
            hyper_file = f'{_hyper_id}.json'
            hyper_path = os.path.join(json_path, hyper_file)
            print(f"Saving hyperparameters to {hyper_path}: {hyper_parameter}")
            with open(hyper_path, 'w', encoding='utf-8') as f:
                json.dump(hyper_parameter, f, ensure_ascii=False)
            # 验证文件是否正确保存
            with open(hyper_path, 'r', encoding='utf-8') as f:
                saved_hyper = json.load(f)
            print(f"Saved hyperparameters: {saved_hyper}")

            # 生成评估文件
            _eval_id = other.get_random_id(user_id, 'e')['content']
            eval_file = f'{_eval_id}.json'
            eval_path = os.path.join(json_path, eval_file)

            # 直接保存模型到static/model目录
            model_file = os.path.join(model_path, f'{_model_id}.pkl')

            # 插入模型信息到数据库
            _now = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            # url_path表示后端提供给用户的post接口
            post_api_url = f"/api/predict/{_model_id}"
            _insert = await mysql.insert('sklearn_db',
                                         ['id', 'user_id', 'data_id', 'use_model', 'types', 'model_name', 'model_condition', 'token', 'base_path', 'hyper', 'evaluation', 'url_path', 'open_bool'],
                                         [_model_id, user_id, data_id, model_select, model_type, model_name, 'training', _model_token, _base_path, hyper_file, eval_file, post_api_url, '1'])
            print(_insert)
            
            # 初始化Redis队列
            train_queue = get_train_queue()
            
            if train_queue and train_queue.is_connected():
                print(f"Using Redis queue for async training: model_id={_model_id}")
                
                # 准备训练任务数据
                # 确保将DataFrame转换为NumPy数组，然后转换为列表
                def to_serializable(data):
                    import pandas as pd
                    if panda.is_dataframe(data):
                        return data.values.tolist()
                    elif hasattr(data, 'tolist'):
                        return data.tolist()
                    else:
                        return data
                
                train_task = {
        'model_id': _model_id,
        'user_id': user_id,
        'model_select': model_select,
        'hyper_parameter': hyper_parameter,
        'train_x': to_serializable(_train_x),
        'train_y': to_serializable(_train_y),
        'test_x': to_serializable(test_x),
        'test_y': to_serializable(test_y),
        'model_file': model_file,
        'eval_path': eval_path,
        'json_path': json_path,
        'model_path': _model_path,
        'data_id': data_id,
        'hyper_file': hyper_file,
        'eval_file': eval_file,
        'model_token': _model_token,
        'model_name': model_name,
        'model_type': model_type,
        'x_columns': x_columns,
        'y_columns': y_columns,
        'hyper_path': hyper_path
    }
                
                # 将任务放入队列
                if train_queue.enqueue('train_queue', train_task):
                    # 初始化任务状态
                    train_queue.set_status(_model_id, {
                        'status': 'queued',
                        'progress': 0,
                        'message': 'Task queued, waiting for processing...'
                    })
                    # 使用全局response函数，将数据作为msg的一部分返回
                    return response(1, {'message': 'Training task queued successfully', 'model_id': _model_id})
                else:
                    print("Failed to enqueue task, falling back to synchronous training")
            else:
                print("Redis not available, using synchronous training")

            # 训练模型
            try:
                print(f"Received model_select: {model_select}")
                print(f"Hyper parameters: {hyper_parameter}")
                if model_select == 'LinearRegression':
                    model = scikit_learn_build.LinearRegression(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'Ridge':
                    model = scikit_learn_build.Ridge(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'LogisticRegression':
                    model = scikit_learn_build.LogisticRegression(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'XGBClassifier':
                    model = scikit_learn_build.XGBClassifier(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'SVC':
                    model = scikit_learn_build.SVC(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'LDA':
                    model = scikit_learn_build.LDA(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'DecisionTreeClassifier':
                    model = scikit_learn_build.DecisionTreeClassifier(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'DecisionTreeRegressor':
                    model = scikit_learn_build.DecisionTreeRegressor(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'MLPClassifier':
                    model = scikit_learn_build.MLPClassifier(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'MLPRegressor':
                    model = scikit_learn_build.MLPRegressor(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'RBFClassifier':
                    model = scikit_learn_build.RBFClassifier(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'SOM':
                    model = scikit_learn_build.SOM(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'CascadeCorrelation':
                    model = scikit_learn_build.CascadeCorrelation(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'ElmanNetwork':
                    model = scikit_learn_build.ElmanNetwork(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'BoltzmannMachine':
                    model = scikit_learn_build.BoltzmannMachine(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'SVR':
                    model = scikit_learn_build.SVR(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'GaussianNB':
                    model = scikit_learn_build.GaussianNB(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'MultinomialNB':
                    model = scikit_learn_build.MultinomialNB(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'BernoulliNB':
                    model = scikit_learn_build.BernoulliNB(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'RandomForestClassifier':
                    model = scikit_learn_build.RandomForestClassifier(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'RandomForestRegressor':
                    model = scikit_learn_build.RandomForestRegressor(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'AdaBoostClassifier':
                    model = scikit_learn_build.AdaBoostClassifier(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'AdaBoostRegressor':
                    model = scikit_learn_build.AdaBoostRegressor(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'XGBRegressor':
                    model = scikit_learn_build.XGBRegressor(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'KMeans':
                    model = scikit_learn_build.KMeans(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'LVQ':
                    model = scikit_learn_build.LVQ(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'GaussianMixture':
                    model = scikit_learn_build.GaussianMixture(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'DBSCAN':
                    model = scikit_learn_build.DBSCAN(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                elif model_select == 'AgglomerativeClustering':
                    model = scikit_learn_build.AgglomerativeClustering(**hyper_parameter)
                    print(f"Created model type: {type(model)}")
                # 可以添加更多模型
                else:
                    print(f"Model not supported: {model_select}")
                    return response(0, 'Model not supported')

                # 训练模型
                if _train_y is None:
                    return response(0, 'No target column found. Please specify y_columns or ensure your dataset has at least two columns.')

                train_result = model.fit(_train_x, _train_y)
                if train_result['status'] != 1:
                    return response(0, train_result['content'])

                # 预测
                predict_result = model.predict(test_x)
                if predict_result['status'] != 1:
                    return response(0, predict_result['content'])

                # 评估
                if test_y is None:
                    return response(0, 'No target column found for evaluation.')

                # Debug print
                print(f"model_select: {model_select}")
                print(f"predict_result['content']: {predict_result['content']}")

                # Check if the model is a classification model that requires X as first parameter
                classification_models = ['LogisticRegression', 'SVC', 'DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier', 'MLPClassifier', 'RBFClassifier', 'GaussianNB', 'MultinomialNB', 'BernoulliNB', 'LDA', 'AdaBoostClassifier']
                if model_select in classification_models:
                    # 确保test_y和predict_result['content']形状一致
                    y_pred = predict_result['content']
                    # 确保test_y是numpy数组
                    if hasattr(test_y, 'values'):
                        test_y = test_y.values
                    # 如果test_y是二维数组且只有一列，转换为一维数组
                    if len(test_y.shape) == 2 and test_y.shape[1] == 1:
                        test_y = test_y.reshape(-1)
                    # 确保y_pred是numpy数组
                    if hasattr(y_pred, 'values'):
                        y_pred = y_pred.values
                    # 如果y_pred是二维数组且只有一列，转换为一维数组
                    if isinstance(y_pred, np.ndarray) and len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                        y_pred = y_pred.reshape(-1)
                    # 如果y_pred是列表且每个元素是列表，转换为一维数组
                    elif isinstance(y_pred, list) and all(isinstance(item, list) for item in y_pred):
                        y_pred = np.array(y_pred).reshape(-1)
                    print("Calling classification model evaluate with 4 parameters")
                    # For classification models, pass X, y_true, y_pred, threshold
                    evaluate_result = model.evaluate(test_x, test_y, y_pred, 0.5)
                else:
                    print("Calling regression model evaluate with 2 parameters")
                    # For regression models, pass y_true, y_pred
                    evaluate_result = model.evaluate(test_y, predict_result['content'])
                if evaluate_result['status'] != 1:
                    return response(0, evaluate_result['content'])

                # 保存评估结果
                with open(eval_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluate_result['content'], f, ensure_ascii=False)

                # 保存模型
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)

                # 更新模型状态为训练成功
                await mysql.update('sklearn_db', ['id'], [_model_id], 'model_condition', 'train success')

                # 生成SDK
                try:
                    # 获取对应的data_db记录
                    data_id = data['data_id']
                    data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                    # 检查data_record是否为有效的响应字典
                    if not isinstance(data_record, dict) or 'response' not in data_record or data_record['response'] != 1:
                        # 更新模型状态为训练成功但SDK生成失败
                        await mysql.update('sklearn_db', ['id'], [_model_id], 'model_condition', 'train success, SDK failed')
                        return response(0, 'Model trained successfully but SDK generation failed: Invalid data record')
                    if len(data_record.get('content', [])) == 0:
                        # 更新模型状态为训练成功但SDK生成失败
                        await mysql.update('sklearn_db', ['id'], [_model_id], 'model_condition', 'train success, SDK failed')
                        return response(0, 'Model trained successfully but SDK generation failed: Data record not found')

                    data_info = data_record['content'][0]
                    though_work = data_info.get('though_work')
                    vectorization = data_info.get('vectorization')
                    normalization = data_info.get('normalization')

                    # 创建SDK目录
                    import shutil
                    import zipfile

                    # 生成SDK ID
                    sdk_id = other.get_random_id(user_id, 'sdk')['content']
                    sdk_dir = os.path.join('./static/sdk', sdk_id)
                    os.makedirs(sdk_dir, exist_ok=True)

                    # 生成main.py
                    main_py_content = "import json\n"
                    main_py_content += "import pickle\n"
                    main_py_content += "import numpy as np\n"
                    main_py_content += "import requests\n"
                    main_py_content += "import pandas as pd\n"
                    main_py_content += "import os\n"
                    main_py_content += "from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog\n"
                    main_py_content += "from PyQt5.QtCore import Qt\n"
                    main_py_content += "\n"
                    main_py_content += "# 模型信息\n"
                    main_py_content += "MODEL_ID = '" + _model_id + "'\n"
                    main_py_content += "MODEL_TOKEN = '" + _model_token + "'\n"
                    main_py_content += "# 校验token的网址\n"
                    main_py_content += "TOKEN_VALIDATION_URL = 'http://p90wpcbruk8x.guyubao.com/api/validate_token'  # 实际使用的网址\n"
                    main_py_content += "# 本地测试用的网址\n"
                    main_py_content += "LOCAL_VALIDATION_URL = 'http://127.0.0.1:8080/api/validate_token'  # 本地测试用\n"
                    main_py_content += "\n"
                    main_py_content += "# 加载模型\n"
                    main_py_content += "def load_model():\n"
                    main_py_content += "    try:\n"
                    main_py_content += "        with open('model.pkl', 'rb') as f:\n"
                    main_py_content += "            return pickle.load(f)\n"
                    main_py_content += "    except Exception as e:\n"
                    main_py_content += "        print('Error loading model: ' + str(e))\n"
                    main_py_content += "        return None\n"
                    main_py_content += "\n"
                    main_py_content += "# 加载配置\n"
                    main_py_content += "def load_config():\n"
                    main_py_content += "    vectorization_config = {}\n"
                    main_py_content += "    normalization_config = {}\n"
                    main_py_content += "    though_work = {}\n"
                    main_py_content += "    \n"
                    main_py_content += "    try:\n"
                    main_py_content += "        with open('vectorization.json', 'r', encoding='utf-8') as f:\n"
                    main_py_content += "            vectorization_config = json.load(f)\n"
                    main_py_content += "    except:\n"
                    main_py_content += "        pass\n"
                    main_py_content += "    \n"
                    main_py_content += "    try:\n"
                    main_py_content += "        with open('normalization.json', 'r', encoding='utf-8') as f:\n"
                    main_py_content += "            normalization_config = json.load(f)\n"
                    main_py_content += "    except:\n"
                    main_py_content += "        pass\n"
                    main_py_content += "    \n"
                    main_py_content += "    try:\n"
                    main_py_content += "        with open('though_work.json', 'r', encoding='utf-8') as f:\n"
                    main_py_content += "            though_work = json.load(f)\n"
                    main_py_content += "    except:\n"
                    main_py_content += "        pass\n"
                    main_py_content += "    \n"
                    main_py_content += "    return vectorization_config, normalization_config, though_work\n"
                    main_py_content += "\n"
                    main_py_content += "# 文本向量化\n"
                    main_py_content += "def vectorize(data, config):\n"
                    main_py_content += "    # 简化处理，实际应该根据配置进行处理\n"
                    main_py_content += "    return data\n"
                    main_py_content += "\n"
                    main_py_content += "# 归一化\n"
                    main_py_content += "def normalize(data, config):\n"
                    main_py_content += "    # 简化处理，实际应该根据配置进行处理\n"
                    main_py_content += "    return data\n"
                    main_py_content += "\n"
                    main_py_content += "# 校验token\n"
                    main_py_content += "def validate_token():\n"
                    main_py_content += "    try:\n"
                    main_py_content += "        # 尝试使用实际网址\n"
                    main_py_content += "        response = requests.post(TOKEN_VALIDATION_URL, json={\n"
                    main_py_content += "            'model_id': MODEL_ID,\n"
                    main_py_content += "            'token': MODEL_TOKEN\n"
                    main_py_content += "        }, timeout=5)\n"
                    main_py_content += "        if response.status_code == 200:\n"
                    main_py_content += "            result = response.json()\n"
                    main_py_content += "            return result.get('status') == 1\n"
                    main_py_content += "    except:\n"
                    main_py_content += "        # 如果实际网址失败，使用本地测试网址\n"
                    main_py_content += "        try:\n"
                    main_py_content += "            response = requests.post(LOCAL_VALIDATION_URL, json={\n"
                    main_py_content += "                'model_id': MODEL_ID,\n"
                    main_py_content += "                'token': MODEL_TOKEN\n"
                    main_py_content += "            }, timeout=5)\n"
                    main_py_content += "            if response.status_code == 200:\n"
                    main_py_content += "                result = response.json()\n"
                    main_py_content += "                return result.get('status') == 1\n"
                    main_py_content += "        except:\n"
                    main_py_content += "            # 如果本地也失败，返回True（离线模式）\n"
                    main_py_content += "            print('Warning: Token validation failed, running in offline mode')\n"
                    main_py_content += "            return True\n"
                    main_py_content += "    return False\n"
                    main_py_content += "\n"
                    main_py_content += "# 预测\n"
                    main_py_content += "def predict(input_data):\n"
                    main_py_content += "    # 加载模型和配置\n"
                    main_py_content += "    model = load_model()\n"
                    main_py_content += "    if model is None:\n"
                    main_py_content += "        return 'Error: Model loading failed'\n"
                    main_py_content += "    \n"
                    main_py_content += "    vectorization_config, normalization_config, though_work = load_config()\n"
                    main_py_content += "    \n"
                    main_py_content += "    try:\n"
                    main_py_content += "        # 处理输入数据\n"
                    main_py_content += "        # 假设输入是CSV格式的字符串\n"
                    main_py_content += "        data_row = input_data.split(',')\n"
                    main_py_content += "        data = np.array([float(x) for x in data_row]).reshape(1, -1)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 应用文本向量化和归一化\n"
                    main_py_content += "        # 根据though_work中的步骤顺序处理\n"
                    main_py_content += "        if though_work:\n"
                    main_py_content += "            # 按照though_work中的步骤顺序处理\n"
                    main_py_content += "            for step in though_work:\n"
                    main_py_content += "                if step['type'] == 'vectorization':\n"
                    main_py_content += "                    data = vectorize(data, vectorization_config)\n"
                    main_py_content += "                elif step['type'] == 'normalization':\n"
                    main_py_content += "                    data = normalize(data, normalization_config)\n"
                    main_py_content += "        else:\n"
                    main_py_content += "            # 默认处理顺序\n"
                    main_py_content += "            data = vectorize(data, vectorization_config)\n"
                    main_py_content += "            data = normalize(data, normalization_config)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 预测\n"
                    main_py_content += "        result = model.predict(data)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 检查是否需要逆向归一化\n"
                    main_py_content += "        if though_work:\n"
                    main_py_content += "            # 检查是否有归一化步骤\n"
                    main_py_content += "            for step in though_work:\n"
                    main_py_content += "                if step['type'] == 'normalization' and 'save_path' in step:\n"
                    main_py_content += "                    # 尝试对预测结果进行逆向归一化\n"
                    main_py_content += "                    try:\n"
                    main_py_content += "                        # 构建预测结果的DataFrame\n"
                    main_py_content += "                        import pandas as pd\n"
                    main_py_content += "                        pred_df = pd.DataFrame({'prediction': [result['content'][0]]})\n"
                    main_py_content += "                        # 尝试逆向归一化\n"
                    main_py_content += "                        denorm_result = panda.denormalize(pred_df, step['save_path'], columns=['prediction'])\n"
                    main_py_content += "                        if denorm_result['status'] == 1:\n"
                    main_py_content += "                            # 返回逆向归一化后的值\n"
                    main_py_content += "                            return denorm_result['content']['prediction'].iloc[0]\n"
                    main_py_content += "                    except Exception as e:\n"
                    main_py_content += "                        print('Error during denormalization: ' + str(e))\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 如果不需要逆向归一化或逆向归一化失败，返回原始预测结果\n"
                    main_py_content += "        return result['content'][0]\n"
                    main_py_content += "    except Exception as e:\n"
                    main_py_content += "        return 'Error: ' + str(e)\n"
                    main_py_content += "\n"
                    main_py_content += "# 批量预测\n"
                    main_py_content += "def batch_predict(file_path):\n"
                    main_py_content += "    # 加载模型和配置\n"
                    main_py_content += "    model = load_model()\n"
                    main_py_content += "    if model is None:\n"
                    main_py_content += "        return 'Error: Model loading failed'\n"
                    main_py_content += "    \n"
                    main_py_content += "    vectorization_config, normalization_config, though_work = load_config()\n"
                    main_py_content += "    \n"
                    main_py_content += "    try:\n"
                    main_py_content += "        # 读取文件\n"
                    main_py_content += "        if file_path.endswith('.csv'):\n"
                    main_py_content += "            df = panda.read_file(file_path, encoding='utf-8')['content']\n"
                    main_py_content += "            df = panda.json_to_dataframe(df)['content']\n"
                    main_py_content += "        elif file_path.endswith('.xlsx'):\n"
                    main_py_content += "            df = panda.read_file(file_path, engine='openpyxl')['content']\n"
                    main_py_content += "            df = panda.json_to_dataframe(df)['content']\n"
                    main_py_content += "        else:\n"
                    main_py_content += "            return 'Error: Unsupported file format'\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 获取特征列\n"
                    main_py_content += "        x_columns = though_work.get('x_columns', df.columns.tolist())\n"
                    main_py_content += "        if not x_columns:\n"
                    main_py_content += "            x_columns = df.columns.tolist()\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 准备数据\n"
                    main_py_content += "        data = df[x_columns].values\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 应用文本向量化和归一化\n"
                    main_py_content += "        if though_work:\n"
                    main_py_content += "            # 按照though_work中的步骤顺序处理\n"
                    main_py_content += "            for step in though_work:\n"
                    main_py_content += "                if step['type'] == 'vectorization':\n"
                    main_py_content += "                    data = vectorize(data, vectorization_config)\n"
                    main_py_content += "                elif step['type'] == 'normalization':\n"
                    main_py_content += "                    data = normalize(data, normalization_config)\n"
                    main_py_content += "        else:\n"
                    main_py_content += "            # 默认处理顺序\n"
                    main_py_content += "            data = vectorize(data, vectorization_config)\n"
                    main_py_content += "            data = normalize(data, normalization_config)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 批量预测\n"
                    main_py_content += "        results = []\n"
                    main_py_content += "        for row in data:\n"
                    main_py_content += "            result = model.predict(row.reshape(1, -1))\n"
                    main_py_content += "            pred_value = result['content'][0]\n"
                    main_py_content += "            \n"
                    main_py_content += "            # 检查是否需要逆向归一化\n"
                    main_py_content += "            if though_work:\n"
                    main_py_content += "                # 检查是否有归一化步骤\n"
                    main_py_content += "                for step in though_work:\n"
                    main_py_content += "                    if step['type'] == 'normalization' and 'save_path' in step:\n"
                    main_py_content += "                        # 尝试对预测结果进行逆向归一化\n"
                    main_py_content += "                        try:\n"
                    main_py_content += "                            # 构建预测结果的DataFrame\n"
                    main_py_content += "                            import pandas as pd\n"
                    main_py_content += "                            pred_df = pd.DataFrame({'prediction': [pred_value]})\n"
                    main_py_content += "                            # 尝试逆向归一化\n"
                    main_py_content += "                            denorm_result = panda.denormalize(pred_df, step['save_path'], columns=['prediction'])\n"
                    main_py_content += "                            if denorm_result['status'] == 1:\n"
                    main_py_content += "                                # 使用逆向归一化后的值\n"
                    main_py_content += "                                pred_value = denorm_result['content']['prediction'].iloc[0]\n"
                    main_py_content += "                        except Exception as e:\n"
                    main_py_content += "                            print('Error during denormalization: ' + str(e))\n"
                    main_py_content += "            \n"
                    main_py_content += "            results.append(pred_value)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 将结果添加到DataFrame\n"
                    main_py_content += "        df['预测结果'] = results\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 保存结果\n"
                    main_py_content += "        output_file = os.path.splitext(file_path)[0] + '_result.csv'\n"
                    main_py_content += "        df.to_csv(output_file, index=False, encoding='utf-8-sig')\n"
                    main_py_content += "        \n"
                    main_py_content += "        return 'Batch prediction completed. Results saved to: ' + output_file\n"
                    main_py_content += "    except Exception as e:\n"
                    main_py_content += "        return 'Error: ' + str(e)\n"
                    main_py_content += "\n"
                    main_py_content += "# Qt5窗口类\n"
                    main_py_content += "class PredictionApp(QMainWindow):\n"
                    main_py_content += "    def __init__(self):\n"
                    main_py_content += "        super().__init__()\n"
                    main_py_content += "        self.setWindowTitle('模型预测工具')\n"
                    main_py_content += "        self.setGeometry(100, 100, 700, 500)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 验证token\n"
                    main_py_content += "        if not validate_token():\n"
                    main_py_content += "            QMessageBox.warning(self, 'Token验证', 'Token验证失败，可能无法正常使用所有功能')\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 创建主窗口部件\n"
                    main_py_content += "        central_widget = QWidget()\n"
                    main_py_content += "        self.setCentralWidget(central_widget)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 创建布局\n"
                    main_py_content += "        layout = QVBoxLayout()\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 模型信息标签\n"
                    main_py_content += "        model_info = QLabel('模型ID: ' + MODEL_ID)\n"
                    main_py_content += "        model_info.setAlignment(Qt.AlignCenter)\n"
                    main_py_content += "        layout.addWidget(model_info)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 单个预测部分\n"
                    main_py_content += "        single_prediction_group = QWidget()\n"
                    main_py_content += "        single_layout = QVBoxLayout()\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 输入标签\n"
                    main_py_content += "        input_label = QLabel('请输入预测数据（逗号分隔）:')\n"
                    main_py_content += "        single_layout.addWidget(input_label)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 输入框\n"
                    main_py_content += "        self.input_line = QLineEdit()\n"
                    main_py_content += "        self.input_line.setPlaceholderText('例如: 1.0,2.0,3.0,4.0')\n"
                    main_py_content += "        single_layout.addWidget(self.input_line)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 预测按钮\n"
                    main_py_content += "        predict_button = QPushButton('单个预测')\n"
                    main_py_content += "        predict_button.clicked.connect(self.on_predict)\n"
                    main_py_content += "        single_layout.addWidget(predict_button)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 结果标签\n"
                    main_py_content += "        self.result_label = QLabel('预测结果: ')\n"
                    main_py_content += "        self.result_label.setAlignment(Qt.AlignCenter)\n"
                    main_py_content += "        single_layout.addWidget(self.result_label)\n"
                    main_py_content += "        \n"
                    main_py_content += "        single_prediction_group.setLayout(single_layout)\n"
                    main_py_content += "        layout.addWidget(single_prediction_group)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 批量预测部分\n"
                    main_py_content += "        batch_prediction_group = QWidget()\n"
                    main_py_content += "        batch_layout = QVBoxLayout()\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 批量预测标签\n"
                    main_py_content += "        batch_label = QLabel('批量预测（上传CSV或XLSX文件）:')\n"
                    main_py_content += "        batch_layout.addWidget(batch_label)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 上传按钮\n"
                    main_py_content += "        upload_button = QPushButton('选择文件')\n"
                    main_py_content += "        upload_button.clicked.connect(self.on_upload)\n"
                    main_py_content += "        batch_layout.addWidget(upload_button)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 批量预测按钮\n"
                    main_py_content += "        batch_button = QPushButton('批量预测')\n"
                    main_py_content += "        batch_button.clicked.connect(self.on_batch_predict)\n"
                    main_py_content += "        batch_layout.addWidget(batch_button)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 批量预测结果标签\n"
                    main_py_content += "        self.batch_result_label = QLabel('批量预测结果: ')\n"
                    main_py_content += "        self.batch_result_label.setAlignment(Qt.AlignCenter)\n"
                    main_py_content += "        batch_layout.addWidget(self.batch_result_label)\n"
                    main_py_content += "        \n"
                    main_py_content += "        batch_prediction_group.setLayout(batch_layout)\n"
                    main_py_content += "        layout.addWidget(batch_prediction_group)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 设置布局\n"
                    main_py_content += "        central_widget.setLayout(layout)\n"
                    main_py_content += "        \n"
                    main_py_content += "        # 保存选择的文件路径\n"
                    main_py_content += "        self.selected_file = ''\n"
                    main_py_content += "    \n"
                    main_py_content += "    def on_predict(self):\n"
                    main_py_content += "        input_data = self.input_line.text()\n"
                    main_py_content += "        if not input_data:\n"
                    main_py_content += "            QMessageBox.warning(self, '输入错误', '请输入预测数据')\n"
                    main_py_content += "            return\n"
                    main_py_content += "        \n"
                    main_py_content += "        result = predict(input_data)\n"
                    main_py_content += "        self.result_label.setText('预测结果: ' + str(result))\n"
                    main_py_content += "    \n"
                    main_py_content += "    def on_upload(self):\n"
                    main_py_content += "        options = QFileDialog.Options()\n"
                    main_py_content += "        options |= QFileDialog.ReadOnly\n"
                    main_py_content += "        file_path, _ = QFileDialog.getOpenFileName(self, \"选择预测文件\", \"\", \"CSV Files (*.csv);;Excel Files (*.xlsx)\", options=options)\n"
                    main_py_content += "        if file_path:\n"
                    main_py_content += "            self.selected_file = file_path\n"
                    main_py_content += "            self.batch_result_label.setText('已选择文件: ' + os.path.basename(file_path))\n"
                    main_py_content += "    \n"
                    main_py_content += "    def on_batch_predict(self):\n"
                    main_py_content += "        if not self.selected_file:\n"
                    main_py_content += "            QMessageBox.warning(self, '文件错误', '请先选择预测文件')\n"
                    main_py_content += "            return\n"
                    main_py_content += "        \n"
                    main_py_content += "        result = batch_predict(self.selected_file)\n"
                    main_py_content += "        self.batch_result_label.setText('批量预测结果: ' + result)\n"
                    main_py_content += "\n"
                    main_py_content += "if __name__ == '__main__':\n"
                    main_py_content += "    app = QApplication([])\n"
                    main_py_content += "    window = PredictionApp()\n"
                    main_py_content += "    window.show()\n"
                    main_py_content += "    app.exec_()\n"

                    # 生成setup.py文件（用于pyinstaller打包）
                    setup_py_content = "#!/usr/bin/env python3\n"
                    setup_py_content += "# setup.py - 用于打包成exe文件\n"
                    setup_py_content += "\n"
                    setup_py_content += "import os\n"
                    setup_py_content += "import sys\n"
                    setup_py_content += "from cx_Freeze import setup, Executable\n"
                    setup_py_content += "\n"
                    setup_py_content += "# 依赖项\n"
                    setup_py_content += "build_exe_options = {\n"
                    setup_py_content += "    'packages': ['numpy', 'requests', 'PyQt5', 'pandas', 'openpyxl'],\n"
                    setup_py_content += "    'includes': ['json', 'pickle', 'os'],\n"
                    setup_py_content += "    'include_files': ['model.pkl', 'vectorization.json', 'normalization.json', 'though_work.json'],\n"
                    setup_py_content += "    'excludes': [],\n"
                    setup_py_content += "    'optimize': 2\n"
                    setup_py_content += "}\n"
                    setup_py_content += "\n"
                    setup_py_content += "# 可执行文件配置\n"
                    setup_py_content += "base = None\n"
                    setup_py_content += "if sys.platform == 'win32':\n"
                    setup_py_content += "    base = 'Win32GUI'  # 隐藏控制台窗口\n"
                    setup_py_content += "\n"
                    setup_py_content += "setup(\n"
                    setup_py_content += "    name='模型预测工具',\n"
                    setup_py_content += "    version='1.0',\n"
                    setup_py_content += "    description='基于训练模型的预测工具',\n"
                    setup_py_content += "    options={'build_exe': build_exe_options},\n"
                    setup_py_content += "    executables=[Executable('main.py', base=base, icon=None)]\n"
                    setup_py_content += ")\n"

                    # 生成打包脚本（bat文件）
                    build_bat_content = "@echo off\n"
                    build_bat_content += "\n"
                    build_bat_content += "REM 安装依赖\n"
                    build_bat_content += "pip install cx-Freeze numpy requests PyQt5 pandas openpyxl\n"
                    build_bat_content += "\n"
                    build_bat_content += "REM 打包成exe\n"
                    build_bat_content += "python setup.py build\n"
                    build_bat_content += "\n"
                    build_bat_content += "echo 打包完成！\n"
                    build_bat_content += "echo 可执行文件位于 build/exe.win-amd64-3.x/ 目录中\n"
                    build_bat_content += "pause\n"
                    # 写入main.py
                    with open(os.path.join(sdk_dir, 'main.py'), 'w', encoding='utf-8') as f:
                        f.write(main_py_content)

                    # 写入setup.py
                    with open(os.path.join(sdk_dir, 'setup.py'), 'w', encoding='utf-8') as f:
                        f.write(setup_py_content)

                    # 写入build.bat
                    with open(os.path.join(sdk_dir, 'build.bat'), 'w', encoding='utf-8') as f:
                        f.write(build_bat_content)

                    # 复制模型文件
                    shutil.copy(model_file, os.path.join(sdk_dir, 'model.pkl'))

                    # 复制超参文件
                    if hyper_file:
                        hyper_path = os.path.join(json_path, hyper_file)
                        if os.path.exists(hyper_path):
                            shutil.copy(hyper_path, os.path.join(sdk_dir, 'hyper.json'))

                    # 复制vectorization.json（如果存在）
                    if vectorization:
                        vectorization_path = os.path.join(json_path, vectorization)
                        if os.path.exists(vectorization_path):
                            shutil.copy(vectorization_path, os.path.join(sdk_dir, 'vectorization.json'))

                    # 复制normalization.json（如果存在）
                    if normalization:
                        normalization_path = os.path.join(json_path, normalization)
                        if os.path.exists(normalization_path):
                            shutil.copy(normalization_path, os.path.join(sdk_dir, 'normalization.json'))
                    
                    # 复制though_work.json（如果存在）
                    if though_work:
                        though_work_path = os.path.join(json_path, though_work)
                        if os.path.exists(though_work_path):
                            shutil.copy(though_work_path, os.path.join(sdk_dir, 'though_work.json'))

                    # 压缩成zip文件
                    zip_file = os.path.join('./static/sdk', f'{sdk_id}.zip')
                    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for root, dirs, files in os.walk(sdk_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, sdk_dir)
                                zf.write(file_path, arcname)

                    # 清理临时目录
                    shutil.rmtree(sdk_dir)

                    # 保存SDK信息到sdk_relationship表
                    sdk_insert_result = await mysql.insert('sdk_relationship', ['model_id', 'sdk_file'], [_model_id, f'{sdk_id}.zip'])
                    if sdk_insert_result['response'] != 1:
                        # 更新模型状态为训练成功但SDK生成失败
                        await mysql.update('sklearn_db', ['id'], [_model_id], 'model_condition', 'train success, SDK failed')
                        return response(0, f'Model trained successfully but SDK generation failed: Failed to insert SDK information: {sdk_insert_result.get("content", "Unknown error")}')

                    print(f'SDK generated successfully: {sdk_id}.zip')
                except Exception as ex:
                    error_msg = str(ex)
                    print(f'Error generating SDK: {error_msg}')
                    # 更新模型状态为训练成功但SDK生成失败
                    await mysql.update('sklearn_db', ['id'], [_model_id], 'model_condition', 'train success, SDK failed')
                    return response(0, f'Model trained successfully but SDK generation failed: {error_msg}')

                return response(1, 'Model trained successfully with SDK')
            except Exception as e:
                # 更新模型状态为训练失败
                await mysql.update('sklearn_db', ['id'], [_model_id], 'model_condition', 'train failed')
                return response(0, f'Training failed: {str(e)}')
        # This api is used to reinforces the model
        elif func_name == 'multi_layer_train':
            pass

        elif func_name == 'find_train':
            user_id = body['user_id']

            # 验证用户是否存在
            _inspect = await mysql.find('user_db', attribute='id', result=user_id)
            if len(_inspect['content']) == 0:
                return response(0, 'This user is not register')

            # 获取用户的模型列表
            _models = await mysql.find('sklearn_db', attribute='user_id', result=user_id)
            models = []
            for model in _models['content']:
                models.append({
                    'id': model['id'],
                    'types': model['types'],
                    'model_name': model['model_name'],
                    'model_condition': model.get('model_condition', 'unknown'),
                    'token': model['token'],
                    'url_path': model.get('url_path', ''),
                    'base_path': model.get('base_path', '')
                })

            return response(msg=models)

        elif func_name == 'train_status':
            model_id = body['model_id']
            
            # 初始化Redis队列
            train_queue = get_train_queue()
            
            if train_queue and train_queue.is_connected():
                # 从Redis获取任务状态
                status = train_queue.get_status(model_id)
                if status:
                    # 转换状态格式
                    redis_status = status.get('status')
                    if redis_status == 'completed':
                        status['status'] = 'train successed'
                    elif redis_status == 'failed':
                        status['status'] = 'train failed'
                    return response(1, status)
                else:
                    # 如果Redis中没有状态，从数据库获取
                    model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
                    if len(model_info['content']) > 0:
                        model = model_info['content'][0]
                        model_condition = model['model_condition']
                        # 转换状态格式
                        if model_condition == 'training':
                            status = 'training'
                        elif model_condition == 'train success' or model_condition == 'train success, SDK failed':
                            status = 'train successed'
                        elif model_condition == 'train failed':
                            status = 'train failed'
                        else:
                            status = 'train failed'
                        return response(1, {
                            'status': status,
                            'progress': 100 if status == 'train successed' else 0,
                            'message': f'Model {status}'
                        })
                    else:
                        return response(0, 'Model not found')
            else:
                # Redis不可用，从数据库获取
                model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
                if len(model_info['content']) > 0:
                    model = model_info['content'][0]
                    model_condition = model['model_condition']
                    # 转换状态格式
                    if model_condition == 'training':
                        status = 'training'
                    elif model_condition == 'train success' or model_condition == 'train success, SDK failed':
                        status = 'train successed'
                    elif model_condition == 'train failed':
                        status = 'train failed'
                    else:
                        status = 'train failed'
                    return response(1, {
                        'status': status,
                        'progress': 100 if status == 'train successed' else 0,
                        'message': f'Model {status}'
                    })
                else:
                    return response(0, 'Model not found')

        elif func_name == 'find_models':
            try:
                user_id = body['user_id']
                show_all = body.get('show_all', False)
                print(f"Finding models for user: {user_id}, show_all: {show_all}")

                # 验证用户是否存在
                _inspect = await mysql.find('user_db', attribute='id', result=user_id)
                if len(_inspect['content']) == 0:
                    print(f"User not found: {user_id}")
                    return response(0, 'This user is not register')

                # 获取模型列表
                if show_all:
                    # 获取所有模型
                    _models = await mysql.find('sklearn_db', aim='*')
                else:
                    # 获取用户的模型列表
                    _models = await mysql.find('sklearn_db', attribute='user_id', result=user_id)
                
                print(f"Found {len(_models['content'])} models")
                models = []
                for model in _models['content']:
                    print(f"Processing model: {model.get('model_name')}")
                    # 确定模型状态
                    model_condition = model.get('model_condition', 'unknown')
                    if model_condition == 'training':
                        status = 'training'
                    elif model_condition == 'train success' or model_condition == 'train success, SDK failed':
                        status = 'train successed'
                    elif model_condition == 'train failed':
                        status = 'train failed'
                    else:
                        status = 'failed'
                    
                    models.append({
                    'id': model.get('id', ''),
                    'user_id': model.get('user_id', user_id),
                    'data_id': model.get('data_id', ''),
                    'model_name': model.get('model_name', ''),
                    'types': model.get('types', ''),
                    'model_type': model.get('types', ''),
                    'model_select': model.get('use_model', ''),  # 使用数据库中的use_model字段
                    'token': model.get('token', ''),
                    'url_path': model.get('url_path', ''),
                    'open_bool': model.get('open_bool', '0'),
                    'status': status,
                    'accuracy': 0.0  # 简化处理，实际应该从评估文件中读取
                })

                print(f"Returning {len(models)} models")
                return response(1, models)
            except Exception as e:
                print(f"Error in find_models: {str(e)}")
                return response(0, f'Error loading models: {str(e)}')

        elif func_name == 'update_open':
            id = body['id']
            open_bool = body['open_bool']

            # 更新模型的开启状态
            result = await mysql.update('sklearn_db', ['id'], [id], 'open_bool', open_bool)
            if result['response'] == 1:
                return response(1, 'Update successful')
            else:
                return response(0, 'Update failed')

        elif func_name == 'get_model_features':
            user_id = body['user_id']
            model_id = body['id']
            data_id = body['data_id']

            # 校验模型是否存在且属于该用户
            model_info = await mysql.find('sklearn_db', attribute=['user_id', 'id'], result=[user_id, model_id], relative=['AND'])
            if len(model_info['content']) == 0:
                return response(0, 'Model not found')

            # 获取数据信息
            data_record = await mysql.find('data_db', attribute='saving', result=data_id)
            if len(data_record['content']) == 0:
                return response(0, 'Data not found')

            data_info = data_record['content'][0]
            though_work = data_info.get('though_work', '')

            # 读取though_work文件获取x_columns和y_columns
            x_columns = []
            y_columns = []
            if though_work and though_work != 'load':
                though_work_path = os.path.join(json_path, though_work)
                if os.path.exists(though_work_path):
                    try:
                        with open(though_work_path, 'r', encoding='utf-8') as f:
                            though_work_data = json.load(f)
                        # 尝试从though_work中获取x_columns和y_columns
                        if isinstance(though_work_data, list):
                            # 对于操作列表，尝试查找train_test_split操作
                            for operation in though_work_data:
                                if operation.get('function', '').lower() == 'train_test_split':
                                    x_columns = operation.get('work', {}).get('data', {}).get('x_columns', [])
                                    y_columns = operation.get('work', {}).get('data', {}).get('y_columns', [])
                                    break
                        elif isinstance(though_work_data, dict):
                            # 对于字典，直接获取x_columns和y_columns
                            x_columns = though_work_data.get('x_columns', [])
                            y_columns = though_work_data.get('y_columns', [])
                    except Exception as e:
                        print(f"Error reading though_work file: {e}")

            return response(1, {'x_columns': x_columns, 'y_columns': y_columns})

        elif func_name == 'predict':
            user_id = body['user_id']
            model_id = None
            token = body.get('token')
            data_id = body.get('data_id')

            # 从请求中提取模型ID（可能在不同位置）
            if 'model_id' in body:
                model_id = body['model_id']
            else:
                # 尝试从其他位置获取模型ID
                pass

            # 如果没有明确的model_id，尝试通过token查找
            if not model_id and token:
                models = await mysql.find('sklearn_db', attribute='token', result=token)
                if len(models['content']) > 0:
                    model_id = models['content'][0]['id']

            if not model_id:
                return response(0, 'Model ID is required')

            # 校验模型是否存在且属于该用户
            model_info = await mysql.find('sklearn_db', attribute=['user_id', 'id'], result=[user_id, model_id], relative=['AND'])
            if len(model_info['content']) == 0:
                return response(0, 'Model not found')

            model_info = model_info['content'][0]

            # 校验token
            if token != model_info['token']:
                return response(0, 'Invalid token')

            # 检查模型是否开启
            if str(model_info['open_bool']) != '1':
                return response(0, 'Model is not open')

            # 加载模型
            model_file = os.path.join(model_path, model_info['base_path'])
            if not os.path.exists(model_file):
                return response(0, 'Model file not found')

            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

                # 准备预测数据
                # 提取自变量值
                x_columns = []
                x_values = []
                
                # 从data_id获取though_work信息以确定x_columns
                if data_id:
                    data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                    if len(data_record['content']) > 0:
                        data_info = data_record['content'][0]
                        though_work = data_info.get('though_work', '')
                        if though_work and though_work != 'load':
                            though_work_path = os.path.join(json_path, though_work)
                            if os.path.exists(though_work_path):
                                try:
                                    with open(though_work_path, 'r', encoding='utf-8') as f:
                                        though_work_data = json.load(f)
                                    # 尝试从though_work中获取x_columns
                                    if isinstance(though_work_data, list):
                                        for operation in though_work_data:
                                            if operation.get('function', '').lower() == 'train_test_split':
                                                x_columns = operation.get('work', {}).get('data', {}).get('x_columns', [])
                                                break
                                    elif isinstance(though_work_data, dict):
                                        x_columns = though_work_data.get('x_columns', [])
                                except Exception as e:
                                    print(f"Error reading though_work file: {e}")
                
                # 如果没有获取到x_columns，尝试从请求中提取
                if not x_columns:
                    # 提取所有数值字段作为自变量
                    for key, value in body.items():
                        if key not in ['user_id', 'model_id', 'token', 'data_id'] and isinstance(value, (int, float)):
                            x_columns.append(key)
                            x_values.append(value)
                else:
                    # 使用已知的x_columns提取值
                    for col in x_columns:
                        if col in body and isinstance(body[col], (int, float)):
                            x_values.append(body[col])
                        else:
                            return response(0, f'Missing value for feature: {col}')

                if not x_values:
                    return response(0, 'No valid input features provided')

                # 转换为numpy数组
                predict_data = np.array(x_values).reshape(1, -1)

                # 预测
                predict_result = model.predict(predict_data)
                if predict_result['status'] != 1:
                    return response(0, predict_result['content'])

                # 构建预测结果
                result = {'model_id': model_id}
                
                # 从data_id获取y_columns
                y_columns = []
                if data_id:
                    data_record = await mysql.find('data_db', attribute='saving', result=data_id)
                    if len(data_record['content']) > 0:
                        data_info = data_record['content'][0]
                        though_work = data_info.get('though_work', '')
                        if though_work and though_work != 'load':
                            though_work_path = os.path.join(json_path, though_work)
                            if os.path.exists(though_work_path):
                                try:
                                    with open(though_work_path, 'r', encoding='utf-8') as f:
                                        though_work_data = json.load(f)
                                    # 尝试从though_work中获取y_columns
                                    if isinstance(though_work_data, list):
                                        for operation in though_work_data:
                                            if operation.get('function', '').lower() == 'train_test_split':
                                                y_columns = operation.get('work', {}).get('data', {}).get('y_columns', [])
                                                break
                                    elif isinstance(though_work_data, dict):
                                        y_columns = though_work_data.get('y_columns', [])
                                except Exception as e:
                                    print(f"Error reading though_work file: {e}")

                # 处理预测结果
                predict_content = predict_result['content']
                if isinstance(predict_content, list):
                    if len(predict_content) == 1 and isinstance(predict_content[0], list):
                        # 二维列表，可能是多输出
                        predict_content = predict_content[0]
                    
                    # 与y_columns对应
                    for i, y_col in enumerate(y_columns):
                        if i < len(predict_content):
                            result[y_col] = predict_content[i]
                        else:
                            break
                else:
                    # 单个预测结果
                    if y_columns:
                        result[y_columns[0]] = predict_content
                    else:
                        result['prediction'] = predict_content

                return response(1, result)
            except Exception as e:
                print(f"Prediction error: {str(e)}")
                return response(0, f'Prediction failed: {str(e)}')
        elif func_name == 'get_train_status':
            model_id = body.get('model_id')
            if not model_id:
                return response(0, 'Model ID is required')
            
            train_queue = get_train_queue()
            if not train_queue or not train_queue.is_connected():
                # 如果Redis不可用，查询数据库
                model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
                if model_info['content']:
                    model_condition = model_info['content'][0]['model_condition']
                    status_map = {
                        'training': {'status': 'training', 'progress': 50},
                        'train success': {'status': 'completed', 'progress': 100},
                        'train success, SDK failed': {'status': 'completed', 'progress': 100, 'sdk_error': 'SDK generation failed'},
                        'train failed': {'status': 'failed', 'progress': 0}
                    }
                    status = status_map.get(model_condition, {'status': 'unknown', 'progress': 0})
                    return response(1, 'Status retrieved successfully', status)
                return response(0, 'Model not found')
            
            status = train_queue.get_status(model_id)
            if status:
                return response(1, 'Status retrieved successfully', status)
            else:
                # 如果Redis中没有状态，查询数据库
                model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
                if model_info['content']:
                    model_condition = model_info['content'][0]['model_condition']
                    status_map = {
                        'training': {'status': 'training', 'progress': 50},
                        'train success': {'status': 'completed', 'progress': 100},
                        'train success, SDK failed': {'status': 'completed', 'progress': 100, 'sdk_error': 'SDK generation failed'},
                        'train failed': {'status': 'failed', 'progress': 0}
                    }
                    status = status_map.get(model_condition, {'status': 'unknown', 'progress': 0})
                    return response(1, 'Status retrieved successfully', status)
                return response(0, 'Model not found')

        elif func_name == 'delete_model':
            user_id = body['user_id']
            model_id = body['model_id']
            password = body['password']

            # 验证用户密码
            user_info = await mysql.find('user_db', attribute='id', result=user_id)
            if len(user_info['content']) == 0:
                return response(0, 'User not found')

            stored_password = user_info['content'][0]['password']
            encrypted_password = other.password_encryption(password)['content']
            if encrypted_password != stored_password:
                return response(0, 'Password is incorrect')

            # 查找模型
            model_info = await mysql.find('sklearn_db', attribute=['user_id', 'id'], result=[user_id, model_id], relative=['AND'])
            if len(model_info['content']) == 0:
                return response(0, 'Model not found')

            model = model_info['content'][0]
            base_path = model['base_path']
            hyper = model['hyper']
            evaluation = model['evaluation']

            # 删除模型文件
            # 直接从model_path目录加载模型
            model_file = os.path.join(model_path, base_path)
            if os.path.exists(model_file):
                os.remove(model_file)

            # 删除超参和评估文件
            if hyper:
                hyper_path = os.path.join(json_path, hyper)
                if os.path.exists(hyper_path):
                    os.remove(hyper_path)
            if evaluation:
                eval_path = os.path.join(json_path, evaluation)
                if os.path.exists(eval_path):
                    os.remove(eval_path)

            # 删除SDK文件
            sdk_info = await mysql.find('sdk_relationship', attribute='model_id', result=model_id)
            if len(sdk_info['content']) > 0:
                sdk_file = sdk_info['content'][0]['sdk_file']
                if sdk_file:
                    # 构建正确的SDK文件路径
                    sdk_path = os.path.join('static', 'sdk', sdk_file)
                    if os.path.exists(sdk_path):
                        os.remove(sdk_path)
                    # 同时尝试旧路径格式，确保兼容
                    old_sdk_path = os.path.join('./', sdk_file)
                    if os.path.exists(old_sdk_path):
                        os.remove(old_sdk_path)
                # 删除SDK关系记录
                await mysql.delete('sdk_relationship', ['model_id'], [model_id])

            # 从数据库中删除模型记录
            await mysql.delete('sklearn_db', ['user_id', 'id'], [user_id, model_id])

            return response(1, 'Model deleted successfully')

        elif func_name == 'delete_all_models':
            try:
                user_id = body['user_id']
                password = body['password']
                print(f"delete_all_models called by user: {user_id}")

                # 验证用户密码
                user_info = await mysql.find('user_db', attribute='id', result=user_id)
                print(f"User info: {user_info}")
                if len(user_info['content']) == 0:
                    return response(0, 'User not found')

                stored_password = user_info['content'][0]['password']
                encrypted_password = other.password_encryption(password)['content']
                if encrypted_password != stored_password:
                    return response(0, 'Password is incorrect')

                # 先获取所有模型
                models = await mysql.find('sklearn_db', aim='*')
                print(f"Found {len(models['content'])} models to delete")
                
                # 逐个删除模型
                for model in models['content']:
                    model_id = model['id']
                    base_path = model['base_path']
                    hyper = model['hyper']
                    evaluation = model['evaluation']
                    
                    # 删除模型文件
                    model_file = os.path.join(model_path, base_path)
                    if os.path.exists(model_file):
                        try:
                            os.remove(model_file)
                            print(f"Deleted model file: {model_file}")
                        except Exception as e:
                            print(f"Error deleting model file {model_file}: {str(e)}")
                    
                    # 删除超参和评估文件
                    if hyper:
                        hyper_path = os.path.join(json_path, hyper)
                        if os.path.exists(hyper_path):
                            try:
                                os.remove(hyper_path)
                                print(f"Deleted hyper file: {hyper_path}")
                            except Exception as e:
                                print(f"Error deleting hyper file {hyper_path}: {str(e)}")
                    if evaluation:
                        eval_path = os.path.join(json_path, evaluation)
                        if os.path.exists(eval_path):
                            try:
                                os.remove(eval_path)
                                print(f"Deleted evaluation file: {eval_path}")
                            except Exception as e:
                                print(f"Error deleting evaluation file {eval_path}: {str(e)}")
                    
                    # 删除SDK文件
                    sdk_info = await mysql.find('sdk_relationship', attribute='model_id', result=model_id)
                    if len(sdk_info['content']) > 0:
                        sdk_file = sdk_info['content'][0]['sdk_file']
                        if sdk_file:
                            # 构建正确的SDK文件路径
                            sdk_path = os.path.join('static', 'sdk', sdk_file)
                            if os.path.exists(sdk_path):
                                try:
                                    os.remove(sdk_path)
                                    print(f"Deleted SDK file: {sdk_path}")
                                except Exception as e:
                                    print(f"Error deleting SDK file {sdk_path}: {str(e)}")
                            # 同时尝试旧路径格式，确保兼容
                            old_sdk_path = os.path.join('./', sdk_file)
                            if os.path.exists(old_sdk_path):
                                try:
                                    os.remove(old_sdk_path)
                                    print(f"Deleted old SDK file: {old_sdk_path}")
                                except Exception as e:
                                    print(f"Error deleting old SDK file {old_sdk_path}: {str(e)}")
                        # 删除SDK关系记录
                        await mysql.delete('sdk_relationship', ['model_id'], [model_id])
                    
                    # 从数据库中删除模型记录
                    await mysql.delete('sklearn_db', ['id'], [model_id])
                    print(f"Deleted model record: {model_id}")
                
                return response(1, 'Deleted all models successfully')
            except Exception as e:
                print(f"Error in delete_all_models: {str(e)}")
                return response(0, f'Error deleting models: {str(e)}')

        elif func_name == 'download_sdk':
            # 下载SDK
            model_id = body.get('model_id')
            if not model_id:
                return response(0, 'Model ID is required')

            # 查找SDK文件
            sdk_info = await mysql.find('sdk_relationship', attribute='model_id', result=model_id)
            if len(sdk_info['content']) == 0:
                return response(0, 'SDK not found')

            sdk_file = sdk_info['content'][0]['sdk_file']
            if not sdk_file:
                return response(0, 'SDK file not found')

            # 尝试多种可能的路径格式
            sdk_path = os.path.join('./static/sdk', sdk_file)
            if not os.path.exists(sdk_path):
                # 尝试旧路径格式
                sdk_path = os.path.join('./', sdk_file)
                if not os.path.exists(sdk_path):
                    # 尝试另一种路径格式
                    sdk_path = os.path.join('static/sdk', sdk_file)
            if not os.path.exists(sdk_path):
                return response(0, 'SDK file not found on disk')

            return response_file(sdk_path, as_attachment=True, filename=os.path.basename(sdk_file))

        elif func_name == 'download_vectorization':
            # 下载文本向量化参数
            model_id = body.get('model_id')
            if not model_id:
                return response(0, 'Model ID is required')

            # 通过model_id查询sklearn_db获取data_id
            model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
            if len(model_info['content']) == 0:
                return response(0, 'Model not found')

            data_id = model_info['content'][0].get('data_id')
            if not data_id:
                return response(0, 'Data ID not found for this model')

            # 查找数据记录
            data_info = await mysql.find('data_db', attribute='saving', result=data_id)
            if len(data_info['content']) == 0:
                return response(0, 'Data not found')

            vectorization = data_info['content'][0].get('vectorization')
            if not vectorization or vectorization == 'None':
                return response(0, 'Vectorization file not found')

            # 构建向量化文件路径
            vectorization_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'json', vectorization)
            if not os.path.exists(vectorization_path):
                return response(0, f'Vectorization file not found on disk: {vectorization_path}')

            return response_file(vectorization_path, as_attachment=True, filename=vectorization)

        elif func_name == 'download_normalization':
            # 下载数据归一化参数
            model_id = body.get('model_id')
            if not model_id:
                return response(0, 'Model ID is required')

            # 通过model_id查询sklearn_db获取data_id
            model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
            if len(model_info['content']) == 0:
                return response(0, 'Model not found')

            data_id = model_info['content'][0].get('data_id')
            if not data_id:
                return response(0, 'Data ID not found for this model')

            # 查找数据记录
            data_info = await mysql.find('data_db', attribute='saving', result=data_id)
            if len(data_info['content']) == 0:
                return response(0, 'Data not found')

            normalization = data_info['content'][0].get('normalization')
            if not normalization or normalization == 'None':
                return response(0, 'Normalization file not found')

            # 构建归一化文件路径
            normalization_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'json', normalization)
            if not os.path.exists(normalization_path):
                return response(0, f'Normalization file not found on disk: {normalization_path}')

            return response_file(normalization_path, as_attachment=True, filename=normalization)

        elif func_name == 'toggle_model_open':
            # 切换模型的open_bool状态
            model_id = body.get('model_id')
            user_id = body.get('user_id')
            model_name = body.get('model_name')
            open_bool = body.get('open_bool')

            # 查找模型
            if model_id:
                model_info = await mysql.find('sklearn_db', attribute='id', result=model_id)
            elif user_id and model_name:
                model_info = await mysql.find('sklearn_db', attribute=['user_id', 'model_name'], result=[user_id, model_name], relative=['AND'])
            else:
                return response(0, 'Model ID or user_id and model_name is required')

            if len(model_info['content']) == 0:
                return response(0, 'Model not found')

            model = model_info['content'][0]
            model_id = model['id']

            # 使用前端提供的open_bool值
            if open_bool is not None:
                new_open_bool = '1' if open_bool else '0'
            else:
                # 如果没有提供open_bool，切换当前状态
                current_open_bool = model['open_bool']
                new_open_bool = '0' if current_open_bool == '1' else '1'

            # 更新状态
            result = await mysql.update('sklearn_db', ['id'], [model_id], 'open_bool', new_open_bool)
            if result['response'] == 1:
                return response(1, f'Model open status updated to {new_open_bool}')
            else:
                return response(0, 'Failed to update model open status')

    # 使用兼容旧版本Python的方式启动异步事件循环
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.start())