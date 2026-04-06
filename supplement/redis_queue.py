import redis
import json
import time
from typing import Any, Dict, Optional

class RedisQueue:
    def __init__(self, host='127.0.0.1', port=6379, db=0, password=None, timeout=5):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.timeout = timeout
        self._client = None
        self._connect()

    def _connect(self):
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                socket_timeout=self.timeout,
                decode_responses=True
            )
            self._client.ping()
            print(f"Redis connected successfully: {self.host}:{self.port}/{self.db}")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self._client = None

    def is_connected(self):
        if self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except:
            return False

    def enqueue(self, queue_name: str, task: Dict[str, Any]) -> bool:
        if not self.is_connected():
            print("Redis is not connected, cannot enqueue task")
            return False
        try:
            task['enqueue_time'] = time.time()
            task_str = json.dumps(task, ensure_ascii=False)
            self._client.rpush(queue_name, task_str)
            return True
        except Exception as e:
            print(f"Enqueue failed: {e}")
            return False

    def dequeue(self, queue_name: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            print("Redis is not connected, cannot dequeue task")
            return None
        try:
            if timeout > 0:
                result = self._client.blpop(queue_name, timeout=timeout)
                if result:
                    _, task_str = result
                    return json.loads(task_str)
            else:
                task_str = self._client.lpop(queue_name)
                if task_str:
                    return json.loads(task_str)
            return None
        except Exception as e:
            print(f"Dequeue failed: {e}")
            return None

    def get_queue_length(self, queue_name: str) -> int:
        if not self.is_connected():
            return 0
        try:
            return self._client.llen(queue_name)
        except:
            return 0

    def set_status(self, task_id: str, status: Dict[str, Any], expire: int = 86400) -> bool:
        if not self.is_connected():
            return False
        try:
            key = f"task_status:{task_id}"
            status['update_time'] = time.time()
            self._client.set(key, json.dumps(status, ensure_ascii=False), ex=expire)
            return True
        except Exception as e:
            print(f"Set status failed: {e}")
            return False

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_connected():
            return None
        try:
            key = f"task_status:{task_id}"
            status_str = self._client.get(key)
            if status_str:
                return json.loads(status_str)
            return None
        except Exception as e:
            print(f"Get status failed: {e}")
            return None

    def delete_status(self, task_id: str) -> bool:
        if not self.is_connected():
            return False
        try:
            key = f"task_status:{task_id}"
            self._client.delete(key)
            return True
        except:
            return False

    def add_progress(self, task_id: str, progress: int, message: str = "") -> bool:
        status = {
            'status': 'running',
            'progress': progress,
            'message': message
        }
        return self.set_status(task_id, status)

    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        status = {
            'status': 'completed',
            'progress': 100,
            'result': result
        }
        return self.set_status(task_id, status)

    def fail_task(self, task_id: str, error: str) -> bool:
        status = {
            'status': 'failed',
            'progress': 0,
            'error': error
        }
        return self.set_status(task_id, status)

train_queue = None

def init_train_queue():
    global train_queue
    setting_path = 'e:\\242821513\\setting.json'
    try:
        with open(setting_path, 'r', encoding='utf-8') as f:
            setting = json.load(f)
        train_queue = RedisQueue(
            host=setting.get('redis_host', '127.0.0.1'),
            port=setting.get('redis_port', 6379),
            db=setting.get('redis_db', 0),
            password=setting.get('redis_password'),
            timeout=setting.get('redis_timeout', 5)
        )
        # 不需要在初始化时清空队列，保持队列中的任务
    except Exception as e:
        print(f"Failed to initialize train queue: {e}")
        train_queue = None

def get_train_queue():
    global train_queue
    if train_queue is None:
        init_train_queue()
    return train_queue
