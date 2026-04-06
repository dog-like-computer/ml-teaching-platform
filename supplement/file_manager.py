# todo-1: important library
from operator import itemgetter
import json
import os

# todo-2: build class
class Manage:
    def __init__(self):
        self.stack = [r'./setting.json',r'./supplement/file_manager.py',r'./app.py',r'./static/system_pic',
                      r'./supplement/mysql_api.py',r'./supplement/ocr_api.py',r'./supplement/encryption.py',
                      r'./supplement/draw.py',r'./supplement/scikit_learn_build.py',r'./static/data',
                      r'./static/base_model',r'./static/html',r'./static/css',r'./static/javascript',
                      r'./static/log',r'./static/model',r'./static/user_pic','./supplement/pd_item.py',
                      r'./static/json']
        self.count = len(self.stack)
        self.status = []

    # This is used to find this item version
    def __integrity(self, file_path, count=1):
        if count == 0:
            return self.status
        self.status.append(os.path.exists(os.path.abspath(file_path[count-1])))
        return self.__integrity(file_path,count-1)

    # This is response of this function file
    def __response(self,section=True,content=None):
        return {"response":section,"content":content}

    # This is used to write the json setting
    def wr_setting(self,methods,penetration=False,internal=None,
                   host='127.0.0.1',port=8080,show_in=True,show_out=True,
                   save_log=True,save_length=20,sql_port=3306,sql_password='123456',
                   sql_admin='root',redis_host='127.0.0.1',redis_port=6379,redis_db=0,
                   redis_password='123456',redis_timeout=5,la='en'):
        _ = self.review()
        if not _['response']:
            return _
        _ = ['penetration','internal','host','port','show_in','show_out','save_log','save_length','sql_port','sql_admin','sql_password',
             'redis_host','redis_port','redis_db','redis_password','redis_timeout','la']
        temp = [penetration,internal,host,port,show_in,show_out,save_log,save_length,sql_port,sql_admin,sql_password,redis_host,redis_port,
                redis_db,redis_password,redis_timeout,la]
        setting_content = json.loads(open(r'./setting.json').read())
        if methods == 'r':
            return self.__response(True,content=setting_content)
        elif methods == 'w':
            for i in range(len(_)):
                setting_content[_[i]] = temp[i]
            with open(r'./setting.json','w',encoding='utf-8') as f:
                # json.dump(setting_content,f,ensure_ascii=False)
                f.write(json.dumps(setting_content))
            f.close()
            return self.__response(True,content=setting_content)
        return self.__response(False,'no this orders')

    # This is used to review file
    def review(self):
        self.status = self.__integrity(self.stack,self.count)[::-1]
        if all(self.status):
            self.status.clear()
            return self.__response(True,'Currently, there is no loss of files on the server backend')
        false_indices = [i for i, val in enumerate(self.status) if not val]
        getter = itemgetter(*false_indices)
        return self.__response(False,f'This code run fail.Because lack of {"".join(getter(self.stack))}')


# todo-3: test way
if __name__ == '__main__':
    manage = Manage()
    print(manage.wr_setting('w',internal='p90wpcbruk8x.guyubao.com'))