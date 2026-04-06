# todo-1 import need library
from xpinyin import Pinyin
from random import randint
import hashlib
import bcrypt
import secrets
import string
import re

# todo-2 build class

txt_path = '../static/json'

class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if not self.root:
            self.root = BSTNode(key)
            return
        current = self.root
        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)
        while True:
            if key_hash % 2 == 0:
                if not current.left:
                    current.left = BSTNode(key)
                    break
                current = current.left
            else:
                if not current.right:
                    current.right = BSTNode(key)
                    break
                current = current.right

    def get_final_key(self):
        result = []
        def post_order(node):
            if node:
                post_order(node.left)
                post_order(node.right)
                result.append(node.key)
        post_order(self.root)
        return result[0] if result else ""

class Other:
    def __init__(self):
        self.p = Pinyin()
        # self.bst_node = BSTNode()
        self.bst = BST()
        pass

    def __response(self,status=1,content=None):
        return {'status':status,'content':content}

    def n_to_id(self,username):
        _ = ''
        punctuation_pattern = re.compile(r'[\u3000-\u303f\uff00-\uffef,\.\?!;:\'"()\[\]<>、，。！？；：""''（）【】《》—～￥…·]')
        clean_text = punctuation_pattern.sub('', username)
        if len(clean_text) > 3:
            clean_text = clean_text[0:3]
        elif len(clean_text) < 3:
            while len(clean_text) < 3:
                clean_text += 'j'
        _ = self.p.get_initials(clean_text).replace('-','').lower()
        while len(_) < 7:
            _ += f'{randint(0,9)}'
        return self.__response(content=_)

    def password_encryption(self,password):
        FIXED_SALT = b"$2b$12$mGFlRI9285LwdEw0kUVdq."
        chinese_pattern = re.compile(r'[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]')
        match = chinese_pattern.search(password)
        if match:
            return self.__response(0,'Password not allow Chinese')
        pwd_bytes = password.encode('utf-8')
        bcrypt_hash = bcrypt.hashpw(pwd_bytes, FIXED_SALT).decode('utf-8')
        self.bst.insert(bcrypt_hash)
        final_hash = self.bst.get_final_key()
        final_cipher = hashlib.sha256(final_hash.encode()).hexdigest()
        return self.__response(content=final_cipher)

    def get_random_token(self,length=32):
        chars = string.ascii_letters + string.digits
        token = ''.join(secrets.choice(chars) for _ in range(length))
        return self.__response(content=token)

    # in feature this function will update
    def open_stopwords(self,file_name):
        with open(txt_path + '/' + file_name,'r',encoding='utf-8') as f:
            content = f.readlines()
        return self.__response(content=content)

    def get_random_id(self,user_id,methods='model'):
        if methods.lower() in ['model','m']:
            _ = f'm{user_id[0:3]}'
        elif methods.lower() in ['data','d']:
            _ = f'd{user_id[0:3]}'
        elif methods.lower() in ['normalization','n']:
            _ = f'n{user_id[0:3]}'
        elif methods.lower() in ['vectorization','v']:
            _ = f'v{user_id[0:3]}'
        elif methods.lower() in ['picture','figure','p','f']:
            _ = f'p{user_id[0:3]}'
        elif methods.lower() in ['work','w']:
            _ = f'w{user_id[0:3]}'
        elif methods.lower() in ['hyper','h']:
            _ = f'h{user_id[0:3]}'
        elif methods.lower() in ['eval','e']:
            _ = f'e{user_id[0:3]}'
        elif methods.lower() in ['sdk','s']:
            _ = f's{user_id[0:3]}'
        else:
            return self.__response(0,'This methods is not allow')
        while len(_) < 10:
            _ += f'{randint(0,9)}'
        return self.__response(content=_)

    def get_train_way(self,way_path):
        with open(way_path,'r',encoding='utf-8') as f:
            file_content = f.readlines()
        f.close()
        return self.__response(content=file_content)

    def create_model_path(self,user_id,model_select,model_type):
        # c || r + user_id[0:3][::-1] + model_select[0:3]--(1+2+3+2+3+4=15)
        _str = f'{model_type[0].lower()}'
        _random = randint(0,99)
        if _random < 10:
            _str += '{:02d}'.format(_random)
        _str += user_id[:3][::-1]
        _random = randint(0,99)
        if _random < 10:
            _str += '{:02d}'.format(_random)
        _str += model_select[:3]
        _random = randint(0,99)
        if _random < 10:
            _str += '{:02d}'.format(_random)
        return self.__response(content=_str)


# todo-3 test way
if __name__ == '__main__':
    other = Other()
    # print(other.n_to_id('你好，'))
    print(other.get_random_token())