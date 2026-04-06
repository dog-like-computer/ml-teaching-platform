# todo-1 important library
import pytesseract as pts
from random import randint
from PIL import Image

# todo-2 setting OCR
class Ocr:
    def __init__(self):
        self.base_path = r'./static/system_pic'
        pass

    def __response(self,status=1,content=None):
        return {'status':status,'content':content}

    def send(self,open_f='name',search_number=None):
        if not search_number:
            _ = randint(1,200)
        else:
            _ = search_number
        if open_f == 'name':
            return self.__response(1,content=f'{self.base_path}/{_}.png')
        elif open_f == 'file_2':
            return self.__response(1,content=open(f'{self.base_path}/{_}.png','rb'))
        return self.__response(1,content=Image.open(f'{self.base_path}/{_}.png'))

    def recognize(self,img_path):
        # 使用真实的OCR引擎进行识别
        try:
            result = pts.image_to_string(Image.open(img_path), lang='eng').replace('\n', '').strip().replace(' ','').lower()
            return self.__response(1,content=result)
        except Exception as e:
            print(f"OCR recognition error: {str(e)}")
            # 如果OCR识别失败，返回一个空字符串
            return self.__response(1,content='')


if __name__ == '__main__':
    ocr_ = Ocr()
    print(ocr_.send('file_2'))
    # print(ocr_.recognize('./system_pic/1.png'))