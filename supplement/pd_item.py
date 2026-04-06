# todo-1 important library
from collections import defaultdict,Counter
import pandas as pd
import numpy as np
import jieba
import math
import json
import os
import re

# todo-2 build data way

data_path = r'./static/data'

class Panda:
    def __init__(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', 1000)
        pd.set_option('display.width', None)

    def __response(self,status=1,content=None):
        return {'status':status,'content':content}

    def find_columns(self,data,types=None):
        # 'int64', 'float64'
        if not types:
            return self.__response(content=data.columns.tolist())
        if isinstance(types,str):
            return self.__response(content=data.select_dtypes(include=[types]).columns.tolist())
        elif isinstance(types,(list,tuple)):
            return self.__response(content=data.select_dtypes(include=types).columns.tolist())

    def read_file(self,file_name,encoding='utf-8',sep=',', sheet_name=None, engine=None):
        _ = file_name.split('.')[-1].lower()
        if _ in ['xlsx','xls']:
            if sheet_name:
                df = pd.read_excel(file_name, sheet_name=sheet_name, engine=engine)
            else:
                df = pd.read_excel(file_name, engine=engine)
        elif _ == 'csv':
            df = pd.read_csv(file_name,encoding=encoding,sep=sep)
        else:
            return self.__response(0,'This file is not have open way')
        df_json = df.to_json(orient='split', force_ascii=False)
        return self.__response(content=df_json)

    def save_file(self,data,save_name,pattern='xlsx',encoding='utf-8', train_index=None, test_index=None, x_columns=None, y_columns=None):
        save_name = f'{save_name}.{pattern}'
        if pattern in ['xlsx','xls']:
            if train_index and test_index and isinstance(train_index, list) and isinstance(test_index, list):
                # 划分了测试集与训练集，保存到不同工作表
                max_index = len(data) - 1
                valid_train_index = [i for i in train_index if 0 <= i <= max_index]
                valid_test_index = [i for i in test_index if 0 <= i <= max_index]
                
                if valid_train_index or valid_test_index:
                    with pd.ExcelWriter(save_name, engine='openpyxl') as writer:
                        if valid_train_index:
                            train_data = data.loc[valid_train_index]
                            train_data.to_excel(writer, sheet_name='train', index=False)
                        if valid_test_index:
                            test_data = data.loc[valid_test_index]
                            test_data.to_excel(writer, sheet_name='test', index=False)
                        # 如果指定了自变量和因变量，添加columns_set表
                        if isinstance(x_columns, list) and isinstance(y_columns, list):
                            # 两者都是列表，无论是否为空
                            columns_data = {'x_columns': x_columns, 'y_columns': y_columns}
                            columns_df = pd.DataFrame([columns_data])
                            columns_df.to_excel(writer, sheet_name='columns_set', index=False)
                        elif isinstance(x_columns, list):
                            # 只指定了自变量
                            columns_data = {'x_columns': x_columns, 'y_columns': []}
                            columns_df = pd.DataFrame([columns_data])
                            columns_df.to_excel(writer, sheet_name='columns_set', index=False)
                        elif isinstance(y_columns, list):
                            # 只指定了因变量
                            columns_data = {'x_columns': [], 'y_columns': y_columns}
                            columns_df = pd.DataFrame([columns_data])
                            columns_df.to_excel(writer, sheet_name='columns_set', index=False)
                else:
                    # 如果没有有效索引，保存整个数据集
                    data.to_excel(save_name,index=False)
            else:
                # 未划分测试集与训练集，直接保存
                data.to_excel(save_name,index=False)
            return self.__response(content='save xlsx || xls ok')
        elif pattern == 'csv':
            if train_index and test_index and isinstance(train_index, list) and isinstance(test_index, list):
                # 划分了测试集与训练集，添加标签列
                labeled_data = data.copy()
                labeled_data['dataset_type'] = 'train'
                max_index = len(data) - 1
                valid_test_index = [i for i in test_index if 0 <= i <= max_index]
                if valid_test_index:
                    labeled_data.loc[valid_test_index, 'dataset_type'] = 'test'
                labeled_data.to_csv(save_name,encoding=encoding,sep=',',index=False)
            else:
                # 未划分测试集与训练集，直接保存
                data.to_csv(save_name,encoding=encoding,sep=',',index=False)
            return self.__response(content='save csv ok')
        return self.__response(0,'This file save false.Please feedback')

    def search(self,data,attribute,result,internal='=',relative='',bottom=0,top=None):
        # we will set paginate way
        pass

    def corr(self):
        pass

    def calculate(self, data, save_name=None):
        _ = data.copy()
        i_lst = []
        col_min_lst = []
        col_max_lst = []
        col_mean_lst = []
        col_std_lst = []
        max_abs_lst = []
        col_median_lst = []
        q1_lst = []
        q3_lst = []
        need_columns = self.find_columns(data, ['int64', 'float64'])['content']
        for i in need_columns:
            i_lst.append(i)
            # 不使用inplace=True，而是创建一个临时系列
            temp_series = _[i].dropna()
            if len(temp_series) > 0:
                col_min_lst.append(temp_series.min())
                col_max_lst.append(temp_series.max())
                col_mean_lst.append(temp_series.mean())
                col_std_lst.append(temp_series.std())
                max_abs_lst.append(temp_series.abs().max())
                col_median_lst.append(temp_series.median())
                q1_lst.append(temp_series.quantile(0.25))
                q3_lst.append(temp_series.quantile(0.75))
            else:
                # 如果所有值都是NaN，添加默认值
                col_min_lst.append(0)
                col_max_lst.append(0)
                col_mean_lst.append(0)
                col_std_lst.append(0)
                max_abs_lst.append(0)
                col_median_lst.append(0)
                q1_lst.append(0)
                q3_lst.append(0)
        param = {
            'col_name': i_lst,
            'col_min': col_min_lst,
            'col_max': col_max_lst,
            'col_std': col_std_lst,
            'col_mean': col_mean_lst,
            'col_median': col_median_lst,
            'max_abs': max_abs_lst,
            'q1': q1_lst,
            'q3': q3_lst,
            'save_path': None
        }
        if not (save_name is None):
            with open(save_name, 'w', encoding='utf-8') as f:
                param['save_path'] = save_name
                f.write(json.dumps(param, ensure_ascii=False))
        return self.__response(content=param)

    def normalization(self, data, methods='Min_Max', detail=None,
                      unique_threshold=0.01, save_name=None,
                      save_path=None, columns=None, is_train=True):
        if is_train:
            # 获取所有可能的数值列，包括需要转换的列
            all_columns = data.columns.tolist()
            if columns:
                all_columns = [col for col in all_columns if col in columns]
            
            # 尝试将列转换为数值类型
            for col in all_columns:
                try:
                    # 尝试转换为float类型
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except Exception as e:
                    print(f"Error converting column {col} to numeric: {e}")
            
            # 现在计算统计信息
            if not detail:
                detail = self.calculate(data, save_name)
            col_stats = {}
            for idx, col in enumerate(detail['content']['col_name']):
                col_stats[col] = {
                    'col_min': detail['content']['col_min'][idx],
                    'col_max': detail['content']['col_max'][idx],
                    'col_mean': detail['content']['col_mean'][idx],
                    'col_std': detail['content']['col_std'][idx],
                    'max_abs': detail['content']['max_abs'][idx],
                    'col_median': detail['content']['col_median'][idx],
                    'q1': detail['content']['q1'][idx],
                    'q3': detail['content']['q3'][idx]
                }
            
            # 现在获取所有数值列
            _columns = self.find_columns(data, ['int64', 'float64'])['content']
            need_columns = []
            for col in _columns:
                if columns and col not in columns:
                    continue
                unique_ratio = data[col].nunique() / len(data[col])
                if set(data[col].unique()) == {0, 1}:
                    continue
                if unique_ratio < unique_threshold:
                    continue
                need_columns.append(col)
            norm_save_dict = {}
            for col_name in need_columns:
                stats = col_stats[col_name]
                if methods in ['Min_Max', 'min_max', 'MM', 'mm']:
                    col_min = stats['col_min']
                    col_max = stats['col_max']
                    if col_max - col_min != 0:
                        data[col_name] = (data[col_name] - col_min) / (col_max - col_min)
                    else:
                        data[col_name] = 0.0
                    norm_save_dict[col_name] = {'col_min': float(col_min), 'col_max': float(col_max)}
                elif methods in ['Max', 'max']:
                    col_max = stats['col_max']
                    data[col_name] = data[col_name] / col_max
                    norm_save_dict[col_name] = {'col_max': float(col_max)}
                elif methods in ['Standard', 'standard', 'z', 'Z', 'Z-score', 'z-score']:
                    col_mean = stats['col_mean']
                    col_std = stats['col_std']
                    if col_std != 0:
                        data[col_name] = (data[col_name] - col_mean) / col_std
                    else:
                        data[col_name] = 0.0
                    norm_save_dict[col_name] = {'col_mean': float(col_mean), 'col_std': float(col_std)}
                elif methods in ['Mean', 'mean']:
                    col_mean = stats['col_mean']
                    col_min = stats['col_min']
                    col_max = stats['col_max']
                    if col_max - col_min != 0:
                        data[col_name] = (data[col_name] - col_mean) / (col_max - col_min)
                    norm_save_dict[col_name] = {'col_mean': float(col_mean), 'col_min': float(col_min),
                                                'col_max': float(col_max)}
                elif methods in ['Max_Abs', 'Max_abs', 'max_abs', 'ma', 'MA', 'Ma', 'mA']:
                    max_abs = stats['max_abs']
                    if max_abs != 0:
                        data[col_name] = data[col_name] / max_abs
                    else:
                        data[col_name] = 0.0
                    norm_save_dict[col_name] = {'max_abs': float(max_abs)}
                elif methods in ['robust_standardize', 'Robust_Standardize', 'Robust_standardize', 'robust_Standardize',
                                 'rs']:
                    q1 = stats['q1']
                    q3 = stats['q3']
                    median = stats['col_median']
                    iqr = q3 - q1
                    if iqr != 0:
                        data[col_name] = (data[col_name] - median) / iqr
                    else:
                        data[col_name] = 0.0
                    norm_save_dict[col_name] = {'q1': float(q1), 'q3': float(q3), 'col_median': float(median)}
                else:
                    return self.__response(0, 'This methods is not allow!')

            if save_path is not None:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(norm_save_dict, f, ensure_ascii=False)
        else:
            # 对于测试集，也要确保列是数值类型
            if columns:
                for col in columns:
                    if col in data.columns:
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                        except Exception as e:
                            print(f"Error converting column {col} to numeric: {e}")
            else:
                for col in data.columns:
                    try:
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except Exception as e:
                        print(f"Error converting column {col} to numeric: {e}")
            
            if save_path is None or not os.path.exists(save_path):
                return self.__response(0, 'Save path is required for test mode')
            with open(save_path, 'r', encoding='utf-8') as f:
                norm_save_dict = json.load(f)
            for col_name, params in norm_save_dict.items():
                if columns and col_name not in columns:
                    continue
                if col_name not in data.columns:
                    continue
                if methods in ['Min_Max', 'min_max', 'MM', 'mm']:
                    col_min = params['col_min']
                    col_max = params['col_max']
                    if col_max - col_min != 0:
                        data[col_name] = (data[col_name] - col_min) / (col_max - col_min)
                    else:
                        data[col_name] = 0.0
                elif methods in ['Max', 'max']:
                    col_max = params['col_max']
                    data[col_name] = data[col_name] / col_max
                elif methods in ['Standard', 'standard', 'z', 'Z', 'Z-score', 'z-score']:
                    col_mean = params['col_mean']
                    col_std = params['col_std']
                    if col_std != 0:
                        data[col_name] = (data[col_name] - col_mean) / col_std
                    else:
                        data[col_name] = 0.0
                elif methods in ['Mean', 'mean']:
                    col_mean = params['col_mean']
                    col_min = params['col_min']
                    col_max = params['col_max']
                    if col_max - col_min != 0:
                        data[col_name] = (data[col_name] - col_mean) / (col_max - col_min)
                elif methods in ['Max_Abs', 'Max_abs', 'max_abs', 'ma', 'MA', 'Ma', 'mA']:
                    max_abs = params['max_abs']
                    if max_abs != 0:
                        data[col_name] = data[col_name] / max_abs
                    else:
                        data[col_name] = 0.0
                elif methods in ['robust_standardize', 'Robust_Standardize', 'Robust_standardize', 'robust_Standardize',
                                 'rs']:
                    q1 = params['q1']
                    q3 = params['q3']
                    median = params['col_median']
                    iqr = q3 - q1
                    if iqr != 0:
                        data[col_name] = (data[col_name] - median) / iqr
                    else:
                        data[col_name] = 0.0

        return self.__response(content=data)

    def denormalize(self, data, save_path, columns=None):
        import pandas as pd
        import json
        import os
        try:
            if not os.path.exists(save_path):
                return self.__response(0, 'Save path does not exist')
            
            with open(save_path, 'r', encoding='utf-8') as f:
                norm_save_dict = json.load(f)
            
            # 确保数据是DataFrame
            if not isinstance(data, pd.DataFrame):
                return self.__response(0, 'Input data must be a DataFrame')
            
            # 对指定列进行逆向归一化
            for col_name, params in norm_save_dict.items():
                if columns and col_name not in columns:
                    continue
                if col_name not in data.columns:
                    continue
                
                # 根据保存的参数判断归一化方法并执行逆向操作
                if 'col_min' in params and 'col_max' in params:
                    # Min-Max归一化逆向操作
                    col_min = params['col_min']
                    col_max = params['col_max']
                    if col_max - col_min != 0:
                        data[col_name] = data[col_name] * (col_max - col_min) + col_min
                elif 'col_max' in params and 'col_min' not in params:
                    # Max归一化逆向操作
                    col_max = params['col_max']
                    data[col_name] = data[col_name] * col_max
                elif 'col_mean' in params and 'col_std' in params:
                    # Standard归一化逆向操作
                    col_mean = params['col_mean']
                    col_std = params['col_std']
                    data[col_name] = data[col_name] * col_std + col_mean
                elif 'col_mean' in params and 'col_min' in params and 'col_max' in params:
                    # Mean归一化逆向操作
                    col_mean = params['col_mean']
                    col_min = params['col_min']
                    col_max = params['col_max']
                    if col_max - col_min != 0:
                        data[col_name] = data[col_name] * (col_max - col_min) + col_mean
                elif 'max_abs' in params:
                    # Max_Abs归一化逆向操作
                    max_abs = params['max_abs']
                    data[col_name] = data[col_name] * max_abs
                elif 'q1' in params and 'q3' in params and 'col_median' in params:
                    # Robust_standardize逆向操作
                    q1 = params['q1']
                    q3 = params['q3']
                    median = params['col_median']
                    iqr = q3 - q1
                    if iqr != 0:
                        data[col_name] = data[col_name] * iqr + median
            
            return self.__response(content=data)
        except Exception as e:
            return self.__response(0, str(e))

    def text_to_number(self, data, methods='label', stopwords=None, replace_bool=True,
                       save_path=None, units=None, columns=None, is_train=True):
        if not stopwords:
            stopwords = set()

        # 默认单位列表，可根据需要扩展
        if units is None:
            units = [
                'kg', 'g', 'mg', 'm', 'cm', 'mm', 'km', 'L', 'mL', '℃', '°C', '°F', '%', 'px', 'dpi', 'kb', 'mb', 'gb',
                'tb'
            ]

        need_columns = self.find_columns(data, ['object'])['content']
        if columns:
            # 只处理指定的列
            need_columns = [col for col in need_columns if col in columns]
        if not need_columns:
            return self.__response(content=data)

        text_save_dict = {}

        for col in need_columns:
            text_series = data[col].fillna("")

            if methods == 'remove_unit':
                numeric_values = []
                # 构建单位正则表达式，按长度降序排序避免短单位先匹配
                units_sorted = sorted(units, key=lambda x: -len(x))
                units_pattern = '|'.join([re.escape(unit) for unit in units_sorted])
                # 新增：匹配"包含单位+数字"的正则（确保只处理有单位的内容）
                has_unit_pattern = re.compile(f'.*({units_pattern}).*[-+]?\\d*\\.?\\d+.*', flags=re.IGNORECASE)

                for text in text_series:
                    text_str = str(text).strip()
                    # ========== 核心修改：仅处理包含单位的文本 ==========
                    if not has_unit_pattern.match(text_str):
                        # 无单位：保留原始文本（或改为 np.nan）
                        numeric_values.append(text_str)  # 保留原文
                        # numeric_values.append(np.nan)  # 也可选择填充NaN，根据需求切换
                        continue

                    # 有单位：执行原逻辑提取数字
                    # 先移除单位
                    text_without_unit = re.sub(units_pattern, '', text_str, flags=re.IGNORECASE)
                    # 再提取数字
                    match = re.search(r'[-+]?\d*\.?\d+', text_without_unit)
                    if match:
                        num_str = match.group()
                        if '.' in num_str:
                            numeric_values.append(float(num_str))
                        else:
                            numeric_values.append(int(num_str))
                    else:
                        # 有单位但无数字：保留原文（或填充0/nan）
                        numeric_values.append(text_str)  # 保留原文
                        # numeric_values.append(0)  # 原逻辑，按需切换
                col_name = col if replace_bool else f"{col}_{methods}"
                data[col_name] = numeric_values
                continue

            # ========== 以下为原有逻辑，无修改 ==========
            tokenized_series = []
            for text in text_series:
                words = jieba.lcut(str(text))
                filtered_words = [word for word in words if word not in stopwords and word.strip()]
                tokenized_series.append(filtered_words)
            
            if is_train:
                # 训练模式：计算并保存参数
                all_words = []
                for words in tokenized_series:
                    all_words.extend(words)
                vocab = {word: idx for idx, word in enumerate(sorted(set(all_words)))}
                vocab_size = len(vocab) if vocab else 1
                numeric_values = []
                if methods == 'label':
                    unique_texts = text_series.unique()
                    text2label = {text: idx for idx, text in enumerate(unique_texts)}
                    numeric_values = [text2label[text] for text in text_series]
                    text_save_dict.update(text2label)
                else:
                    bow_vectors = []
                    for words in tokenized_series:
                        vec = np.zeros(vocab_size)
                        word_count = Counter(words)
                        for word, count in word_count.items():
                            if word in vocab:
                                vec[vocab[word]] = count
                        bow_vectors.append(vec)
                    tfidf_vectors = []
                    if 'tfidf' in methods:
                        doc_count = len(tokenized_series)
                        word_doc_count = Counter()
                        for words in tokenized_series:
                            word_doc_count.update(set(words))
                        idf_dict = {}
                        for word in vocab:
                            doc_with_word = word_doc_count.get(word, 0)
                            idf_dict[word] = math.log(doc_count / (doc_with_word + 1))
                        for i, words in enumerate(tokenized_series):
                            vec = np.zeros(vocab_size)
                            total_words = len(words)
                            if total_words > 0:
                                word_count = Counter(words)
                                for word, count in word_count.items():
                                    if word in vocab:
                                        tf = count / total_words
                                        idf = idf_dict.get(word, 0)
                                        vec[vocab[word]] = tf * idf
                            norm = np.linalg.norm(vec)
                            vec = vec / (norm + 1e-8)
                            tfidf_vectors.append(vec)
                    for i in range(len(tokenized_series)):
                        if methods == 'bow_sum':
                            val = float(np.sum(bow_vectors[i]))
                        elif methods == 'bow_mean':
                            val = float(np.mean(bow_vectors[i]))
                        elif methods == 'tfidf_sum':
                            val = float(np.sum(tfidf_vectors[i]))
                        elif methods == 'tfidf_mean':
                            val = float(np.mean(tfidf_vectors[i]))
                        else:
                            val = 0.0
                        numeric_values.append(val)
                    text_save_dict.update(vocab)
            else:
                # 测试模式：加载训练集的参数
                if save_path is None or not os.path.exists(save_path):
                    return self.__response(0, 'Save path is required for test mode')

                with open(save_path, 'r', encoding='utf-8') as f:
                    text_save_dict = json.load(f)

                numeric_values = []
                if methods == 'label':
                    # 对于标签编码，使用训练集的映射
                    for text in text_series:
                        # 如果测试集中的文本在训练集中没有出现，分配一个默认值
                        numeric_values.append(text_save_dict.get(text, 0))
                else:
                    # 对于其他方法，使用训练集的词汇表
                    vocab = text_save_dict
                    vocab_size = len(vocab) if vocab else 1
                    bow_vectors = []
                    for words in tokenized_series:
                        vec = np.zeros(vocab_size)
                        word_count = Counter(words)
                        for word, count in word_count.items():
                            if word in vocab:
                                vec[vocab[word]] = count
                        bow_vectors.append(vec)
                    tfidf_vectors = []
                    if 'tfidf' in methods:
                        # 对于TF-IDF，使用训练集的IDF值
                        # 这里简化处理，只使用词频
                        for i, words in enumerate(tokenized_series):
                            vec = np.zeros(vocab_size)
                            total_words = len(words)
                            if total_words > 0:
                                word_count = Counter(words)
                                for word, count in word_count.items():
                                    if word in vocab:
                                        vec[vocab[word]] = count / total_words
                            norm = np.linalg.norm(vec)
                            vec = vec / (norm + 1e-8)
                            tfidf_vectors.append(vec)
                    for i in range(len(tokenized_series)):
                        if methods == 'bow_sum':
                            val = float(np.sum(bow_vectors[i]))
                        elif methods == 'bow_mean':
                            val = float(np.mean(bow_vectors[i]))
                        elif methods == 'tfidf_sum':
                            val = float(np.sum(tfidf_vectors[i]))
                        elif methods == 'tfidf_mean':
                            val = float(np.mean(tfidf_vectors[i]))
                        else:
                            val = 0.0
                        numeric_values.append(val)
            
            col_name = col if replace_bool else f"{col}_{methods}"
            data[col_name] = numeric_values
            if methods == 'label':
                data[col_name] = data[col_name].astype(int)
            else:
                data[col_name] = data[col_name].astype(float)

        if save_path is not None and methods != 'remove_unit' and is_train:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(text_save_dict, f, ensure_ascii=False)

        return self.__response(content=data)

    def fill_na(self,data,methods=None, columns=None):
        _columns = data.columns
        if columns:
            # 只处理指定的列
            _columns = [col for col in _columns if col in columns]
        _columns_type = self.find_columns(data,['int64', 'float64'])['content']
        _calculate = self.calculate(data)['content']
        if methods is None:
            methods = ['0'] * len(_columns)
        while len(methods) < len(_columns):
            methods.append('0')
        for index, col in enumerate(_columns):
            if methods[index] == '0':
                data[col].fillna(0,inplace=True)
            elif methods[index].lower() in ['mean','median']:
                if col in _columns_type:
                    _index = _columns_type.index(col)
                    _mean = _calculate[f'col_{methods[index].lower()}'][_index]
                    data[col].fillna(_mean,inplace=True)
                else:
                    data[col].fillna(0,inplace=True)
            elif methods[index] == 'mode':
                data[col].fillna(data[col].mode()[0],inplace=True)
            elif methods[index] in ['ffill','bfill']:
                data[col].fillna(method=methods[index],inplace=True)
            else:
                return self.__response(0, 'This function will be update as soon as')
        return self.__response(content=data)

    def drop_duplicate(self,data,subset=None):
        if subset is None:
            data.drop_duplicates(inplace=True)
        elif isinstance(subset,str):
            data.drop_duplicates(subset=[subset], inplace=True)
        elif isinstance(subset,(tuple,list)):
            data.drop_duplicates(subset=subset, inplace=True)
        else:
            self.__response(0,'This subset setting fail')
        return self.__response(content=data)

    def drop_outlier(self,data,methods='multiple',multiple=3, columns=None):
        # 先尝试将所有列转换为数值类型
        all_columns = data.columns.tolist()
        if columns:
            all_columns = [col for col in all_columns if col in columns]
        
        for col in all_columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                print(f"Error converting column {col} to numeric: {e}")
        
        # 现在计算统计信息
        _calculate = self.calculate(data)['content']
        _columns = _calculate['col_name']
        if columns:
            # 只处理指定的列
            _columns = [col for col in _columns if col in columns]
        _mean = _calculate['col_mean']
        _std = _calculate['col_std']
        if methods.lower() == 'multiple':
            for i, col in enumerate(_columns):
                lower = _mean[i] - multiple * _std[i]
                upper = _mean[i] + multiple * _std[i]
                data = data[(data[col] >= lower) & (data[col] <= upper)]
        elif methods.lower() == 'iqr':
            _q1 = _calculate['q1']
            _q3 = _calculate['q3']
            for i, col in enumerate(_columns):
                iqr = _q3[i] - _q1[i]
                lower = _q1[i] - 1.5 * iqr
                upper = _q3[i] + 1.5 * iqr
                data = data[(data[col] >= lower) & (data[col] <= upper)]
        else:
            return self.__response(0,'This function is not allow to use')
        return self.__response(content=data)

    def train_test_split(self, data, test_size=0.2, random_state=None, x_columns=None, y_columns=None, columns=None):
        from sklearn.model_selection import train_test_split
        
        # 确定要使用的列
        if x_columns and y_columns:
            # 使用指定的自变量和因变量
            X = data[x_columns]
            y = data[y_columns]
            train_x, test_x, train_y, test_y = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            # 重建完整的训练集和测试集
            train_data = pd.concat([train_x, train_y], axis=1)
            test_data = pd.concat([test_x, test_y], axis=1)
        elif columns:
            # 使用指定的列
            selected_data = data[columns]
            train_data, test_data = train_test_split(
                selected_data, test_size=test_size, random_state=random_state
            )
        else:
            # 使用所有列
            train_data, test_data = train_test_split(
                data, test_size=test_size, random_state=random_state
            )
        
        # 获取训练集和测试集的索引
        train_index = train_data.index.tolist()
        test_index = test_data.index.tolist()
        
        return self.__response(content={
            'train': {'index': train_index},
            'test': {'index': test_index}
        })

    def json_to_dataframe(self,json_data):
        try:
            json_data = json.loads(json_data)
        except:
            pass
        index = [int(i) for i in json_data['index']]
        columns = json_data['columns']
        _data = json_data['data']
        df = pd.DataFrame(
            data=_data,
            columns=columns,
            index=index
        )
        return self.__response(content=df)

    def dataframe_to_json(self,data):
        data_json = data.to_json(orient='split', force_ascii=False)
        return self.__response(content=data_json)

    def concat(self, dataframes, ignore_index=False):
        try:
            result = pd.concat(dataframes, ignore_index=ignore_index)
            return self.__response(content=result)
        except Exception as e:
            return self.__response(status=0, content=str(e))

    def is_integer(self, value):
        try:
            return pd.api.types.is_integer(value)
        except Exception as e:
            return False

    def is_float(self, value):
        try:
            return pd.api.types.is_float(value)
        except Exception as e:
            return False

    def series_std(self, data):
        try:
            return pd.Series(data).std() if data else 0
        except Exception as e:
            return 0

    def is_dataframe(self, data):
        try:
            return isinstance(data, pd.DataFrame)
        except Exception as e:
            return False

    def filter_data(self, data, column, value):
        try:
            if self.is_dataframe(data):
                return self.__response(content=data[data[column] == value])
            else:
                return self.__response(status=0, content='Input is not a DataFrame')
        except Exception as e:
            return self.__response(status=0, content=str(e))

    def is_categorical(self, data, column, threshold=0.1, max_unique=20):
        try:
            if self.is_dataframe(data) and column in data.columns:
                unique_values = len(set(data[column]))
                total_values = len(data[column])
                return self.__response(content=unique_values < total_values * threshold and unique_values <= max_unique)
            else:
                return self.__response(status=0, content='Input is not a DataFrame or column not found')
        except Exception as e:
            return self.__response(status=0, content=str(e))


# todo-3 test way
if __name__ == '__main__':
    panda = Panda()
    data = pd.read_excel('./static/data/python_book.xlsx')
    _ = panda.text_to_number(data,'tfidf_mean')
    _1 = panda.calculate(_['content'])
    print(_1)
