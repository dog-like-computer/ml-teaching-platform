# 后端多语言资源文件

i18n = {
    'cn': {
        # 通用响应
        'success': '成功',
        'error': '错误',
        'not_found': '未找到',
        'invalid_input': '无效的输入',
        'unauthorized': '未授权',
        'server_error': '服务器错误',
        'bad_request': '错误的请求',
        'method_not_allowed': '该路由不允许此方法',
        'route_not_found': '该路由未找到',
        'parse_error': '解析错误',
        'empty_request_data': '空请求数据',
        
        # 数据相关
        'data_upload_success': '数据上传成功',
        'data_upload_failed': '数据上传失败',
        'dataset_not_found': '数据集未找到',
        'dataset_deleted': '数据集已删除',
        'dataset_delete_failed': '数据集删除失败',
        'sheet_updated': '工作表名称已更新',
        'sheet_update_failed': '工作表名称更新失败',
        'file_not_found': '文件未找到',
        'unsupported_file_type': '不支持的文件类型',
        'error_reading_excel': '读取Excel文件错误',
        'error_reading_csv': '读取CSV文件错误',
        'missing_columns': '缺少列',
        'train_test_split_failed': '训练集测试集划分失败',
        'preprocess_function_not_supported': '预处理函数不支持',
        
        # 模型相关
        'model_train_success': '模型训练成功',
        'model_train_failed': '模型训练失败',
        'model_not_found': '模型未找到',
        'model_deleted': '模型已删除',
        'model_delete_failed': '模型删除失败',
        'model_renamed': '模型名称已更新',
        'model_rename_failed': '模型名称更新失败',
        'token_updated': '令牌已更新',
        'token_update_failed': '令牌更新失败',
        'no_sdk_available': '无可用的SDK文件',
        'sdk_not_found': 'SDK文件未找到',
        'error_loading_hyperparams': '加载超参数错误',
        'hyperparams_not_found': '超参数文件未找到',
        
        # 预测相关
        'prediction_success': '预测成功',
        'prediction_failed': '预测失败',
        'no_model_available': '无可用模型',
        'invalid_model': '无效的模型',
        'no_valid_input_features': '未提供有效的输入特征',
        
        # 用户相关
        'login_success': '登录成功',
        'login_failed': '登录失败',
        'register_success': '注册成功',
        'register_failed': '注册失败',
        'password_reset': '密码已重置',
        'password_reset_failed': '密码重置失败',
        'account_recovered': '账户已恢复',
        'account_recovery_failed': '账户恢复失败',
        'info_updated': '个人信息已更新',
        'info_update_failed': '个人信息更新失败',
        'account_deleted': '账户已删除',
        'account_delete_failed': '账户删除失败',
        'user_id_required': '用户ID是必需的',
        'user_id_phone_mismatch': '用户ID和电话号码不匹配',
        'phone_common': '该手机号已存在',
        'user_not_created': '该用户未创建，请注册。',
        'password_incorrect': '密码错误，请重置。',
        'load_success': '加载成功',
        'user_id_error': '用户ID错误',
        'system_error': '系统遇到问题',
        'please_load_first': '请先加载数据',
        'user_not_registered': '该用户未注册',
        'id_not_registered': '该ID未注册',
        'create_user_success': '创建用户成功。现在请加载数据。您的ID是',
        
        # 上传相关
        'upload_success': '上传成功',
        'upload_failed': '上传失败',
        'not_multipart_form_data': '请求不是multipart/form-data类型',
        'file_parameter_error': '文件参数错误（需要"file"）',
        'upload_file_null': '上传文件为空',
        
        # 验证码相关
        'captcha_generated': '验证码已生成',
        'captcha_invalid': '验证码无效',
        
        # 其他
        'operation_success': '操作成功',
        'operation_failed': '操作失败',
        'invalid_parameters': '无效的参数',
        'database_error': '数据库错误',
        'redis_error': 'Redis错误',
        'file_open_error': '无法打开该文件，请联系客服',
        'file_open_temporary_error': '暂时无法打开该文件，请联系客服',
        'function_not_allowed': '该功能不允许使用',
        'update_not_allowed': '该更新不允许',
        'cannot_call_other_models': '不能调用其他用户的模型',
        'invalid_token': '无效的令牌',
        'model_not_open': '模型未开放',
        'model_file_not_found': '模型文件未找到'
    },
    'en': {
        # General responses
        'success': 'Success',
        'error': 'Error',
        'not_found': 'Not found',
        'invalid_input': 'Invalid input',
        'unauthorized': 'Unauthorized',
        'server_error': 'Server error',
        'bad_request': 'Bad Request',
        'method_not_allowed': 'This route does not allow this method',
        'route_not_found': 'This route is not found',
        'parse_error': 'Parse error',
        'empty_request_data': 'Empty request data',
        
        # Data related
        'data_upload_success': 'Data uploaded successfully',
        'data_upload_failed': 'Data upload failed',
        'dataset_not_found': 'Dataset not found',
        'dataset_deleted': 'Dataset deleted',
        'dataset_delete_failed': 'Dataset deletion failed',
        'sheet_updated': 'Sheet name updated',
        'sheet_update_failed': 'Sheet name update failed',
        'file_not_found': 'File not found',
        'unsupported_file_type': 'Unsupported file type',
        'error_reading_excel': 'Error reading Excel file',
        'error_reading_csv': 'Error reading CSV file',
        'missing_columns': 'Missing columns',
        'train_test_split_failed': 'Train test split failed',
        'preprocess_function_not_supported': 'Preprocess function not supported',
        
        # Model related
        'model_train_success': 'Model trained successfully',
        'model_train_failed': 'Model training failed',
        'model_not_found': 'Model not found',
        'model_deleted': 'Model deleted',
        'model_delete_failed': 'Model deletion failed',
        'model_renamed': 'Model name updated',
        'model_rename_failed': 'Model name update failed',
        'token_updated': 'Token updated',
        'token_update_failed': 'Token update failed',
        'no_sdk_available': 'No SDK file available',
        'sdk_not_found': 'SDK file not found',
        'error_loading_hyperparams': 'Error loading hyperparams',
        'hyperparams_not_found': 'Hyperparams file not found',
        
        # Prediction related
        'prediction_success': 'Prediction successful',
        'prediction_failed': 'Prediction failed',
        'no_model_available': 'No models available',
        'invalid_model': 'Invalid model',
        'no_valid_input_features': 'No valid input features provided',
        
        # User related
        'login_success': 'Login successful',
        'login_failed': 'Login failed',
        'register_success': 'Registration successful',
        'register_failed': 'Registration failed',
        'password_reset': 'Password reset',
        'password_reset_failed': 'Password reset failed',
        'account_recovered': 'Account recovered',
        'account_recovery_failed': 'Account recovery failed',
        'info_updated': 'Personal information updated',
        'info_update_failed': 'Personal information update failed',
        'account_deleted': 'Account deleted',
        'account_delete_failed': 'Account deletion failed',
        'user_id_required': 'User ID is required',
        'user_id_phone_mismatch': 'User ID and phone number do not match',
        'phone_common': 'This phone is common',
        'user_not_created': 'This user is not created.Please register.',
        'password_incorrect': 'This password is false.Please reset.',
        'load_success': 'Load success',
        'user_id_error': 'User ID error',
        'system_error': 'System find some trouble',
        'please_load_first': 'Please load at first',
        'user_not_registered': 'This user is not register',
        'id_not_registered': 'This id is not register',
        'create_user_success': 'Create user success. Now please load.Your id is',
        
        # Upload related
        'upload_success': 'Upload success',
        'upload_failed': 'File upload fail',
        'not_multipart_form_data': 'Request：is not multipart/form-data type',
        'file_parameter_error': 'File is mistake（need "file"）',
        'upload_file_null': 'Upload file is null',
        
        # Captcha related
        'captcha_generated': 'Captcha generated',
        'captcha_invalid': 'Invalid captcha',
        
        # Other
        'operation_success': 'Operation successful',
        'operation_failed': 'Operation failed',
        'invalid_parameters': 'Invalid parameters',
        'database_error': 'Database error',
        'redis_error': 'Redis error',
        'file_open_error': 'This file we don`t have way to open.Please communicate with customer service',
        'file_open_temporary_error': 'This file is not have way to open temporarily.Please communicate with customer service',
        'function_not_allowed': 'This function is not allow to use',
        'update_not_allowed': 'This update is not allow',
        'cannot_call_other_models': 'Cannot call other users\' models',
        'invalid_token': 'Invalid token',
        'model_not_open': 'Model is not open',
        'model_file_not_found': 'Model file not found'
    }
}

# 获取语言设置
def get_language():
    import json
    try:
        with open('setting.json', 'r', encoding='utf-8') as f:
            setting = json.load(f)
        return setting.get('la', 'cn')
    except Exception:
        return 'cn'

# 获取翻译
def translate(key, lang=None):
    if lang is None:
        lang = get_language()
    return i18n.get(lang, i18n['cn']).get(key, key)
