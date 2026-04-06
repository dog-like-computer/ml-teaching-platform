// 验证码初始化
let recoverVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
const recoverVerifyImg = document.getElementById('recover-verify-img');
recoverVerifyImg.src = `/static/system_pic/${recoverVerifyCodeNum}.png`;
recoverVerifyImg.onclick = function() {
    recoverVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
    this.src = `/static/system_pic/${recoverVerifyCodeNum}.png`;
    document.getElementById('recover-verify').value = '';
    document.getElementById('recover-error').style.display = 'none';
    document.getElementById('success-msg').style.display = 'none';
};

// 找回账户提交逻辑（适配后端新逻辑 + 兼容非JSON响应）
const recoverBtn = document.getElementById('recover-btn');
recoverBtn.addEventListener('click', async function() {
    const username = document.getElementById('recover-username').value.trim();
    const phone = document.getElementById('recover-phone').value.trim();
    const verifyCode = document.getElementById('recover-verify').value.trim();
    const recoverError = document.getElementById('recover-error');
    const successMsg = document.getElementById('success-msg');
    const userIdDisplay = document.getElementById('user-id-display');

    // 清空提示
    recoverError.style.display = 'none';
    successMsg.style.display = 'none';

    // 前端校验
    if (!username || !phone || !verifyCode) {
        recoverError.style.display = 'block';
        recoverError.textContent = 'Please fill in all required fields';
        return;
    }

    if (!/^1[3-9]\d{9}$/.test(phone)) {
        recoverError.style.display = 'block';
        recoverError.textContent = 'Invalid phone number (must be 11 digits)';
        return;
    }

    // 禁用按钮防止重复提交
    recoverBtn.disabled = true;
    recoverBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';

    try {
        console.log('=== 找回账户请求 ===');
        console.log('请求参数：', {
            username: username,
            phone: phone,
            imgCode: `${recoverVerifyCodeNum}.png`,
            inputCode: verifyCode
        });

        // 发送请求
        const response = await fetch('/api/recover_account', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                phone: phone,
                imgCode: `${recoverVerifyCodeNum}.png`,
                inputCode: verifyCode
            })
        });

        // 读取原始响应（兼容各种异常）
        const rawResponse = await response.text().catch(err => {
            console.error('读取响应失败：', err);
            return '';
        });
        console.log('后端原始响应：', rawResponse);

        // 处理响应（兼容JSON/非JSON格式）
        let isSuccess = false;
        let responseMsg = '';
        let userId = '';

        // 尝试解析JSON
        try {
            const result = JSON.parse(rawResponse);
            console.log('解析后的JSON：', result);

            // 验证后端返回的status
            if (result.status === 1) {
                isSuccess = true;
                responseMsg = result.msg || '';
                // 提取用户ID（匹配后端格式：Your id is {ID}）
                const idMatch = responseMsg.match(/Your id is (\w+)/);
                userId = idMatch ? idMatch[1] : (result.content?.[0]?.id || '');
            } else {
                isSuccess = false;
                responseMsg = result.msg || 'Account not found';
            }
        } catch (e) {
            // 非JSON响应处理（兼容后端返回None/纯文本）
            console.log('响应不是合法JSON，进入兼容处理：', e);

            // 判断是否成功（根据返回文本特征）
            if (rawResponse.includes('Find your id success') || rawResponse.includes('Your id is')) {
                isSuccess = true;
                // 提取ID
                const idMatch = rawResponse.match(/Your id is (\w+)/);
                userId = idMatch ? idMatch[1] : 'unknown';
                responseMsg = 'Find your id success';
            }
            // 验证码错误
            else if (rawResponse.includes('mistake') || rawResponse.includes('captcha')) {
                isSuccess = false;
                responseMsg = 'Captcha is incorrect';
            }
            // 其他错误
            else if (rawResponse === 'None' || rawResponse.trim() === '') {
                isSuccess = false;
                responseMsg = 'Account not found, please check your information';
            }
            // 其他文本提示
            else {
                isSuccess = false;
                responseMsg = rawResponse || 'Failed to find account';
            }
        }

        // 处理成功场景
        if (isSuccess && userId) {
            successMsg.style.display = 'flex';
            userIdDisplay.textContent = userId;
            recoverError.style.display = 'none';
        }
        // 处理失败场景
        else {
            successMsg.style.display = 'none';
            recoverError.style.display = 'block';
            recoverError.textContent = responseMsg || 'Account not found, please check your username and phone';
        }

    } catch (networkError) {
        // 网络错误处理
        console.error('网络请求异常：', networkError);
        successMsg.style.display = 'none';
        recoverError.style.display = 'block';
        recoverError.textContent = 'Network error, please check your connection';
    } finally {
        // 恢复按钮状态
        recoverBtn.disabled = false;
        recoverBtn.innerHTML = '<i class="fas fa-search"></i> Find Account';
    }
});