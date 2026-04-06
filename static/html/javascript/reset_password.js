// 验证码初始化
let resetVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
const resetVerifyImg = document.getElementById('reset-verify-img');
resetVerifyImg.src = `/static/system_pic/${resetVerifyCodeNum}.png`;
resetVerifyImg.onclick = function() {
    resetVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
    this.src = `/static/system_pic/${resetVerifyCodeNum}.png`;
    document.getElementById('reset-verify').value = '';
    document.getElementById('reset-error').style.display = 'none';
};

// 新密码格式校验
const resetNewPwd = document.getElementById('reset-newpwd');
const newPwdTip = document.getElementById('newpwd-tip');
resetNewPwd.oninput = function() {
    const pwd = this.value;
    const pwdReg = /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+-=\[\]{};':"\\|,.<>/?]).{8,16}$/;
    if (pwd.length === 0) {
        newPwdTip.className = 'password-tip';
        newPwdTip.textContent = '';
    } else if (!pwdReg.test(pwd)) {
        newPwdTip.className = 'password-tip error';
        newPwdTip.textContent = 'Password must contain letters, numbers, special chars (8-16 chars)';
    } else {
        newPwdTip.className = 'password-tip success';
        newPwdTip.textContent = 'Password format is valid';
    }
};

// 确认新密码校验
const resetConfirmPwd = document.getElementById('reset-confirmpwd');
const confirmPwdError = document.getElementById('confirmpwd-error');
resetConfirmPwd.oninput = function() {
    const newPwd = resetNewPwd.value;
    const confirmPwd = this.value;
    if (confirmPwd.length === 0) {
        confirmPwdError.style.display = 'none';
    } else if (newPwd !== confirmPwd) {
        confirmPwdError.style.display = 'block';
        confirmPwdError.textContent = 'Passwords do not match';
    } else {
        confirmPwdError.style.display = 'none';
    }
};

// 重置密码提交逻辑（修复解析问题 + 增加调试日志）
const resetBtn = document.getElementById('reset-btn');
resetBtn.addEventListener('click', async function() {
    const userId = document.getElementById('reset-id').value.trim();
    const phone = document.getElementById('reset-phone').value.trim();
    const verifyCode = document.getElementById('reset-verify').value.trim();
    const newPwd = resetNewPwd.value.trim();
    const confirmPwd = resetConfirmPwd.value.trim();
    const resetError = document.getElementById('reset-error');

    // 清空错误提示
    resetError.style.display = 'none';

    // 前端校验
    if (!userId || !phone || !verifyCode || !newPwd || !confirmPwd) {
        resetError.style.display = 'block';
        resetError.textContent = 'Please fill in all required fields';
        return;
    }

    if (!/^1[3-9]\d{9}$/.test(phone)) {
        resetError.style.display = 'block';
        resetError.textContent = 'Invalid phone number (must be 11 digits)';
        return;
    }

    const pwdReg = /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+-=\[\]{};':"\\|,.<>/?]).{8,16}$/;
    if (!pwdReg.test(newPwd)) {
        resetError.style.display = 'block';
        resetError.textContent = 'New password format is invalid';
        return;
    }

    if (newPwd !== confirmPwd) {
        resetError.style.display = 'block';
        resetError.textContent = 'Passwords do not match';
        return;
    }

    // 禁用按钮防止重复提交
    resetBtn.disabled = true;
    resetBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    try {
        console.log('=== 开始请求重置密码 ===');
        console.log('请求参数：', {
            id: userId,
            phone: phone,
            imgCode: `${resetVerifyCodeNum}.png`,
            inputCode: verifyCode,
            password: newPwd
        });

        const response = await fetch('/api/reset_password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                id: userId,
                phone: phone,
                imgCode: `${resetVerifyCodeNum}.png`,
                inputCode: verifyCode,
                password: newPwd // 匹配后端参数名
            })
        });

        // 打印响应状态和原始数据（调试用）
        console.log('响应状态码：', response.status);
        const rawResult = await response.text();
        console.log('响应原始数据：', rawResult);

        // 解析JSON（增加容错）
        let result;
        try {
            result = JSON.parse(rawResult);
            console.log('解析后的响应：', result);
        } catch (e) {
            console.error('JSON解析失败：', e);
            throw new Error('Invalid response format from server');
        }

        // 核心：严格判断后端返回的status为1
        if (result && result.status === 1) {
            resetError.style.display = 'none';

            // 提取新密码（兼容多种格式）
            const msg = result.msg || '';
            // 增强正则：匹配任意字符（包括特殊字符）
            const pwdMatch = msg.match(/Your new password is (.+)/);
            const newPassword = pwdMatch ? pwdMatch[1] : newPwd; // 兜底用输入的密码

            console.log('提取的新密码：', newPassword);
            // 展示重置成功弹窗
            showResetSuccessModal(newPassword);
        } else {
            resetError.style.display = 'block';
            // 适配后端错误提示
            const errorMsg = result?.msg || 'Password reset failed';
            if (errorMsg === 'This image verification code is mistake') {
                resetError.textContent = 'Captcha is incorrect';
            } else {
                resetError.textContent = errorMsg || 'Password reset failed, please check your information';
            }
            console.log('重置失败原因：', errorMsg);
        }
    } catch (error) {
        console.error('请求异常：', error);
        resetError.style.display = 'block';
        resetError.textContent = 'Network error, please try again later';
    } finally {
        // 恢复按钮状态
        resetBtn.disabled = false;
        resetBtn.innerHTML = '<i class="fas fa-save"></i> Reset Password';
    }
});

// 重置密码成功弹窗（确保能正常显示）
function showResetSuccessModal(newPassword) {
    // 先移除可能存在的旧弹窗
    const oldModal = document.querySelector('.reset-success-modal');
    if (oldModal) oldModal.remove();

    // 创建弹窗样式（确保样式生效）
    let style = document.getElementById('reset-modal-style');
    if (!style) {
        style = document.createElement('style');
        style.id = 'reset-modal-style';
        style.textContent = `
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .reset-success-modal {
                position: fixed !important;
                top: 0 !important;
                left: 0 !important;
                width: 100% !important;
                height: 100% !important;
                background-color: rgba(0,0,0,0.5) !important;
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                z-index: 99999 !important; /* 提高层级避免被遮挡 */
            }
            .reset-modal-content {
                background-color: white !important;
                padding: 40px !important;
                border-radius: 12px !important;
                width: 100% !important;
                max-width: 400px !important;
                text-align: center !important;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
                animation: fadeIn 0.3s ease !important;
            }
            .reset-modal-icon {
                color: #52c41a !important;
                font-size: 50px !important;
                margin-bottom: 20px !important;
            }
            .reset-modal-title {
                color: #333 !important;
                font-size: 22px !important;
                font-weight: 600 !important;
                margin-bottom: 10px !important;
            }
            .reset-modal-desc {
                color: #666 !important;
                margin-bottom: 25px !important;
                line-height: 1.5 !important;
            }
            .new-pwd-box {
                background-color: #f9f9f9 !important;
                padding: 15px !important;
                border-radius: 8px !important;
                margin-bottom: 25px !important;
            }
            .new-pwd-label {
                color: #777 !important;
                font-size: 14px !important;
                margin-bottom: 5px !important;
                display: block !important;
            }
            .new-pwd-value {
                color: #667eea !important;
                font-size: 20px !important;
                font-weight: 600 !important;
                letter-spacing: 1px !important;
            }
            .reset-modal-note {
                color: #ff4d4f !important;
                font-size: 13px !important;
                margin-bottom: 30px !important;
                font-weight: 500 !important;
            }
            .reset-modal-close-btn {
                width: 100% !important;
                padding: 12px !important;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                font-size: 16px !important;
                cursor: pointer !important;
                font-weight: 500 !important;
                transition: all 0.3s ease !important;
            }
            .reset-modal-close-btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 15px rgba(102, 126, 234, 0.2) !important;
            }
        `;
        document.head.appendChild(style);
    }

    // 创建弹窗容器
    const modal = document.createElement('div');
    modal.className = 'reset-success-modal';

    // 弹窗内容（确保HTML结构正确）
    modal.innerHTML = `
        <div class="reset-modal-content">
            <div class="reset-modal-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h3 class="reset-modal-title">Password Reset Success!</h3>
            <p class="reset-modal-desc">Your password has been updated successfully.</p>
            <div class="new-pwd-box">
                <span class="new-pwd-label">Your New Password</span>
                <span class="new-pwd-value">${newPassword}</span>
            </div>
            <p class="reset-modal-note">⚠️ Please remember your new password!</p>
            <button class="reset-modal-close-btn">Back to Login</button>
        </div>
    `;

    // 强制添加到body最前面（避免被遮挡）
    document.body.insertBefore(modal, document.body.firstChild);

    // 关闭弹窗事件（确保绑定成功）
    const closeBtn = modal.querySelector('.reset-modal-close-btn');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.remove();
            window.location.href = '/'; // 跳回登录页
        });
    } else {
        console.error('关闭按钮未找到');
    }

    // 点击遮罩层关闭
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
            window.location.href = '/';
        }
    });

    // 确保弹窗可见（调试用）
    console.log('成功弹窗已创建：', modal);
}