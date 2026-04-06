// Initialize login verification code
let loginVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
const loginVerifyImg = document.getElementById('login-verify-img');
loginVerifyImg.src = `/static/system_pic/${loginVerifyCodeNum}.png`;
loginVerifyImg.onclick = function() {
    loginVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
    this.src = `/static/system_pic/${loginVerifyCodeNum}.png`;
    // Clear input box and error prompt after clicking verification code
    document.getElementById('login-verify').value = '';
    document.getElementById('login-error').style.display = 'none';
};

// Initialize registration verification code
let regVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
const regVerifyImg = document.getElementById('reg-verify-img');
regVerifyImg.src = `/static/system_pic/${regVerifyCodeNum}.png`;
regVerifyImg.onclick = function() {
    regVerifyCodeNum = Math.floor(Math.random() * 200) + 1;
    this.src = `/static/system_pic/${regVerifyCodeNum}.png`;
    // Clear input box and error prompt after clicking verification code
    document.getElementById('reg-verify').value = '';
    document.getElementById('reg-error').style.display = 'none';
};

// Tab switch logic (added smooth transition)
const tabs = document.querySelectorAll('.tab');
const formBoxes = document.querySelectorAll('.form-box');
tabs.forEach(tab => {
    tab.addEventListener('click', function() {
        // Clear all error prompts before switching
        document.getElementById('login-error').style.display = 'none';
        document.getElementById('reg-error').style.display = 'none';
        document.getElementById('repwd-error').style.display = 'none';

        tabs.forEach(t => t.classList.remove('active'));
        formBoxes.forEach(f => f.classList.remove('active'));
        this.classList.add('active');
        const tabId = this.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    });
});

// Real-time password verification logic (optimized prompt text)
const regPwd = document.getElementById('reg-password');
const pwdTip = document.getElementById('pwd-tip');
regPwd.oninput = function() {
    const pwd = this.value;
    const pwdReg = /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+-=\[\]{};':"\\|,.<>/?]).{8,16}$/;
    if (pwd.length === 0) {
        pwdTip.className = 'password-tip';
        pwdTip.textContent = '';
    } else if (!pwdReg.test(pwd)) {
        pwdTip.className = 'password-tip error';
        pwdTip.textContent = 'Password must contain letters, numbers, special chars (8-16 chars)';
    } else {
        pwdTip.className = 'password-tip success';
        pwdTip.textContent = 'Password format is valid';
    }
};

// Confirm password verification logic
const regRepwd = document.getElementById('reg-repassword');
const repwdError = document.getElementById('repwd-error');
regRepwd.oninput = function() {
    const pwd = regPwd.value;
    const repwd = this.value;
    if (repwd.length === 0) {
        repwdError.style.display = 'none';
    } else if (pwd !== repwd) {
        repwdError.style.display = 'block';
        repwdError.textContent = 'Passwords do not match';
    } else {
        repwdError.style.display = 'none';
    }
};

// Login API submission logic (fixed: add localStorage to store user_id + optimized jump path)
const loginBtn = document.getElementById('login-btn');
loginBtn.addEventListener('click', async function() {
    const loginId = document.getElementById('login-id').value.trim();
    const loginPwd = document.getElementById('login-password').value.trim();
    const loginVerify = document.getElementById('login-verify').value.trim();
    const loginError = document.getElementById('login-error');

    // Frontend basic verification
    if (!loginId || !loginPwd || !loginVerify) {
        loginError.style.display = 'block';
        loginError.textContent = 'Please fill in all required fields';
        return;
    }

    // Disable button to prevent duplicate submission
    loginBtn.disabled = true;
    loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    try {
        const response = await fetch('/api/load_verification', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                id: loginId,
                password: loginPwd,
                imgCode: `${loginVerifyCodeNum}.png`,
                inputCode: loginVerify
            })
        });
        const result = await response.json();

        // Core: directly display the msg returned by the backend
        if (result.status === 1) {
            loginError.style.display = 'none';
            // 🌟 Key fix: store user_id in localStorage for home.js to read
            localStorage.setItem('user_id', loginId);
            alert('Login success! Redirecting to home page...');
            // Optimized: use route path to jump to home page
            window.location.href = '/home';
        } else {
            loginError.style.display = 'block';
            // Prioritize displaying backend msg, use default value if none
            loginError.textContent = result.msg || 'Login failed, please check your information';
        }
    } catch (error) {
        loginError.style.display = 'block';
        loginError.textContent = 'Network error, please try again later';
    } finally {
        // Restore button state
        loginBtn.disabled = false;
        loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Login';
    }
});

// Registration API submission logic (accurately display backend error msg + extract and display user ID)
const regBtn = document.getElementById('reg-btn');
regBtn.addEventListener('click', async function() {
    const username = document.getElementById('reg-username').value.trim();
    const phone = document.getElementById('reg-phone').value.trim();
    const password = regPwd.value.trim();
    const repassword = regRepwd.value.trim();
    const regVerify = document.getElementById('reg-verify').value.trim();
    const regError = document.getElementById('reg-error');

    // Clear previous error prompts
    regError.style.display = 'none';

    // 1. Frontend basic verification
    if (!username || !phone || !password || !repassword || !regVerify) {
        regError.style.display = 'block';
        regError.textContent = 'Please fill in all required fields';
        return;
    }

    // 2. Phone number format verification
    if (!/^1[3-9]\d{9}$/.test(phone)) {
        regError.style.display = 'block';
        regError.textContent = 'Invalid phone number (must be 11 digits)';
        return;
    }

    // 3. Password complexity verification
    const pwdReg = /^(?=.*[a-zA-Z])(?=.*\d)(?=.*[!@#$%^&*()_+-=\[\]{};':"\\|,.<>/?]).{8,16}$/;
    if (!pwdReg.test(password)) {
        regError.style.display = 'block';
        regError.textContent = 'Password format invalid (letters + numbers + special chars, 8-16 chars)';
        return;
    }

    // 4. Verify that the two passwords are consistent
    if (password !== repassword) {
        regError.style.display = 'block';
        regError.textContent = 'Passwords do not match';
        return;
    }

    // Disable button to prevent duplicate submission
    regBtn.disabled = true;
    regBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';

    try {
        const response = await fetch('/api/register_verification', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                phone: phone,
                password: password,
                imgCode: `${regVerifyCodeNum}.png`,
                inputCode: regVerify
            })
        });
        const result = await response.json();

        // Core: parse the msg returned by the backend, extract the user ID and display it
        if (result.status === 1) {
            regError.style.display = 'none';

            // Extract user ID (match backend return format: Your id is {ID})
            const msg = result.msg || '';
            const idMatch = msg.match(/Your id is (\w+)/);
            const userId = idMatch ? idMatch[1] : 'unknown';

            // Custom popup to display registration success and user ID
            showSuccessModal(userId);

            // Clear registration form
            document.getElementById('reg-username').value = '';
            document.getElementById('reg-phone').value = '';
            document.getElementById('reg-password').value = '';
            document.getElementById('reg-repassword').value = '';
            document.getElementById('reg-verify').value = '';
            pwdTip.textContent = '';
            pwdTip.className = 'password-tip';

            // Delay switching to login tab (optional, will also switch after closing the popup)
            setTimeout(() => {
                tabs.forEach(t => t.classList.remove('active'));
                formBoxes.forEach(f => f.classList.remove('active'));
                document.querySelector('.tab[data-tab="login"]').classList.add('active');
                document.getElementById('login').classList.add('active');
            }, 3000);
        } else {
            regError.style.display = 'block';
            // Prioritize displaying backend msg, use default value if none
            regError.textContent = result.msg || 'Register failed, please check your information';
        }
    } catch (error) {
        regError.style.display = 'block';
        regError.textContent = 'Network error, please try again later';
    } finally {
        // Restore button state
        regBtn.disabled = false;
        regBtn.innerHTML = '<i class="fas fa-user-plus"></i> Register';
    }
});

// Copy user ID function (newly added)
function copyUserId(userId) {
    navigator.clipboard.writeText(userId);
    alert('User ID copied to clipboard!');
}

// Registration success popup (display user ID + add copy function)
function showSuccessModal(userId) {
    // Create popup animation style
    const style = document.createElement('style');
    style.textContent = `
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .success-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        .modal-content {
            background-color: white;
            padding: 40px;
            border-radius: 12px;
            width: 100%;
            max-width: 400px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            animation: fadeIn 0.3s ease;
        }
        .modal-icon {
            color: #52c41a;
            font-size: 50px;
            margin-bottom: 20px;
        }
        .modal-title {
            color: #333;
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .modal-desc {
            color: #666;
            margin-bottom: 25px;
            line-height: 1.5;
        }
        .user-id-box {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .user-id-label {
            color: #777;
            font-size: 14px;
            margin-bottom: 5px;
            display: block;
        }
        .user-id-container {
            display: flex;
            align-items: center;
            gap: 10px;
            justify-content: center;
        }
        .user-id-value {
            color: #667eea;
            font-size: 20px;
            font-weight: 600;
            letter-spacing: 1px;
        }
        .copy-btn {
            background: #eee;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .copy-btn:hover {
            background: #e0e0e0;
        }
        .modal-note {
            color: #777;
            font-size: 13px;
            margin-bottom: 30px;
        }
        .modal-close-btn {
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .modal-close-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(102, 126, 234, 0.2);
        }
    `;
    document.head.appendChild(style);

    // Create popup container
    const modal = document.createElement('div');
    modal.className = 'success-modal';

    // Popup content (added copy button)
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <h3 class="modal-title">Register Success!</h3>
            <p class="modal-desc">Your account has been created successfully.</p>
            <div class="user-id-box">
                <span class="user-id-label">Your User ID</span>
                <div class="user-id-container">
                    <span class="user-id-value">${userId}</span>
                    <button class="copy-btn" onclick="copyUserId('${userId}')">
                        <i class="fas fa-copy"></i>
                    </button>
                </div>
            </div>
            <p class="modal-note">Please remember your ID for future login</p>
            <button class="modal-close-btn">OK, Go to Login</button>
        </div>
    `;

    // Add to page
    document.body.appendChild(modal);

    // Close popup event
    const closeBtn = modal.querySelector('.modal-close-btn');
    closeBtn.addEventListener('click', () => {
        modal.remove();
        // Switch to login tab
        tabs.forEach(t => t.classList.remove('active'));
        formBoxes.forEach(f => f.classList.remove('active'));
        document.querySelector('.tab[data-tab="login"]').classList.add('active');
        document.getElementById('login').classList.add('active');
    });

    // Click mask layer to close popup
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
            // Switch to login tab
            tabs.forEach(t => t.classList.remove('active'));
            formBoxes.forEach(f => f.classList.remove('active'));
            document.querySelector('.tab[data-tab="login"]').classList.add('active');
            document.getElementById('login').classList.add('active');
        }
    });
}