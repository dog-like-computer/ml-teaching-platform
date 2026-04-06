// Execute after page loading is complete
document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const updateForm = document.getElementById('update-user-form');
    const tips = document.getElementById('tips');
    const loading = document.getElementById('loading');
    const userInfoDisplay = document.getElementById('user-info-display');
    const backBtn = document.getElementById('back-btn');

    // DOM elements for displaying user info
    const displayUserId = document.getElementById('display-user-id');
    const displayUsername = document.getElementById('display-username');
    const displayPhone = document.getElementById('display-phone');
    const displayIdentity = document.getElementById('display-identity');
    const displayHeadUrl = document.getElementById('display-head-url');

    // Hidden user ID input
    const updateUserId = document.getElementById('update-user-id');

    // Form elements
    const updateUsername = document.getElementById('update-username');
    const updatePhone = document.getElementById('update-phone');
    const updatePassword = document.getElementById('update-password');
    const verifyPassword = document.getElementById('verify-password');
    const updateIdentity = document.getElementById('update-identity');
    const updateHead = document.getElementById('update-head');

    // --------------------------
    // 核心：页面初始化自动获取用户ID并加载信息
    // --------------------------
    // 方式1：从URL参数获取user_id（推荐，如：?user_id=123）
    function getUserIdFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('user_id');
    }

    // 方式2：从localStorage获取（登录后存储的用户ID）
    function getUserIdFromStorage() {
        return localStorage.getItem('user_id');
    }

    // 方式3：如果有其他方式（如Token解析），可在此扩展
    function getCurrentUserId() {
        // 优先从URL获取，其次从本地存储
        return getUserIdFromUrl() || getUserIdFromStorage() || '';
    }

    // Tips message display function
    function showTips(text, isSuccess = true) {
        tips.className = `tips ${isSuccess ? 'success' : 'error'}`;
        tips.textContent = text;
        // Hide tips after 3 seconds
        setTimeout(() => {
            tips.style.display = 'none';
        }, 3000);
    }

    // Load user information automatically
    async function loadUserInfo() {
        const userId = getCurrentUserId();

        // 校验用户ID是否存在
        if (!userId) {
            loading.textContent = 'User ID not found! Please check your login status.';
            showTips('User ID not found! Please login first.', false);
            return;
        }

        try {
            // Send request to search_user_info interface
            const response = await fetch('/api/search_user_info', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId
                })
            });

            const result = await response.json();
            if (result.status === 1) {
                // Hide loading, show info
                loading.classList.add('hidden');
                userInfoDisplay.classList.remove('hidden');

                // Render user information
                const userInfo = result.msg;
                displayUserId.textContent = userInfo.user_id || '--';
                displayUsername.textContent = userInfo.username || '--';
                displayPhone.textContent = userInfo.phone || '--';
                displayIdentity.textContent = userInfo.identity || '--';

                // Avatar display
                const AVATAR_ROOT_PATH = '/static/user_pic/';
                let avatarUrl = `${AVATAR_ROOT_PATH}primary.jpg`;
                if (userInfo.head_url) {
                    let fileName = userInfo.head_url
                        .replace('./static/user_pic', '')
                        .replace('/static/user_pic', '')
                        .replace('user_pic', '')
                        .replace(/^\//, '');
                    if (!fileName.endsWith('.jpg')) fileName = 'primary.jpg';
                    avatarUrl = `${AVATAR_ROOT_PATH}${fileName}`;
                }
                displayHeadUrl.src = avatarUrl;

                // Auto-fill update form
                updateUserId.value = userInfo.user_id;
                updateUsername.value = userInfo.username || '';
                updatePhone.value = userInfo.phone || '';
                updateIdentity.value = userInfo.identity || '';

                showTips('User information loaded successfully');
            } else {
                loading.textContent = 'Failed to load user information';
                showTips(result.msg || 'Load user information failed', false);
            }
        } catch (error) {
            loading.textContent = 'Network error! Please try again later.';
            showTips('Network error: ' + error.message, false);
        }
    }

    // --------------------------
    // Update user information logic
    // --------------------------
    updateForm.addEventListener('submit', async function(e) {
        e.preventDefault(); // Prevent default form submission

        // Get form data
        const userId = updateUserId.value.trim();
        const username = updateUsername.value.trim();
        const phone = updatePhone.value.trim();
        const password = updatePassword.value.trim();
        const verifyPasswordValue = verifyPassword.value.trim();
        const identity = updateIdentity.value.trim();
        const headFile = updateHead.files[0];

        if (!userId) {
            showTips('User ID is required', false);
            return;
        }

        if (!verifyPasswordValue) {
            showTips('Verify password is required', false);
            return;
        }

        try {
            let formData = new FormData();
            // Basic parameters
            formData.append('func_name', 'update_user');
            formData.append('id', userId);
            if (username) formData.append('username', username);
            if (phone) formData.append('phone', phone);
            if (password) formData.append('password', password);
            if (identity) formData.append('identity', identity);
            // Avatar file (add if exists)
            if (headFile) formData.append('head', headFile);
            // Verify password
            formData.append('verify_password', verifyPasswordValue);

            // Send update request
            const response = await fetch('/api/update_user', {
                method: 'POST',
                body: formData // Use FormData for file upload
            });

            const result = await response.json();
            if (result.status === 1) {
                showTips('Update successfully');
                // Clear password and file input
                updatePassword.value = '';
                verifyPassword.value = '';
                updateHead.value = '';
                // Redirect to home page after successful update
                setTimeout(() => {
                    window.location.href = '/home';
                }, 1000); // Wait 1 second to show success message
            } else {
                showTips(result.msg || 'Update failed', false);
            }
        } catch (error) {
            showTips('Network error: ' + error.message, false);
        }
    });

    // --------------------------
    // Reset form logic
    // --------------------------
    document.getElementById('reset-btn').addEventListener('click', function(e) {
        e.preventDefault();
        // Reset form to original values (not empty)
        updateUsername.value = displayUsername.textContent === '--' ? '' : displayUsername.textContent;
        updatePhone.value = displayPhone.textContent === '--' ? '' : displayPhone.textContent;
        updateIdentity.value = displayIdentity.textContent === '--' ? '' : displayIdentity.textContent;
        updatePassword.value = '';
        verifyPassword.value = '';
        updateHead.value = '';
        showTips('Form reset successfully');
    });

    // --------------------------
    // Back button functionality
    // --------------------------
    backBtn.addEventListener('click', function() {
        window.location.href = '/home';
    });

    // --------------------------
    // Avatar click to select file
    // --------------------------
    displayHeadUrl.addEventListener('click', function() {
        updateHead.click();
    });

    // Preview new avatar before upload
    updateHead.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                displayHeadUrl.src = event.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // Initial load user information
    loadUserInfo();
});