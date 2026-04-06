// Global Configuration
const userId = localStorage.getItem('user_id')?.trim();
const API_BASE_URL = '/api';
const AVATAR_ROOT_PATH = '/static/user_pic/';

// Copy to clipboard function
function copyToClipboard(text) {
    if (text === '-' || !text) return;
    
    navigator.clipboard.writeText(text)
        .then(() => {
            // Show success message
            const message = document.createElement('div');
            message.className = 'copy-success';
            message.textContent = 'Token copied to clipboard!';
            message.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #4CAF50;
                color: white;
                padding: 10px 15px;
                border-radius: 4px;
                z-index: 1000;
                animation: fadeInOut 2s ease-in-out;
            `;
            document.body.appendChild(message);
            
            setTimeout(() => {
                message.remove();
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy: ', err);
        });
}

// Add CSS for animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translateY(-20px); }
        20% { opacity: 1; transform: translateY(0); }
        80% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .pulse {
        animation: pulse 1s ease-in-out;
    }
`;
document.head.appendChild(style);

// Global State
let modelSelectChangeHandler = null;
let availableColumns = [];
let userInfo = {};
let trainingStatusIntervals = new Map(); // 存储训练状态查询的定时器
let currentModule = 'dataset'; // 当前模块，默认为dataset
let isLoadingModels = false; // 标记是否正在加载模型列表

// Initialize App
document.addEventListener('DOMContentLoaded', async () => {
    // Authentication Check
    if (!userId || userId === 'null' || userId === 'undefined') {
        alert('Please login first!');
        window.location.href = '/static/html/index.html';
        return;
    }

    // Initialize Core Functions
    await initUserInfo();
    initUserMenu();
    initModuleSwitch();
    await loadUserDatasets();
    await loadUserModels(); // Load models on init

    // Bind Events
    bindUploadForm();
    bindTrainForm();
    bindTrainingWayChange();
    bindPredictModule();
    bindLoadColumnsBtn();
    bindDatasetModalEvents();
    bindModelModalEvents(); // Bind model modal events
    bindRefreshButton(); // Bind refresh button event
});

/**
 * Mask Phone Number (Hide Middle 4 Digits)
 */
function maskPhoneNumber(phone) {
    if (!phone || phone.length !== 11) return phone || 'N/A';
    return phone.replace(/(\d{3})\d{4}(\d{4})/, '$1****$2');
}

/**
 * Initialize User Information
 */
async function initUserInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/search_user_info`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId }),
        });

        const data = await response.json();
        if (![200, 1].includes(data?.status)) throw new Error(`API Error: ${data.status}`);

        const user = data.msg || {};
        const fullName = user.username || user.name || userId || "User";
        const shortName = fullName.length > 5 ? `${fullName.substring(0, 5)}...` : fullName;
        const maskedPhone = maskPhoneNumber(user.phone);

        // Avatar Handling
        let avatarUrl = `${AVATAR_ROOT_PATH}primary.jpg`;
        if (user.head_url) {
            let fileName = user.head_url
                .replace('./static/user_pic', '')
                .replace('/static/user_pic', '')
                .replace('user_pic', '')
                .replace(/^\//, '');
            if (!fileName.endsWith('.jpg')) fileName = 'primary.jpg';
            avatarUrl = `${AVATAR_ROOT_PATH}${fileName}`;
        }

        // Update UI
        document.getElementById("user-name-short").innerText = shortName;
        document.getElementById("dropdown-username").innerText = fullName;
        document.getElementById("dropdown-user-id").innerText = user.user_id || "";
        document.getElementById("dropdown-phone").innerText = maskedPhone;

        const avatarElement = document.getElementById("user-avatar-small");
        avatarElement.src = avatarUrl;
        avatarElement.onerror = () => {
            avatarElement.src = `${AVATAR_ROOT_PATH}primary.jpg`;
        };

    } catch (err) {
        console.error("Failed to load user info:", err);
        document.getElementById("user-name-short").innerText = "User";
        document.getElementById("user-avatar-small").src = `${AVATAR_ROOT_PATH}primary.jpg`;
    }
}

/**
 * Initialize User Menu
 */
function initUserMenu() {
    const menuBtn = document.getElementById('user-menu-btn');
    const dropdown = document.getElementById('user-dropdown');

    menuBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        dropdown.classList.toggle('show');
    });

    document.addEventListener('click', () => dropdown.classList.remove('show'));
    dropdown.addEventListener('click', (e) => e.stopPropagation());

    // Menu Actions
    document.getElementById('update-info-btn').addEventListener('click', () => {
        window.location.href = '/user_info';
    });

    document.getElementById('logout-btn').addEventListener('click', () => {
        localStorage.removeItem('user_id');
        alert('Logged out successfully!');
        window.location.href = '/static/html/index.html';
    });

    document.getElementById('delete-account-btn').addEventListener('click', () => {
        if (confirm('Are you sure to delete your account? This action cannot be undone!')) {
            localStorage.removeItem('user_id');
            alert('Account deleted successfully!');
            window.location.href = '/static/html/index.html';
        }
    });
}

/**
 * Initialize Module Switching
 */
function initModuleSwitch() {
    const moduleBtns = document.querySelectorAll('.module-btn');
    const moduleContents = document.querySelectorAll('.module-content');

    moduleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            moduleBtns.forEach(b => b.classList.remove('active'));
            moduleContents.forEach(c => c.classList.remove('active'));

            const moduleName = btn.dataset.module;
            currentModule = moduleName;
            btn.classList.add('active');
            document.getElementById(`${moduleName}-module`).classList.add('active');

            // 停止所有训练状态检查
            stopAllTrainingStatusChecks();

            // Refresh predict model dropdown when switching to predict module
            if (moduleName === 'predict') {
                updatePredictModelDropdown();
            }
            
            // Load models when switching to model module
            if (moduleName === 'model') {
                loadUserModels();
            }
        });
    });

    // Expose switch method for global use
    window.switchToModelModule = function() {
        const modelBtn = document.querySelector('.module-btn[data-module="model"]');
        if (modelBtn) {
            modelBtn.click(); // Simulate click to switch to model module
        }
    };
}

/**
 * Bind Refresh Button Event
 */
function bindRefreshButton() {
    // 在model模块添加刷新按钮
    const modelModule = document.getElementById('model-module');
    if (modelModule) {
        // 检查是否已经有刷新按钮
        let refreshBtn = modelModule.querySelector('.refresh-models-btn');
        if (!refreshBtn) {
            // 创建刷新按钮
            refreshBtn = document.createElement('button');
            refreshBtn.className = 'btn secondary refresh-models-btn';
            refreshBtn.textContent = 'Refresh Models';
            refreshBtn.style.marginBottom = '10px';
            refreshBtn.style.marginRight = '10px';
            
            // 添加到模型列表上方
            const modelList = document.getElementById('model-list');
            if (modelList) {
                modelList.parentNode.insertBefore(refreshBtn, modelList);
            }
        }
        
        // 绑定点击事件
        refreshBtn.addEventListener('click', async () => {
            await loadUserModels();
        });
    }
}

/**
 * Load User Datasets
 */
async function loadUserDatasets() {
    const datasetList = document.getElementById('dataset-list');
    const trainDatasetSelect = document.getElementById('train-dataset');
    const loadColumnsBtn = document.getElementById('load-columns-btn');

    // Loading State
    datasetList.innerHTML = '<div class="loading">Loading datasets...</div>';
    trainDatasetSelect.innerHTML = '<option value="">-- Loading Datasets --</option>';
    loadColumnsBtn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/find_data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, find_way: 'all' })
        });

        if (!response.ok) throw new Error(`Server Error: ${response.status}`);
        const data = await response.json();

        if (data.status === 1 || data.status === 200) {
            const datasets = Array.isArray(data.msg) ? data.msg : [];

            // Update Dataset List
            if (datasets.length === 0) {
                datasetList.innerHTML = '<p class="empty-dataset">No datasets uploaded yet.</p>';
            } else {
                datasetList.innerHTML = '';
                datasets.forEach((ds, index) => {
                    const item = document.createElement('div');
                    item.className = 'dataset-item';
                    item.dataset.datasetId = ds.saving;

                    item.innerHTML = `
                        <div class="dataset-header">
                            <span class="dataset-index">${index + 1}</span>
                            <strong class="dataset-filename">${ds.sheet_name || ds.saving || `Dataset ${index+1}`}</strong>
                            <span class="dataset-tag ${ds.types}">${ds.types || 'Unknown'}</span>
                        </div>
                        <div class="dataset-details">
                            <div class="detail-item">
                                <label>File Name:</label>
                                <span>${ds.saving || '-'}</span>
                            </div>
                            <div class="detail-item">
                                <label>Updated Time:</label>
                                <span>${ds.update_time || '-'}</span>
                            </div>
                            <div class="detail-item">
                                <label>Process Status:</label>
                                <span class="status-${ds.though_work}">${ds.though_work || '-'}</span>
                            </div>
                            <div class="detail-item">
                                <label>User ID:</label>
                                <span>${ds.user_id || '-'}</span>
                            </div>
                        </div>
                        <div class="dataset-item-actions">
                            <button class="btn secondary small dataset-action-btn update-sheet-btn" data-saving="${ds.saving}" data-sheet="${ds.sheet_name || ''}">Update Sheet Name</button>
                            <button class="btn danger small dataset-action-btn delete-dataset-btn" data-id="${ds.saving}">Delete Dataset</button>
                        </div>
                    `;

                    // Click to Detail Page
                    item.addEventListener('click', (e) => {
                        if (!e.target.closest('.dataset-item-actions')) {
                            const datasetId = ds.saving;
                            if (datasetId) {
                                window.location.href = `/static/html/detail_data.html?data_id=${encodeURIComponent(datasetId)}`;
                            } else {
                                alert('Dataset ID is missing!');
                            }
                        }
                    });

                    datasetList.appendChild(item);
                });

                // Bind Dataset Actions
                bindDatasetActionButtons();
            }

            // Update Training Module Dataset Select
            trainDatasetSelect.innerHTML = '<option value="">-- Select Dataset --</option>';
            datasets.forEach((ds) => {
                const option = document.createElement('option');
                option.value = ds.saving;
                // 以sheet_name为主要显示目标，加上saving作为区分
                const displayName = ds.sheet_name || ds.saving;
                option.textContent = `${displayName} (${ds.saving}) (${ds.types})`;
                trainDatasetSelect.appendChild(option);
            });

            // Enable Load Columns Button
            if (datasets.length > 0) loadColumnsBtn.disabled = false;

        } else {
            const errorMsg = data.msg || 'Failed to load datasets';
            datasetList.innerHTML = `<p class="error">${errorMsg}</p>`;
            trainDatasetSelect.innerHTML = '<option value="">-- Load Failed --</option>';
        }

    } catch (error) {
        datasetList.innerHTML = `<p class="error">Network error: ${error.message}</p>`;
        trainDatasetSelect.innerHTML = '<option value="">-- Load Failed --</option>';
    }
}

/**
 * Load User Models
 */
async function loadUserModels() {
    // 防止重复加载
    if (isLoadingModels) {
        console.log('Models are already loading, skipping...');
        return;
    }
    
    const modelList = document.getElementById('model-list');
    const totalModelsEl = document.getElementById('total-models');
    const successModelsEl = document.getElementById('success-models');
    const failedModelsEl = document.getElementById('failed-models');

    // Loading State
    modelList.innerHTML = '<div class="loading">Loading models...</div>';
    totalModelsEl.textContent = '0';
    successModelsEl.textContent = '0';
    failedModelsEl.textContent = '0';

    isLoadingModels = true;

    try {
        // 每次都从 localStorage 重新获取 userId
        const currentUserId = localStorage.getItem('user_id')?.trim();
        console.log('Loading user models...');
        console.log('User ID:', currentUserId);
        
        // Check if userId exists
        if (!currentUserId || currentUserId === 'null' || currentUserId === 'undefined') {
            throw new Error('User ID is missing or invalid');
        }
        
        const response = await fetch(`${API_BASE_URL}/find_models`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: currentUserId })
        });

        console.log('Response status:', response.status);
        
        // 即使状态码不是 200，也要尝试解析 JSON 响应
        let data;
        try {
            data = await response.json();
            console.log('Response data:', data);
        } catch (error) {
            console.error('Error parsing response:', error);
            modelList.innerHTML = '<p class="error">Failed to parse server response</p>';
            return;
        }

        if (!data) {
            modelList.innerHTML = '<p class="error">No response data received</p>';
            return;
        }
        
        // 确保 data 是对象类型
        if (typeof data !== 'object' || data === null) {
            console.error('Invalid response format:', data);
            modelList.innerHTML = '<p class="error">Invalid server response format</p>';
            return;
        }
        
        if (data.status === 1 || data.status === 200) {
            const models = Array.isArray(data.msg) ? data.msg : [];
            console.log('Models data:', models);

            // Update statistics
            const total = models.length;
            const success = models.filter(m => m.status === 'success' || m.status === 1 || m.status === 'train successed').length;
            const failed = models.filter(m => m.status === 'failed' || m.status === 0 || m.status === 'train failed').length;
            const training = models.filter(m => m.status === 'training').length;

            totalModelsEl.textContent = total;
            successModelsEl.textContent = success;
            failedModelsEl.textContent = failed;

            // 移除自动训练状态查询，只在用户手动操作时才查询
            stopAllTrainingStatusChecks();

            // Update Model List
            if (models.length === 0) {
                modelList.innerHTML = '<p class="empty-model">No trained models yet.</p>';
            } else {
                modelList.innerHTML = '';
                models.forEach((model, index) => {
                    const item = document.createElement('div');
                    item.className = 'model-item';
                    item.dataset.modelId = model.id;

                    // Determine status tag
                    let statusClass, statusText;
                    if (model.status === 'success' || model.status === 1 || model.status === 'train successed') {
                        statusClass = 'success';
                        statusText = 'Completed';
                    } else if (model.status === 'failed' || model.status === 0 || model.status === 'train failed') {
                        statusClass = 'failed';
                        statusText = 'Failed';
                    } else if (model.status === 'training') {
                        statusClass = 'training';
                        statusText = 'Training';
                    } else {
                        statusClass = 'unknown';
                        statusText = 'Unknown';
                    }

                    item.innerHTML = `
                        <div class="model-header">
                            <span class="model-index">${index + 1}</span>
                            <strong class="model-name">${model.model_name || `Model ${index+1}`}</strong>
                            <span class="model-tag ${statusClass}">${statusText}</span>
                            <span class="model-tag ${model.model_type}">${model.model_type || 'Unknown'}</span>
                        </div>
                        <div class="model-details">
                            <div class="detail-item">
                                <label>User ID:</label>
                                <span>${model.user_id || '-'}</span>
                            </div>
                            <div class="detail-item">
                                <label>Dataset:</label>
                                <span>${model.data_id || model.data?.data_id || '-'}</span>
                            </div>
                            <div class="detail-item">
                                <label>Types:</label>
                                <span>${model.types || model.model_type || '-'}</span>
                            </div>
                            <div class="detail-item">
                                <label>Token:</label>
                                <span>${model.token || '-'}</span>
                                ${model.token ? `<button class="btn secondary small copy-btn" onclick="copyToClipboard('${model.token}'); event.stopPropagation();">Copy</button>` : ''}
                            </div>
                            <div class="detail-item">
                                <label>URL Path:</label>
                                <span>${model.url_path ? `${(true ? 'p90wpcbruk8x.guyubao.com' : '127.0.0.1:8080')}/${model.url_path}` : '-'}</span>
                                ${model.url_path ? `<button class="btn secondary small copy-btn" onclick="copyToClipboard('${(true ? 'p90wpcbruk8x.guyubao.com' : '127.0.0.1:8080')}/${model.url_path}'); event.stopPropagation();">Copy</button>` : ''}
                            </div>
                            <div class="detail-item">
                                <label>Open:</label>
                                <label class="switch" onclick="event.stopPropagation();">
                                    <input type="checkbox" ${model.open_bool === 1 || model.open_bool === true ? 'checked' : ''} onchange="toggleModelOpen('${model.id}', this.checked); event.stopPropagation();">
                                    <span class="slider round"></span>
                                </label>
                            </div>
                        </div>
                        <div class="model-item-actions">
                            <button class="btn secondary small model-action-btn rename-model-btn" data-id="${model.id}" data-name="${model.model_name}">Rename</button>
                            <button class="btn secondary small model-action-btn update-token-btn" data-id="${model.id}" data-token="${model.token || ''}">Update Token</button>
                            <button class="btn danger small model-action-btn delete-model-btn" data-id="${model.id}">Delete Model</button>
                        </div>
                    `;

                    // Click to Detail Page
                    item.addEventListener('click', (e) => {
                        if (!e.target.closest('.model-item-actions')) {
                            const modelId = model.id;
                            if (modelId) {
                                window.location.href = `/detail_model?user_id=${encodeURIComponent(currentUserId)}&model_id=${encodeURIComponent(modelId)}`;
                            } else {
                                alert('Model ID is missing!');
                            }
                        }
                    });

                    modelList.appendChild(item);
                });

                // Bind Model Actions
                bindModelActionButtons();
            }

            // Update prediction dropdown
            updatePredictModelDropdown(models);
        } else {
            const errorMsg = typeof data.msg === 'string' ? data.msg : 'Failed to load models';
            console.error('Error loading models:', errorMsg);
            modelList.innerHTML = `<p class="error">${errorMsg}</p>`;
        }

    } catch (error) {
        console.error('Error loading user models:', error);
        modelList.innerHTML = `<p class="error">Network error: ${error.message}</p>`;
    } finally {
        isLoadingModels = false;
    }
}

/**
 * Update Predict Model Dropdown
 */
function updatePredictModelDropdown(models = null) {
    const predictModelSelect = document.getElementById('predict-model');

    // If models not provided, fetch them
    if (!models) {
        fetch(`${API_BASE_URL}/find_models`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 1 || data.status === 200) {
                populatePredictModels(data.msg || []);
            }
        })
        .catch(() => {
            predictModelSelect.innerHTML = '<option value="">-- Load Failed --</option>';
        });
        return;
    }

    // Populate dropdown with models
    populatePredictModels(models);
}

/**
 * Populate Predict Model Dropdown
 */
function populatePredictModels(models) {
    const predictModelSelect = document.getElementById('predict-model');
    predictModelSelect.innerHTML = '<option value="">-- Select Model --</option>';

    // Filter only successful models
    const successfulModels = models.filter(m => m.status === 'success' || m.status === 1 || m.status === 'train successed');

    if (successfulModels.length === 0) {
        predictModelSelect.innerHTML = '<option value="">-- No Trained Models Available --</option>';
        return;
    }

    successfulModels.forEach(model => {
        const option = document.createElement('option');
        option.value = model.id;
        // 添加模型ID的一部分、模型算法和数据集信息，以便区分同名模型
        const modelIdShort = model.id.substring(0, 6); // 取模型ID的前6个字符
        const modelAlgorithm = model.model_select || 'Unknown';
        option.textContent = `${model.model_name} (${model.model_type}, ${modelAlgorithm}) [${modelIdShort}]`;
        // 添加数据集信息作为title属性，鼠标悬停时显示
        option.title = `Dataset: ${model.data_id || 'Unknown'}\nModel ID: ${model.id}\nAlgorithm: ${modelAlgorithm}`;
        predictModelSelect.appendChild(option);
    });
}

/**
 * Bind Dataset Action Buttons (Delete/Update)
 */
function bindDatasetActionButtons() {
    // Delete Dataset
    document.querySelectorAll('.delete-dataset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            openDeleteDatasetModal(btn.getAttribute('data-id'));
        });
    });

    // Update Sheet Name
    document.querySelectorAll('.update-sheet-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            openUpdateSheetModal(
                btn.getAttribute('data-saving'),
                btn.getAttribute('data-sheet')
            );
        });
    });
}

/**
 * Bind Model Action Buttons (Delete/Rename)
 */
function bindModelActionButtons() {
    // Delete Model
    document.querySelectorAll('.delete-model-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            openDeleteModelModal(btn.getAttribute('data-id'));
        });
    });

    // Rename Model
    document.querySelectorAll('.rename-model-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            openRenameModelModal(
                btn.getAttribute('data-id'),
                btn.getAttribute('data-name')
            );
        });
    });

    // Update Token
    document.querySelectorAll('.update-token-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            openUpdateTokenModal(
                btn.getAttribute('data-id'),
                btn.getAttribute('data-token')
            );
        });
    });
}

/**
 * Bind Dataset Modal Events
 */
function bindDatasetModalEvents() {
    // Delete Modal
    const deleteModal = document.getElementById('delete-dataset-modal');
    document.getElementById('delete-modal-close').addEventListener('click', () => {
        deleteModal.style.display = 'none';
        document.getElementById('delete-password').value = '';
        showMessage('dataset-actions-result', '', '');
    });

    document.getElementById('cancel-delete').addEventListener('click', () => {
        deleteModal.style.display = 'none';
        document.getElementById('delete-password').value = '';
        showMessage('dataset-actions-result', '', '');
    });

    document.getElementById('confirm-delete').addEventListener('click', async () => {
        const datasetId = document.getElementById('delete-data-id').value;
        const password = document.getElementById('delete-password').value.trim();

        if (!password) {
            showMessage('dataset-actions-result', 'Please enter your password!', 'error');
            return;
        }

        try {
            showMessage('dataset-actions-result', 'Deleting dataset...', 'loading');

            const response = await fetch(`${API_BASE_URL}/delete_data`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    data_id: datasetId,
                    password: password
                })
            });

            const data = await response.json();
            if (data.status === 1 || data.status === 200) {
                showMessage('dataset-actions-result', 'Dataset deleted successfully!', 'success');
                deleteModal.style.display = 'none';
                document.getElementById('delete-password').value = '';
                await loadUserDatasets();
            } else {
                showMessage('dataset-actions-result', data.msg || 'Delete failed', 'error');
            }

        } catch (error) {
            showMessage('dataset-actions-result', `Network error: ${error.message}`, 'error');
        }
    });

    // Update Sheet Modal
    const updateModal = document.getElementById('update-sheet-modal');
    document.getElementById('update-modal-close').addEventListener('click', () => {
        updateModal.style.display = 'none';
        document.getElementById('new-sheet-name').value = '';
        showMessage('dataset-actions-result', '', '');
    });

    document.getElementById('cancel-update').addEventListener('click', () => {
        updateModal.style.display = 'none';
        document.getElementById('new-sheet-name').value = '';
        showMessage('dataset-actions-result', '', '');
    });

    document.getElementById('confirm-update').addEventListener('click', async () => {
        const saving = document.getElementById('update-saving').value;
        const newSheetName = document.getElementById('new-sheet-name').value.trim();

        if (!newSheetName) {
            showMessage('dataset-actions-result', 'Please enter new sheet name!', 'error');
            return;
        }

        try {
            showMessage('dataset-actions-result', 'Updating sheet name...', 'loading');

            const response = await fetch(`${API_BASE_URL}/update_sheet_name`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    saving: saving,
                    sheet_name: newSheetName
                })
            });

            const data = await response.json();
            if (data.status === 1 || data.status === 200) {
                showMessage('dataset-actions-result', 'Sheet name updated successfully!', 'success');
                updateModal.style.display = 'none';
                document.getElementById('new-sheet-name').value = '';
                await loadUserDatasets();
            } else {
                showMessage('dataset-actions-result', data.msg || 'Update failed', 'error');
            }

        } catch (error) {
            showMessage('dataset-actions-result', `Network error: ${error.message}`, 'error');
        }
    });

    // Close Modal When Click Outside
    window.addEventListener('click', (e) => {
        if (e.target === deleteModal) deleteModal.style.display = 'none';
        if (e.target === updateModal) updateModal.style.display = 'none';
    });
}

/**
 * Bind Model Modal Events
 */
function bindModelModalEvents() {
    // Delete Model Modal
    const deleteModal = document.getElementById('delete-model-modal');
    document.getElementById('delete-model-modal-close').addEventListener('click', () => {
        deleteModal.style.display = 'none';
        document.getElementById('delete-model-password').value = '';
        showMessage('model-actions-result', '', '');
    });

    document.getElementById('cancel-delete-model').addEventListener('click', () => {
        deleteModal.style.display = 'none';
        document.getElementById('delete-model-password').value = '';
        showMessage('model-actions-result', '', '');
    });

    document.getElementById('confirm-delete-model').addEventListener('click', async () => {
        const modelId = document.getElementById('delete-model-id').value;
        const password = document.getElementById('delete-model-password').value.trim();

        if (!password) {
            showMessage('model-actions-result', 'Please enter your password!', 'error');
            return;
        }

        try {
            showMessage('model-actions-result', 'Deleting model...', 'loading');

            const response = await fetch(`${API_BASE_URL}/delete_model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    model_id: modelId,
                    password: password
                })
            });

            const data = await response.json();
            if (data.status === 1 || data.status === 200) {
                showMessage('model-actions-result', 'Model deleted successfully!', 'success');
                deleteModal.style.display = 'none';
                document.getElementById('delete-model-password').value = '';
                await loadUserModels();
            } else {
                showMessage('model-actions-result', data.msg || 'Delete failed', 'error');
            }

        } catch (error) {
            showMessage('model-actions-result', `Network error: ${error.message}`, 'error');
        }
    });

    // Rename Model Modal
    const renameModal = document.getElementById('rename-model-modal');
    document.getElementById('rename-model-modal-close').addEventListener('click', () => {
        renameModal.style.display = 'none';
        document.getElementById('new-model-name').value = '';
        showMessage('model-actions-result', '', '');
    });

    document.getElementById('cancel-rename-model').addEventListener('click', () => {
        renameModal.style.display = 'none';
        document.getElementById('new-model-name').value = '';
        showMessage('model-actions-result', '', '');
    });

    document.getElementById('confirm-rename-model').addEventListener('click', async () => {
        const modelId = document.getElementById('rename-model-id').value;
        const newModelName = document.getElementById('new-model-name').value.trim();

        if (!newModelName) {
            showMessage('model-actions-result', 'Please enter new model name!', 'error');
            return;
        }

        try {
            showMessage('model-actions-result', 'Updating model name...', 'loading');

            const response = await fetch(`${API_BASE_URL}/rename_model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    model_id: modelId,
                    new_name: newModelName
                })
            });

            const data = await response.json();
            if (data.status === 1 || data.status === 200) {
                showMessage('model-actions-result', 'Model name updated successfully!', 'success');
                renameModal.style.display = 'none';
                document.getElementById('new-model-name').value = '';
                await loadUserModels();
            } else {
                showMessage('model-actions-result', data.msg || 'Update failed', 'error');
            }

        } catch (error) {
            showMessage('model-actions-result', `Network error: ${error.message}`, 'error');
        }
    });

    // Close Modal When Click Outside
    window.addEventListener('click', (e) => {
        if (e.target === deleteModal) deleteModal.style.display = 'none';
        if (e.target === renameModal) renameModal.style.display = 'none';
        if (e.target === updateTokenModal) updateTokenModal.style.display = 'none';
    });

    // Update Token Modal
    const updateTokenModal = document.getElementById('update-token-modal');
    document.getElementById('update-token-modal-close').addEventListener('click', () => {
        updateTokenModal.style.display = 'none';
        document.getElementById('new-token').value = '';
        showMessage('model-actions-result', '', '');
    });

    document.getElementById('cancel-update-token').addEventListener('click', () => {
        updateTokenModal.style.display = 'none';
        document.getElementById('new-token').value = '';
        showMessage('model-actions-result', '', '');
    });

    document.getElementById('confirm-update-token').addEventListener('click', async () => {
        const modelId = document.getElementById('update-token-model-id').value;
        const newToken = document.getElementById('new-token').value.trim();

        if (!newToken) {
            showMessage('model-actions-result', 'Please enter new token!', 'error');
            return;
        }

        try {
            showMessage('model-actions-result', 'Updating token...', 'loading');

            const response = await fetch(`${API_BASE_URL}/update_model_token`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    model_id: modelId,
                    new_token: newToken
                })
            });

            const data = await response.json();
            if (data.status === 1 || data.status === 200) {
                showMessage('model-actions-result', 'Token updated successfully!', 'success');
                updateTokenModal.style.display = 'none';
                document.getElementById('new-token').value = '';
                await loadUserModels();
            } else {
                showMessage('model-actions-result', data.msg || 'Update failed', 'error');
            }

        } catch (error) {
            showMessage('model-actions-result', `Network error: ${error.message}`, 'error');
        }
    });
}

/**
 * Open Delete Dataset Modal
 */
function openDeleteDatasetModal(datasetId) {
    document.getElementById('delete-data-id').value = datasetId;
    document.getElementById('delete-password').value = '';
    document.getElementById('delete-dataset-modal').style.display = 'flex';
    showMessage('dataset-actions-result', '', '');
}

/**
 * Open Update Sheet Modal
 */
function openUpdateSheetModal(saving, currentSheetName) {
    document.getElementById('update-saving').value = saving;
    document.getElementById('new-sheet-name').value = currentSheetName;
    document.getElementById('update-sheet-modal').style.display = 'flex';
    showMessage('dataset-actions-result', '', '');
}

/**
 * Open Delete Model Modal
 */
function openDeleteModelModal(modelId) {
    document.getElementById('delete-model-id').value = modelId;
    document.getElementById('delete-model-password').value = '';
    document.getElementById('delete-model-modal').style.display = 'flex';
    showMessage('model-actions-result', '', '');
}

/**
 * Open Rename Model Modal
 */
function openRenameModelModal(modelId, currentName) {
    document.getElementById('rename-model-id').value = modelId;
    document.getElementById('new-model-name').value = currentName;
    document.getElementById('rename-model-modal').style.display = 'flex';
    showMessage('model-actions-result', '', '');
}

/**
 * Open Update Token Modal
 */
function openUpdateTokenModal(modelId, currentToken) {
    document.getElementById('update-token-model-id').value = modelId;
    document.getElementById('new-token').value = currentToken;
    document.getElementById('update-token-modal').style.display = 'flex';
    showMessage('model-actions-result', '', '');
}

// Toggle model open status
async function toggleModelOpen(modelId, newStatus) {
    try {
        console.log('Toggling model open status:', modelId, newStatus);
        const response = await fetch(`${API_BASE_URL}/toggle_model_open`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                model_id: modelId, 
                open_bool: newStatus === 'true' || newStatus === true
            })
        });

        console.log('Response status:', response.status);
        const data = await response.json();
        console.log('Response data:', data);
        if (data.status === 1 || data.status === 200) {
            // Refresh model list
            await loadUserModels();
        } else {
            showMessage('model-actions-result', data.msg || 'Failed to update model status', 'error');
        }
    } catch (error) {
        console.error('Error toggling model open status:', error);
        showMessage('model-actions-result', 'Network error', 'error');
    }
}

/**
 * 查询训练状态
 */
async function checkTrainingStatus(modelId) {
    // 只在model模块时才查询
    if (currentModule !== 'model') {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/train_status`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
        });

        if (!response.ok) throw new Error(`Failed to check training status: ${response.status}`);
        const data = await response.json();

        if (data.status === 1 || data.status === 200) {
            const status = data.msg?.status;
            console.log(`Model ${modelId} training status:`, status);

            // 如果训练完成或失败，停止查询
            if (status === 'train successed' || status === 'train failed') {
                stopTrainingStatusCheck(modelId);
                // 不自动刷新模型列表，用户需要手动刷新
                console.log('Model training completed. Please click "Refresh Models" to update the list.');
            }
        }
    } catch (error) {
        console.error('Error checking training status:', error);
    }
}

/**
 * 开始训练状态查询
 */
function startTrainingStatusCheck(modelId) {
    // 如果已经有定时器，直接返回
    if (trainingStatusIntervals.has(modelId)) {
        console.log(`Training status check for model ${modelId} already running`);
        return;
    }

    // 每3分钟查询一次
    const intervalId = setInterval(() => {
        checkTrainingStatus(modelId);
    }, 180000);

    // 存储定时器ID
    trainingStatusIntervals.set(modelId, intervalId);
    console.log(`Started training status check for model ${modelId}`);

    // 延迟1秒后再查询，避免在加载模型列表时频繁调用
    setTimeout(() => {
        checkTrainingStatus(modelId);
    }, 1000);
}

/**
 * 停止训练状态查询
 */
function stopTrainingStatusCheck(modelId) {
    const intervalId = trainingStatusIntervals.get(modelId);
    if (intervalId) {
        clearInterval(intervalId);
        trainingStatusIntervals.delete(modelId);
        console.log(`Stopped training status check for model ${modelId}`);
    }
}

/**
 * 停止所有训练状态查询
 */
function stopAllTrainingStatusChecks() {
    trainingStatusIntervals.forEach((intervalId, modelId) => {
        clearInterval(intervalId);
        console.log(`Stopped training status check for model ${modelId}`);
    });
    trainingStatusIntervals.clear();
}

/**
 * Bind Load Columns Button (Optimized Dropdown Version)
 */
function bindLoadColumnsBtn() {
    const loadColumnsBtn = document.getElementById('load-columns-btn');
    const trainDatasetSelect = document.getElementById('train-dataset');
    const columnsContainer = document.getElementById('columns-container');
    const availableColumnsEl = document.getElementById('available-columns');
    const xTagGroup = document.getElementById('x-tag-group');
    const yTagGroup = document.getElementById('y-tag-group');
    const xColumnSelect = document.getElementById('x-column-select');
    const yColumnSelect = document.getElementById('y-column-select');
    const addXBtn = document.getElementById('add-x-btn');
    const addYBtn = document.getElementById('add-y-btn');

    // Selected Columns State
    let selectedX = [];
    let selectedY = [];

    // Reset On Dataset Change
    trainDatasetSelect.addEventListener('change', () => {
        const selectedDataset = trainDatasetSelect.value;
        loadColumnsBtn.disabled = !selectedDataset;

        // Reset All States
        columnsContainer.style.display = 'none';
        availableColumnsEl.textContent = '--';
        xTagGroup.innerHTML = '';
        yTagGroup.innerHTML = '';
        xColumnSelect.innerHTML = '<option value="">-- Select Column --</option>';
        yColumnSelect.innerHTML = '<option value="">-- Select Column --</option>';
        selectedX = [];
        selectedY = [];
        availableColumns = [];
        
        // 根据文件类型显示或隐藏列配置
        if (selectedDataset) {
            const isXlsxFile = selectedDataset.toLowerCase().endsWith('.xlsx') || selectedDataset.toLowerCase().endsWith('.xls');
            const columnConfigSection = document.querySelector('.form-group:has(#load-columns-btn)');
            if (columnConfigSection) {
                columnConfigSection.style.display = isXlsxFile ? 'none' : 'block';
            }
        } else {
            // 显示列配置
            const columnConfigSection = document.querySelector('.form-group:has(#load-columns-btn)');
            if (columnConfigSection) {
                columnConfigSection.style.display = 'block';
            }
        }
    });

    // Load Columns Click Event
    loadColumnsBtn.addEventListener('click', async () => {
        const dataId = trainDatasetSelect.value;
        if (!dataId) {
            showMessage('train-result', 'Please select a dataset first!', 'error');
            return;
        }

        // 对于xlsx文件，提示用户已经包含列信息
        const isXlsxFile = dataId.toLowerCase().endsWith('.xlsx') || dataId.toLowerCase().endsWith('.xls');
        if (isXlsxFile) {
            showMessage('train-result', 'XLSX file already contains column information, no need to select columns.', 'info');
            return;
        }

        try {
            showMessage('train-result', 'Loading columns from dataset...', 'loading');

            const response = await fetch(`${API_BASE_URL}/find_columns`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data_id: dataId })
            });

            if (!response.ok) throw new Error(`Failed to load columns: ${response.status}`);
            const data = await response.json();

            if (data.status === 1 || data.status === 200) {
                availableColumns = data.msg || [];
                if (availableColumns.length === 0) {
                    showMessage('train-result', 'No columns found in dataset!', 'error');
                    return;
                }

                // Reset Selected Columns
                selectedX = [];
                selectedY = [];

                // Update Dropdown Options
                updateColumnSelects();
                // Render Empty Tags
                renderTags();

                // Show Columns Container
                columnsContainer.style.display = 'block';
                availableColumnsEl.textContent = availableColumns.join(', ');
                showMessage('train-result', 'Columns loaded successfully!', 'success');

            } else {
                showMessage('train-result', data.msg || 'Failed to load columns', 'error');
            }

        } catch (error) {
            showMessage('train-result', `Network error: ${error.message}`, 'error');
        }
    });

    /**
     * Get Available Columns (Not Selected by X/Y)
     */
    function getAvailableColumns() {
        return availableColumns.filter(col => !selectedX.includes(col) && !selectedY.includes(col));
    }

    /**
     * Update Dropdown Options
     */
    function updateColumnSelects() {
        const available = getAvailableColumns();

        // Update X Column Select
        xColumnSelect.innerHTML = '<option value="">-- Select Column --</option>';
        available.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col; // Full column name display
            xColumnSelect.appendChild(option);
        });

        // Update Y Column Select
        yColumnSelect.innerHTML = '<option value="">-- Select Column --</option>';
        available.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col; // Full column name display
            yColumnSelect.appendChild(option);
        });
    }

    /**
     * Render XY Column Tags
     */
    function renderTags() {
        // Clear Existing Tags
        xTagGroup.innerHTML = '';
        yTagGroup.innerHTML = '';

        // Render X Tags
        selectedX.forEach(col => {
            const tag = document.createElement('div');
            tag.className = 'tag';
            tag.innerHTML = `${col} <button class="tag-del" data-col="${col}" data-type="x">&times;</button>`;
            xTagGroup.appendChild(tag);
        });

        // Render Y Tags
        selectedY.forEach(col => {
            const tag = document.createElement('div');
            tag.className = 'tag';
            tag.innerHTML = `${col} <button class="tag-del" data-col="${col}" data-type="y">&times;</button>`;
            yTagGroup.appendChild(tag);
        });

        // Bind Delete Events
        document.querySelectorAll('.tag-del').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const col = e.target.dataset.col;
                const type = e.target.dataset.type;

                // Remove from Selected Arrays
                if (type === 'x') {
                    selectedX = selectedX.filter(c => c !== col);
                } else {
                    selectedY = selectedY.filter(c => c !== col);
                }

                // Update UI
                renderTags();
                updateColumnSelects();
            });
        });
    }

    // Add X Column Button
    addXBtn.addEventListener('click', () => {
        const selectedCol = xColumnSelect.value;
        if (!selectedCol) {
            alert('Please select a column from the dropdown!');
            return;
        }

        // Add to Selected X
        selectedX.push(selectedCol);

        // Update UI
        renderTags();
        updateColumnSelects();

        // Reset Select
        xColumnSelect.value = '';
    });

    // Add Y Column Button
    addYBtn.addEventListener('click', () => {
        const selectedCol = yColumnSelect.value;
        if (!selectedCol) {
            alert('Please select a column from the dropdown!');
            return;
        }

        // Add to Selected Y
        selectedY.push(selectedCol);

        // Update UI
        renderTags();
        updateColumnSelects();

        // Reset Select
        yColumnSelect.value = '';
    });

    // Expose Selected Columns to Global
    window.getSelectedColumns = () => ({
        x: [...selectedX],
        y: [...selectedY]
    });
}

/**
 * Bind Training Method Change Event
 */
function bindTrainingWayChange() {
    const trainWaySelect = document.getElementById('train-way');
    const modelSelect = document.getElementById('train-model');

    trainWaySelect.addEventListener('change', async (e) => {
        const trainWay = e.target.value;
        modelSelect.innerHTML = '<option value="">-- Loading --</option>';
        modelSelect.disabled = true;
        document.getElementById('hyperparams-container').style.display = 'none';

        // Remove Previous Handler
        if (modelSelectChangeHandler) {
            modelSelect.removeEventListener('change', modelSelectChangeHandler);
            modelSelectChangeHandler = null;
        }

        if (!trainWay) {
            modelSelect.innerHTML = '<option value="">-- Select Training Method First --</option>';
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/find_model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, train_way: trainWay })
            });

            if (!response.ok) throw new Error(`Server Error: ${response.status}`);
            const data = await response.json();

            if (data.status === 1 || data.status === 200) {
                const modelList = Array.isArray(data.msg) ? data.msg : [];
                loadSelectOptions(modelSelect, modelList);
                modelSelect.disabled = false;

                // Bind New Model Select Handler
                modelSelectChangeHandler = async () => {
                    await loadModelHyperparams(modelSelect.value);
                };
                modelSelect.addEventListener('change', modelSelectChangeHandler);

            } else {
                modelSelect.innerHTML = '<option value="">-- Load Failed --</option>';
                showMessage('train-result', `Failed to load models: ${data.msg}`, 'error');
            }

        } catch (error) {
            modelSelect.innerHTML = '<option value="">-- Load Failed --</option>';
            showMessage('train-result', `Network error: ${error.message}`, 'error');
        }
    });
}

/**
 * Load Model Hyperparameters
 */
async function loadModelHyperparams(modelName) {
    const container = document.getElementById('hyperparams-container');
    const form = document.getElementById('hyperparams-form');

    // Reset
    form.innerHTML = '';
    container.style.display = 'none';

    if (!modelName) return;

    try {
        const response = await fetch(`${API_BASE_URL}/get_model_hyperparams`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });

        if (!response.ok) throw new Error(`Failed to load hyperparams: ${response.status}`);
        const data = await response.json();

        if (data.status === 1 || data.status === 200) {
            const hyperparamsConfig = data.msg || {};

            // Create Hyperparameter Form
            form.className = 'hyperparams-grid';
            for (const [paramName, config] of Object.entries(hyperparamsConfig)) {
                // Skip 'model' parameter as it shouldn't be displayed as input field
                if (paramName === 'model') {
                    continue;
                }
                
                const group = document.createElement('div');
                group.className = 'hyperparam-group';
                group.setAttribute('data-param', paramName);

                // Label with Tooltip
                const label = document.createElement('label');
                label.textContent = paramName.replace(/_/g, ' ');
                if (config.note) label.title = config.note;
                group.appendChild(label);

                // Input Element
                let input;
                if (config.options && Array.isArray(config.options)) {
                    input = document.createElement('select');
                    input.name = paramName;
                    if (config.note) input.title = config.note;

                    config.options.forEach(opt => {
                        const option = document.createElement('option');
                        option.value = opt;
                        option.textContent = opt;
                        if (opt === config.default) option.selected = true;
                        input.appendChild(option);
                    });

                } else if (config.range && Array.isArray(config.range)) {
                    input = document.createElement('input');
                    input.type = 'number';
                    input.name = paramName;
                    input.min = config.range[0];
                    input.max = config.range[1];
                    input.step = config.type === 'float' ? 'any' : '1';
                    input.value = config.default;
                    // Remove required attribute
                    input.required = false;
                    if (config.note) input.title = config.note;

                } else if (config.type === 'bool') {
                    input = document.createElement('input');
                    input.type = 'checkbox';
                    input.name = paramName;
                    input.checked = config.default === true;
                    input.value = input.checked ? 'true' : 'false';
                    if (config.note) input.title = config.note;

                    input.addEventListener('change', () => {
                        input.value = input.checked ? 'true' : 'false';
                    });

                } else {
                    input = document.createElement('input');
                    input.type = config.type && config.type.includes('int') ? 'number' : 'text';
                    input.name = paramName;
                    input.value = config.default ?? '';
                    // Remove required attribute
                    input.required = false;
                    if (config.note) input.title = config.note;
                }

                group.appendChild(input);
                form.appendChild(group);
            }

            // Show Hyperparameter Container
            container.style.display = 'block';

        } else {
            showMessage('train-result', 'Failed to load hyperparameters', 'error');
        }

    } catch (error) {
        showMessage('train-result', `Network error: ${error.message}`, 'error');
    }
}

/**
 * Populate Select Options
 */
function loadSelectOptions(select, options) {
    select.innerHTML = '';
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select --';
    select.appendChild(defaultOption);

    if (Array.isArray(options) && options.length > 0) {
        options.forEach(item => {
            if (item && item.trim()) {
                const option = document.createElement('option');
                option.value = item.trim();
                option.textContent = item.trim();
                select.appendChild(option);
            }
        });
    } else {
        const noOption = document.createElement('option');
        noOption.value = '';
        noOption.textContent = 'No Options Available';
        select.appendChild(noOption);
    }
}

/**
 * Bind Upload Form
 */
function bindUploadForm() {
    const form = document.getElementById('upload-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const dataFile = document.getElementById('data-file');

        // Validation
        if (!dataFile.files || dataFile.files.length === 0) {
            showMessage('upload-result', 'Please select a file first!', 'error');
            return;
        }

        const allowedTypes = [
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ];
        const fileType = dataFile.files[0].type;
        if (!allowedTypes.includes(fileType)) {
            showMessage('upload-result', 'Only CSV/Excel files are allowed!', 'error');
            return;
        }

        // Upload File
        showMessage('upload-result', 'Uploading file...', 'loading');
        const formData = new FormData();
        formData.append('user_id', userId);
        formData.append('types', document.getElementById('data-type').value);
        formData.append('file', dataFile.files[0]);

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
            const data = await response.json();

            if (data.status === 1 || data.status === 200) {
                showMessage('upload-result', 'File uploaded successfully!', 'success');
                form.reset();
                await loadUserDatasets();
            } else {
                showMessage('upload-result', `Upload failed: ${data.msg}`, 'error');
            }

        } catch (error) {
            showMessage('upload-result', `Network error: ${error.message}`, 'error');
        }
    });
}

/**
 * Bind Training Form
 */
function bindTrainForm() {
    console.log('bindTrainForm function called');
    const form = document.getElementById('train-form');
    console.log('Form element found:', form);
    
    if (!form) {
        console.error('Train form not found!');
        return;
    }
    
    form.addEventListener('submit', async (e) => {
        console.log('Form submit event triggered');
        e.preventDefault();

        // Get Form Values
        const customModelName = document.getElementById('model-name').value.trim();
        const trainDataset = document.getElementById('train-dataset').value;
        const trainWay = document.getElementById('train-way').value; // model_type (Training Method)
        const trainModel = document.getElementById('train-model').value; // model_select (Training Model)
        const { x: xColumnsArray, y: yColumnsArray } = window.getSelectedColumns ? window.getSelectedColumns() : { x: [], y: [] };

        console.log('Form values:', {
            customModelName,
            trainDataset,
            trainWay,
            trainModel,
            xColumnsArray,
            yColumnsArray
        });

        // Validation
        if (!customModelName) {
            showMessage('train-result', 'Please enter a custom model name!', 'error');
            return;
        }
        if (!trainDataset) {
            showMessage('train-result', 'Please select a dataset!', 'error');
            return;
        }
        if (!trainWay || !trainModel) {
            showMessage('train-result', 'Please complete all selections!', 'error');
            return;
        }
        
        // 根据文件类型验证列选择
        const isXlsxFile = trainDataset.toLowerCase().endsWith('.xlsx') || trainDataset.toLowerCase().endsWith('.xls');
        if (!isXlsxFile) {
            // 对于csv文件，需要指定自变量与因变量
            if (xColumnsArray.length === 0) {
                showMessage('train-result', 'Please add X columns!', 'error');
                return;
            }
            if (yColumnsArray.length === 0) {
                showMessage('train-result', 'Please add Y columns!', 'error');
                return;
            }
        }

        // Collect Hyperparameters
        const hyperparams = {};
        document.querySelectorAll('.hyperparam-group').forEach(group => {
            const paramName = group.getAttribute('data-param');
            const input = group.querySelector('input, select');

            if (!input) return;

            // Get Input Value
            let value;
            if (input.type === 'checkbox') {
                value = input.checked;
            } else if (input.type === 'number') {
                value = input.valueAsNumber;
            } else {
                value = input.value.trim();
            }

            // Keep key even if value is empty (let backend handle it)
            hyperparams[paramName] = value;
        });

        // Construct JSON structure required by backend
        const requestData = {
            model_name: customModelName, // Use user-defined model name
            model_select: trainModel,
            model_type: trainWay,
            data: {
                data_id: trainDataset
            },
            hyper_parameter: hyperparams
        };
        
        // 对于csv文件，添加x_columns和y_columns
        if (!isXlsxFile) {
            requestData.data.x_columns = xColumnsArray;
            requestData.data.y_columns = yColumnsArray;
        }

        console.log('Request data:', requestData);

        // Start Training
        showMessage('train-result', 'Training model...', 'loading');

        try {
            console.log('Sending request to:', `${API_BASE_URL}/train`);
            console.log('User ID:', userId);
            
            const response = await fetch(`${API_BASE_URL}/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    ...requestData // Expand JSON structure required by backend
                })
            });

            console.log('Response received:', response);
            
            if (!response.ok) throw new Error(`Training failed: ${response.status}`);
            const data = await response.json();
            console.log('Response data:', data);

            if (data.status === 1 || data.status === 200) {
                showMessage('train-result', 'Model training started successfully!', 'success');

                // 获取模型ID
                const modelId = data.msg?.model_id || data.msg?.id;
                if (modelId) {
                    console.log('Model ID received:', modelId);
                }

                // Auto switch to model module
                window.switchToModelModule();
                
                // 不自动查询训练状态，用户需要手动刷新查看状态
                console.log('Model training started. Please click "Refresh Models" to check status.');

                // Reset Form
                form.reset();
                document.getElementById('train-way').value = '';
                document.getElementById('train-model').value = '';
                document.getElementById('hyperparams-container').style.display = 'none';

                // 不需要再次刷新模型列表，因为switchToModelModule已经调用了loadUserModels()

            } else {
                showMessage('train-result', `Training failed: ${data.msg}`, 'error');
            }

        } catch (error) {
            console.error('Error sending request:', error);
            showMessage('train-result', `Network error: ${error.message}`, 'error');
            // 训练失败时，不重置表单和用户信息
            // 只显示错误消息，保持用户状态不变
        }
    });
}

/**
 * Bind Prediction Module Events
 */
function bindPredictModule() {
    const modelSelect = document.getElementById('predict-model');
    const predictBtn = document.getElementById('predict-btn');
    
    // 存储当前模型信息
    let currentModel = null;
    let modelFeatures = null;

    // 模型选择事件
    modelSelect.addEventListener('change', async () => {
        const selectedModelId = modelSelect.value;
        const modelInfoDiv = document.getElementById('model-info');
        const inputFormContainer = document.getElementById('input-form-container');
        const inputForm = document.getElementById('input-form');
        
        // 重置状态
        predictBtn.disabled = true;
        modelInfoDiv.style.display = 'none';
        inputFormContainer.style.display = 'none';
        inputForm.innerHTML = '';
        showMessage('predict-result', '', '');

        if (!selectedModelId) {
            return;
        }

        // 获取模型信息
        try {
            showMessage('predict-result', 'Loading model information...', 'loading');
            
            // 从模型列表中找到当前模型
            const response = await fetch(`${API_BASE_URL}/find_models`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId })
            });

            if (!response.ok) throw new Error(`Failed to load models: ${response.status}`);
            const data = await response.json();

            if (data.status === 1 || data.status === 200) {
                const models = Array.isArray(data.msg) ? data.msg : [];
                currentModel = models.find(m => m.id === selectedModelId);

                if (!currentModel) {
                    showMessage('predict-result', 'Model not found!', 'error');
                    return;
                }

                // 显示模型信息
                document.getElementById('model-name-display').textContent = currentModel.model_name || 'Unknown';
                document.getElementById('model-dataset-display').textContent = currentModel.data_id || 'Unknown';
                document.getElementById('model-type-display').textContent = currentModel.types || currentModel.model_type || 'Unknown';
                modelInfoDiv.style.display = 'block';

                // 获取模型特征信息
                showMessage('predict-result', 'Loading model features...', 'loading');
                
                const featuresResponse = await fetch(`${API_BASE_URL}/get_model_features`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        id: selectedModelId,
                        user_id: userId,
                        data_id: currentModel.data_id
                    })
                });

                if (!featuresResponse.ok) throw new Error(`Failed to load features: ${featuresResponse.status}`);
                const featuresData = await featuresResponse.json();

                if (featuresData.status === 1 || featuresData.status === 200) {
                    modelFeatures = featuresData.msg;
                    if (modelFeatures && modelFeatures.x_columns) {
                        // 生成输入表单
                        generateInputForm(modelFeatures.x_columns);
                        inputFormContainer.style.display = 'block';
                        predictBtn.disabled = false;
                        showMessage('predict-result', '', '');
                    } else {
                        showMessage('predict-result', 'Failed to load model features!', 'error');
                    }
                } else {
                    showMessage('predict-result', `Failed to load features: ${featuresData.msg}`, 'error');
                }
            } else {
                showMessage('predict-result', `Failed to load models: ${data.msg}`, 'error');
            }
        } catch (error) {
            showMessage('predict-result', `Network error: ${error.message}`, 'error');
        }
    });

    // 预测按钮事件
    predictBtn.addEventListener('click', async () => {
        if (!currentModel || !modelFeatures) {
            showMessage('predict-result', 'Please select a model first!', 'error');
            return;
        }

        // 收集输入数据
        const inputData = { user_id: userId };
        let isValid = true;

        modelFeatures.x_columns.forEach(col => {
            const input = document.getElementById(`input-${col}`);
            if (!input) return;

            const value = input.value.trim();
            if (!value) {
                showMessage('predict-result', `Please enter value for ${col}!`, 'error');
                isValid = false;
                return;
            }

            // 转换为数字
            const numValue = parseFloat(value);
            if (isNaN(numValue)) {
                showMessage('predict-result', `Invalid value for ${col}: must be a number!`, 'error');
                isValid = false;
                return;
            }

            inputData[col] = numValue;
        });

        if (!isValid) return;

        // 添加其他必要信息
        inputData.data_id = currentModel.data_id;
        inputData.token = currentModel.token;

        // 运行预测
        showMessage('predict-result', 'Running prediction...', 'loading');

        try {
            // 使用新的预测接口：/api/predict/{model_id}
            const selectedModelId = modelSelect.value;
            const response = await fetch(`${API_BASE_URL}/predict/${selectedModelId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(inputData)
            });

            if (!response.ok) throw new Error(`Prediction failed: ${response.status}`);
            const data = await response.json();

            if (data.status === 1 || data.status === 200) {
                let result = data.msg;
                if (result) {
                    let resultHtml = '<h3>Prediction Results</h3><div class="prediction-results">';
                    
                    // 处理嵌套的响应格式
                    if (result.status !== undefined && result.content !== undefined) {
                        result = result.content;
                    }
                    
                    // 显示预测结果
                    if (Array.isArray(result)) {
                        // 如果是数组，直接显示
                        resultHtml += `<div class="result-item"><strong>Prediction:</strong> ${result.join(', ')}</div>`;
                    } else if (typeof result === 'object') {
                        // 如果是对象，遍历显示
                        for (const [key, value] of Object.entries(result)) {
                            resultHtml += `<div class="result-item"><strong>${key}:</strong> ${value}</div>`;
                        }
                    } else {
                        // 如果是单个值，直接显示
                        resultHtml += `<div class="result-item"><strong>Prediction:</strong> ${result}</div>`;
                    }
                    
                    resultHtml += '</div>';
                    document.getElementById('predict-result').innerHTML = resultHtml;
                    document.getElementById('predict-result').className = 'success';
                } else {
                    showMessage('predict-result', 'Prediction completed but no results returned!', 'info');
                }
            } else {
                showMessage('predict-result', `Prediction failed: ${data.msg}`, 'error');
            }

        } catch (error) {
            showMessage('predict-result', `Network error: ${error.message}`, 'error');
        }
    });
}

/**
 * Generate Input Form for Model Features
 */
function generateInputForm(xColumns) {
    const inputForm = document.getElementById('input-form');
    inputForm.innerHTML = '';

    xColumns.forEach(col => {
        const formGroup = document.createElement('div');
        formGroup.className = 'form-group input-feature';

        const label = document.createElement('label');
        label.htmlFor = `input-${col}`;
        label.textContent = col;

        const input = document.createElement('input');
        input.type = 'number';
        input.id = `input-${col}`;
        input.name = col;
        input.step = 'any';
        input.placeholder = `Enter value for ${col}`;
        input.required = true;

        formGroup.appendChild(label);
        formGroup.appendChild(input);
        inputForm.appendChild(formGroup);
    });
}

/**
 * Show Feedback Message
 */
function showMessage(elementId, message, type = '') {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Clear Previous State
    element.className = '';
    element.textContent = '';

    if (!message) return;

    // Set Message Content and Style
    element.textContent = message;
    if (type) element.classList.add(type);

    // Auto Clear (Except Loading)
    if (type && type !== 'loading') {
        setTimeout(() => {
            element.textContent = '';
            element.className = '';
        }, 5000);
    }
}