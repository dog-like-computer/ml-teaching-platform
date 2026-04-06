// 全局配置
const API_BASE_URL = '/api';

// 全局状态管理
const state = {
    userId: localStorage.getItem('user_id')?.trim() || '',
    currentData: null,
    originalData: null,
    filteredData: null,
    dataId: '',
    pagination: {
        currentPage: 1,
        pageSize: 10,
        totalRows: 0,
        filteredRows: 0,
        totalPages: 0
    },
    filter: {
        activeConditions: 1,
        maxConditions: 3,
        conditions: [
            { column: 'all', operator: 'contains', value: '' },
            { column: 'all', operator: 'contains', value: '' },
            { column: 'all', operator: 'contains', value: '' }
        ],
        debounceTimer: null,
        debounceDelay: 300
    },
    editing: {
        isEditing: false,
        currentCell: null
    },
    methods: {
        normalization: [],
        vectorization: []
    },
    selectedMethods: {
        normalization: '',
        vectorization: '',
        preprocess: '',
        split: ''
    },
    splitInfo: {
        trainIndex: [],
        testIndex: [],
        xColumns: null,
        yColumns: null
    },
    operationLogs: {
        delete: '',
        update: '',
        vectorization: '',
        normalization: '',
        preprocess: '',
        data_id: '',
        result_data: {}
    }
};

// ========== 核心工具函数（保留原始数据类型） ==========
/**
 * 安全解析JSON，保留原始数据类型（数字/布尔值不转字符串）
 * @param {string} jsonStr - 原始JSON字符串
 * @returns {any} 解析后的数据
 */
function safeJsonParse(jsonStr) {
    const sanitized = jsonStr
        .replace(/\bNaN\b/g, 'null')
        .replace(/\bInfinity\b/g, 'null')
        .replace(/\b-Infinity\b/g, 'null');

    return JSON.parse(sanitized, (key, value) => {
        if (value === null) return '';
        return value;
    });
}

/**
 * 还原单元格数据类型（编辑后恢复原始类型）
 * @param {any} originalValue - 原始值（保留类型）
 * @param {string} inputValue - 编辑后的字符串值
 * @returns {any} 还原类型后的值
 */
function restoreCellType(originalValue, inputValue) {
    if (inputValue === '' || inputValue === 'null') return null;

    switch (typeof originalValue) {
        case 'number':
            const num = parseFloat(inputValue);
            return isNaN(num) ? originalValue : num;
        case 'boolean':
            return inputValue.toLowerCase() === 'true';
        case 'object':
            return originalValue;
        default:
            return inputValue;
    }
}

/**
 * 获取DOM元素（简化代码）
 * @param {string} id - 元素ID
 * @returns {HTMLElement|null}
 */
function el(id) {
    return document.getElementById(id);
}

// ========== 数据加载与渲染 ==========
/**
 * 加载数据集详情（保留原始类型）
 */
async function loadDatasetDetail() {
    const loading = el('loading');
    if (loading) loading.style.display = 'block';

    try {
        // 从URL中获取sheet参数
        const urlParams = new URLSearchParams(window.location.search);
        const sheetName = urlParams.get('sheet');
        
        const response = await fetch(`${API_BASE_URL}/find_data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: state.userId,
                find_way: 'single',
                data_id: state.dataId,
                sheet_name: sheetName
            })
        });

        if (!response.ok) throw new Error(`服务器错误: ${response.status}`);
        const result = await response.json();

        if (![200, 1].includes(result.status) || !result.msg) {
            throw new Error('无效的API响应');
        }

        state.currentData = safeJsonParse(result.msg);
        state.originalData = JSON.parse(JSON.stringify(state.currentData));

        // 确保数据结构统一，添加index字段解决长度不匹配问题
        if (!state.currentData.index) {
            const dataLength = state.currentData.data?.length || state.currentData.length || 0;
            state.currentData.index = Array.from({ length: dataLength }, (_, i) => i);
        }

        // 检查是否存在dataset_type列（CSV文件）或根据工作表名称确定数据集类型（XLSX文件）
        const columns = state.currentData.columns || (Array.isArray(state.currentData[0]) ? [] : Object.keys(state.currentData[0] || {}));
        let rowsSource = state.currentData.data || state.currentData;
        
        // 重置splitInfo
        state.splitInfo.trainIndex = [];
        state.splitInfo.testIndex = [];
        
        // 检查是否存在dataset_type列
        if (columns.includes('dataset_type')) {
            // CSV文件：根据dataset_type列确定训练集和测试集
            rowsSource.forEach((row, index) => {
                const datasetType = Array.isArray(row) ? row[columns.indexOf('dataset_type')] : row['dataset_type'];
                if (datasetType === 'train') {
                    state.splitInfo.trainIndex.push(index);
                } else if (datasetType === 'test') {
                    state.splitInfo.testIndex.push(index);
                }
            });
            
            // 按数据集类型排序：训练集在前，测试集在后
            if (state.splitInfo.trainIndex.length > 0 || state.splitInfo.testIndex.length > 0) {
                const sortedRows = [];
                const sortedIndex = [];
                
                // 先添加训练集数据
                state.splitInfo.trainIndex.forEach(index => {
                    sortedRows.push(rowsSource[index]);
                    sortedIndex.push(index);
                });
                
                // 再添加测试集数据
                state.splitInfo.testIndex.forEach(index => {
                    sortedRows.push(rowsSource[index]);
                    sortedIndex.push(index);
                });
                
                // 更新数据和索引
                if (state.currentData.data) {
                    state.currentData.data = sortedRows;
                } else {
                    state.currentData = sortedRows;
                }
                state.currentData.index = sortedIndex;
                
                // 重新构建splitInfo，确保索引正确
                state.splitInfo.trainIndex = Array.from({ length: state.splitInfo.trainIndex.length }, (_, i) => i);
                state.splitInfo.testIndex = Array.from({ length: state.splitInfo.testIndex.length }, (_, i) => i + state.splitInfo.trainIndex.length);
            }
        } else {
            // XLSX文件：同时加载训练集和测试集
            const dataTypes = state.dataId.split('.').pop().toLowerCase();
            if (['xlsx', 'xls'].includes(dataTypes)) {
                // 加载训练集
                const trainResponse = await fetch(`${API_BASE_URL}/find_data`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: state.userId,
                        find_way: 'single',
                        data_id: state.dataId,
                        sheet_name: 'train'
                    })
                });
                
                if (trainResponse.ok) {
                    const trainResult = await trainResponse.json();
                    if ([200, 1].includes(trainResult.status) && trainResult.msg) {
                        const trainData = safeJsonParse(trainResult.msg);
                        const trainRows = trainData.data || trainData;
                        
                        // 加载测试集
                        const testResponse = await fetch(`${API_BASE_URL}/find_data`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                user_id: state.userId,
                                find_way: 'single',
                                data_id: state.dataId,
                                sheet_name: 'test'
                            })
                        });
                        
                        if (testResponse.ok) {
                            const testResult = await testResponse.json();
                            if ([200, 1].includes(testResult.status) && testResult.msg) {
                                const testData = safeJsonParse(testResult.msg);
                                const testRows = testData.data || testData;
                                
                                // 合并训练集和测试集，训练集在前
                                const mergedRows = [...trainRows, ...testRows];
                                
                                // 更新数据和索引
                                if (state.currentData.data) {
                                    state.currentData.data = mergedRows;
                                } else {
                                    state.currentData = mergedRows;
                                }
                                state.currentData.index = Array.from({ length: mergedRows.length }, (_, i) => i);
                                
                                // 更新splitInfo
                                state.splitInfo.trainIndex = Array.from({ length: trainRows.length }, (_, i) => i);
                                state.splitInfo.testIndex = Array.from({ length: testRows.length }, (_, i) => i + trainRows.length);
                            }
                        }
                    }
                }
            } else {
                // 其他文件类型，根据URL参数确定数据集类型
                const urlParams = new URLSearchParams(window.location.search);
                const sheetName = urlParams.get('sheet');
                if (sheetName) {
                    if (sheetName.toLowerCase() === 'train') {
                        // 所有数据都是训练集
                        rowsSource.forEach((_, index) => {
                            state.splitInfo.trainIndex.push(index);
                        });
                    } else if (sheetName.toLowerCase() === 'test') {
                        // 所有数据都是测试集
                        rowsSource.forEach((_, index) => {
                            state.splitInfo.testIndex.push(index);
                        });
                    }
                }
            }
        }

        state.pagination.totalRows = state.currentData.data?.length || state.currentData.length || 0;
        state.pagination.filteredRows = state.pagination.totalRows;

        renderFilterColumns();
        renderColumnSelects();
        calculatePagination();
        renderPagedDataTable();

    } catch (error) {
        console.error('加载数据失败:', error);
        alert(`加载数据失败: ${error.message}`);
    } finally {
        if (loading) loading.style.display = 'none';
    }
}

/**
 * 加载归一化/向量化方法列表
 */
async function loadFunctionMethods() {
    try {
        const response = await fetch(`${API_BASE_URL}/find_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: state.userId,
                train_way: 'all'
            })
        });

        if (!response.ok) throw new Error(`服务器错误: ${response.status}`);
        const result = await response.json();

        if ((result.status === 200 || result.status === 1) && result.msg && typeof result.msg === 'object') {
            state.methods.normalization = result.msg.normalization || [];
            state.methods.vectorization = result.msg.vectorization || [];
        } else if ((result.status === 200 || result.status === 1) && Array.isArray(result.msg)) {
            if (result.msg.some(item => ['Min_Max', 'Standard'].includes(item))) {
                state.methods.normalization = result.msg;
            } else if (result.msg.some(item => ['bow_sum', 'tfidf_mean', 'remove_unit'].includes(item))) {
                state.methods.vectorization = result.msg;
            }
        }

        renderFunctionSelects();

    } catch (error) {
        console.error('加载方法失败:', error);
        fallbackFunctionMethods();
        renderFunctionSelects();
    }
}

/**
 * 降级方案：默认方法列表
 */
function fallbackFunctionMethods() {
    state.methods.normalization = ['Min_Max', 'Max', 'Standard', 'Mean', 'Max_Abs', 'robust_standardize'];
    state.methods.vectorization = ['label', 'bow_sum', 'bow_mean', 'tfidf_sum', 'tfidf_mean', 'remove_unit'];
}

/**
 * 渲染方法下拉框
 */
function renderFunctionSelects() {
    const normSelect = el('normalization-select');
    const vecSelect = el('vectorization-select');
    const vecUnitInput = el('vectorization-unit');

    if (!normSelect || !vecSelect || !vecUnitInput) return;

    // 渲染归一化方法
    normSelect.innerHTML = '<option value="">Normalization</option>';
    state.methods.normalization.forEach(method => {
        const option = document.createElement('option');
        option.value = method;
        option.textContent = method.replace('_', ' ');
        normSelect.appendChild(option);
    });

    // 渲染向量化方法
    vecSelect.innerHTML = '<option value="">Vectorization</option>';
    state.methods.vectorization.forEach(method => {
        const option = document.createElement('option');
        option.value = method;
        option.textContent = method.replace('_', ' ');
        vecSelect.appendChild(option);
    });

    // 向量化方法切换事件
    vecSelect.addEventListener('change', (e) => {
        const selectedMethod = e.target.value;
        const applyVecBtn = el('apply-vectorize-btn');

        if (selectedMethod === 'remove_unit') {
            vecUnitInput.disabled = false;
            vecUnitInput.placeholder = 'Units (comma separated: kg,cm,℃,元)';
            if (applyVecBtn) applyVecBtn.disabled = !selectedMethod;
        } else {
            vecUnitInput.disabled = selectedMethod ? false : true;
            vecUnitInput.placeholder = 'Units (optional)';
            if (applyVecBtn) applyVecBtn.disabled = !selectedMethod;
        }
    });

    normSelect.disabled = false;
    vecSelect.disabled = false;
    vecUnitInput.disabled = true;
}

/**
 * 重新生成连续索引（解决删除行后索引不匹配核心方法）
 * @param {Object} data - 当前数据集
 */
function regenerateContinuousIndex(data) {
    if (!data) return;
    const dataLength = data.data?.length || data.length || 0;
    // 重新生成从0开始的连续索引
    data.index = Array.from({ length: dataLength }, (_, i) => i);
}

/**
 * 渲染分页数据表格（保留原始类型）
 */
function renderPagedDataTable() {
    const tableHeader = el('table-header');
    const tableBody = el('table-body');
    if (!tableHeader || !tableBody) return;

    tableHeader.innerHTML = '';
    tableBody.innerHTML = '';

    const columns = state.currentData.columns || (Array.isArray(state.currentData[0]) ? [] : Object.keys(state.currentData[0] || {}));
    const rowsSource = state.currentData.data || state.currentData;
    const currentPageRows = getCurrentPageData();

    // 渲染表头
    const headerRow = document.createElement('tr');
    headerRow.appendChild(document.createElement('th'));

    // 添加数据划分标识列（如果已进行数据划分）
    if (state.splitInfo.trainIndex.length > 0 || state.splitInfo.testIndex.length > 0) {
        const splitTh = document.createElement('th');
        splitTh.textContent = 'Dataset';
        headerRow.appendChild(splitTh);
    }

    columns.forEach((col, idx) => {
        const th = document.createElement('th');
        th.textContent = col;

        const deleteBtn = document.createElement('span');
        deleteBtn.className = 'delete-col';
        deleteBtn.textContent = '×';
        deleteBtn.dataset.colIndex = idx;
        th.appendChild(deleteBtn);

        headerRow.appendChild(th);
    });
    tableHeader.appendChild(headerRow);

    // 渲染表体
    currentPageRows.forEach((row, pageRowIdx) => {
        // 修复：使用重新生成的连续索引，而非原始索引
        const originalRowIdx = parseInt((state.pagination.currentPage - 1) * state.pagination.pageSize + pageRowIdx);
        const tr = document.createElement('tr');

        // 删除行按钮
        const deleteCell = document.createElement('td');
        deleteCell.className = 'delete-row';
        deleteCell.textContent = '×';
        deleteCell.dataset.rowIndex = originalRowIdx;
        tr.appendChild(deleteCell);

        // 添加数据划分标识列（如果已进行数据划分）
        if (state.splitInfo.trainIndex.length > 0 || state.splitInfo.testIndex.length > 0) {
            const splitCell = document.createElement('td');
            if (state.splitInfo.trainIndex.includes(originalRowIdx)) {
                splitCell.textContent = 'Train';
                splitCell.style.backgroundColor = '#e6f7ff';
            } else if (state.splitInfo.testIndex.includes(originalRowIdx)) {
                splitCell.textContent = 'Test';
                splitCell.style.backgroundColor = '#fff7e6';
            } else {
                splitCell.textContent = '';
            }
            tr.appendChild(splitCell);
        }

        // 渲染单元格
        columns.forEach((col, colIdx) => {
            const td = document.createElement('td');
            const cellValue = Array.isArray(row) ? row[colIdx] : row[col];
            td.textContent = cellValue === null || cellValue === undefined ? '' : String(cellValue);

            td.dataset.originalValueJson = JSON.stringify(cellValue);
            td.dataset.rowIndex = originalRowIdx;
            td.dataset.colIndex = colIdx;
            td.dataset.colName = col;

            td.addEventListener('dblclick', () => startCellEdit(td));
            td.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    confirmCellEdit();
                }
            });

            tr.appendChild(td);
        });

        tableBody.appendChild(tr);
    });

    updatePaginationUI();
}

/**
 * 获取当前页数据
 * @returns {Array} 当前页数据行
 */
function getCurrentPageData() {
    const rows = state.filteredData || state.currentData.data || state.currentData;
    const start = (state.pagination.currentPage - 1) * state.pagination.pageSize;
    const end = start + state.pagination.pageSize;
    return rows.slice(start, end);
}

/**
 * 计算分页信息
 */
function calculatePagination() {
    state.pagination.totalPages = Math.ceil(state.pagination.filteredRows / state.pagination.pageSize);
    state.pagination.currentPage = Math.min(Math.max(state.pagination.currentPage, 1), state.pagination.totalPages);
}

/**
 * 更新分页UI
 */
function updatePaginationUI() {
    if (el('total-rows')) el('total-rows').textContent = state.pagination.totalRows;
    if (el('filtered-rows')) el('filtered-rows').textContent = state.pagination.filteredRows;
    if (el('current-page')) el('current-page').textContent = state.pagination.currentPage;
    if (el('total-pages')) el('total-pages').textContent = state.pagination.totalPages;
    if (el('prev-page')) el('prev-page').disabled = state.pagination.currentPage <= 1;
    if (el('next-page')) el('next-page').disabled = state.pagination.currentPage >= state.pagination.totalPages;
    if (el('page-size-select')) el('page-size-select').value = state.pagination.pageSize;
}

/**
 * 渲染筛选列下拉框
 */
function renderFilterColumns() {
    const selects = document.querySelectorAll('.filter-column');
    if (!selects.length || !state.currentData) return;

    const columns = state.currentData.columns || Object.keys(state.currentData[0] || {});
    selects.forEach(select => {
        select.innerHTML = '<option value="all">All Columns</option>';
        columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            select.appendChild(opt);
        });
    });
}

/**
 * 渲染所有列选择下拉框
 */
function renderColumnSelects() {
    if (!state.currentData) return;

    const columns = state.currentData.columns || Object.keys(state.currentData[0] || {});
    
    // 填充统一的列选择
    const normList = el('normalization-columns-list');
    if (normList) {
        normList.innerHTML = '';
        columns.forEach(col => {
            const div = document.createElement('div');
            div.style.padding = '4px 8px';
            div.innerHTML = `<input type="checkbox" value="${col}" style="margin-right: 8px;"><label>${col}</label>`;
            normList.appendChild(div);
        });
    }
    
    // 填充数据划分的X列选择（复选框列表）
    const xColumnsList = el('x-columns-list');
    if (xColumnsList) {
        xColumnsList.innerHTML = '';
        columns.forEach(col => {
            const checkboxItem = document.createElement('div');
            checkboxItem.className = 'checkbox-item';
            checkboxItem.innerHTML = `
                <input type="checkbox" id="x-col-${col}" value="${col}">
                <label for="x-col-${col}">${col}</label>
            `;
            xColumnsList.appendChild(checkboxItem);
        });
    }
    
    // 填充数据划分的Y列选择（复选框列表）
    const yColumnsList = el('y-columns-list');
    if (yColumnsList) {
        yColumnsList.innerHTML = '';
        columns.forEach(col => {
            const checkboxItem = document.createElement('div');
            checkboxItem.className = 'checkbox-item';
            checkboxItem.innerHTML = `
                <input type="checkbox" id="y-col-${col}" value="${col}">
                <label for="y-col-${col}">${col}</label>
            `;
            yColumnsList.appendChild(checkboxItem);
        });
    }
    
    // 初始化全选/取消全选按钮
    initCheckboxActions();
    
    // 添加列选择按钮的事件监听器
    addColumnSelectorEvents();
}

/**
 * 初始化复选框全选/取消全选按钮
 */
function initCheckboxActions() {
    // X列全选
    const selectAllX = el('select-all-x');
    if (selectAllX) {
        selectAllX.addEventListener('click', () => {
            const xCheckboxes = document.querySelectorAll('#x-columns-list input[type="checkbox"]');
            xCheckboxes.forEach(checkbox => {
                checkbox.checked = true;
            });
        });
    }
    
    // X列取消全选
    const deselectAllX = el('deselect-all-x');
    if (deselectAllX) {
        deselectAllX.addEventListener('click', () => {
            const xCheckboxes = document.querySelectorAll('#x-columns-list input[type="checkbox"]');
            xCheckboxes.forEach(checkbox => {
                checkbox.checked = false;
            });
        });
    }
    
    // Y列全选
    const selectAllY = el('select-all-y');
    if (selectAllY) {
        selectAllY.addEventListener('click', () => {
            const yCheckboxes = document.querySelectorAll('#y-columns-list input[type="checkbox"]');
            yCheckboxes.forEach(checkbox => {
                checkbox.checked = true;
            });
        });
    }
    
    // Y列取消全选
    const deselectAllY = el('deselect-all-y');
    if (deselectAllY) {
        deselectAllY.addEventListener('click', () => {
            const yCheckboxes = document.querySelectorAll('#y-columns-list input[type="checkbox"]');
            yCheckboxes.forEach(checkbox => {
                checkbox.checked = false;
            });
        });
    }
}



/**
 * 添加列选择器的事件监听器
 */
function addColumnSelectorEvents() {
    // 归一化列选择
    const normBtn = el('normalization-columns-btn');
    const normDropdown = el('normalization-columns-dropdown');
    if (normBtn && normDropdown) {
        normBtn.addEventListener('click', function() {
            normDropdown.style.display = normDropdown.style.display === 'none' ? 'block' : 'none';
        });
    }
    
    // 向量化列选择
    const vecBtn = el('vectorization-columns-btn');
    const vecDropdown = el('vectorization-columns-dropdown');
    if (vecBtn && vecDropdown) {
        vecBtn.addEventListener('click', function() {
            vecDropdown.style.display = vecDropdown.style.display === 'none' ? 'block' : 'none';
        });
    }
    
    // 全选/取消全选按钮
    document.querySelectorAll('.select-all-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const target = this.getAttribute('data-target');
            const list = el(`${target}-columns-list`);
            if (list) {
                list.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                    checkbox.checked = true;
                });
            }
        });
    });
    
    document.querySelectorAll('.deselect-all-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const target = this.getAttribute('data-target');
            const list = el(`${target}-columns-list`);
            if (list) {
                list.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                    checkbox.checked = false;
                });
            }
        });
    });
    
    // 点击页面其他地方关闭下拉框
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.column-selector')) {
            document.querySelectorAll('.columns-dropdown').forEach(dropdown => {
                dropdown.style.display = 'none';
            });
        }
    });
}

// ========== 单元格编辑（保留原始类型） ==========
/**
 * 开始编辑单元格
 * @param {HTMLElement} cell - 单元格元素
 */
function startCellEdit(cell) {
    if (!cell || state.editing.isEditing) return;

    state.editing.isEditing = true;
    state.editing.currentCell = cell;
    cell.contentEditable = true;
    cell.classList.add('editing');
    cell.focus();

    setTimeout(() => {
        const range = document.createRange();
        const sel = window.getSelection();
        range.selectNodeContents(cell);
        sel.removeAllRanges();
        sel.addRange(range);
    }, 0);
}

/**
 * 确认单元格编辑
 */
function confirmCellEdit() {
    if (!state.editing.isEditing || !state.editing.currentCell) return;

    const cell = state.editing.currentCell;
    const originalValue = JSON.parse(cell.dataset.originalValueJson);
    const inputValue = cell.textContent.trim();
    const newValue = restoreCellType(originalValue, inputValue);

    if (JSON.stringify(originalValue) !== JSON.stringify(newValue)) {
        const rowIdx = parseInt(cell.dataset.rowIndex);
        const colName = cell.dataset.colName;
        const colIdx = parseInt(cell.dataset.colIndex);

        updateCellData(rowIdx, colIdx, colName, newValue);

        cell.textContent = newValue === null || newValue === undefined ? '' : String(newValue);
        cell.dataset.originalValueJson = JSON.stringify(newValue);

        const updateLog = `update row${rowIdx} and column${colName} from ${JSON.stringify(originalValue)} to ${JSON.stringify(newValue)}`;
        state.operationLogs.update = state.operationLogs.update
            ? `${state.operationLogs.update};${updateLog}`
            : updateLog;
        state.operationLogs.result_data = state.currentData;
    }

    cell.contentEditable = false;
    cell.classList.remove('editing');
    state.editing.isEditing = false;
    state.editing.currentCell = null;
}

/**
 * 更新单元格数据（保留类型）
 * @param {number} rowIdx - 行索引
 * @param {number} colIdx - 列索引
 * @param {string} colName - 列名
 * @param {any} value - 新值（保留类型）
 */
function updateCellData(rowIdx, colIdx, colName, value) {
    if (!state.currentData) return;

    const rowsSource = state.currentData.data || state.currentData;
    if (Array.isArray(rowsSource[rowIdx])) {
        rowsSource[rowIdx][colIdx] = value;
    } else if (rowsSource[rowIdx]) {
        rowsSource[rowIdx][colName] = value;
    }
}

// ========== 数据划分功能 ==========
/**
 * 初始化数据划分事件监听
 */
function initSplitEvents() {
    console.log('initSplitEvents called');
    const splitToggleBtn = el('split-toggle-btn');
    const applySplitBtn = el('apply-split-btn');
    const splitConfig = el('split-config');

    console.log('splitToggleBtn:', splitToggleBtn);
    console.log('applySplitBtn:', applySplitBtn);
    console.log('splitConfig:', splitConfig);

    if (!splitToggleBtn || !applySplitBtn || !splitConfig) {
        console.log('Missing elements, returning');
        return;
    }

    // 数据划分按钮切换
    let splitActive = false;
    splitToggleBtn.addEventListener('click', () => {
        console.log('splitToggleBtn clicked');
        splitActive = !splitActive;
        state.selectedMethods.split = splitActive ? 'train_test_split' : '';

        console.log('splitActive:', splitActive);
        console.log('state.selectedMethods.split:', state.selectedMethods.split);

        applySplitBtn.disabled = !state.selectedMethods.split;

        if (state.selectedMethods.split) {
            console.log('Showing split config');
            splitConfig.style.display = 'block';
            // 给配置面板添加active类，确保其显示
            const splitConfigPanel = document.getElementById('split-config-panel');
            console.log('splitConfigPanel:', splitConfigPanel);
            if (splitConfigPanel) {
                splitConfigPanel.classList.add('active');
                console.log('Added active class to splitConfigPanel');
            }
        } else {
            console.log('Hiding split config');
            splitConfig.style.display = 'none';
            // 移除active类
            const splitConfigPanel = document.getElementById('split-config-panel');
            if (splitConfigPanel) {
                splitConfigPanel.classList.remove('active');
            }
        }
    });

    // 应用数据划分按钮
    applySplitBtn.addEventListener('click', applySplit);

    // 启用按钮
    splitToggleBtn.disabled = false;
    console.log('initSplitEvents completed');
}

/**
 * 应用数据划分
 */
async function applySplit() {
    if (!state.currentData || !state.selectedMethods.split) {
        alert('未选择数据划分方法！');
        return;
    }

    const testSizeInput = el('split-test-size'); // 使用配置面板中的输入框

    const testSize = parseFloat(testSizeInput.value) || 0.2;
    
    // 从复选框列表中获取X列和Y列
    const xColumns = [];
    const yColumns = [];
    
    // 获取X列
    const xCheckboxes = document.querySelectorAll('#x-columns-list input[type="checkbox"]:checked');
    xCheckboxes.forEach(checkbox => {
        xColumns.push(checkbox.value);
    });
    
    // 获取Y列
    const yCheckboxes = document.querySelectorAll('#y-columns-list input[type="checkbox"]:checked');
    yCheckboxes.forEach(checkbox => {
        yColumns.push(checkbox.value);
    });
    
    const columns = [];
    
    // 当没有选择列时使用默认逻辑：X为所有列除了最后一列，Y为最后一列
           const allColumns = state.currentData.columns || Object.keys(state.currentData[0] || {});
           const xColumnsParam = xColumns.length > 0 ? xColumns : (allColumns.length > 1 ? allColumns.slice(0, -1) : []);
           const yColumnsParam = yColumns.length > 0 ? yColumns : (allColumns.length > 0 ? [allColumns[allColumns.length - 1]] : []);
           const columnsParam = columns.length > 0 ? columns : allColumns;

    console.log('Applying split with:');
    console.log('dataId:', state.dataId);
    console.log('testSize:', testSize);
    console.log('xColumns:', xColumns);
    console.log('yColumns:', yColumns);

    const funcLoading = el('func-loading');
    if (funcLoading) funcLoading.style.display = 'flex';

    try {
        console.log('Sending request to:', `${API_BASE_URL}/train_test_split`);
        const response = await fetch(`${API_BASE_URL}/train_test_split`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                data_id: state.dataId,
                test_size: testSize,
                x_columns: xColumnsParam,
                y_columns: yColumnsParam,
                columns: columnsParam
            })
        });

        console.log('Response status:', response.status);
        console.log('Response ok:', response.ok);

        if (!response.ok) throw new Error(`服务器错误: ${response.status}`);
        const result = await response.json();

        console.log('Response result:', result);

        if ((result.status === 200 || result.status === 1) && result.msg) {
            state.splitInfo.trainIndex = result.msg.train_index;
            state.splitInfo.testIndex = result.msg.test_index;
            state.splitInfo.xColumns = xColumns;
            state.splitInfo.yColumns = yColumns;

            console.log('Split successful!');
            console.log('Train index:', state.splitInfo.trainIndex);
            console.log('Test index:', state.splitInfo.testIndex);

            alert('数据划分成功！');
            renderPagedDataTable();
        } else {
            throw new Error(result.msg || '数据划分失败');
        }

    } catch (error) {
        console.error('数据划分失败:', error);
        console.error('Error stack:', error.stack);
        alert(`数据划分失败: ${error.message}`);
        state.selectedMethods.split = '';
    } finally {
        if (funcLoading) funcLoading.style.display = 'none';
    }
}

// ========== 数据操作（删除行/列 + 归一化/向量化/预处理） ==========
/**
 * 删除列
 * @param {number} colIdx - 列索引
 */
function deleteColumn(colIdx) {
    if (!confirm('确定删除该列吗？')) return;
    colIdx = parseInt(colIdx);

    const columns = state.currentData.columns || Object.keys(state.currentData[0] || {});
    const colName = columns[colIdx];

    const deleteLog = `delete column ${colName}`;
    state.operationLogs.delete = state.operationLogs.delete
        ? `${state.operationLogs.delete};${deleteLog}`
        : deleteLog;

    if (state.currentData.columns) state.currentData.columns.splice(colIdx, 1);
    const rowsSource = state.currentData.data || state.currentData;
    rowsSource.forEach(row => {
        if (Array.isArray(row)) row.splice(colIdx, 1);
        else delete row[colName];
    });

    // 修复：删除列后重新生成索引
    regenerateContinuousIndex(state.currentData);

    state.pagination.filteredRows = rowsSource.length;
    state.pagination.totalRows = rowsSource.length;
    calculatePagination();
    renderFilterColumns();
    renderPagedDataTable();

    alert(`列 "${colName}" 删除成功！`);
}

/**
 * 删除行
 * @param {number} rowIdx - 行索引
 */
function deleteRow(rowIdx) {
    if (!confirm('确定删除该行吗？')) return;
    rowIdx = parseInt(rowIdx);

    const rowsSource = state.currentData.data || state.currentData;
    if (rowIdx < 0 || rowIdx >= rowsSource.length) {
        alert('无效的行索引！');
        return;
    }

    const deleteLog = `delete row ${rowIdx}`;
    state.operationLogs.delete = state.operationLogs.delete
        ? `${state.operationLogs.delete};${deleteLog}`
        : deleteLog;

    // 删除指定行
    rowsSource.splice(rowIdx, 1);

    // 核心修复：删除行后重新生成连续索引，解决长度不匹配问题
    regenerateContinuousIndex(state.currentData);

    state.pagination.filteredRows = rowsSource.length;
    state.pagination.totalRows = rowsSource.length;
    calculatePagination();
    renderPagedDataTable();

    alert(`行 ${rowIdx} 删除成功！`);
}

/**
 * 应用归一化
 */
async function applyNormalization(method) {
    if (!state.currentData) {
        alert('无数据集可处理！');
        return;
    }

    state.selectedMethods.normalization = method;
    const funcLoading = el('func-loading');
    if (funcLoading) funcLoading.style.display = 'flex';

    try {
        // 确保请求数据包含连续索引
        let requestData = {
            ...state.currentData
        };
        regenerateContinuousIndex(requestData);

        const normList = el('normalization-columns-list');
        const selectedColumns = Array.from(normList.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        const columns = selectedColumns.length > 0 ? selectedColumns : state.currentData.columns;

        const response = await fetch(`${API_BASE_URL}/normalization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                normalization: method,
                data: requestData,
                columns: columns,
                user_id: state.userId,
                data_id: state.dataId
            })
        });

        if (!response.ok) throw new Error(`服务器错误: ${response.status}`);
        const result = await response.json();

        if ((result.status === 200 || result.status === 1) && result.msg) {
            const processedData = safeJsonParse(result.msg);
            // 修复：处理返回数据后重新生成索引
            regenerateContinuousIndex(processedData);
            
            // 使用全部归一化结果，包括训练集和测试集
            state.currentData = processedData;

            state.pagination.filteredRows = state.currentData.data?.length || state.currentData.length || 0;
            state.pagination.totalRows = state.pagination.filteredRows;

            state.operationLogs.normalization = method;
            state.operationLogs.result_data = state.currentData;

            calculatePagination();
            renderColumnSelects();
            renderPagedDataTable();
            // 确保数据划分配置面板的显示状态
            if (state.selectedMethods.split) {
                const splitConfig = el('split-config');
                const splitConfigPanel = el('split-config-panel');
                if (splitConfig) splitConfig.style.display = 'block';
                if (splitConfigPanel) splitConfigPanel.classList.add('active');
            }
            alert(`归一化（${method}）应用成功！`);
        } else {
            throw new Error(result.message || '归一化失败');
        }

    } catch (error) {
        console.error('归一化失败:', error);
        alert(`归一化失败: ${error.message}`);
        state.selectedMethods.normalization = '';
    } finally {
        if (funcLoading) funcLoading.style.display = 'none';
    }
}

/**
 * 应用向量化
 */
async function applyVectorization(method) {
    if (!state.currentData) {
        alert('无数据集可处理！');
        return;
    }

    const vecUnitInput = el('vectorization-unit');
    let units = [];
    if (vecUnitInput && vecUnitInput.value) {
        units = vecUnitInput.value.split(',').map(unit => unit.trim()).filter(unit => unit);
    }

    state.selectedMethods.vectorization = method;
    const funcLoading = el('func-loading');
    if (funcLoading) funcLoading.style.display = 'flex';

    try {
        // 确保请求数据包含连续索引
        let requestData = {
            ...state.currentData
        };
        regenerateContinuousIndex(requestData);

        const normList = el('normalization-columns-list');
        const selectedColumns = Array.from(normList.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        const columns = selectedColumns.length > 0 ? selectedColumns : state.currentData.columns;

        const requestBody = {
            vectorization: method,
            data: requestData,
            unit: units,
            columns: columns,
            user_id: state.userId,
            data_id: state.dataId
        };

        const response = await fetch(`${API_BASE_URL}/vectorization`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error(`服务器错误: ${response.status}`);
        const result = await response.json();

        if ((result.status === 200 || result.status === 1) && result.msg) {
            const processedData = safeJsonParse(result.msg);
            // 修复：处理返回数据后重新生成索引
            regenerateContinuousIndex(processedData);
            
            // 使用全部向量化结果，包括训练集和测试集
            state.currentData = processedData;

            state.pagination.filteredRows = state.currentData.data?.length || state.currentData.length || 0;
            state.pagination.totalRows = state.pagination.filteredRows;

            state.operationLogs.vectorization = method;
            state.operationLogs.result_data = state.currentData;

            calculatePagination();
            renderColumnSelects();
            renderPagedDataTable();
            // 确保数据划分配置面板的显示状态
            if (state.selectedMethods.split) {
                const splitConfig = el('split-config');
                const splitConfigPanel = el('split-config-panel');
                if (splitConfig) splitConfig.style.display = 'block';
                if (splitConfigPanel) splitConfigPanel.classList.add('active');
            }

            const unitMsg = units.length ? ` (单位: ${units.join(', ')})` : '';
            alert(`向量化（${method}）应用成功！${unitMsg}`);
        } else {
            throw new Error(result.message || '向量化失败');
        }

    } catch (error) {
        console.error('向量化失败:', error);
        alert(`向量化失败: ${error.message}`);
        state.selectedMethods.vectorization = '';
    } finally {
        if (funcLoading) funcLoading.style.display = 'none';
    }
}

/**
 * 初始化预处理事件监听
 */
function initPreprocessEvents() {
    const preprocessSelect = el('preprocess-select');
    const applyPreprocessBtn = el('apply-preprocess-btn');
    const preprocessConfig = el('preprocess-config');

    if (!preprocessSelect || !applyPreprocessBtn) return;

    // 预处理下拉框切换
    preprocessSelect.addEventListener('change', (e) => {
        state.selectedMethods.preprocess = e.target.value;

        applyPreprocessBtn.disabled = !state.selectedMethods.preprocess;

        if (state.selectedMethods.preprocess) {
            preprocessConfig.style.display = 'block';

            // const preprocessTitle = el('preprocess-config-title');
            // if (preprocessTitle) {
            //     preprocessTitle.textContent = `Configure: ${getPreprocessName(state.selectedMethods.preprocess)}`;
            // }

            document.querySelectorAll('.config-panel').forEach(panel => {
                panel.style.display = 'none';
            });

            if (state.selectedMethods.preprocess === 'duplicate') {
                const duplicateConfig = el('duplicate-config');
                if (duplicateConfig) duplicateConfig.style.display = 'flex';
            } else if (state.selectedMethods.preprocess === 'fill_na') {
                const fillnaConfig = el('fillna-config');
                if (fillnaConfig) fillnaConfig.style.display = 'flex';
            } else if (state.selectedMethods.preprocess === 'outlier') {
                const outlierConfig = el('outlier-config');
                if (outlierConfig) outlierConfig.style.display = 'flex';
            }
        } else {
            preprocessConfig.style.display = 'none';
        }
    });

    // 应用预处理按钮
    applyPreprocessBtn.addEventListener('click', applyPreprocess);

    preprocessSelect.disabled = false;
}

/**
 * 获取预处理功能名称（用于显示）
 */
function getPreprocessName(funcName) {
    const names = {
        'duplicate': 'Remove Duplicates',
        'fill_na': 'Fill Missing Values',
        'outlier': 'Remove Outliers'
    };
    return names[funcName] || funcName;
}

/**
 * 应用数据预处理
 */
async function applyPreprocess() {
    if (!state.currentData || !state.selectedMethods.preprocess) {
        alert('未选择预处理方法！');
        return;
    }

    const funcLoading = el('func-loading');
    if (funcLoading) funcLoading.style.display = 'flex';

    try {
        // 确保请求数据包含连续索引
        const requestData = {
            ...state.currentData
        };
        regenerateContinuousIndex(requestData);

        const requestBody = {
            func_name: state.selectedMethods.preprocess,
            data: requestData,
            user_id: state.userId,
            data_id: state.dataId
        };

        if (state.selectedMethods.preprocess === 'duplicate') {
            const subsetInput = el('duplicate-subset');
            const subset = subsetInput ? (subsetInput.value ? subsetInput.value.split(',').map(s => s.trim()) : null) : null;
            requestBody.subset = subset;
        } else if (state.selectedMethods.preprocess === 'fill_na') {
            const methodsInput = el('fillna-methods');
            const methods = methodsInput ? (methodsInput.value ? methodsInput.value.split(',').map(m => m.trim()) : ['0']) : ['0'];
            requestBody.methods = methods;
        } else if (state.selectedMethods.preprocess === 'outlier') {
            const methodsSelect = el('outlier-methods');
            const multipleInput = el('outlier-multiple');

            requestBody.methods = methodsSelect ? methodsSelect.value : '';
            requestBody.multiple = multipleInput ? (parseFloat(multipleInput.value) || 3) : 3;
        }

        // 直接调用单独的预处理接口，而不是通过 preprocess 接口
        let apiEndpoint;
        if (state.selectedMethods.preprocess === 'duplicate') {
            apiEndpoint = `${API_BASE_URL}/duplicate`;
        } else if (state.selectedMethods.preprocess === 'fill_na') {
            apiEndpoint = `${API_BASE_URL}/fill_na`;
        } else if (state.selectedMethods.preprocess === 'outlier') {
            apiEndpoint = `${API_BASE_URL}/outlier`;
        } else {
            throw new Error('不支持的预处理方法');
        }

        const response = await fetch(apiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error(`服务器错误: ${response.status}`);
        const result = await response.json();

        if ((result.status === 200 || result.status === 1) && result.msg) {
            const processedData = safeJsonParse(result.msg);
            // 修复：处理返回数据后重新生成索引
            regenerateContinuousIndex(processedData);
            state.currentData = processedData;

            state.pagination.filteredRows = state.currentData.data?.length || state.currentData.length || 0;
            state.pagination.totalRows = state.pagination.filteredRows;

            state.operationLogs.preprocess = state.selectedMethods.preprocess;
            state.operationLogs.result_data = state.currentData;

            calculatePagination();
            renderColumnSelects();
            renderPagedDataTable();
            // 确保数据划分配置面板的显示状态
            if (state.selectedMethods.split) {
                const splitConfig = el('split-config');
                const splitConfigPanel = el('split-config-panel');
                if (splitConfig) splitConfig.style.display = 'block';
                if (splitConfigPanel) splitConfigPanel.classList.add('active');
            }

            alert(`${getPreprocessName(state.selectedMethods.preprocess)} 应用成功！`);
        } else {
            throw new Error(result.message || '预处理失败');
        }

    } catch (error) {
        console.error('预处理失败:', error);
        alert(`${getPreprocessName(state.selectedMethods.preprocess)} 失败: ${error.message}`);
    } finally {
        if (funcLoading) funcLoading.style.display = 'none';
    }
}

// ========== 实时筛选功能 ==========
/**
 * 初始化实时筛选事件
 */
function initRealTimeFilterEvents() {
    const addConditionBtn = el('add-condition');
    if (addConditionBtn) {
        addConditionBtn.addEventListener('click', addFilterCondition);
        updateAddConditionButtonState();
    }

    const resetFilterBtn = el('reset-filter');
    if (resetFilterBtn) {
        resetFilterBtn.addEventListener('click', resetAdvancedFilter);
    }

    const advancedFilter = document.querySelector('.advanced-filter');
    if (advancedFilter) {
        advancedFilter.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-condition')) {
                const conditionIndex = parseInt(e.target.closest('.filter-condition')?.dataset.index || 0);
                removeFilterCondition(conditionIndex);
            }
        });
    }

    document.querySelectorAll('.filter-condition').forEach((group, index) => {
        const columnSelect = group.querySelector('.filter-column');
        const operatorSelect = group.querySelector('.filter-operator');
        const valueInput = group.querySelector('.filter-value');

        if (columnSelect) {
            columnSelect.addEventListener('change', (e) => {
                state.filter.conditions[index].column = e.target.value;
                triggerRealTimeFilter();
            });
        }

        if (operatorSelect) {
            operatorSelect.addEventListener('change', (e) => {
                state.filter.conditions[index].operator = e.target.value;
                triggerRealTimeFilter();
            });
        }

        if (valueInput) {
            valueInput.addEventListener('input', (e) => {
                state.filter.conditions[index].value = e.target.value.trim();
                triggerRealTimeFilter();
            });
            valueInput.addEventListener('paste', (e) => {
                setTimeout(() => {
                    state.filter.conditions[index].value = valueInput.value.trim();
                    triggerRealTimeFilter();
                }, 0);
            });
            valueInput.addEventListener('keyup', (e) => {
                state.filter.conditions[index].value = e.target.value.trim();
                triggerRealTimeFilter();
            });
        }
    });
}

/**
 * 触发实时筛选（带防抖）
 */
function triggerRealTimeFilter() {
    if (state.filter.debounceTimer) {
        clearTimeout(state.filter.debounceTimer);
    }
    state.filter.debounceTimer = setTimeout(() => {
        if (state.currentData) {
            applyAdvancedFilter();
        }
    }, state.filter.debounceDelay);
}

/**
 * 添加筛选条件
 */
function addFilterCondition() {
    if (state.filter.activeConditions >= state.filter.maxConditions) return;

    state.filter.activeConditions++;
    const conditionGroups = document.querySelectorAll('.filter-condition');

    for (let i = 0; i < conditionGroups.length; i++) {
        if (i < state.filter.activeConditions) {
            conditionGroups[i].classList.add('active');
            const removeBtn = conditionGroups[i].querySelector('.remove-condition');
            if (removeBtn) {
                removeBtn.disabled = false;
            }
        }
    }

    updateAddConditionButtonState();
    triggerRealTimeFilter();
}

/**
 * 移除筛选条件
 * @param {number} index - 条件索引
 */
function removeFilterCondition(index) {
    if (state.filter.activeConditions <= 1) return;

    state.filter.conditions[index] = { column: 'all', operator: 'contains', value: '' };
    const conditionGroup = document.querySelector(`.filter-condition[data-index="${index}"]`);

    if (conditionGroup) {
        conditionGroup.classList.remove('active');

        const valueInput = conditionGroup.querySelector('.filter-value');
        const columnSelect = conditionGroup.querySelector('.filter-column');
        const operatorSelect = conditionGroup.querySelector('.filter-operator');

        if (valueInput) valueInput.value = '';
        if (columnSelect) columnSelect.value = 'all';
        if (operatorSelect) operatorSelect.value = 'contains';

        const removeBtn = conditionGroup.querySelector('.remove-condition');
        if (removeBtn) {
            removeBtn.disabled = true;
        }
    }

    state.filter.activeConditions--;
    updateAddConditionButtonState();
    triggerRealTimeFilter();
}

/**
 * 更新添加条件按钮状态
 */
function updateAddConditionButtonState() {
    const addBtn = el('add-condition');
    if (addBtn) {
        addBtn.disabled = state.filter.activeConditions >= state.filter.maxConditions;
    }
}

/**
 * 应用高级筛选（重新实现）
 */
function applyAdvancedFilter() {
    // 收集有效的筛选条件
    const validConditions = [];
    for (let i = 0; i < state.filter.activeConditions; i++) {
        const condition = state.filter.conditions[i];
        if (condition.value.trim()) {
            validConditions.push(condition);
        }
    }

    const rows = state.currentData.data || state.currentData;
    let filteredRows = rows;

    if (validConditions.length > 0) {
        const columns = state.currentData.columns || Object.keys(state.currentData[0] || {});
        
        filteredRows = rows.filter(row => {
            // 确保row是有效的
            if (!row || (typeof row !== 'object' && !Array.isArray(row))) {
                return false;
            }
            
            // 检查所有条件是否都满足
            return validConditions.every(condition => {
                let values = [];
                
                // 获取需要比较的值
                if (condition.column === 'all') {
                    // 所有列
                    if (Array.isArray(row)) {
                        values = row;
                    } else if (typeof row === 'object') {
                        values = Object.values(row);
                    }
                } else {
                    // 特定列
                    if (Array.isArray(row)) {
                        const colIndex = columns.indexOf(condition.column);
                        if (colIndex !== -1 && colIndex < row.length) {
                            values = [row[colIndex]];
                        }
                    } else if (typeof row === 'object') {
                        if (condition.column in row) {
                            values = [row[condition.column]];
                        }
                    }
                }
                
                // 检查是否有任何值满足条件
                return values.some(val => {
                    if (val === null || val === undefined) {
                        return false;
                    }
                    
                    return compareValues(val, condition.value, condition.operator);
                });
            });
        });
        
        // 存储筛选后的数据
        state.filteredData = filteredRows;
    } else {
        // 没有筛选条件时，清空筛选数据
        state.filteredData = null;
    }

    // 更新分页信息
    state.pagination.filteredRows = filteredRows.length;
    state.pagination.currentPage = 1;
    calculatePagination();

    // 更新UI
    if (el('filtered-rows')) {
        el('filtered-rows').textContent = state.pagination.filteredRows;
    }
    renderPagedDataTable();
}

/**
 * 比较值和条件
 * @param {any} val - 要比较的值
 * @param {string} conditionValue - 条件值
 * @param {string} operator - 操作符
 * @returns {boolean} 是否满足条件
 */
function compareValues(val, conditionValue, operator) {
    try {
        switch (operator) {
            case 'contains':
                return String(val).toLowerCase().includes(String(conditionValue).toLowerCase());
                
            case 'equals':
                // 处理不同类型的比较
                if (typeof val === 'number') {
                    const numValue = parseFloat(conditionValue);
                    return !isNaN(numValue) && val === numValue;
                } else if (typeof val === 'boolean') {
                    const boolValue = conditionValue.toLowerCase() === 'true';
                    return val === boolValue;
                } else {
                    return String(val) === conditionValue;
                }
                
            case 'greater':
                const valNum = parseFloat(val);
                const condNum = parseFloat(conditionValue);
                return !isNaN(valNum) && !isNaN(condNum) && valNum > condNum;
                
            case 'less':
                const valNumLess = parseFloat(val);
                const condNumLess = parseFloat(conditionValue);
                return !isNaN(valNumLess) && !isNaN(condNumLess) && valNumLess < condNumLess;
                
            case 'greater_eq':
                const valNumGte = parseFloat(val);
                const condNumGte = parseFloat(conditionValue);
                return !isNaN(valNumGte) && !isNaN(condNumGte) && valNumGte >= condNumGte;
                
            case 'less_eq':
                const valNumLte = parseFloat(val);
                const condNumLte = parseFloat(conditionValue);
                return !isNaN(valNumLte) && !isNaN(condNumLte) && valNumLte <= condNumLte;
                
            default:
                return false;
        }
    } catch (error) {
        console.error('比较值时出错:', error);
        return false;
    }
}

/**
 * 重置高级筛选
 */
function resetAdvancedFilter() {
    state.filter.conditions = [
        { column: 'all', operator: 'contains', value: '' },
        { column: 'all', operator: 'contains', value: '' },
        { column: 'all', operator: 'contains', value: '' }
    ];
    state.filter.activeConditions = 1;
    state.filteredData = null;

    document.querySelectorAll('.filter-condition').forEach((group, index) => {
        const columnSelect = group.querySelector('.filter-column');
        const operatorSelect = group.querySelector('.filter-operator');
        const valueInput = group.querySelector('.filter-value');
        const removeBtn = group.querySelector('.remove-condition');

        if (columnSelect) columnSelect.value = 'all';
        if (operatorSelect) operatorSelect.value = 'contains';
        if (valueInput) valueInput.value = '';

        if (index < state.filter.activeConditions) {
            group.classList.add('active');
            if (removeBtn) removeBtn.disabled = true;
        } else {
            group.classList.remove('active');
            if (removeBtn) removeBtn.disabled = true;
        }
    });

    updateAddConditionButtonState();
    state.pagination.filteredRows = state.pagination.totalRows;
    state.pagination.currentPage = 1;
    calculatePagination();

    if (el('filtered-rows')) el('filtered-rows').textContent = state.pagination.filteredRows;
    renderPagedDataTable();
}

// ========== 数据下载功能 ==========
/**
 * 下载数据
 */
async function downloadData() {
    const funcLoading = el('func-loading');
    if (funcLoading) funcLoading.style.display = 'flex';

    try {
        const response = await fetch(`${API_BASE_URL}/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: state.userId,
                saving: state.dataId
            })
        });

        if (!response.ok) throw new Error(`服务器错误: ${response.status}`);

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = state.dataId;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        alert('数据下载成功！');

    } catch (error) {
        console.error('下载失败:', error);
        alert(`下载失败: ${error.message}`);
    } finally {
        if (funcLoading) funcLoading.style.display = 'none';
    }
}

// ========== 核心：提交数据（彻底解决索引错误） ==========
/**
 * 提交数据变更到后端
 */
async function submitDataChanges() {
    if (!confirm('确定保存所有修改吗？')) return;

    // 1. 收集表单参数
    const pattern = el('pattern-select')?.value.trim() || 'csv';
    const encoding = el('encoding-input')?.value.trim() || 'utf-8';
    const sheet_name = el('sheet-name-input')?.value.trim() || 'Sheet1';
    const types = el('types-select')?.value.trim() || '';

    // 2. 构建operation数组（所有索引均为整数）
    const operationSteps = [];
    let stepCounter = 1;

    // 处理删除操作
    if (state.operationLogs.delete) {
        state.operationLogs.delete.split(';').forEach(op => {
            const colMatch = op.match(/delete column (\w+)/);
            const rowMatch = op.match(/delete row (\d+)/);

            if (colMatch) {
                operationSteps.push({
                    "step": stepCounter++,
                    "function": "delete",
                    "work": {
                        "area": {
                            "index": -1, // 强制整数
                            "columns": colMatch[1],
                            "primary": "",
                            "change": ""
                        }
                    }
                });
            } else if (rowMatch) {
                operationSteps.push({
                    "step": stepCounter++,
                    "function": "delete",
                    "work": {
                        "area": {
                            "index": parseInt(rowMatch[1]), // 强制整数
                            "columns": "",
                            "primary": "",
                            "change": ""
                        }
                    }
                });
            }
        });
    }

    // 处理更新操作
    if (state.operationLogs.update) {
        state.operationLogs.update.split(';').forEach(op => {
            const match = op.match(/update row(\d+) and column(\w+) from (.+) to (.+)/);
            if (match) {
                operationSteps.push({
                    "step": stepCounter++,
                    "function": "update",
                    "work": {
                        "area": {
                            "index": parseInt(match[1]), // 强制整数
                            "columns": match[2],
                            "primary": JSON.parse(match[3]),
                            "change": JSON.parse(match[4])
                        }
                    }
                });
            }
        });
    }

    // 处理数据划分操作（必须在归一化和向量化之前）
    if (state.splitInfo.trainIndex.length > 0 || state.splitInfo.testIndex.length > 0) {
        // 从复选框列表中获取X列和Y列
        const xColumns = [];
        const yColumns = [];
        
        // 获取X列
        const xCheckboxes = document.querySelectorAll('#x-columns-list input[type="checkbox"]:checked');
        xCheckboxes.forEach(checkbox => {
            xColumns.push(checkbox.value);
        });
        
        // 获取Y列
        const yCheckboxes = document.querySelectorAll('#y-columns-list input[type="checkbox"]:checked');
        yCheckboxes.forEach(checkbox => {
            yColumns.push(checkbox.value);
        });
        
        operationSteps.push({
            "step": stepCounter++,
            "function": "train_test_split",
            "work": {
                "data": {
                    "x_columns": xColumns.length > 0 ? xColumns : (() => {
                        const allColumns = state.currentData.columns || Object.keys(state.currentData[0] || {});
                        return allColumns.length > 1 ? allColumns.slice(0, -1) : [];
                    })(),
                    "y_columns": yColumns.length > 0 ? yColumns : (() => {
                        const allColumns = state.currentData.columns || Object.keys(state.currentData[0] || {});
                        return allColumns.length > 0 ? [allColumns[allColumns.length - 1]] : [];
                    })(),
                    "columns": null, // 暂时设置为null，因为前端没有单独的columns选择
                    "train_index": state.splitInfo.trainIndex,
                    "test_index": state.splitInfo.testIndex
                }
            }
        });
    }

    // 处理归一化操作
    if (state.operationLogs.normalization) {
        const normList = el('normalization-columns-list');
        const selectedColumns = Array.from(normList.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        operationSteps.push({
            "step": stepCounter++,
            "function": "normalization",
            "work": {
                "method": [state.operationLogs.normalization],
                "subset": selectedColumns.length > 0 ? selectedColumns : null
            }
        });
    }

    // 处理向量化操作
    if (state.operationLogs.vectorization) {
        const normList = el('normalization-columns-list');
        const selectedColumns = Array.from(normList.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        operationSteps.push({
            "step": stepCounter++,
            "function": "vectorization",
            "work": {
                "method": [state.operationLogs.vectorization],
                "subset": selectedColumns.length > 0 ? selectedColumns : null
            }
        });
    }

    // 处理预处理操作
    if (state.operationLogs.preprocess) {
        const funcMap = {
            'fill_na': 'fillna',
            'duplicate': 'drop_duplicate',
            'outlier': 'drop_outlier'
        };
        const func = funcMap[state.operationLogs.preprocess] || state.operationLogs.preprocess;

        const work = {};
        if (func === 'fillna') {
            const methodsInput = el('fillna-methods');
            const methods = methodsInput ? (methodsInput.value ? methodsInput.value.split(',').map(m => m.trim()) : ['0']) : ['0'];
            work.method = methods;
            // 获取选择的列
            const normList = el('normalization-columns-list');
            const selectedColumns = Array.from(normList.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
            work.subset = selectedColumns.length > 0 ? selectedColumns : null;
        } else if (func === 'drop_duplicate') {
            // const subsetInput = el('duplicate-subset');
            // const subset = subsetInput ? (subsetInput.value ? subsetInput.value.split(',').map(s => s.trim()) : []) : [];
            // work.subset = subset;
            // 使用统一的列选择框
            const normList = el('normalization-columns-list');
            const selectedColumns = Array.from(normList.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
            work.subset = selectedColumns.length > 0 ? selectedColumns : null;
        } else if (func === 'drop_outlier') {
            const methodsSelect = el('outlier-methods');
            const methodVal = methodsSelect ? methodsSelect.value : '';
            work.method = methodVal ? [methodVal] : [];
            // 获取选择的列
            const normList = el('normalization-columns-list');
            const selectedColumns = Array.from(normList.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
            work.subset = selectedColumns.length > 0 ? selectedColumns : null;
        }

        operationSteps.push({
            "step": stepCounter++,
            "function": func,
            "work": work
        });
    }

    // 3. 检查空操作
    if (operationSteps.length === 0) {
        alert('至少需要进行一次修改操作！');
        return;
    }

    // 4. 构建result_data（Pandas兼容格式，确保索引连续）
    const resultData = {
        "index": Array.from(
            { length: state.currentData.data?.length || state.currentData.length },
            (_, i) => parseInt(i) // 强制整数索引
        ),
        "columns": state.currentData.columns || Object.keys(state.currentData[0] || {}),
        "data": state.currentData.data || state.currentData
    };

    // 5. 构建最终请求体
    const requestBody = {
        "user_id": state.userId,
        "data_id": state.dataId,
        "operation": operationSteps,
        "result_data": resultData,
        "pattern": pattern,
        "encoding": encoding,
        "sheet_name": sheet_name,
        "types": types
    };

    console.log('最终提交数据:', JSON.stringify(requestBody, null, 2));

    // 6. 提交请求
    const loading = el('func-loading');
    if (loading) loading.style.display = 'flex';

    try {
        const response = await fetch(`${API_BASE_URL}/saving_data`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json; charset=utf-8' },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) throw new Error(`请求失败: ${response.status}`);
        const result = await response.json();

        if ([200, 1].includes(result.status)) {
            alert('保存成功！');
            state.operationLogs = {
                delete: '',
                update: '',
                vectorization: '',
                normalization: '',
                preprocess: '',
                data_id: state.dataId,
                result_data: {}
            };
            if (el('save-config-modal')) el('save-config-modal').style.display = 'none';
        } else {
            throw new Error(result.message || '保存失败');
        }
    } catch (error) {
        console.error('保存失败:', error);
        alert(`保存失败: ${error.message}`);
    } finally {
        if (loading) loading.style.display = 'none';
    }
}

// ========== 初始化事件监听 ==========
/**
 * 初始化所有事件监听
 */
function initEventListeners() {
    // 工作表选择
    if (el('sheet-select')) {
        el('sheet-select').addEventListener('change', (e) => {
            const selectedSheet = e.target.value;
            // 构建新的URL，保持其他参数不变
            const url = new URL(window.location.href);
            url.searchParams.set('sheet', selectedSheet);
            window.location.href = url.toString();
        });
    }

    // 返回按钮
    if (el('back-btn')) {
        el('back-btn').addEventListener('click', () => {
            window.location.href = '/static/html/home.html';
        });
    }

    // 保存按钮（打开弹窗）
    if (el('save-btn')) {
        el('save-btn').addEventListener('click', () => {
            if (el('pattern-select')) {
                el('pattern-select').value = state.dataId.split('.').pop() || 'csv';
            }
            if (el('encoding-input')) el('encoding-input').value = 'utf-8';
            if (el('sheet-name-input')) el('sheet-name-input').value = 'Sheet1';
            if (el('types-select')) el('types-select').value = '';

            if (el('save-config-modal')) el('save-config-modal').style.display = 'block';
        });
    }

    // 下载按钮
    if (el('download-btn')) {
        el('download-btn').addEventListener('click', downloadData);
    }

    // 确认保存
    if (el('confirm-save-btn')) {
        el('confirm-save-btn').addEventListener('click', submitDataChanges);
    }

    // 取消保存
    if (el('cancel-save-btn')) {
        el('cancel-save-btn').addEventListener('click', () => {
            if (el('save-config-modal')) el('save-config-modal').style.display = 'none';
        });
    }

    // 关闭弹窗
    if (el('close-modal')) {
        el('close-modal').addEventListener('click', () => {
            if (el('save-config-modal')) el('save-config-modal').style.display = 'none';
        });
    }

    // 取消按钮
    if (el('cancel-save-btn')) {
        el('cancel-save-btn').addEventListener('click', () => {
            if (el('save-config-modal')) el('save-config-modal').style.display = 'none';
        });
    }

    // 分页事件
    if (el('page-size-select')) {
        el('page-size-select').addEventListener('change', (e) => {
            state.pagination.pageSize = parseInt(e.target.value);
            state.pagination.currentPage = 1;
            calculatePagination();
            renderPagedDataTable();
        });
    }

    if (el('prev-page')) {
        el('prev-page').addEventListener('click', () => {
            if (state.pagination.currentPage > 1) {
                state.pagination.currentPage--;
                renderPagedDataTable();
                updatePaginationUI();
            }
        });
    }

    if (el('next-page')) {
        el('next-page').addEventListener('click', () => {
            if (state.pagination.currentPage < state.pagination.totalPages) {
                state.pagination.currentPage++;
                renderPagedDataTable();
                updatePaginationUI();
            }
        });
    }

    // 归一化应用按钮
    const applyNormBtn = el('apply-normalize-btn');
    const normSelect = el('normalization-select');
    if (applyNormBtn && normSelect) {
        applyNormBtn.addEventListener('click', () => {
            const selectedMethod = normSelect.value;
            if (selectedMethod) {
                applyNormalization(selectedMethod);
            } else {
                alert('请选择归一化方法！');
            }
        });

        normSelect.addEventListener('change', (e) => {
            applyNormBtn.disabled = !e.target.value;
        });
    }

    // 向量化应用按钮
    const applyVecBtn = el('apply-vectorize-btn');
    const vecSelect = el('vectorization-select');
    if (applyVecBtn && vecSelect) {
        applyVecBtn.addEventListener('click', () => {
            const selectedMethod = vecSelect.value;
            if (selectedMethod) {
                applyVectorization(selectedMethod);
            } else {
                alert('请选择向量化方法！');
            }
        });

        vecSelect.addEventListener('change', (e) => {
            applyVecBtn.disabled = !e.target.value;
        });
    }

    // 删除列/行事件委托
    if (el('table-header')) {
        el('table-header').addEventListener('click', (e) => {
            if (e.target.classList.contains('delete-col')) {
                deleteColumn(e.target.dataset.colIndex);
            }
        });
    }

    if (el('table-body')) {
        el('table-body').addEventListener('click', (e) => {
            if (e.target.classList.contains('delete-row')) {
                deleteRow(e.target.dataset.rowIndex);
            }
        });
    }

    // 全局编辑事件
    document.addEventListener('keydown', (e) => {
        if (state.editing.isEditing && e.key === 'Escape') {
            const cell = state.editing.currentCell;
            if (cell) {
                cell.textContent = JSON.parse(cell.dataset.originalValueJson);
                cell.contentEditable = false;
                cell.classList.remove('editing');
            }
            state.editing.isEditing = false;
            state.editing.currentCell = null;
        }
    });

    // 初始化预处理事件
    initPreprocessEvents();

    // 初始化数据划分事件
    initSplitEvents();

    // 初始化实时筛选事件
    initRealTimeFilterEvents();
}

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', async () => {
    // 登录验证
    if (!state.userId) {
        alert('请先登录！');
        window.location.href = '/static/html/home.html';
        return;
    }

    // 获取data_id
    const urlParams = new URLSearchParams(window.location.search);
    state.dataId = urlParams.get('data_id');
    if (!state.dataId) {
        alert('缺少数据集ID！');
        window.location.href = '/static/html/home.html';
        return;
    }
    state.operationLogs.data_id = state.dataId;

    // 初始化事件
    initEventListeners();

    // 加载方法列表
    await loadFunctionMethods();

    // 加载数据
    await loadDatasetDetail();
    
    // 设置工作表选择框的当前值
    const sheetName = urlParams.get('sheet');
    if (el('sheet-select') && sheetName) {
        el('sheet-select').value = sheetName;
    }
});