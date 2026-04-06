// 获取URL参数
function getUrlParams() {
    const params = {};
    const searchParams = new URLSearchParams(window.location.search);
    for (const [key, value] of searchParams.entries()) {
        params[key] = value;
    }
    console.log('URL Parameters:', params);
    return params;
}

// 显示提示信息（小窗口形式）
function showTips(msg, type = 'info', element = null) {
    // 创建小窗口容器
    const modal = document.createElement('div');
    modal.className = `tips-modal ${type}`;
    modal.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        z-index: 1000;
        max-width: 400px;
        text-align: center;
    `;
    
    // 创建消息内容
    const message = document.createElement('div');
    message.textContent = msg;
    message.style.marginBottom = '15px';
    
    // 创建关闭按钮
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.style.cssText = `
        padding: 8px 16px;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    `;
    closeBtn.onclick = function() {
        modal.remove();
        overlay.remove();
    };
    
    // 创建遮罩层
    const overlay = document.createElement('div');
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: 999;
    `;
    overlay.onclick = function() {
        modal.remove();
        overlay.remove();
    };
    
    // 组装小窗口
    modal.appendChild(message);
    modal.appendChild(closeBtn);
    
    // 添加到页面
    document.body.appendChild(overlay);
    document.body.appendChild(modal);
    
    // 3秒后自动关闭
    setTimeout(() => {
        try {
            modal.remove();
            overlay.remove();
        } catch (e) {
            // 忽略已被手动关闭的情况
        }
    }, 3000);
}

// 全局变量存储模型列表
let modelList = [];
let currentModelIndex = -1;

// 获取用户的所有模型
async function fetchModelList(user_id) {
    try {
        const response = await fetch('/api/find_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: user_id
            })
        });
        
        const result = await response.json();
        if (result.status === 1 || result.status === 200) {
            modelList = result.msg || [];
            console.log('Fetched model list:', modelList);
            return modelList;
        } else {
            console.error('Failed to fetch model list:', result.msg);
            return [];
        }
    } catch (error) {
        console.error('Error fetching model list:', error);
        return [];
    }
}

// 加载模型详情
async function loadModelDetail() {
    const params = getUrlParams();
    const user_id = params.user_id;
    const model_id = params.model_id;
    
    console.log('URL Params:', params);
    console.log('User ID:', user_id);
    console.log('Model ID:', model_id);
    
    if (!user_id || !model_id) {
        showTips('Missing required parameters', 'error');
        return;
    }
    
    // 显示加载状态
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = 'block';
    }
    
    try {
        // 获取模型列表
        await fetchModelList(user_id);
        
        // 找到当前模型在列表中的位置
        currentModelIndex = modelList.findIndex(model => model.id === model_id);
        console.log('Current model index:', currentModelIndex);
        
        console.log('Sending request to /api/detail_model');
        const response = await fetch('/api/detail_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: user_id,
                model_id: model_id
            })
        });
        
        console.log('Response status:', response.status);
        console.log('Response headers:', response.headers);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Response data:', result);
        
        if (result.status === 1 || result.status === 200) {
            const data = result.msg;
            console.log('Model data:', data);
            
            // 打印data对象的结构
            console.log('Data structure:', JSON.stringify(data, null, 2));
            
            // 检查model_info对象是否存在
            console.log('model_info exists:', data.model_info !== undefined && data.model_info !== null);
            if (data.model_info) {
                console.log('model_info properties:', Object.keys(data.model_info));
                console.log('model_info.name:', data.model_info.name);
            }
            
            // 打印所有可能的模型名称字段
            console.log('Possible model name fields:');
            console.log('data.model_info.name:', data.model_info ? data.model_info.name : 'N/A');
            console.log('data.name:', data.name);
            console.log('data.model_name:', data.model_name);
            console.log('data.name_model:', data.name_model);
            
            // 打印所有可能的模型类型字段
            console.log('Possible model type fields:');
            console.log('data.types:', data.types);
            console.log('data.type:', data.type);
            console.log('data.model_info.type:', data.model_info ? data.model_info.type : 'N/A');
            console.log('data.model_info.types:', data.model_info ? data.model_info.types : 'N/A');
            
            // 打印所有可能的模型状态字段
            console.log('Possible model status fields:');
            console.log('data.status:', data.status);
            console.log('data.model_info.status:', data.model_info ? data.model_info.status : 'N/A');
            
            // 打印所有可能的模型开放状态字段
            console.log('Possible model open_bool fields:');
            console.log('data.open_bool:', data.open_bool);
            console.log('data.model_info.open_bool:', data.model_info ? data.model_info.open_bool : 'N/A');
            
            // 打印完整的data对象，以便查看实际的数据结构
            console.log('Complete data object:', data);
            
            // 优先渲染核心信息，让用户尽快看到内容
            console.log('Calling renderModelInfo...');
            // 使用DOMContentLoaded事件确保DOM已经完全加载
            if (document.readyState === 'complete') {
                // DOM已经完全加载，直接调用
                renderModelInfo(data);
                console.log('Calling renderHyperParams...');
                renderHyperParams(data.hyper_params);
                console.log('Calling renderTrainTestInfo...');
                renderTrainTestInfo(data.train_test_info, data);
                console.log('Calling renderDownloadLinks...');
                renderDownloadLinks(data.downloads);
                console.log('Calling setupOpenBoolToggle...');
                const openBool = data.model_info ? data.model_info.open_bool : data.open_bool;
                setupOpenBoolToggle(openBool, user_id, model_id);
            } else {
                // DOM还没有完全加载，等待DOMContentLoaded事件
                document.addEventListener('DOMContentLoaded', () => {
                    renderModelInfo(data);
                    console.log('Calling renderHyperParams...');
                    renderHyperParams(data.hyper_params);
                    console.log('Calling renderTrainTestInfo...');
                    renderTrainTestInfo(data.train_test_info, data);
                    console.log('Calling renderDownloadLinks...');
                    renderDownloadLinks(data.downloads);
                    console.log('Calling setupOpenBoolToggle...');
                    const openBool = data.model_info ? data.model_info.open_bool : data.open_bool;
                    setupOpenBoolToggle(openBool, user_id, model_id);
                });
            }
            
            // 异步渲染评估指标和数据集统计，不阻塞页面加载
            console.log('Calling renderEvaluationMetrics...');
            renderEvaluationMetrics(data.evaluation_metrics);
            
            // 传递data_id给initDatasetStats，异步执行
            const data_id = data.data_id;
            console.log('Passing data_id to initDatasetStats:', data_id);
            if (data_id) {
                // 延迟执行，让核心内容先加载完成
                setTimeout(() => {
                    initDatasetStats(data_id);
                }, 100);
            }
        } else {
            showTips(result.msg || 'Failed to load model details', 'error');
        }
    } catch (error) {
        console.error('Error loading model details:', error);
        showTips(`Error loading model details: ${error.message}`, 'error');
    } finally {
        // 隐藏加载状态
        if (loading) {
            loading.style.display = 'none';
        }
    }
}

// 导航到上一个模型
function navigateToPrevModel() {
    const params = getUrlParams();
    const user_id = params.user_id;
    const current_model_id = params.model_id;
    
    // 找到当前模型在列表中的位置
    const currentIndex = modelList.findIndex(model => model.id === current_model_id);
    
    if (currentIndex > 0) {
        const prevModel = modelList[currentIndex - 1];
        window.location.href = `/detail_model?user_id=${user_id}&model_id=${prevModel.id}`;
    } else {
        showTips('No previous model', 'info');
    }
}

// 导航到下一个模型
function navigateToNextModel() {
    const params = getUrlParams();
    const user_id = params.user_id;
    const current_model_id = params.model_id;
    
    // 找到当前模型在列表中的位置
    const currentIndex = modelList.findIndex(model => model.id === current_model_id);
    
    if (currentIndex < modelList.length - 1) {
        const nextModel = modelList[currentIndex + 1];
        window.location.href = `/detail_model?user_id=${user_id}&model_id=${nextModel.id}`;
    } else {
        showTips('No next model', 'info');
    }
}

// 渲染模型信息
function renderModelInfo(data) {
    console.log('Rendering model info:', data);
    try {
        // 直接从model_info.name获取模型名称
        let modelName = 'N/A';
        console.log('data:', data);
        console.log('data.model_info:', data ? data.model_info : 'data is null/undefined');
        if (data) {
            console.log('data has model_info:', 'model_info' in data);
            if (data.model_info) {
                console.log('data.model_info has name:', 'name' in data.model_info);
                console.log('data.model_info.name:', data.model_info.name);
                console.log('typeof data.model_info.name:', typeof data.model_info.name);
                if (data.model_info.name) {
                    modelName = data.model_info.name;
                    console.log('Model name from model_info.name:', modelName);
                } else {
                    console.error('model_info.name is empty or null');
                }
            } else {
                console.error('data.model_info is null/undefined');
            }
        } else {
            console.error('data is null/undefined');
        }
        
        console.log('Final model name:', modelName);
        
        // 检查model-name元素是否存在
        console.log('Looking for model-name element...');
        console.log('Document readyState:', document.readyState);
        console.log('Current URL:', window.location.href);
        
        // 打印整个文档的HTML结构
        console.log('Document HTML:', document.documentElement.outerHTML.substring(0, 1000));
        
        // 尝试使用getElementById
        const modelNameElement = document.getElementById('model-name');
        console.log('getElementById result:', modelNameElement);
        
        // 尝试使用querySelector
        const modelNameElement2 = document.querySelector('#model-name');
        console.log('querySelector result:', modelNameElement2);
        
        // 打印所有h1元素
        const h1Elements = document.getElementsByTagName('h1');
        console.log('All h1 elements:', h1Elements);
        for (let i = 0; i < h1Elements.length; i++) {
            console.log('h1 element', i, ':', h1Elements[i]);
            console.log('h1 element', i, 'id:', h1Elements[i].id);
            console.log('h1 element', i, 'innerHTML:', h1Elements[i].innerHTML);
        }
        
        // 尝试使用querySelector找到model-title，然后找到其中的span
        const modelTitleElement = document.querySelector('#model-title');
        console.log('model-title element:', modelTitleElement);
        if (modelTitleElement) {
            const spanElement = modelTitleElement.querySelector('span');
            console.log('span element inside model-title:', spanElement);
            if (spanElement) {
                console.log('span element id:', spanElement.id);
                console.log('Setting model name to:', modelName);
                spanElement.textContent = modelName;
                spanElement.title = modelName;
                console.log('Model name set successfully:', spanElement.textContent);
            }
        }
        
        // 尝试直接修改model-title元素的文本内容
        const modelTitleElement2 = document.getElementById('model-title');
        if (modelTitleElement2) {
            console.log('model-title element found, setting text content');
            modelTitleElement2.innerHTML = `Model: <span id="model-name">${modelName}</span>`;
            console.log('Model title updated:', modelTitleElement2.innerHTML);
        }
        
        if (modelNameElement) {
            console.log('Setting model name to:', modelName);
            modelNameElement.textContent = modelName;
            modelNameElement.title = modelName;
            console.log('Model name set successfully:', modelNameElement.textContent);
        } else {
            console.error('model-name element not found');
        }
        
        // 检查model-type元素是否存在
        const modelTypeElement = document.getElementById('model-type');
        if (modelTypeElement) {
            const modelType = data.model_info && data.model_info.type ? data.model_info.type : (data.types ? data.types : (data.type ? data.type : 'N/A'));
            modelTypeElement.textContent = modelType;
            console.log('Model type set to:', modelType);
        }
        
        // 检查model-status元素是否存在
        const modelStatusElement = document.getElementById('model-status');
        if (modelStatusElement) {
            const modelStatus = data.model_info && data.model_info.status ? data.model_info.status : (data.status ? data.status : (data.model_condition ? data.model_condition : 'N/A'));
            modelStatusElement.textContent = modelStatus;
            console.log('Model status set to:', modelStatus);
        }
        
        // 检查open-bool-toggle元素是否存在
        const openBoolElement = document.getElementById('open-bool-toggle');
        if (openBoolElement) {
            const openBool = data.model_info && data.model_info.open_bool ? data.model_info.open_bool : data.open_bool;
            openBoolElement.checked = openBool === 1 || openBool === true;
            console.log('Open bool set to:', openBool);
        }
        
        // 打印所有元素的最终状态
        console.log('Final element states:');
        console.log('model-name:', document.getElementById('model-name') ? document.getElementById('model-name').textContent : 'Element not found');
        console.log('model-type:', document.getElementById('model-type') ? document.getElementById('model-type').textContent : 'Element not found');
        console.log('model-status:', document.getElementById('model-status') ? document.getElementById('model-status').textContent : 'Element not found');
        console.log('open-bool-toggle:', document.getElementById('open-bool-toggle') ? document.getElementById('open-bool-toggle').checked : 'Element not found');
    } catch (error) {
        console.error('Error rendering model info:', error);
    }
}

// 渲染超参信息
function renderHyperParams(hyper_params) {
    console.log('Rendering hyper params:', hyper_params);
    try {
        const container = document.getElementById('hyper-params');
        container.innerHTML = '';
        
        if (!hyper_params || Object.keys(hyper_params).length === 0) {
            container.innerHTML = '<p>No hyperparameters available</p>';
            return;
        }
        
        const table = document.createElement('table');
        table.className = 'hyper-params-table';
        
        // 添加表头
        const headerRow = table.insertRow();
        const keyHeader = headerRow.insertCell();
        const valueHeader = headerRow.insertCell();
        keyHeader.textContent = 'Parameter';
        valueHeader.textContent = 'Value';
        keyHeader.style.fontWeight = 'bold';
        valueHeader.style.fontWeight = 'bold';
        
        function addRow(key, value) {
            const row = table.insertRow();
            const keyCell = row.insertCell();
            const valueCell = row.insertCell();
            
            keyCell.textContent = key;
            valueCell.textContent = typeof value === 'object' ? JSON.stringify(value) : value;
        }
        
        for (const [key, value] of Object.entries(hyper_params)) {
            addRow(key, value);
        }
        
        container.appendChild(table);
    } catch (error) {
        console.error('Error rendering hyper params:', error);
    }
}

// 渲染评估指标
function renderEvaluationMetrics(evaluation_metrics) {
    console.log('Rendering evaluation metrics:', evaluation_metrics);
    try {
        const container = document.getElementById('evaluation-chart');
        container.style.height = '400px';
        
        // 添加下载按钮
        const containerParent = container.parentElement;
        let downloadBtn = containerParent.querySelector('.download-chart-btn');
        if (!downloadBtn) {
            downloadBtn = document.createElement('button');
            downloadBtn.className = 'btn secondary download-chart-btn';
            downloadBtn.textContent = 'Download Chart';
            downloadBtn.style.marginTop = '10px';
            containerParent.appendChild(downloadBtn);
        }
        
        // 检查echarts是否已定义
        if (typeof echarts === 'undefined') {
            console.error('ECharts is not loaded');
            container.innerHTML = '<p>ECharts library is not loaded. Please refresh the page.</p>';
            return;
        }
        
        const chart = echarts.init(container);
        
        if (!evaluation_metrics || Object.keys(evaluation_metrics).length === 0) {
            chart.setOption({});
            return;
        }
        
        // 处理嵌套的评估指标结构
        let metrics = [];
        let values = [];
        
        // 优先使用平均指标
        if (evaluation_metrics.average_metrics) {
            for (const [key, value] of Object.entries(evaluation_metrics.average_metrics)) {
                metrics.push(key);
                values.push(value);
            }
        } else {
            // 如果没有平均指标，使用所有指标
            for (const [key, value] of Object.entries(evaluation_metrics)) {
                if (typeof value !== 'object') {
                    metrics.push(key);
                    values.push(value);
                }
            }
        }
        
        if (metrics.length === 0) {
            chart.setOption({});
            return;
        }
        
        chart.setOption({
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: metrics,
                axisLabel: {
                    rotate: 45
                }
            },
            yAxis: {
                type: 'value',
                min: 0,
                max: 1
            },
            series: [{
                name: 'Score',
                type: 'bar',
                data: values,
                itemStyle: {
                    color: '#4CAF50'
                },
                label: {
                    show: true,
                    position: 'top',
                    formatter: function(params) {
                        return params.value.toFixed(4);
                    }
                }
            }]
        });
        
        // 下载按钮点击事件
        downloadBtn.onclick = function() {
            if (typeof echarts !== 'undefined') {
                const dataURL = chart.getDataURL({ type: 'png', pixelRatio: 2 });
                const link = document.createElement('a');
                link.href = dataURL;
                link.download = '1_evaluation_metrics.png';
                link.click();
            }
        };
        
        window.addEventListener('resize', () => {
            if (typeof echarts !== 'undefined') {
                chart.resize();
            }
        });
        
        // 渲染混淆矩阵
        renderConfusionMatrix(evaluation_metrics);
    } catch (error) {
        console.error('Error rendering evaluation metrics:', error);
    }
}

// 渲染混淆矩阵
function renderConfusionMatrix(evaluation_metrics) {
    console.log('Rendering confusion matrix:', evaluation_metrics);
    try {
        // 检查是否有混淆矩阵数据
        let confusionMatrixData = null;
        let classes = [];
        
        // 尝试从多个位置获取混淆矩阵数据
        // 1. 从metrics_per_output中获取
        if (evaluation_metrics && evaluation_metrics.metrics_per_output) {
            for (const [key, metrics] of Object.entries(evaluation_metrics.metrics_per_output)) {
                if (metrics.confusion_matrix && metrics.classes) {
                    confusionMatrixData = metrics.confusion_matrix;
                    classes = metrics.classes;
                    break;
                }
            }
        }
        
        // 2. 直接从evaluation_metrics中获取
        if (!confusionMatrixData && evaluation_metrics && evaluation_metrics.confusion_matrix) {
            confusionMatrixData = evaluation_metrics.confusion_matrix;
            // 尝试获取类别信息
            classes = evaluation_metrics.classes || [];
        }
        
        // 3. 尝试从其他可能的位置获取
        if (!confusionMatrixData && evaluation_metrics && evaluation_metrics.average_metrics && evaluation_metrics.average_metrics.confusion_matrix) {
            confusionMatrixData = evaluation_metrics.average_metrics.confusion_matrix;
            classes = evaluation_metrics.average_metrics.classes || [];
        }
        
        if (!confusionMatrixData || confusionMatrixData.length === 0) {
            console.log('No confusion matrix data found');
            return;
        }
        
        // 创建混淆矩阵容器
        let cmContainer = document.getElementById('confusion-matrix');
        if (!cmContainer) {
            const evalContainer = document.querySelector('#evaluation-chart').parentElement;
            if (evalContainer) {
                cmContainer = document.createElement('div');
                cmContainer.id = 'confusion-matrix';
                cmContainer.style.marginTop = '30px';
                cmContainer.innerHTML = '<h3>Confusion Matrix</h3><div id="confusion-matrix-chart" style="height: 400px;"></div>';
                evalContainer.appendChild(cmContainer);
            } else {
                return;
            }
        }
        
        const chartContainer = document.getElementById('confusion-matrix-chart');
        if (!chartContainer) return;
        
        // 检查echarts是否已定义
        if (typeof echarts === 'undefined') {
            console.error('ECharts is not loaded');
            chartContainer.innerHTML = '<p>ECharts library is not loaded. Please refresh the page.</p>';
            return;
        }
        
        const chart = echarts.init(chartContainer);
        
        // 准备数据
        const data = [];
        for (let i = 0; i < confusionMatrixData.length; i++) {
            for (let j = 0; j < confusionMatrixData[i].length; j++) {
                data.push([j, i, confusionMatrixData[i][j]]);
            }
        }
        
        // 如果没有类别信息，生成默认类别
        if (classes.length === 0) {
            for (let i = 0; i < confusionMatrixData.length; i++) {
                classes.push(`Class ${i}`);
            }
        }
        
        chart.setOption({
            tooltip: {
                position: 'top',
                formatter: function(params) {
                    return `True: ${classes[params.value[1]]}<br/>Predicted: ${classes[params.value[0]]}<br/>Count: ${params.value[2]}`;
                }
            },
            grid: {
                height: '50%',
                top: '10%'
            },
            xAxis: {
                type: 'category',
                data: classes,
                splitArea: {
                    show: true
                },
                name: 'Predicted'
            },
            yAxis: {
                type: 'category',
                data: classes,
                splitArea: {
                    show: true
                },
                name: 'True',
                inverse: true
            },
            visualMap: {
                min: 0,
                max: Math.max(...confusionMatrixData.flat()),
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '5%'
            },
            series: [{
                name: 'Confusion Matrix',
                type: 'heatmap',
                data: data,
                label: {
                    show: true,
                    formatter: function(params) {
                        return params.value[2];
                    }
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        });
        
        // 添加下载按钮
        let downloadBtn = cmContainer.querySelector('.download-chart-btn');
        if (!downloadBtn) {
            downloadBtn = document.createElement('button');
            downloadBtn.className = 'btn secondary download-chart-btn';
            downloadBtn.textContent = 'Download Confusion Matrix';
            downloadBtn.style.marginTop = '10px';
            cmContainer.appendChild(downloadBtn);
        }
        
        // 下载按钮点击事件
        downloadBtn.onclick = function() {
            if (typeof echarts !== 'undefined') {
                const dataURL = chart.getDataURL({ type: 'png', pixelRatio: 2 });
                const link = document.createElement('a');
                link.href = dataURL;
                link.download = '2_confusion_matrix.png';
                link.click();
            }
        };
        
        window.addEventListener('resize', () => {
            if (typeof echarts !== 'undefined') {
                chart.resize();
            }
        });
    } catch (error) {
        console.error('Error rendering confusion matrix:', error);
    }
}

// 填充选择框选项
async function populateSelectOptions(data_id) {
    try {
        if (!data_id) {
            console.error('data_id is undefined');
            return;
        }
        
        console.log('Populating select options with data_id:', data_id);
        
        // 获取数据集列名
        const response = await fetch('/api/find_columns', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ data_id: data_id })
        });
        
        const result = await response.json();
        if (result.status === 1 || result.status === 200) {
            const columns = result.msg || [];
            console.log('Columns fetched:', columns);
            
            // 填充数据集统计的选择框
            const columnSelect = document.getElementById('column-select');
            if (columnSelect) {
                columnSelect.innerHTML = '';
                // 添加默认选项
                const defaultOption = document.createElement('option');
                defaultOption.value = '';
                defaultOption.textContent = '-- Select Column --';
                columnSelect.appendChild(defaultOption);
                
                if (columns.length > 0) {
                    columns.forEach(column => {
                        const option = document.createElement('option');
                        option.value = column;
                        option.textContent = column;
                        columnSelect.appendChild(option);
                    });
                } else {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'No columns available';
                    columnSelect.appendChild(option);
                }
            }
            
            // 填充相关系数的选择框（隐藏的）
            const corrColumnsSelect = document.getElementById('corr-columns-select');
            if (corrColumnsSelect) {
                corrColumnsSelect.innerHTML = '';
                if (columns.length > 0) {
                    columns.forEach(column => {
                        const option = document.createElement('option');
                        option.value = column;
                        option.textContent = column;
                        corrColumnsSelect.appendChild(option);
                    });
                } else {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'No columns available';
                    corrColumnsSelect.appendChild(option);
                }
            }
            
            // 填充相关系数的复选框组（移动端友好）
            const corrColumnsContainer = document.getElementById('corr-columns-container');
            if (corrColumnsContainer) {
                corrColumnsContainer.innerHTML = '';
                if (columns.length > 0) {
                    columns.forEach(column => {
                        const checkboxDiv = document.createElement('div');
                        checkboxDiv.className = 'checkbox-item';
                        checkboxDiv.innerHTML = `
                            <input type="checkbox" id="corr-checkbox-${column}" value="${column}">
                            <label for="corr-checkbox-${column}">${column}</label>
                        `;
                        corrColumnsContainer.appendChild(checkboxDiv);
                    });
                } else {
                    corrColumnsContainer.innerHTML = '<p>No columns available</p>';
                }
            }
        }
    } catch (error) {
        console.error('Error populating select options:', error);
    }
}

// 加载列统计信息
async function loadColumnStats(data_id, column) {
    try {
        console.log('Loading column stats for:', column, 'data_id:', data_id);
        
        // 检查data_id是否存在
        if (!data_id) {
            console.error('data_id is undefined');
            showTips('Data ID is not available', 'error');
            return;
        }
        
        // 显示加载状态
        const columnStats = document.getElementById('column-stats');
        if (columnStats) {
            columnStats.innerHTML = 'Loading stats...';
        }
        
        // 调用后端API获取实际数据
        const response = await fetch('/api/get_column_stats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                data_id: data_id, 
                column: column 
            })
        });
        
        const result = await response.json();
        if (result.status === 1 || result.status === 200) {
            const { stats, data, isCategorical, isNormalized, originalRange } = result.msg;
            
            console.log('Fetched data for column', column, 'isCategorical:', isCategorical, 'isNormalized:', isNormalized);
            console.log('Stats:', stats);
            console.log('Data length:', data.length);
            
            // 计算频率和比例
            const frequency = {};
            data.forEach(value => {
                const key = value.toString();
                frequency[key] = (frequency[key] || 0) + 1;
            });
            
            const total = data.length;
            const proportions = {};
            for (const [value, count] of Object.entries(frequency)) {
                proportions[value] = count / total;
            }
            
            // 显示统计信息（横向放置）
            if (columnStats) {
                columnStats.innerHTML = `
                    <div class="stats-horizontal">
                        <div class="stat-item">
                            <span class="stat-label">Mean:</span>
                            <span class="stat-value">${isNaN(stats.mean) ? 'N/A' : stats.mean.toFixed(4)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Median:</span>
                            <span class="stat-value">${isNaN(stats.median) ? 'N/A' : stats.median.toFixed(4)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Min:</span>
                            <span class="stat-value">${isNaN(stats.min) ? 'N/A' : stats.min.toFixed(4)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Max:</span>
                            <span class="stat-value">${isNaN(stats.max) ? 'N/A' : stats.max.toFixed(4)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Std:</span>
                            <span class="stat-value">${isNaN(stats.std) ? 'N/A' : stats.std.toFixed(4)}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Count:</span>
                            <span class="stat-value">${isNaN(stats.count) ? 'N/A' : stats.count}</span>
                        </div>
                        ${isNormalized && originalRange ? `
                        <div class="stat-item">
                            <span class="stat-label">Original Range:</span>
                            <span class="stat-value">${originalRange.min.toFixed(4)}-${originalRange.max.toFixed(4)}</span>
                        </div>
                        ` : ''}
                    </div>
                `;
            }
            
            // 显示图表（并排显示）
            const columnChart = document.getElementById('column-chart');
            if (columnChart) {
                columnChart.style.height = '300px';
                columnChart.style.display = 'flex';
                columnChart.style.gap = '20px';
                
                // 清空容器
                columnChart.innerHTML = '';
                
                // 检查echarts是否已定义
                if (typeof echarts === 'undefined') {
                    console.error('ECharts is not loaded');
                    columnChart.innerHTML = '<p>ECharts library is not loaded. Please refresh the page.</p>';
                    return;
                }
                
                // 创建第一个图表容器
                const chart1Container = document.createElement('div');
                chart1Container.style.flex = '1';
                chart1Container.style.height = '100%';
                columnChart.appendChild(chart1Container);
                
                // 创建第二个图表容器（用于频率和比例）
                const chart2Container = document.createElement('div');
                chart2Container.style.flex = '1';
                chart2Container.style.height = '100%';
                columnChart.appendChild(chart2Container);
                
                // 第一个图表
                const chart1 = echarts.init(chart1Container);
                
                if (isCategorical) {
                    // 对于分类数据，显示频率分布
                    const values = Object.keys(frequency).sort((a, b) => parseFloat(a) - parseFloat(b));
                    const counts = values.map(value => frequency[value]);
                    
                    chart1.setOption({
                        title: {
                            text: `${column} Frequency`,
                            textStyle: {
                                fontSize: 14
                            }
                        },
                        tooltip: {
                            trigger: 'axis'
                        },
                        xAxis: {
                            type: 'category',
                            data: values.length > 0 ? values : ['No data'],
                            name: 'Value'
                        },
                        yAxis: {
                            type: 'value',
                            name: 'Frequency',
                            min: 0,
                            max: counts.length > 0 ? Math.max(...counts) + 1 : 1
                        },
                        series: [{
                            name: 'Frequency',
                            type: 'bar',
                            data: counts.length > 0 ? counts : [0],
                            itemStyle: {
                                color: '#4CAF50'
                            },
                            label: {
                                show: true,
                                position: 'top',
                                formatter: function(params) {
                                    return params.value;
                                }
                            }
                        }]
                    });
                } else {
                    // 对于连续数据，显示箱型图
                    // 检查是否有足够的数据点
                    if (data.length < 4) {
                        // 数据点不足，显示散点图
                        chart1.setOption({
                            title: {
                                text: `${column} Data Points`,
                                textStyle: {
                                    fontSize: 14
                                }
                            },
                            tooltip: {
                                trigger: 'item'
                            },
                            xAxis: {
                                type: 'category',
                                data: data.map((_, index) => `Point ${index + 1}`),
                                name: 'Index'
                            },
                            yAxis: {
                                type: 'value',
                                name: 'Value',
                                min: Math.floor(stats.min) - 1,
                                max: Math.ceil(stats.max) + 1
                            },
                            series: [{
                                name: 'Data Points',
                                type: 'scatter',
                                data: data,
                                itemStyle: {
                                    color: '#4CAF50'
                                },
                                symbolSize: 10,
                                label: {
                                    show: true,
                                    position: 'top',
                                    formatter: function(params) {
                                        return params.value.toFixed(4);
                                    }
                                }
                            }]
                        });
                    } else {
                        // 数据点足够，显示箱型图
                        chart1.setOption({
                            title: {
                                text: `${column} Box Plot`,
                                textStyle: {
                                    fontSize: 14
                                }
                            },
                            tooltip: {
                                trigger: 'item',
                                formatter: function(params) {
                                    const boxData = params.data;
                                    return `
                                        Min: ${boxData[1].toFixed(4)}<br/>
                                        Q1: ${boxData[2].toFixed(4)}<br/>
                                        Median: ${boxData[3].toFixed(4)}<br/>
                                        Q3: ${boxData[4].toFixed(4)}<br/>
                                        Max: ${boxData[5].toFixed(4)}
                                    `;
                                }
                            },
                            xAxis: {
                                type: 'category',
                                data: [column],
                                boundaryGap: true
                            },
                            yAxis: {
                                type: 'value',
                                name: 'Value',
                                min: Math.floor(stats.min),
                                max: Math.ceil(stats.max)
                            },
                            series: [{
                                name: 'Box Plot',
                                type: 'boxplot',
                                data: [
                                    [
                                        stats.min,  // 最小值
                                        stats.min + (stats.max - stats.min) * 0.25,  // Q1
                                        stats.median,  // 中位数
                                        stats.min + (stats.max - stats.min) * 0.75,  // Q3
                                        stats.max,  // 最大值
                                        stats.mean  // 均值
                                    ]
                                ],
                                itemStyle: {
                                    color: '#4CAF50'
                                }
                            }]
                        });
                    }
                }
                
                // 第二个图表（频率和比例）
                const chart2 = echarts.init(chart2Container);
                
                if (isCategorical) {
                    // 对于分类数据，显示比例
                    const values = Object.keys(frequency).sort((a, b) => parseFloat(a) - parseFloat(b));
                    const props = values.map(value => proportions[value]);
                    
                    chart2.setOption({
                        title: {
                            text: `${column} Proportion`,
                            textStyle: {
                                fontSize: 14
                            }
                        },
                        tooltip: {
                            trigger: 'axis',
                            formatter: function(params) {
                                const value = params[0].name;
                                const proportion = params[0].value.toFixed(4);
                                return `Value: ${value}<br/>Proportion: ${proportion}`;
                            }
                        },
                        xAxis: {
                            type: 'category',
                            data: values.length > 0 ? values : ['No data'],
                            name: 'Value'
                        },
                        yAxis: {
                            type: 'value',
                            name: 'Proportion',
                            max: 1
                        },
                        series: [{
                            name: 'Proportion',
                            type: 'line',
                            data: props.length > 0 ? props : [0],
                            itemStyle: {
                                color: '#2196F3'
                            },
                            lineStyle: {
                                width: 2
                            },
                            symbol: 'circle',
                            symbolSize: 8,
                            label: {
                                show: true,
                                position: 'top',
                                formatter: function(params) {
                                    return params.value.toFixed(4);
                                }
                            }
                        }]
                    });
                } else {
                    // 对于连续数据，显示数据分布
                    // 检查数据点数量
                    if (data.length < 5) {
                        // 数据点不足，显示简单的计数
                        chart2.setOption({
                            title: {
                                text: `${column} Data Count`,
                                textStyle: {
                                    fontSize: 14
                                }
                            },
                            tooltip: {
                                trigger: 'item'
                            },
                            xAxis: {
                                type: 'category',
                                data: ['Data Points'],
                                name: 'Type'
                            },
                            yAxis: {
                                type: 'value',
                                name: 'Count',
                                min: 0,
                                max: data.length + 1
                            },
                            series: [{
                                name: 'Count',
                                type: 'bar',
                                data: [data.length],
                                itemStyle: {
                                    color: '#2196F3'
                                }
                            }]
                        });
                    } else {
                        // 数据点足够，显示分布
                        // 计算bin的范围
                        const binCount = Math.min(10, data.length); // 对于小样本，减少bin数量
                        const binWidth = (stats.max - stats.min) / binCount;
                        const binLabels = [];
                        const binCounts = Array(binCount).fill(0);
                        
                        // 计算每个bin的计数
                        data.forEach(value => {
                            const binIndex = Math.min(Math.floor((value - stats.min) / binWidth), binCount - 1);
                            binCounts[binIndex]++;
                        });
                        
                        // 生成bin标签
                        for (let i = 0; i < binCount; i++) {
                            const start = stats.min + i * binWidth;
                            const end = start + binWidth;
                            binLabels.push(`${start.toFixed(1)}-${end.toFixed(1)}`);
                        }
                        
                        chart2.setOption({
                            title: {
                                text: `${column} Distribution`,
                                textStyle: {
                                    fontSize: 14
                                }
                            },
                            tooltip: {
                                trigger: 'axis',
                                formatter: function(params) {
                                    const binRange = params[0].name;
                                    const count = params[0].value;
                                    return `Range: ${binRange}<br/>Count: ${count}`;
                                }
                            },
                            grid: {
                                left: '3%',
                                right: '4%',
                                bottom: '15%',
                                containLabel: true
                            },
                            xAxis: {
                                type: 'category',
                                data: binLabels,
                                name: 'Value Range',
                                axisLabel: {
                                    rotate: 60,
                                    fontSize: 10
                                }
                            },
                            yAxis: {
                                type: 'value',
                                name: 'Count'
                            },
                            series: [{
                            name: 'Distribution',
                            type: 'bar',
                            data: binCounts,
                            itemStyle: {
                                color: '#2196F3'
                            },
                            label: {
                                show: true,
                                position: 'top',
                                formatter: function(params) {
                                    return params.value;
                                }
                            }
                        }]
                        });
                    }
                }
                
                // 添加下载按钮
                const containerParent = columnChart.parentElement;
                let downloadBtn = containerParent.querySelector('.download-chart-btn');
                if (!downloadBtn) {
                    downloadBtn = document.createElement('button');
                    downloadBtn.className = 'btn secondary download-chart-btn';
                    downloadBtn.textContent = 'Download Charts';
                    downloadBtn.style.marginTop = '10px';
                    containerParent.appendChild(downloadBtn);
                }
                
                // 下载按钮点击事件
                downloadBtn.onclick = function() {
                    if (typeof echarts !== 'undefined') {
                        // 下载第一个图表
                        const dataURL1 = chart1.getDataURL({ type: 'png', pixelRatio: 2 });
                        const link1 = document.createElement('a');
                        link1.href = dataURL1;
                        link1.download = `2_${column}_chart1.png`;
                        link1.click();
                        
                        // 下载第二个图表
                        const dataURL2 = chart2.getDataURL({ type: 'png', pixelRatio: 2 });
                        const link2 = document.createElement('a');
                        link2.href = dataURL2;
                        link2.download = `3_${column}_chart2.png`;
                        link2.click();
                    }
                };
                
                // 响应式调整
                window.addEventListener('resize', () => {
                    if (typeof echarts !== 'undefined') {
                        chart1.resize();
                        chart2.resize();
                    }
                });
            }
            
            // 添加CSS样式
            if (!document.getElementById('stats-styles')) {
                const style = document.createElement('style');
                style.id = 'stats-styles';
                style.textContent = `
                    .stats-horizontal {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        margin-bottom: 20px;
                    }
                    .stats-horizontal .stat-item {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        min-width: 100px;
                    }
                    .stats-horizontal .stat-label {
                        font-size: 12px;
                        color: #666;
                        margin-bottom: 5px;
                    }
                    .stats-horizontal .stat-value {
                        font-size: 16px;
                        font-weight: bold;
                        color: #333;
                    }
                    .checkbox-group {
                        display: flex;
                        flex-wrap: wrap;
                        flex-direction: row;
                        gap: 15px;
                        margin-top: 10px;
                        max-height: 200px;
                        overflow-y: auto;
                        padding: 10px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }
                    .checkbox-item {
                        display: flex;
                        align-items: center;
                        margin-right: 20px;
                        margin-bottom: 5px;
                        white-space: nowrap;
                        flex-shrink: 0;
                    }
                    .checkbox-item input[type="checkbox"] {
                        margin-right: 5px;
                    }
                    .checkbox-item label {
                        font-size: 14px;
                        cursor: pointer;
                    }
                `;
                document.head.appendChild(style);
            }
        } else {
            showTips(result.msg || 'Failed to load column stats', 'error');
        }
    } catch (error) {
        console.error('Error loading column stats:', error);
        showTips('Error loading column stats', 'error');
    }
}

// 全局变量存储ECharts实例
let correlationChartInstance = null;

// 计算相关系数
async function calculateCorrelation(data_id, columns, correlationType) {
    try {
        console.log('Calculating correlation for columns:', columns, 'Type:', correlationType);
        
        // 显示加载状态
        const correlationChart = document.getElementById('correlation-chart');
        if (correlationChart) {
            correlationChart.innerHTML = 'Calculating correlation...';
        }
        
        // 限制最多5列
        if (columns.length > 5) {
            showTips('Please select at most 5 columns for correlation calculation', 'error');
            return;
        }
        
        // 调用后端API计算相关系数
        const response = await fetch('/api/calculate_correlation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                data_id: data_id, 
                columns: columns, 
                correlation_type: correlationType 
            })
        });
        
        const result = await response.json();
        if (result.status === 1 || result.status === 200) {
            const { correlation_matrix, columns: returnedColumns } = result.msg;
            
            // 显示相关系数矩阵
            const correlationChart = document.getElementById('correlation-chart');
            if (correlationChart) {
                correlationChart.style.height = '400px';
                
                // 销毁之前的ECharts实例
                if (correlationChartInstance) {
                    correlationChartInstance.dispose();
                }
                
                // 检查echarts是否已定义
                if (typeof echarts === 'undefined') {
                    console.error('ECharts is not loaded');
                    correlationChart.innerHTML = '<p>ECharts library is not loaded. Please refresh the page.</p>';
                    return;
                }
                
                // 创建新的ECharts实例
                correlationChartInstance = echarts.init(correlationChart);
                
                correlationChartInstance.setOption({
                    title: {
                        text: `Correlation Matrix (${correlationType})`
                    },
                    tooltip: {
                        position: 'top'
                    },
                    grid: {
                        height: '50%',
                        top: '10%'
                    },
                    xAxis: {
                        type: 'category',
                        data: returnedColumns,
                        splitArea: {
                            show: true
                        }
                    },
                    yAxis: {
                        type: 'category',
                        data: returnedColumns,
                        splitArea: {
                            show: true
                        },
                        inverse: true
                    },
                    visualMap: {
                        min: -1,
                        max: 1,
                        calculable: true,
                        orient: 'horizontal',
                        left: 'center',
                        bottom: '5%',
                        inRange: {
                            color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
                        }
                    },
                    series: [{
                        name: 'Correlation',
                        type: 'heatmap',
                        data: correlation_matrix.flatMap((row, i) => {
                            return row.map((value, j) => [j, i, value]);
                        }),
                        label: {
                            show: true,
                            formatter: function(params) {
                                return params.value[2].toFixed(3);
                            }
                        },
                        emphasis: {
                            itemStyle: {
                                shadowBlur: 10,
                                shadowColor: 'rgba(0, 0, 0, 0.5)'
                            }
                        }
                    }]
                });
                
                // 添加下载按钮
                const containerParent = correlationChart.parentElement;
                let downloadBtn = containerParent.querySelector('.download-chart-btn');
                if (!downloadBtn) {
                    downloadBtn = document.createElement('button');
                    downloadBtn.className = 'btn secondary download-chart-btn';
                    downloadBtn.textContent = 'Download Chart';
                    downloadBtn.style.marginTop = '10px';
                    containerParent.appendChild(downloadBtn);
                }
                
                // 下载按钮点击事件
                downloadBtn.onclick = function() {
                    const dataURL = correlationChartInstance.getDataURL({ type: 'png', pixelRatio: 2 });
                    const link = document.createElement('a');
                    link.href = dataURL;
                    link.download = `4_correlation_matrix_${correlationType}.png`;
                    link.click();
                };
                
                // 移除之前的resize事件监听器
                window.removeEventListener('resize', handleCorrelationChartResize);
                
                // 添加新的resize事件监听器
                function handleCorrelationChartResize() {
                    if (correlationChartInstance) {
                        correlationChartInstance.resize();
                    }
                }
                window.addEventListener('resize', handleCorrelationChartResize);
            }
        } else {
            showTips(result.msg || 'Failed to calculate correlation', 'error');
        }
    } catch (error) {
        console.error('Error calculating correlation:', error);
        showTips('Error calculating correlation', 'error');
    }
}

// 初始化数据集统计和相关系数部分
function initDatasetStats(data_id) {
    try {
        // 填充选择框选项
        populateSelectOptions(data_id);
        
        // 添加选择列后的事件监听
        const columnSelect = document.getElementById('column-select');
        if (columnSelect) {
            columnSelect.addEventListener('change', async () => {
                const selectedColumn = columnSelect.value;
                if (selectedColumn) {
                    await loadColumnStats(data_id, selectedColumn);
                }
            });
        }
        
        // 添加计算相关系数的事件监听
        const calculateCorrBtn = document.getElementById('calculate-corr');
        if (calculateCorrBtn) {
            calculateCorrBtn.addEventListener('click', async () => {
                // 从复选框组中获取选中的列
                const checkboxes = document.querySelectorAll('#corr-columns-container input[type="checkbox"]:checked');
                const selectedColumns = Array.from(checkboxes).map(checkbox => checkbox.value);
                const corrTypeSelect = document.getElementById('corr-type-select');
                const correlationType = corrTypeSelect ? corrTypeSelect.value : 'pearson';
                
                if (selectedColumns.length >= 2) {
                    await calculateCorrelation(data_id, selectedColumns, correlationType);
                } else {
                    showTips('Please select at least two columns', 'error');
                }
            });
        }
    } catch (error) {
        console.error('Error initializing dataset stats:', error);
    }
}

// 渲染训练集测试集信息
function renderTrainTestInfo(train_test_info, data) {
    console.log('Rendering train test info:', train_test_info);
    try {
        const isSplit = train_test_info.is_split || false;
        const totalSize = train_test_info.total_size || 0;
        const details = train_test_info.details || '';
        
        document.getElementById('train-size').textContent = train_test_info.train_size || 0;
        document.getElementById('test-size').textContent = train_test_info.test_size || 0;
        document.getElementById('test-ratio').textContent = train_test_info.ratio || 0;
        
        // 获取模型信息（使用数据库中的 use_model 字段）
        let modelName = data.model_info?.use_model || data.use_model;
        if (!modelName) {
            // 如果 use_model 不存在，尝试其他字段
            modelName = data.model_info?.name || data.data_info?.sheet_name || data.model_info?.id || data.name || data.model_name || data.name_model || 'Unknown Model';
        }
        
        // 显示总数据量和模型信息
        const totalSizeElement = document.getElementById('total-size');
        if (totalSizeElement) {
            // 如果元素存在，更新其值部分
            const valueElement = totalSizeElement.querySelector('.value');
            if (valueElement) {
                valueElement.textContent = `${totalSize} (${modelName})`;
            }
        } else {
            // 如果不存在total-size元素，创建一个与其他信息项相同结构的元素
            const trainTestContainer = document.querySelector('.train-test-info');
            if (trainTestContainer) {
                const totalElement = document.createElement('div');
                totalElement.id = 'total-size';
                totalElement.className = 'info-item';
                totalElement.innerHTML = `
                    <span class="label">Total Size:</span>
                    <span class="value">${totalSize} (${modelName})</span>
                `;
                trainTestContainer.appendChild(totalElement);
            }
        }
        
        // 移除多余的信息显示
        // 移除split-info元素
        const splitInfo = document.getElementById('split-info');
        if (splitInfo) {
            splitInfo.remove();
        }
        
        // 移除split-details元素
        const detailsElement = document.getElementById('split-details');
        if (detailsElement) {
            detailsElement.remove();
        }
    } catch (error) {
        console.error('Error rendering train test info:', error);
    }
}

// 渲染下载链接
function renderDownloadLinks(downloads) {
    console.log('Rendering download links:', downloads);
    try {
        // 获取URL参数中的user_id和model_id
        const params = getUrlParams();
        const user_id = params.user_id;
        const model_id = params.model_id;
        
        // 为SDK下载按钮添加点击事件
        const downloadSdkBtn = document.getElementById('download-sdk');
        downloadSdkBtn.href = '#';
        downloadSdkBtn.onclick = async function(e) {
            e.preventDefault();
            const btn = this;
            
            try {
                // 直接使用URL参数中的模型ID和用户ID
                const modelId = model_id;
                const userId = user_id;
                
                console.log('Downloading SDK for model:', modelId, 'user:', userId);
                
                // 发送下载请求
                const downloadResponse = await fetch('/api/download_sdk', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        user_id: userId,
                        model_id: modelId
                    })
                });
                
                console.log('Download response status:', downloadResponse.status);
                console.log('Download response headers:', downloadResponse.headers);
                
                // 检查响应状态
                if (downloadResponse.ok) {
                    // 检查响应是否为文件
                    const contentDisposition = downloadResponse.headers.get('Content-Disposition');
                    const contentType = downloadResponse.headers.get('Content-Type');
                    
                    console.log('Content-Disposition:', contentDisposition);
                    console.log('Content-Type:', contentType);
                    
                    if (contentDisposition || contentType === 'application/zip') {
                        // 处理文件下载
                        const blob = await downloadResponse.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        // 尝试从Content-Disposition头中获取文件名
                        let filename = 'sdk.zip';
                        if (contentDisposition) {
                            // 优先匹配filename参数
                            let match = contentDisposition.match(/filename=([^;]+)/);
                            if (match) {
                                // 移除可能的引号
                                filename = match[1].replace(/['"]/g, '');
                            } else {
                                // 再匹配filename*参数
                                match = contentDisposition.match(/filename\*=([^;]+)/);
                                if (match) {
                                    filename = decodeURIComponent(match[1].replace(/'/g, ''));
                                }
                            }
                        }
                        // 移除文件名中的UTF-8前缀（如果存在）
                        if (filename.startsWith('UTF-8')) {
                            filename = filename.substring(5);
                        }
                        // 移除文件名中的任何UTF-8相关前缀
                        filename = filename.replace('UTF-8', '');
                        filename = filename.replace('utf-8', '');
                        filename = filename.replace('UTF8', '');
                        filename = filename.replace('utf8', '');
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        // 尝试解析错误响应
                        try {
                            const errorData = await downloadResponse.json();
                            console.log('Error data:', errorData);
                            showTips(errorData.msg || 'Failed to download SDK', 'error', btn);
                        } catch (e) {
                            console.error('Error parsing response:', e);
                            showTips('Failed to download SDK', 'error', btn);
                        }
                    }
                } else {
                    // 尝试解析错误响应
                    try {
                        const errorData = await downloadResponse.json();
                        console.log('Error data:', errorData);
                        showTips(errorData.msg || 'Failed to download SDK', 'error', btn);
                    } catch (e) {
                        console.error('Error parsing error response:', e);
                        showTips('Failed to download SDK', 'error', btn);
                    }
                }
            } catch (error) {
                console.error('Error downloading SDK:', error);
                showTips(`Error downloading SDK: ${error.message}`, 'error', btn);
            }
        };
        
        // 为向量化和归一化下载按钮添加点击事件
        const downloadVectorizationBtn = document.getElementById('download-vectorization');
        downloadVectorizationBtn.href = '#';
        downloadVectorizationBtn.onclick = async function(e) {
            e.preventDefault();
            const btn = this;
            
            try {
                // 直接使用URL参数中的模型ID
                const modelId = model_id;
                
                console.log('Downloading vectorization for model:', modelId);
                
                // 发送下载请求
                const downloadResponse = await fetch('/api/download_vectorization', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model_id: modelId
                    })
                });
                
                console.log('Vectorization download response status:', downloadResponse.status);
                
                // 检查响应状态
                if (downloadResponse.ok) {
                    // 检查响应是否为文件
                    const contentDisposition = downloadResponse.headers.get('Content-Disposition');
                    const contentType = downloadResponse.headers.get('Content-Type');
                    
                    console.log('Content-Disposition:', contentDisposition);
                    console.log('Content-Type:', contentType);
                    
                    if (contentDisposition || contentType === 'application/json') {
                        // 处理文件下载
                        const blob = await downloadResponse.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        // 尝试从Content-Disposition头中获取文件名
                        let filename = 'vectorization.json';
                        if (contentDisposition) {
                            // 优先匹配filename参数
                            let match = contentDisposition.match(/filename=([^;]+)/);
                            if (match) {
                                // 移除可能的引号
                                filename = match[1].replace(/['"]/g, '');
                            } else {
                                // 再匹配filename*参数
                                match = contentDisposition.match(/filename\*=([^;]+)/);
                                if (match) {
                                    filename = decodeURIComponent(match[1].replace(/'/g, ''));
                                }
                            }
                        }
                        // 移除文件名中的UTF-8前缀（如果存在）
                        if (filename.startsWith('UTF-8')) {
                            filename = filename.substring(5);
                        }
                        // 移除文件名中的任何UTF-8相关前缀
                        filename = filename.replace('UTF-8', '');
                        filename = filename.replace('utf-8', '');
                        filename = filename.replace('UTF8', '');
                        filename = filename.replace('utf8', '');
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        // 尝试解析为JSON
                        try {
                            const responseData = await downloadResponse.json();
                            console.log('Error data:', responseData);
                            showTips(responseData.msg || 'Failed to download vectorization file', 'error', btn);
                        } catch (e) {
                            console.error('Error parsing response:', e);
                            showTips('Failed to download vectorization file', 'error', btn);
                        }
                    }
                } else {
                    // 尝试解析错误响应
                    try {
                        const errorData = await downloadResponse.json();
                        console.log('Error data:', errorData);
                        showTips(errorData.msg || 'Failed to download vectorization file', 'error', btn);
                    } catch (e) {
                        console.error('Error parsing error response:', e);
                        showTips('Failed to download vectorization file', 'error', btn);
                    }
                }
            } catch (error) {
                console.error('Error downloading vectorization file:', error);
                showTips(`Error downloading vectorization file: ${error.message}`, 'error', btn);
            }
        };
        
        const downloadNormalizationBtn = document.getElementById('download-normalization');
        downloadNormalizationBtn.href = '#';
        downloadNormalizationBtn.onclick = async function(e) {
            e.preventDefault();
            const btn = this;
            
            try {
                // 直接使用URL参数中的模型ID
                const modelId = model_id;
                
                console.log('Downloading normalization for model:', modelId);
                
                // 发送下载请求
                const downloadResponse = await fetch('/api/download_normalization', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        model_id: modelId
                    })
                });
                
                console.log('Normalization download response status:', downloadResponse.status);
                
                // 检查响应状态
                if (downloadResponse.ok) {
                    // 检查响应是否为文件
                    const contentDisposition = downloadResponse.headers.get('Content-Disposition');
                    const contentType = downloadResponse.headers.get('Content-Type');
                    
                    console.log('Content-Disposition:', contentDisposition);
                    console.log('Content-Type:', contentType);
                    
                    if (contentDisposition || contentType === 'application/json') {
                        // 处理文件下载
                        const blob = await downloadResponse.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        // 尝试从Content-Disposition头中获取文件名
                        let filename = 'normalization.json';
                        if (contentDisposition) {
                            // 优先匹配filename参数
                            let match = contentDisposition.match(/filename=([^;]+)/);
                            if (match) {
                                // 移除可能的引号
                                filename = match[1].replace(/['"]/g, '');
                            } else {
                                // 再匹配filename*参数
                                match = contentDisposition.match(/filename\*=([^;]+)/);
                                if (match) {
                                    filename = decodeURIComponent(match[1].replace(/'/g, ''));
                                }
                            }
                        }
                        // 移除文件名中的UTF-8前缀（如果存在）
                        if (filename.startsWith('UTF-8')) {
                            filename = filename.substring(5);
                        }
                        // 移除文件名中的任何UTF-8相关前缀
                        filename = filename.replace('UTF-8', '');
                        filename = filename.replace('utf-8', '');
                        filename = filename.replace('UTF8', '');
                        filename = filename.replace('utf8', '');
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                    } else {
                        // 尝试解析为JSON
                        try {
                            const responseData = await downloadResponse.json();
                            console.log('Error data:', responseData);
                            showTips(responseData.msg || 'Failed to download normalization file', 'error', btn);
                        } catch (e) {
                            console.error('Error parsing response:', e);
                            showTips('Failed to download normalization file', 'error', btn);
                        }
                    }
                } else {
                    // 尝试解析错误响应
                    try {
                        const errorData = await downloadResponse.json();
                        console.log('Error data:', errorData);
                        showTips(errorData.msg || 'Failed to download normalization file', 'error', btn);
                    } catch (e) {
                        console.error('Error parsing error response:', e);
                        showTips('Failed to download normalization file', 'error', btn);
                    }
                }
            } catch (error) {
                console.error('Error downloading normalization file:', error);
                showTips(`Error downloading normalization file: ${error.message}`, 'error', btn);
            }
        };
    } catch (error) {
        console.error('Error rendering download links:', error);
    }
}

// 设置open_bool开关
function setupOpenBoolToggle(open_bool, user_id, model_id) {
    const toggle = document.getElementById('open-bool-toggle');
    toggle.checked = open_bool;
    
    toggle.addEventListener('change', async () => {
        try {
            const response = await fetch('/api/toggle_model_open', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_id: user_id,
                    model_id: model_id,
                    open_bool: toggle.checked
                })
            });
            
            const result = await response.json();
            
            if (result.status === 1) {
                showTips('Model status updated successfully');
            } else {
                showTips('Failed to update model status', 'error');
                toggle.checked = open_bool; // 恢复原状态
            }
        } catch (error) {
            console.error('Error updating model status:', error);
            showTips('Error updating model status', 'error');
            toggle.checked = open_bool; // 恢复原状态
        }
    });
}

// 全局变量存储刷新定时器
let modelStatusRefreshInterval = null;

// 初始化页面
async function init() {
    console.log('Initializing page...');
    
    // 提前加载CSS样式
    if (!document.getElementById('stats-styles')) {
        const style = document.createElement('style');
        style.id = 'stats-styles';
        style.textContent = `
            .stats-horizontal {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin-bottom: 20px;
            }
            .stats-horizontal .stat-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                min-width: 100px;
            }
            .stats-horizontal .stat-label {
                font-size: 12px;
                color: #666;
                margin-bottom: 5px;
            }
            .stats-horizontal .stat-value {
                font-size: 16px;
                font-weight: bold;
                color: #333;
            }
            .checkbox-group {
                display: flex;
                flex-wrap: wrap;
                flex-direction: row;
                gap: 15px;
                margin-top: 10px;
                max-height: 200px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
                margin-right: 20px;
                margin-bottom: 5px;
                white-space: nowrap;
                flex-shrink: 0;
            }
            .checkbox-item input[type="checkbox"] {
                margin-right: 5px;
            }
            .checkbox-item label {
                font-size: 14px;
                cursor: pointer;
            }
        `;
        document.head.appendChild(style);
    }
    
    await loadModelDetail();
    // 不要在这里调用initDatasetStats，因为data_id还没有获取到
    // initDatasetStats();
    console.log('Page initialized');
    
    // 启动模型状态自动刷新，每10秒一次
    startModelStatusRefresh();
}

// 启动模型状态自动刷新
function startModelStatusRefresh() {
    // 清除现有的定时器
    if (modelStatusRefreshInterval) {
        clearInterval(modelStatusRefreshInterval);
    }
    
    // 设置新的定时器，每3分钟刷新一次
    modelStatusRefreshInterval = setInterval(async () => {
        console.log('Auto refreshing model status...');
        try {
            await loadModelDetail();
        } catch (error) {
            console.error('Error during auto refresh:', error);
        }
    }, 180000); // 3分钟刷新一次
    
    console.log('Model status auto refresh started with 3min interval');
}

// 停止模型状态自动刷新
function stopModelStatusRefresh() {
    if (modelStatusRefreshInterval) {
        clearInterval(modelStatusRefreshInterval);
        modelStatusRefreshInterval = null;
        console.log('Model status auto refresh stopped');
    }
}

// 页面加载完成后初始化
console.log('Adding DOMContentLoaded listener');
window.addEventListener('DOMContentLoaded', async () => {
    console.log('DOMContentLoaded event fired');
    
    // 等待一小段时间，确保所有DOM元素都已加载
    setTimeout(async () => {
        console.log('Initializing after DOM fully loaded');
        await init();
        
        // 添加导航按钮事件监听器
        const prevModelBtn = document.getElementById('prev-model-btn');
        const nextModelBtn = document.getElementById('next-model-btn');
        
        if (prevModelBtn) {
            prevModelBtn.addEventListener('click', navigateToPrevModel);
            console.log('Added event listener for previous model button');
        }
        
        if (nextModelBtn) {
            nextModelBtn.addEventListener('click', navigateToNextModel);
            console.log('Added event listener for next model button');
        }
    }, 500);
});