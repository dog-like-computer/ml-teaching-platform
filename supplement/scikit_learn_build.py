# todo-1 important need library
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy.special import expit, comb
from collections import Counter

from scipy.linalg import eigh
import pandas as pd
import numpy as np
import pickle
import json
import os

# todo-2.0 build scikit-learn class (based)
class SkLearn:
    def __init__(self):
        pass

    def _response(self,status=1,content=None):
        return {'status':status,'content':content}

# this include MLP some function
def sigmoid(x):
    return expit(x)

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def linear(x):
    return x

def linear_deriv(x):
    return np.ones_like(x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def init_population(pop_size, weight_shapes):
    population = []
    for _ in range(pop_size):
        individual = []
        for shape in weight_shapes:
            w = np.random.normal(0, 0.1, shape)
            individual.append(w)
        population.append(individual)
    return population

def fitness(individual, X, y, forward_fn):
    y_pred = forward_fn(X, individual)
    if len(y.shape) == 2:
        loss = -np.mean(y * np.log(y_pred + 1e-10))
    else:
        loss = np.mean((y - y_pred) ** 2)
    return 1 / (loss + 1e-10)

def selection(population, fitness_scores, elite_size=2):
    elite = np.argsort(fitness_scores)[-elite_size:]
    elite_individuals = [population[i] for i in elite]
    total_fitness = sum(fitness_scores)
    probs = [score / total_fitness for score in fitness_scores]
    selected = np.random.choice(len(population), size=len(population) - elite_size, p=probs)
    selected_individuals = [population[i] for i in selected]
    return elite_individuals + selected_individuals

def crossover(parent1, parent2, crossover_rate=0.8):
    if np.random.random() > crossover_rate:
        return parent1.copy()
    child = []
    for w1, w2 in zip(parent1, parent2):
        mask = np.random.rand(*w1.shape) < 0.5
        child_w = np.where(mask, w1, w2)
        child.append(child_w)
    return child

def mutation(individual, mutation_rate=0.01, mutation_strength=0.05):
    mutated = []
    for w in individual:
        mutation = np.random.normal(0, mutation_strength, w.shape)
        mask = np.random.rand(*w.shape) < mutation_rate
        mutated_w = w + mask * mutation
        mutated.append(mutated_w)
    return mutated

def genetic_algorithm(X, y, forward_fn, weight_shapes, pop_size=10, generations=5, elite_size=2):
    population = init_population(pop_size, weight_shapes)
    for _ in range(generations):
        fitness_scores = [fitness(ind, X, y, forward_fn) for ind in population]
        population = selection(population, fitness_scores, elite_size)
        new_population = []
        for i in range(len(population)):
            parent1 = population[i]
            parent2 = population[np.random.randint(len(population))]
            child = crossover(parent1, parent2)
            child = mutation(child)
            new_population.append(child)
        population = new_population
    fitness_scores = [fitness(ind, X, y, forward_fn) for ind in population]
    best_ind = population[np.argmax(fitness_scores)]
    return best_ind

def rbf(x, center, sigma):
    # 确保 x 和 center 形状匹配
    if x.ndim == 2 and center.ndim == 1:
        # x: (n_samples, n_features), center: (n_features,)
        return np.exp(-np.sum((x - center) ** 2, axis=1) / (2 * sigma ** 2 + 1e-10))
    elif x.ndim == 2 and center.ndim == 2:
        # x: (n_samples, n_features), center: (n_centers, n_features)
        # 计算每个样本到每个中心的距离
        return np.exp(-np.sum((x[:, np.newaxis, :] - center) ** 2, axis=2) / (2 * sigma ** 2 + 1e-10))
    else:
        # 其他情况
        return np.exp(-np.sum((x - center) ** 2) / (2 * sigma ** 2 + 1e-10))

def euclidean_dist(x1, x2=None):
    return np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))

def som_neighborhood(d, i, sigma):
    return np.exp(-(d ** 2) / (2 * sigma ** 2 + 1e-10))

def boltzmann_prob(x, beta):
    return 1 / (1 + np.exp(-beta * x))

def linear_kernel(X1, X2):
    """线性核: K(x1,x2) = x1·x2"""
    return np.dot(X1, X2.T)

def rbf_kernel(X1, X2, gamma='scale'):
    """RBF核（高斯核）: K(x1,x2) = exp(-gamma·||x1-x2||²)"""
    if gamma == 'scale':
        gamma = 1.0 / X1.shape[1]
    elif gamma == 'auto':
        gamma = 1.0
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * dist)

def poly_kernel(X1, X2, degree=3, coef0=1.0):
    """多项式核: K(x1,x2) = (gamma·x1·x2 + coef0)^degree"""
    return (np.dot(X1, X2.T) + coef0) ** degree

def sigmoid_kernel(X1, X2, gamma='scale', coef0=0.0):
    """Sigmoid核: K(x1,x2) = tanh(gamma·x1·x2 + coef0)"""
    if gamma == 'scale':
        gamma = 1.0 / X1.shape[1]
    elif gamma == 'auto':
        gamma = 1.0
    return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

def get_kernel(kernel, gamma='scale', degree=3, coef0=1.0):
    """核函数选择器"""
    if kernel == 'linear':
        def linear_kernel_wrapper(X1, X2):
            return linear_kernel(X1, X2)
        return linear_kernel_wrapper
    elif kernel == 'rbf':
        def rbf_kernel_wrapper(X1, X2):
            return rbf_kernel(X1, X2, gamma)
        return rbf_kernel_wrapper
    elif kernel == 'poly':
        def poly_kernel_wrapper(X1, X2):
            return poly_kernel(X1, X2, degree, coef0)
        return poly_kernel_wrapper
    elif kernel == 'sigmoid':
        def sigmoid_kernel_wrapper(X1, X2):
            return sigmoid_kernel(X1, X2, gamma, coef0)
        return sigmoid_kernel_wrapper
    else:
        raise ValueError(f"不支持的核函数: {kernel}")

def smo_solver(X, y, C, kernel, tol=1e-3, max_iter=1000):
    n_samples, n_features = X.shape
    alpha = np.zeros(n_samples)  # 拉格朗日乘子（对偶变量）
    b = 0.0  # 偏置项
    E = np.zeros(n_samples)  # 误差缓存
    kernel_mat = kernel(X, X)  # 核矩阵

    iter_num = 0
    total_iter = 0
    
    # 优化：预计算所有样本的误差
    def compute_error(i):
        return np.sum(alpha * y * kernel_mat[:, i]) + b - y[i]

    # 优化：添加误差阈值，当所有误差都小于阈值时提前停止
    def all_errors_below_tol():
        for i in range(n_samples):
            E[i] = compute_error(i)
            if (y[i] * E[i] < -tol and alpha[i] < C) or (y[i] * E[i] > tol and alpha[i] > 0):
                return False
        return True

    # 优化：批量计算误差，减少循环次数
    def batch_compute_error():
        for i in range(n_samples):
            E[i] = compute_error(i)

    # 首先批量计算一次误差
    batch_compute_error()

    while iter_num < max_iter and total_iter < max_iter * 10:
        # 优化：提前检查所有样本是否满足KKT条件
        if all_errors_below_tol():
            break
            
        alpha_changed = 0
        # 优化：优先选择违反KKT条件最严重的样本
        non_bound_indices = np.where((alpha > 0) & (alpha < C))[0]
        bound_indices = np.where((alpha == 0) | (alpha == C))[0]
        # 先检查非边界样本，再检查边界样本
        check_indices = np.concatenate([non_bound_indices, bound_indices])
        
        # 优化：批量计算所有样本的误差，减少重复计算
        batch_compute_error()
        
        # 优化：优先处理违反KKT条件最严重的样本
        kkt_violations = []
        for i in check_indices:
            if (y[i] * E[i] < -tol and alpha[i] < C) or (y[i] * E[i] > tol and alpha[i] > 0):
                # 计算违反程度
                violation = abs(y[i] * E[i])
                kkt_violations.append((violation, i))
        
        # 按违反程度排序，优先处理最严重的
        kkt_violations.sort(reverse=True, key=lambda x: x[0])
        prioritized_indices = [i for _, i in kkt_violations]
        
        for i in prioritized_indices:
            # 违反KKT条件则选择i作为第一个变量
            if (y[i] * E[i] < -tol and alpha[i] < C) or (y[i] * E[i] > tol and alpha[i] > 0):
                # 优化：选择使|E_i - E_j|最大的j，加快收敛
                if len(non_bound_indices) > 0:
                    # 从非边界样本中选择
                    j = non_bound_indices[np.argmax(np.abs(E[non_bound_indices] - E[i]))]
                else:
                    # 随机选择
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)

                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()

                # 计算上下界L和H（软间隔约束）
                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[i] + alpha[j])
                if L == H:
                    continue

                # 计算η
                eta = 2 * kernel_mat[i, j] - kernel_mat[i, i] - kernel_mat[j, j]
                if eta >= 0:
                    continue

                # 更新αj
                alpha[j] -= y[j] * (E[i] - E[j]) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                # 若αj变化过小则跳过
                if abs(alpha[j] - alpha_j_old) < tol:
                    continue

                # 更新αi
                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                # 更新偏置b
                b1 = b - E[i] - y[i] * (alpha[i] - alpha_i_old) * kernel_mat[i, i] - y[j] * (alpha[j] - alpha_j_old) * \
                     kernel_mat[i, j]
                b2 = b - E[j] - y[i] * (alpha[i] - alpha_i_old) * kernel_mat[i, j] - y[j] * (alpha[j] - alpha_j_old) * \
                     kernel_mat[j, j]
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                # 优化：只更新受影响的样本的误差，而不是所有样本
                for k in [i, j]:
                    if 0 < alpha[k] < C:
                        E[k] = compute_error(k)

                alpha_changed += 1
                # 优化：每处理10个样本检查一次收敛，避免不必要的计算
                if alpha_changed >= 10:
                    break

        if alpha_changed == 0:
            iter_num += 1
        else:
            iter_num = 0
        total_iter += 1

    # 提取支持向量（α>0的样本）
    sv_idx = alpha > tol
    sv_alpha = alpha[sv_idx]
    sv_X = X[sv_idx]
    sv_y = y[sv_idx]

    return sv_alpha, sv_X, sv_y, b

def xgb_loss_gradient(y_true, y_pred, loss_type):
    """XGBoost 一阶/二阶梯度计算（泰勒展开核心）"""
    if loss_type == 'binary':  # 二分类（逻辑损失）
        p = sigmoid(y_pred)
        grad = p - y_true  # 一阶导数
        hess = p * (1 - p) # 二阶导数
    elif loss_type == 'regression':  # 回归（平方损失）
        grad = y_pred - y_true  # 一阶导数
        hess = np.ones_like(y_pred) # 二阶导数
    elif loss_type == 'multiclass': # 多分类（交叉熵）
        p = softmax(y_pred)
        grad = p - y_true  # 一阶导数
        hess = p * (1 - p) # 二阶导数
    else:
        raise ValueError(f"不支持的损失类型：{loss_type}")
    return grad, hess

def manhattan_dist(X1, X2=None):
    if X2 is None:
        X2 = X1
    return cdist(X1, X2, metric='cityblock')

def cosine_dist(X1, X2=None):
    if X2 is None:
        X2 = X1
    return cdist(X1, X2, metric='cosine')

def get_distance_fn(metric='euclidean'):
    if metric == 'euclidean':
        return euclidean_dist
    elif metric == 'manhattan':
        return manhattan_dist
    elif metric == 'cosine':
        return cosine_dist
    else:
        raise ValueError(f"不支持的距离度量：{metric}")

# todo-2.1 build scikit-learn evaluation
class Evaluation(SkLearn):
    def __init__(self):
        super(Evaluation, self).__init__()

    def mse(self,y_predict,y_result,calculate='all'):
        y_predict = np.array(y_predict)
        y_result = np.array(y_result)
        if calculate == 'all':
            return self._response(content={'mse':np.mean((y_predict - y_result) ** 2)})
        return self._response(content={'mse': np.mean((y_predict - y_result) ** 2,axis=0)})

    def mase(self,y_predict,y_result,calculate='all'):
        y_predict = np.array(y_predict)
        y_result = np.array(y_result)
        if calculate == 'all':
            return self._response(content={'mase':(np.mean(np.abs(y_result - y_predict))) / (np.mean(np.abs(y_result[1:] - y_result[:-1])))})
        return self._response(content={'mase': (np.mean(np.abs(y_result - y_predict),axis=0)) / (np.mean(np.abs(y_result[1:] - y_result[:-1]),axis=0))})

    # This function only use to two class question
    def roc(self, y_predict, y_probability=None, calculate='all'):
        y_predict = np.array(y_predict, dtype=int)
        y_prob = np.array(y_predict, dtype=float) if y_probability is None else np.array(y_probability, dtype=float)
        if y_predict.shape != y_prob.shape:
            return self._response(0, 'Label shape != probability shape')
        if len(y_predict.shape) == 1:
            p = np.sum(y_predict == 1)
            n = np.sum(y_predict == 0)
            if p == 0 or n == 0:
                return self._response(content={'roc': {'error': 'No positive/negative samples'}})
            unique = np.unique(y_prob)
            unique_sort = np.sort(unique)[::-1]
            thresholds = np.concatenate([[unique_sort[0] + 0.1], unique_sort])
            sort_idx = np.argsort(y_prob)
            y_pred_sort = y_predict[sort_idx]
            y_prob_sort = y_prob[sort_idx]
            fpr, tpr = [], []
            for th in thresholds:
                pred_label = (y_prob_sort >= th).astype(int)
                TP = np.sum((y_pred_sort == 1) & (pred_label == 1))
                FP = np.sum((y_pred_sort == 0) & (pred_label == 1))
                fpr.append(FP / n if n != 0 else 0.0)
                tpr.append(TP / p if p != 0 else 0.0)
            res = {
                'fpr': np.round(np.array(fpr), 4),
                'tpr': np.round(np.array(tpr), 4),
                'thresholds': np.round(thresholds, 4)
            }
            return self._response(content={'roc': res})
        n_labels = y_predict.shape[1]
        all_fpr = []
        all_tpr = []
        all_thresholds = []
        for idx in range(n_labels):
            pred = y_predict[:, idx]
            prob = y_prob[:, idx]
            p = np.sum(pred == 1)
            n = np.sum(pred == 0)
            if p == 0 or n == 0:
                all_fpr.append(np.array([np.nan]))
                all_tpr.append(np.array([np.nan]))
                all_thresholds.append(np.array([np.nan]))
                continue
            unique = np.unique(prob)
            unique_sort = np.sort(unique)[::-1]
            thresholds = np.concatenate([[unique_sort[0] + 0.1], unique_sort])
            sort_idx = np.argsort(prob)
            pred_sort = pred[sort_idx]
            prob_sort = prob[sort_idx]
            fpr, tpr = [], []
            for th in thresholds:
                pred_label = (prob_sort >= th).astype(int)
                TP = np.sum((pred_sort == 1) & (pred_label == 1))
                FP = np.sum((pred_sort == 0) & (pred_label == 1))
                fpr.append(FP / n if n != 0 else 0.0)
                tpr.append(TP / p if p != 0 else 0.0)
            all_fpr.append(np.array(fpr))
            all_tpr.append(np.array(tpr))
            all_thresholds.append(thresholds)
        max_len = max([len(arr) for arr in all_thresholds])
        for i in range(n_labels):
            pad_len = max_len - len(all_fpr[i])
            if pad_len > 0:
                all_fpr[i] = np.pad(all_fpr[i], (0, pad_len), constant_values=np.nan)
                all_tpr[i] = np.pad(all_tpr[i], (0, pad_len), constant_values=np.nan)
                all_thresholds[i] = np.pad(all_thresholds[i], (0, pad_len), constant_values=np.nan)
        res = {
            'fpr': np.round(np.array(all_fpr), 4),  # 二维数组 (n_labels, max_len)
            'tpr': np.round(np.array(all_tpr), 4),  # 二维数组 (n_labels, max_len)
            'thresholds': np.round(np.array(all_thresholds), 4)  # 二维数组 (n_labels, max_len)
        }
        return self._response(content={'roc': res})

    # This function only use to two class question: bottom of roc
    def auc(self, roc_result, tpr=None, calculate='all'):
        if isinstance(roc_result, (np.ndarray, list)) and isinstance(tpr, (np.ndarray, list)):
            fpr = np.array(roc_result, dtype=float)
            tpr = np.array(tpr, dtype=float)
            if len(fpr.shape) == 1:
                sorted_idx = np.argsort(fpr)
                delta_fpr = np.diff(fpr[sorted_idx])
                avg_tpr = (tpr[sorted_idx][1:] + tpr[sorted_idx][:-1]) / 2
                auc = max(0.0, min(1.0, np.sum(delta_fpr * avg_tpr)))
                return self._response(content={'auc': round(auc, 4)})
            elif len(fpr.shape) == 2:
                auc_list = []
                for i in range(fpr.shape[0]):
                    if np.isnan(fpr[i]).all() or np.isnan(tpr[i]).all():
                        auc_list.append(np.nan)
                        continue
                    valid_idx = ~np.isnan(fpr[i])
                    fpr_valid = fpr[i][valid_idx]
                    tpr_valid = tpr[i][valid_idx]
                    sorted_idx = np.argsort(fpr_valid)
                    delta_fpr = np.diff(fpr_valid[sorted_idx])
                    avg_tpr = (tpr_valid[sorted_idx][1:] + tpr_valid[sorted_idx][:-1]) / 2
                    auc = max(0.0, min(1.0, np.sum(delta_fpr * avg_tpr)))
                    auc_list.append(round(auc, 4))
                if calculate == 'all':
                    valid_auc = [a for a in auc_list if not np.isnan(a)]
                    auc_res = round(np.mean(valid_auc), 4) if valid_auc else np.nan
                else:
                    auc_res = np.array(auc_list)
                return self._response(content={'auc': auc_res})
        elif isinstance(roc_result, dict) and tpr is None:
            roc_content = roc_result.get('content', {}).get('roc', {})
            if 'error' in roc_content:
                return self._response(content={'auc': roc_content['error']})
            fpr = roc_content['fpr']
            tpr = roc_content['tpr']
            return self.auc(fpr, tpr, calculate=calculate)
        else:
            return self._response(0, "Invalid input: support (fpr,tpr) array or roc() result dict")

    def f1(self, y_result, y_predict=None, y_probability=None, threshold=0.5):
        y_result = np.array(y_result, dtype=int)
        if y_predict is not None and y_probability is not None:
            return self._response(0, content='Loss parameter: y_predict和y_probability')
        elif y_predict is None and y_probability is None:
            return self._response(0, content='Loss parameter: y_predict或y_probability')
        elif y_probability is not None:
            y_probability = np.array(y_probability, dtype=float)
            if y_result.shape != y_probability.shape:
                return self._response(0, content='Result shape != Probability shape')
            y_predict = (y_probability >= threshold).astype(int)
        elif y_predict is not None:
            y_predict = np.array(y_predict, dtype=int)
            if y_result.shape != y_predict.shape:
                return self._response(0, content='Result shape != Predict shape')
        precision_res = self.precision(y_result, y_predict)
        if precision_res['status'] == 0:
            return self._response(0, content=f"Precision计算错误: {precision_res['content']}")
        precision = precision_res['content']['precision']
        recall_res = self.recall(y_result, y_predict)
        if recall_res['status'] == 0:
            return self._response(0, content=f"Recall计算错误: {recall_res['content']}")
        recall = recall_res['content']['recall']
        if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
            f1_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        else:
            f1_val = np.where(
                (precision + recall) != 0,
                2 * (precision * recall) / (precision + recall),
                0.0
            )
        return self._response(content={
            'f1': f1_val,
            'precision': precision,
            'recall': recall
        })

    def precision(self, y_result, y_predict):
        y_result = np.array(y_result, dtype=int)
        y_predict = np.array(y_predict, dtype=int)
        if y_result.shape != y_predict.shape:
            return self._response(0, 'y_result shape != y_predict shape')
        if len(y_result.shape) == 1:
            TP = np.sum((y_result == 1) & (y_predict == 1))
            FP = np.sum((y_result == 0) & (y_predict == 1))
            precision_val = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        else:
            n_labels = y_result.shape[1]
            precision_vals = []
            for idx in range(n_labels):
                res_col = y_result[:, idx]
                pred_col = y_predict[:, idx]
                TP = np.sum((res_col == 1) & (pred_col == 1))
                FP = np.sum((res_col == 0) & (pred_col == 1))
                precision_vals.append(TP / (TP + FP) if (TP + FP) != 0 else 0.0)
            precision_val = np.array(precision_vals)
        return self._response(content={'precision': precision_val})

    def recall(self, y_result, y_predict):
        y_result = np.array(y_result, dtype=int)
        y_predict = np.array(y_predict, dtype=int)
        if y_result.shape != y_predict.shape:
            return self._response(0, 'y_result shape != y_predict shape')
        if len(y_result.shape) == 1:
            TP = np.sum((y_result == 1) & (y_predict == 1))
            FN = np.sum((y_result == 1) & (y_predict == 0))
            recall_val = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        else:
            n_labels = y_result.shape[1]
            recall_vals = []
            for idx in range(n_labels):
                res_col = y_result[:, idx]
                pred_col = y_predict[:, idx]
                TP = np.sum((res_col == 1) & (pred_col == 1))
                FN = np.sum((res_col == 1) & (pred_col == 0))
                recall_vals.append(TP / (TP + FN) if (TP + FN) != 0 else 0.0)
            recall_val = np.array(recall_vals)
        return self._response(content={'recall': recall_val})

    def bias(self, y_true, y_predict):
        y_true = np.array(y_true, dtype=float)
        y_predict = np.array(y_predict, dtype=float)
        if y_true.shape != y_predict.shape:
            return self._response(0, 'y_true shape != y_predict shape')
        if y_true.size == 0:
            return self._response(0, 'Input array length is zero')
        if len(y_true.shape) == 1:
            bias_val = np.mean(y_predict) - np.mean(y_true)
        else:
            bias_val = np.mean(y_predict, axis=0) - np.mean(y_true, axis=0)
        return self._response(content={'bias': bias_val})

    def variance(self, y_predict):
        y_predict = np.array(y_predict, dtype=float)
        if y_predict.size == 0:
            return self._response(0, 'Input array length is zero')
        if len(y_predict.shape) == 1:
            variance_val = np.var(y_predict, ddof=0)
        else:
            variance_val = np.var(y_predict, axis=0, ddof=0)
        return self._response(content={'variance': variance_val})

    def r2(self, y_true, y_predict):
        y_true = np.array(y_true, dtype=float)
        y_predict = np.array(y_predict, dtype=float)
        if y_true.shape != y_predict.shape:
            return self._response(0, 'y_true shape != y_predict shape')
        if y_true.size == 0:
            return self._response(0, 'Input array length is zero')
        if len(y_true.shape) == 1:
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot == 0:
                r2_val = 1.0 if np.allclose(y_predict, y_true) else 0.0
            else:
                ss_res = np.sum((y_true - y_predict) ** 2)
                r2_val = 1 - (ss_res / ss_tot)
        else:
            n_labels = y_true.shape[1]
            r2_vals = []
            for idx in range(n_labels):
                true_col = y_true[:, idx]
                pred_col = y_predict[:, idx]
                ss_tot = np.sum((true_col - np.mean(true_col)) ** 2)
                if ss_tot == 0:
                    r2_val = 1.0 if np.allclose(pred_col, true_col) else 0.0
                else:
                    ss_res = np.sum((true_col - pred_col) ** 2)
                    r2_val = 1 - (ss_res / ss_tot)
                r2_vals.append(r2_val)
            r2_val = np.array(r2_vals)
        return self._response(content={'r2': r2_val})

    def mae(self, y_predict, y_result, calculate='all'):
        """平均绝对误差 Mean Absolute Error"""
        y_predict = np.array(y_predict)
        y_result = np.array(y_result)
        if calculate == 'all':
            return self._response(content={'mae': np.mean(np.abs(y_predict - y_result))})
        return self._response(content={'mae': np.mean(np.abs(y_predict - y_result), axis=0)})

    def accuracy(self, y_true, y_pred, calculate='all'):
        # 转换为numpy数组并统一类型
        y_true = np.array(y_true, dtype=np.int64)
        y_pred = np.array(y_pred, dtype=np.int64)

        # 维度校验
        if y_true.shape != y_pred.shape:
            return self._response(0, content="y_true和y_pred的形状必须完全一致")

        # 单因变量转二维，统一处理逻辑
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        n_samples, n_outputs = y_true.shape
        acc_per_output = []

        # 逐因变量计算准确率
        for i in range(n_outputs):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            acc = float(np.mean(y_t == y_p))  # 转为原生float
            acc_per_output.append(acc)

        # 计算平均准确率
        avg_acc = float(np.mean(acc_per_output))

        # 根据calculate参数控制输出格式
        if calculate == 'all':
            # calculate='all'：仅返回平均准确率（float）
            content = {"accuracy": avg_acc}
        else:
            # calculate!='all'：仅返回各因变量准确率
            if n_outputs == 1:
                content = {"accuracy": acc_per_output[0]}  # 单因变量返回单个float
            else:
                content = {"accuracy": acc_per_output}  # 多因变量返回列表

        return self._response(1, content=content)

    def silhouette_score(self, X, labels):
        n_samples = len(X)
        if len(np.unique(labels)) < 2:
            return 0.0
        dist_mat = euclidean_dist(X)
        s = np.zeros(n_samples)
        for i in range(n_samples):
            same = labels == labels[i]
            a = np.mean(dist_mat[i, same])
            b = np.inf
            for c in np.unique(labels):
                if c == labels[i]: continue
                other = labels == c
                avg = np.mean(dist_mat[i, other])
                if avg < b: b = avg
            s[i] = (b - a) / max(a, b)
        return np.mean(s)

    def calinski_harabasz_score(self, X, labels):
        n_samples = len(X)
        n_clusters = len(np.unique(labels))
        if n_clusters < 2: return 0.0
        g_mean = np.mean(X, axis=0)
        w_var = 0.0
        b_var = 0.0
        for c in np.unique(labels):
            clu = X[labels == c]
            c_mean = np.mean(clu, axis=0)
            w_var += np.sum((clu - c_mean) ** 2)
            b_var += len(clu) * np.sum((c_mean - g_mean) ** 2)
        return (b_var / (n_clusters - 1)) / (w_var / (n_samples - n_clusters))

    def davies_bouldin_score(self, X, labels):
        n_clusters = len(np.unique(labels))
        if n_clusters < 2: return 0.0
        centers = []
        avg_d = []
        for c in np.unique(labels):
            clu = X[labels == c]
            ct = np.mean(clu, axis=0)
            centers.append(ct)
            avg_d.append(np.mean(euclidean_dist(clu, ct.reshape(1, -1))))
        centers = np.array(centers)
        avg_d = np.array(avg_d)
        db = 0.0
        for i in range(n_clusters):
            mx = 0.0
            for j in range(n_clusters):
                if i == j: continue
                r = (avg_d[i] + avg_d[j]) / euclidean_dist(centers[i].reshape(1, -1), centers[j].reshape(1, -1))[0][0]
                if r > mx: mx = r
            db += mx
        return db / n_clusters

    def adjusted_rand_score(self, y_true, y_pred):
        # 计算C(n, 2)的辅助函数
        def comb2(n):
            return n * (n - 1) / 2 if n >= 2 else 0
        
        yt = np.array(y_true)
        yp = np.array(y_pred)
        lbs = np.unique(np.concatenate([yt, yp]))
        n = len(yt)
        cm = np.zeros((len(lbs), len(lbs)), dtype=int)
        d = {l: i for i, l in enumerate(lbs)}
        for t, p in zip(yt, yp):
            cm[d[t], d[p]] += 1
        sc = sum(comb2(x) for x in np.sum(cm, axis=1))
        sk = sum(comb2(x) for x in np.sum(cm, axis=0))
        sn = sum(comb2(x) for x in cm.ravel())
        ex = sc * sk / comb2(n) if comb2(n) != 0 else 0
        mx = (sc + sk) / 2
        return (sn - ex) / (mx - ex) if mx != ex else 0

    def confusion_matrix(self, y_true, y_pred):
        """计算混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            dict: 包含混淆矩阵、类别标签和矩阵维度的字典
        """
        y_true = np.array(y_true, dtype=np.int64)
        y_pred = np.array(y_pred, dtype=np.int64)
        
        # 维度校验
        if y_true.shape != y_pred.shape:
            return self._response(0, content="y_true和y_pred的形状必须完全一致")
        
        # 单因变量转一维，统一处理逻辑
        if len(y_true.shape) == 2 and y_true.shape[1] == 1:
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)
        
        # 获取所有唯一类别
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        # 创建类别到索引的映射
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        # 初始化混淆矩阵
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        # 填充混淆矩阵
        for t, p in zip(y_true, y_pred):
            if t in class_to_idx and p in class_to_idx:
                cm[class_to_idx[t], class_to_idx[p]] += 1
        
        return self._response(1, content={
            "confusion_matrix": cm.tolist(),
            "classes": classes.tolist(),
            "shape": cm.shape
        })

# todo-2.2 build scikit-learn data divide
class Divide(SkLearn):
    def __init__(self):
        super(Divide, self).__init__()

    def leave_aside(self, data, columns_x=None, columns_y=None, class_col=None, to_dataframe=True):
        class_cols, class_vals, data_np = self._preprocess_class_param(data, class_col)
        if isinstance(class_cols, str) and class_cols == "error":
            return self._response(0, class_vals)

        n_samples = len(data_np)
        if n_samples < 2:
            return self._response(0, "The number of samples must be at least 2")

        x_indices, y_indices = self._get_column_indices(data, columns_x, columns_y)
        splits = []

        if class_cols is None:
            for i in range(n_samples):
                train_np = np.delete(data_np, i, axis=0)
                test_np = data_np[i:i + 1]
                train_x, train_y = self._split_data(train_np, data, x_indices, y_indices, columns_x, columns_y,
                                                    to_dataframe)
                test_x, test_y = self._split_data(test_np, data, x_indices, y_indices, columns_x, columns_y,
                                                  to_dataframe)
                splits.append({'train': {'x': train_x, 'y': train_y}, 'test': {'x': test_x, 'y': test_y}})
        else:
            class_comb = [tuple(row) if len(class_cols) > 1 else row[0] for row in class_vals]
            class_comb_unique = list(set(class_comb))
            for comb in class_comb_unique:
                comb_indices = [i for i, c in enumerate(class_comb) if c == comb]
                for idx in comb_indices:
                    train_np = np.delete(data_np, idx, axis=0)
                    test_np = data_np[idx:idx + 1]
                    train_x, train_y = self._split_data(train_np, data, x_indices, y_indices, columns_x, columns_y,
                                                        to_dataframe)
                    test_x, test_y = self._split_data(test_np, data, x_indices, y_indices, columns_x, columns_y,
                                                      to_dataframe)
                    splits.append({'train': {'x': train_x, 'y': train_y}, 'test': {'x': test_x, 'y': test_y}})
        return self._response(1, splits)

    def cross_validation(self, data, k, columns_x=None, columns_y=None, class_col=None, to_dataframe=True):
        if k == 1:
            return self._response(0, "K == 1 is not allowed")

        class_cols, class_vals, data_np = self._preprocess_class_param(data, class_col)
        if isinstance(class_cols, str) and class_cols == "error":
            return self._response(0, class_vals)

        n_samples = len(data_np)
        if k < 2 or k > n_samples:
            return self._response(0, f"k must satisfy 2 <= k <= {n_samples} (current number of samples)")

        x_indices, y_indices = self._get_column_indices(data, columns_x, columns_y)
        splits = []

        if class_cols is None:
            shuffled_data = np.random.permutation(data_np)
            fold_size = n_samples // k
            for i in range(k):
                start = i * fold_size
                end = start + fold_size if i < k - 1 else n_samples
                test_np = shuffled_data[start:end]
                train_np = np.concatenate([shuffled_data[:start], shuffled_data[end:]])
                train_x, train_y = self._split_data(train_np, data, x_indices, y_indices, columns_x, columns_y,
                                                    to_dataframe)
                test_x, test_y = self._split_data(test_np, data, x_indices, y_indices, columns_x, columns_y,
                                                  to_dataframe)
                splits.append({'train': {'x': train_x, 'y': train_y}, 'test': {'x': test_x, 'y': test_y}})
        else:
            class_comb = [tuple(row) if len(class_cols) > 1 else row[0] for row in class_vals]
            class_comb_counter = Counter(class_comb)
            for comb, cnt in class_comb_counter.items():
                if cnt < k:
                    return self._response(0,
                                          f"Class combination {comb} has only {cnt} samples, less than k={k}, stratified k-fold is not possible")

            all_test_indices = [[] for _ in range(k)]
            for comb in class_comb_counter.keys():
                comb_indices = np.array([i for i, c in enumerate(class_comb) if c == comb])
                np.random.shuffle(comb_indices)
                comb_folds = np.array_split(comb_indices, k)
                for i in range(k):
                    all_test_indices[i].extend(comb_folds[i].tolist())

            for test_indices in all_test_indices:
                test_indices = np.array(test_indices, dtype=int)
                train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
                train_np = data_np[train_indices]
                test_np = data_np[test_indices]
                train_x, train_y = self._split_data(train_np, data, x_indices, y_indices, columns_x, columns_y,
                                                    to_dataframe)
                test_x, test_y = self._split_data(test_np, data, x_indices, y_indices, columns_x, columns_y,
                                                  to_dataframe)
                splits.append({'train': {'x': train_x, 'y': train_y}, 'test': {'x': test_x, 'y': test_y}})
        return self._response(1, splits)

    def self_help(self, data, columns_x=None, columns_y=None, class_col=None, n_times=100, random_state=None,
                  to_dataframe=True):
        if random_state is not None:
            np.random.seed(random_state)

        class_cols, class_vals, data_np = self._preprocess_class_param(data, class_col)
        if isinstance(class_cols, str) and class_cols == "error":
            return self._response(0, class_vals)

        n_samples = len(data_np)
        x_indices, y_indices = self._get_column_indices(data, columns_x, columns_y)
        splits = []

        if class_cols is None:
            for _ in range(n_times):
                train_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                train_np = data_np[train_indices]
                all_indices = np.arange(n_samples)
                test_indices = np.setdiff1d(all_indices, train_indices)
                if len(test_indices) == 0:
                    test_indices = [n_samples - 1]
                test_np = data_np[test_indices]
                train_x, train_y = self._split_data(train_np, data, x_indices, y_indices, columns_x, columns_y,
                                                    to_dataframe)
                test_x, test_y = self._split_data(test_np, data, x_indices, y_indices, columns_x, columns_y,
                                                  to_dataframe)
                splits.append({'train': {'x': train_x, 'y': train_y}, 'test': {'x': test_x, 'y': test_y}})
        else:
            class_comb = [tuple(row) if len(class_cols) > 1 else row[0] for row in class_vals]
            class_comb_counter = Counter(class_comb)
            total_cnt = sum(class_comb_counter.values())
            class_ratios = {comb: cnt / total_cnt for comb, cnt in class_comb_counter.items()}

            for _ in range(n_times):
                train_indices = []
                for comb, ratio in class_ratios.items():
                    comb_indices = np.array([i for i, c in enumerate(class_comb) if c == comb])
                    comb_sample_cnt = int(round(ratio * n_samples))
                    if comb_sample_cnt == 0:
                        comb_sample_cnt = 1
                    comb_train_indices = np.random.choice(comb_indices, size=comb_sample_cnt, replace=True)
                    train_indices.extend(comb_train_indices)

                train_indices = np.array(train_indices)
                train_np = data_np[train_indices]
                all_indices = np.arange(n_samples)
                test_indices = np.setdiff1d(all_indices, train_indices)
                if len(test_indices) == 0:
                    test_indices = [n_samples - 1]
                test_np = data_np[test_indices]
                train_x, train_y = self._split_data(train_np, data, x_indices, y_indices, columns_x, columns_y,
                                                    to_dataframe)
                test_x, test_y = self._split_data(test_np, data, x_indices, y_indices, columns_x, columns_y,
                                                  to_dataframe)
                splits.append({'train': {'x': train_x, 'y': train_y}, 'test': {'x': test_x, 'y': test_y}})
        return self._response(1, splits)

    def train_test_split(self, data, test_size=0.2, random_state=None, columns_x=None, columns_y=None, class_col=None,
                         to_dataframe=True):
        if random_state is not None:
            np.random.seed(random_state)

        if not (0 < test_size < 1):
            return self._response(0, "test_size must satisfy 0 < test_size < 1")
        class_cols, class_vals, data_np = self._preprocess_class_param(data, class_col)
        if isinstance(class_cols, str) and class_cols == "error":
            return self._response(0, class_vals)
        # print('a')
        n_samples = len(data_np)
        test_n = int(round(n_samples * test_size))
        # print('b')
        if test_n < 1 or test_n >= n_samples:
            return self._response(0,
                                  f"test_size={test_size} is invalid, current samples: {n_samples}, test set needs at least 1 and less than total samples")
        # print('c')
        x_indices, y_indices = self._get_column_indices(data, columns_x, columns_y)
        # print('d')
        if class_cols is None:
            all_indices = np.arange(n_samples)
            np.random.shuffle(all_indices)
            test_indices = all_indices[:test_n]
            train_indices = all_indices[test_n:]
        else:
            class_comb = [tuple(row) if len(class_cols) > 1 else row[0] for row in class_vals]
            class_comb_counter = Counter(class_comb)
            train_indices = []
            test_indices = []
            for comb, cnt in class_comb_counter.items():
                comb_indices = np.array([i for i, c in enumerate(class_comb) if c == comb])
                np.random.shuffle(comb_indices)
                comb_test_n = int(round(cnt * test_size))
                comb_test_n = max(1, comb_test_n)
                comb_test_n = min(comb_test_n, cnt - 1)

                comb_test_indices = comb_indices[:comb_test_n]
                comb_train_indices = comb_indices[comb_test_n:]

                test_indices.extend(comb_test_indices)
                train_indices.extend(comb_train_indices)

            if len(test_indices) > test_n:
                test_indices = test_indices[:test_n]
            elif len(test_indices) < test_n:
                remaining_indices = np.setdiff1d(np.arange(n_samples), test_indices)
                add_indices = np.random.choice(remaining_indices, size=test_n - len(test_indices), replace=False)
                test_indices.extend(add_indices)
                train_indices = np.setdiff1d(train_indices, add_indices)

        train_np = data_np[train_indices]
        test_np = data_np[test_indices]
        print(0)
        train_x, train_y = self._split_data(train_np, data, x_indices, y_indices, columns_x, columns_y, to_dataframe)
        test_x, test_y = self._split_data(test_np, data, x_indices, y_indices, columns_x, columns_y, to_dataframe)

        result = {'train': {'x': train_x, 'y': train_y}, 'test': {'x': test_x, 'y': test_y}}
        return self._response(1, result)

    def _preprocess_class_param(self, data, class_col):
        class_cols = None
        if class_col is not None:
            if isinstance(class_col, str):
                class_cols = [class_col]
            elif isinstance(class_col, (list, tuple)):
                class_cols = list(class_col)
            else:
                return "error", "class_col only supports str/list/tuple format", None

        # 统一转换为NumPy数组
        if isinstance(data, pd.DataFrame):
            data_np = data.values
        elif isinstance(data, np.ndarray):
            data_np = data
        else:
            return "error", "data must be pandas.DataFrame or numpy.ndarray", None

        # 无分层列的情况
        if class_cols is None:
            return None, None, data_np

        # 有分层列：校验并提取类别值
        class_vals = None
        if isinstance(data, pd.DataFrame):
            # DataFrame：校验列名
            missing_cols = [col for col in class_cols if col not in data.columns]
            if missing_cols:
                return "error", f"class columns {missing_cols} do not exist in the dataset", data_np
            class_vals = data[class_cols].values
        else:
            # NumPy数组：校验索引
            non_int_cols = [col for col in class_cols if not isinstance(col, int)]
            if non_int_cols:
                return "error", "NumPy array only supports integer indices for class columns", data_np
            out_range_cols = [col for col in class_cols if col < 0 or col >= data_np.shape[1]]
            if out_range_cols:
                return "error", f"class indices {out_range_cols} are out of the dataset column range", data_np
            class_vals = data_np[:, class_cols]

        # 移除“无重复类别报错”逻辑（兼容回归数据）
        return class_cols, class_vals, data_np

    def _get_column_indices(self, data, columns_x, columns_y):
        x_indices = None
        # 处理特征列索引（支持多列）
        if columns_x is not None and isinstance(data, pd.DataFrame):
            if isinstance(columns_x, str):
                x_indices = [data.columns.get_loc(columns_x)]
            elif isinstance(columns_x, (list, tuple)):
                # columns_x = columns_x[0]
                # print(type(columns_x))
                x_indices = [data.columns.get_loc(col) for col in columns_x]
                # print(x_indices)

        y_indices = None
        # 处理标签列索引（支持多列）
        if columns_y is not None and isinstance(data, pd.DataFrame):
            if isinstance(columns_y, str):
                y_indices = [data.columns.get_loc(columns_y)]
            elif isinstance(columns_y, (list, tuple)):
                y_indices = [data.columns.get_loc(col) for col in columns_y]
        return x_indices, y_indices

    def _split_data(self, data_np, original_data, x_indices, y_indices, columns_x, columns_y, to_dataframe):
        """修复核心：保留多列标签的二维维度，不展平"""
        # 返回DataFrame格式
        # print(1)
        if to_dataframe and isinstance(original_data, pd.DataFrame):
            df = pd.DataFrame(data_np, columns=original_data.columns)
            x = df[columns_x] if columns_x is not None and len(columns_x) > 0 else df
            y = df[columns_y] if columns_y is not None and len(columns_y) > 0 else None
            # print(2)
        # 返回NumPy数组（核心修复：保留多列维度）
        else:
            # print(3)
            # 特征：x_indices为列表且非空时，返回(n_samples, n_features)，否则返回(n_samples, n_cols)
            x = data_np[:, x_indices] if x_indices is not None and len(x_indices) > 0 else data_np
            # 标签：y_indices为列表且非空时，返回(n_samples, n_outputs)，否则返回(n_samples,)或None
            if y_indices is not None and len(y_indices) > 0:
                y = data_np[:, y_indices]
                # 单标签时保持一维，多标签时保持二维（避免展平）
                if len(y.shape) == 2 and y.shape[1] == 1:
                    y = y.reshape(-1)
                # print(4)
            else:
                y = None
        # print(5)
        return x, y

# todo-2.3 build scikit-learn LinearRegression
class LinearRegression(Evaluation):
    def __init__(
        self,
        fit_intercept=True,
        solver="normal_eq",
        alpha=0.0,
        max_iter=1000,
        tol=1e-3,
        learning_rate=0.01
    ):
        super().__init__()
        self.weights = None
        self.intercept = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self._n_features_in = None

    def _add_bias_term(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            return self._response(
                status=0,
                content="Input error: X and y must have the same number of samples"
            )

        self._n_features_in = X.shape[1] if len(X.shape) > 1 else 1
        X_with_bias = self._add_bias_term(X)

        try:
            if self.solver == "normal_eq":
                X_T = X_with_bias.T
                n_features = X_with_bias.shape[1]
                I = np.eye(n_features)
                if self.fit_intercept:
                    I[0, 0] = 0
                theta = np.linalg.inv(X_T @ X_with_bias + self.alpha * I) @ X_T @ y
            elif self.solver == "sgd":
                theta = np.zeros((X_with_bias.shape[1], 1))
                loss_history = []
                for i in range(self.max_iter):
                    idx = np.random.randint(X_with_bias.shape[0])
                    X_i = X_with_bias[idx:idx+1]
                    y_i = y[idx:idx+1]
                    y_pred = X_i @ theta
                    error = y_pred - y_i
                    grad = X_i.T @ error + self.alpha * theta
                    if self.fit_intercept:
                        grad[0] -= self.alpha * theta[0]
                    theta -= self.learning_rate * grad
                    loss = np.mean((X_with_bias @ theta - y) ** 2) + self.alpha * np.sum(theta[1:] ** 2)
                    loss_history.append(loss)
                    if i > 0 and abs(loss_history[-2] - loss_history[-1]) < self.tol:
                        break
            else:
                return self._response(
                    status=0,
                    content=f"不支持的solver: {self.solver}，可选'normal_eq'或'sgd'"
                )

            if self.fit_intercept:
                self.intercept = theta[0, 0]
                self.weights = theta[1:, 0]
            else:
                self.intercept = 0.0
                self.weights = theta[:, 0]

            return self._response(
                status=1,
                content={
                    "weights": self.weights,
                    "bias": self.intercept,
                    "n_features": len(self.weights)
                }
            )
        except np.linalg.LinAlgError:
            return self._response(
                status=0,
                content="Singular matrix error (collinear features), cannot train model"
            )

    def predict(self, X):
        if self.weights is None or self.intercept is None:
            return self._response(
                status=0,
                content="Model not trained yet - call fit() first"
            )

        X = np.array(X, dtype=np.float64)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._n_features_in:
            return self._response(
                status=0,
                content=f"Input error: X must have {self._n_features_in} features"
            )

        y_pred = X @ self.weights + self.intercept

        return self._response(
            status=1,
            content=y_pred
        )

    def evaluate(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.array(y_pred, dtype=np.float64).reshape(-1)

        if len(y_true) != len(y_pred):
            return self._response(
                status=0,
                content="Input error: y_true and y_pred must have the same length"
            )

        mse_value = self.mse(y_pred, y_true)['content']['mse']
        r2_value = self.r2(y_true, y_pred)['content']['r2']
        bias_value = self.bias(y_true, y_pred)['content']['bias']
        variance_value = self.variance(y_pred)['content']['variance']
        mase_value = self.mase(y_pred,y_true)['content']['mase']

        return self._response(content={'mse':mse_value,'r2':r2_value,'bias':bias_value,'variance':variance_value,'mase':mase_value})

# todo-2.4 build scikit-learn Ridge
class Ridge(Evaluation):
    def __init__(
            self,
            alpha=1.0,
            fit_intercept=True,
            solver="cholesky",
            max_iter=1000,
            tol=1e-3,
            learning_rate=0.01,
            copy_X=True,
            random_state=None
    ):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.copy_X = copy_X
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = None
        self._n_features_in = None

    def _add_bias_term(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y):
        if self.copy_X:
            X = np.array(X, dtype=np.float64).copy()
            y = np.array(y, dtype=np.float64).copy()
        else:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.float64)

        y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            return self._response(
                status=0,
                content="Input error: X and y must have the same number of samples"
            )

        self._n_features_in = X.shape[1] if len(X.shape) > 1 else 1
        X_with_bias = self._add_bias_term(X)
        n_features = X_with_bias.shape[1]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        try:
            if self.solver == "cholesky":
                reg_matrix = self.alpha * np.eye(n_features)
                if self.fit_intercept:
                    reg_matrix[0, 0] = 0
                X_T = X_with_bias.T
                theta, _, _, _ = np.linalg.lstsq(X_T @ X_with_bias + reg_matrix, X_T @ y, rcond=None)
            elif self.solver == "sgd":
                theta = np.zeros((n_features, 1))
                loss_history = []
                for i in range(self.max_iter):
                    idx = np.random.randint(X_with_bias.shape[0])
                    X_i = X_with_bias[idx:idx + 1]
                    y_i = y[idx:idx + 1]
                    y_pred = X_i @ theta
                    error = y_pred - y_i
                    grad = X_i.T @ error + self.alpha * theta
                    if self.fit_intercept:
                        grad[0] -= self.alpha * theta[0]
                    theta -= self.learning_rate * grad
                    loss = np.mean((X_with_bias @ theta - y) ** 2) + self.alpha * np.sum(theta[1:] ** 2)
                    loss_history.append(loss)
                    if i > 0 and abs(loss_history[-2] - loss_history[-1]) < self.tol:
                        break
            else:
                return self._response(
                    status=0,
                    content=f"不支持的solver: {self.solver}，可选'cholesky'或'sgd'"
                )

            if self.fit_intercept:
                self.intercept_ = theta[0, 0]
                self.coef_ = theta[1:, 0]
            else:
                self.intercept_ = 0.0
                self.coef_ = theta[:, 0]

            return self._response(
                status=1,
                content={
                    "coef_": self.coef_,
                    "intercept_": self.intercept_,
                    "alpha": self.alpha,
                    "n_features_in": self._n_features_in
                }
            )
        except np.linalg.LinAlgError:
            return self._response(
                status=0,
                content="Singular matrix error (collinear features), cannot train model"
            )

    def predict(self, X):
        if self.coef_ is None or self.intercept_ is None:
            return self._response(
                status=0,
                content="Model not trained yet - call fit() first"
            )

        X = np.array(X, dtype=np.float64)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self._n_features_in:
            return self._response(
                status=0,
                content=f"Input error: X must have {self._n_features_in} features"
            )

        y_pred = X @ self.coef_ + self.intercept_
        return self._response(
            status=1,
            content=y_pred
        )

    def score(self, X, y):
        pred_res = self.predict(X)
        if pred_res['status'] == 0:
            return pred_res

        y_pred = pred_res['content']
        y_true = np.array(y, dtype=np.float64).reshape(-1)

        return self.r2(y_true, y_pred)

    def evaluate(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.array(y_pred, dtype=np.float64).reshape(-1)

        if len(y_true) != len(y_pred):
            return self._response(
                status=0,
                content="Input error: y_true and y_pred must have the same length"
            )

        mse_res = self.mse(y_pred, y_true)
        r2_res = self.r2(y_true, y_pred)
        bias_res = self.bias(y_true, y_pred)
        variance_res = self.variance(y_pred)
        mase_res = self.mase(y_pred, y_true)

        content = {
            "mse": mse_res['content']['mse'],
            "r2": r2_res['content']['r2'],
            "bias": bias_res['content']['bias'],
            "variance": variance_res['content']['variance'],
            "mase": mase_res['content']['mase']
        }
        return self._response(1, content)

# todo-2.5 build scikit-learn LogisticRegression
class LogisticRegression(Evaluation):
    def __init__(
            self,
            penalty='l2',
            C=1.0,
            fit_intercept=True,
            solver='sgd',
            max_iter=1000,
            tol=1e-4,
            random_state=None,
            copy_X=True,
            l1_ratio=None
    ):
        super().__init__()
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.copy_X = copy_X
        self.l1_ratio = l1_ratio

        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.loss_history = []
        self.n_outputs = None

    def _add_bias_term(self, X):
        X = np.array(X, dtype=np.float64)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def _sigmoid(self, z):
        return expit(z)

    def _compute_loss(self, y_true, y_prob):
        epsilon = 1e-10
        loss_per_output = -np.mean(y_true * np.log(y_prob + epsilon) + (1 - y_true) * np.log(1 - y_prob + epsilon),
                                   axis=0)
        total_loss = np.mean(loss_per_output)

        # 添加正则化损失
        if self.penalty == 'l2':
            reg_loss = 0.5 * (1 / self.C) * np.sum(self.coef_ ** 2)
        elif self.penalty == 'l1':
            reg_loss = (1 / self.C) * np.sum(np.abs(self.coef_))
        elif self.penalty == 'elasticnet':
            l1_reg = (1 / self.C) * self.l1_ratio * np.sum(np.abs(self.coef_))
            l2_reg = 0.5 * (1 / self.C) * (1 - self.l1_ratio) * np.sum(self.coef_ ** 2)
            reg_loss = l1_reg + l2_reg
        else:
            reg_loss = 0.0

        return total_loss + reg_loss

    def fit(self, X, y):
        if self.copy_X:
            X = np.array(X, dtype=np.float64).copy()
            y = np.array(y, dtype=np.float64).copy()
        else:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.float64)

        if len(y.shape) == 1:
            self.n_outputs = 1
            y = y.reshape(-1, 1)
        else:
            self.n_outputs = y.shape[1]

        self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
        X_with_bias = self._add_bias_term(X)
        n_samples, n_features = X_with_bias.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 初始化权重和截距
        self.coef_ = np.zeros((n_features - (1 if self.fit_intercept else 0), self.n_outputs))
        self.intercept_ = np.zeros(self.n_outputs) if self.fit_intercept else None

        self.loss_history = []
        for i in range(self.max_iter):
            if self.fit_intercept:
                z = X_with_bias @ np.vstack([self.intercept_.reshape(1, -1), self.coef_])
            else:
                z = X_with_bias @ self.coef_
            y_prob = self._sigmoid(z)

            # 计算梯度
            gradient = (1 / n_samples) * X_with_bias.T @ (y_prob - y)

            # 添加正则化梯度
            if self.penalty == 'l2':
                reg_grad = (1 / self.C) * np.vstack([np.zeros((1, self.n_outputs)), self.coef_])
            elif self.penalty == 'l1':
                reg_grad = (1 / self.C) * np.vstack([np.zeros((1, self.n_outputs)), np.sign(self.coef_)])
            elif self.penalty == 'elasticnet':
                l1_grad = (1 / self.C) * self.l1_ratio * np.vstack([np.zeros((1, self.n_outputs)), np.sign(self.coef_)])
                l2_grad = (1 / self.C) * (1 - self.l1_ratio) * np.vstack([np.zeros((1, self.n_outputs)), self.coef_])
                reg_grad = l1_grad + l2_grad
            else:
                reg_grad = 0.0

            gradient += reg_grad

            # 更新参数
            if self.fit_intercept:
                self.intercept_ -= self.tol * gradient[0, :]
                self.coef_ -= self.tol * gradient[1:, :]
            else:
                self.coef_ -= self.tol * gradient

            # 记录损失
            if i % 10 == 0:
                loss = self._compute_loss(y, y_prob)
                self.loss_history.append(loss)
                # 收敛判断
                if len(self.loss_history) > 1 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                    break

        return self._response(
            status=1,
            content={
                "coef_": self.coef_ if self.n_outputs > 1 else self.coef_.reshape(-1),
                "intercept_": self.intercept_ if self.n_outputs > 1 else self.intercept_[
                    0] if self.fit_intercept else None,
                "n_features_in_": self.n_features_in_,
                "n_outputs": self.n_outputs,
                "final_loss": self.loss_history[-1] if self.loss_history else None,
                "max_iter": self.max_iter,
                "penalty": self.penalty,
                "C": self.C,
                "fit_intercept": self.fit_intercept,
                "solver": self.solver
            }
        )

    def predict_proba(self, X):
        if self.coef_ is None or self.intercept_ is None:
            return self._response(
                status=0,
                content="Model not trained yet - call fit() first"
            )
        X = np.array(X, dtype=np.float64)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X_with_bias = self._add_bias_term(X)

        if self.fit_intercept:
            z = X_with_bias @ np.vstack([self.intercept_.reshape(1, -1), self.coef_])
        else:
            z = X_with_bias @ self.coef_
        y_prob = self._sigmoid(z)

        y_prob_reshaped = y_prob if self.n_outputs > 1 else y_prob.reshape(-1)
        return self._response(
            status=1,
            content=y_prob_reshaped
        )

    def predict(self, X, threshold=0.5):
        proba_response = self.predict_proba(X)
        if proba_response["status"] == 0:
            return proba_response
        y_prob = proba_response["content"]
        y_pred = (y_prob >= threshold).astype(int)
        return self._response(
            status=1,
            content=y_pred
        )

    def evaluate(self, X, y_true, y_pred, threshold=0.5):
        y_true = np.array(y_true, dtype=np.int64)
        y_pred = np.array(y_pred, dtype=np.float64)

        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        if y_true.shape[0] != y_pred.shape[0]:
            return self._response(status=0, content="y_true和y_pred样本数必须一致")
        if y_true.shape[1] != y_pred.shape[1]:
            return self._response(status=0, content="y_true和y_pred因变量数必须一致")
        n_outputs = y_true.shape[1]

        if np.all((y_pred >= 0) & (y_pred <= 1)):
            y_pred = (y_pred >= threshold).astype(int)

        y_prob_resp = self.predict_proba(X)
        if y_prob_resp['status'] != 1:
            return y_prob_resp
        y_prob = y_prob_resp['content']
        if len(y_prob.shape) == 1:
            y_prob = y_prob.reshape(-1, 1)

        metrics_per_output = {}
        avg_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}

        for i in range(n_outputs):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            y_pba = y_prob[:, i] if (len(y_prob.shape) >= 2 and y_prob.shape[1] >= i + 1) else np.array([])

            f1_resp = self.f1(y_t, y_p)
            prec_resp = self.precision(y_t, y_p)
            rec_resp = self.recall(y_t, y_p)
            f1 = f1_resp['content']['f1'] if f1_resp['status'] == 1 else 0.0
            prec = prec_resp['content']['precision'] if prec_resp['status'] == 1 else 0.0
            rec = rec_resp['content']['recall'] if rec_resp['status'] == 1 else 0.0
            acc = np.mean(y_t == y_p)

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_t, y_p)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            roc_resp = self.roc(y_t, y_pba)
            roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
            auc = 0.0
            if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                    auc_val = auc_resp['content']['auc']
                    if not np.isnan(auc_val) and not np.isinf(auc_val):
                        auc = auc_val

            metrics_per_output[f"output_{i + 1}"] = {
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                "confusion_matrix": cm, "classes": classes
            }

            avg_metrics["auc"] += auc
            avg_metrics["accuracy"] += acc
            avg_metrics["precision"] += prec
            avg_metrics["recall"] += rec
            avg_metrics["f1"] += f1

        avg_metrics = {k: v / n_outputs for k, v in avg_metrics.items()}

        return self._response(
            status=1,
            content={
                "metrics_per_output": metrics_per_output,
                "average_metrics": avg_metrics,
                "n_outputs": n_outputs
            }
        )

# todo-2.6 build scikit-learn LDA
class LDA(Evaluation):
    def __init__(
            self,
            n_components=None,
            solver='svd',
            shrinkage=None,
            priors=None,
            store_covariance=False,
            tol=1e-4,
            covariance_estimator=None
    ):
        super().__init__()
        self.n_components = n_components
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.store_covariance = store_covariance
        self.tol = tol
        self.covariance_estimator = covariance_estimator

        self.coef_ = None
        self.means_ = None
        self.priors_ = None
        self.classes_ = None
        self.X_mean_ = None
        self.covariance_ = None
        self.n_features_in_ = None
        self.n_classes_ = None

    def _shrink_cov(self, cov_matrix):
        if self.shrinkage is None or self.shrinkage == 0:
            return cov_matrix

        shrinkage = np.clip(self.shrinkage, 1e-6, 1.0)
        diag_cov = np.diag(np.diag(cov_matrix))
        shrunk_cov = (1 - shrinkage) * cov_matrix + shrinkage * diag_cov
        return shrunk_cov

    def fit(self, X, y):
        # 转换为numpy数组
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        # 确保y是一维数组
        if y.ndim > 1:
            y = y.ravel()

        # 检查输入维度
        if X.ndim != 2:
            return self._response(0, content="X必须是二维数组")
        if len(y) != X.shape[0]:
            return self._response(0, content=f"y的长度与X的样本数不一致: y={len(y)}, X={X.shape[0]}")

        # 检查solver
        if self.solver not in ['svd', 'eigen']:
            return self._response(0, content=f"不支持的solver: {self.solver}，可选'svd'或'eigen'")

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # 获取类别
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # 计算n_components
        if self.n_components is None:
            self.n_components = min(self.n_classes_ - 1, n_features)
        else:
            if self.n_components > self.n_classes_ - 1 or self.n_components > n_features:
                return self._response(
                    0,
                    content=f"n_components 最大为 {min(self.n_classes_ - 1, n_features)}，当前输入 {self.n_components}"
                )

        # 计算先验概率
        if self.priors is None:
            self.priors_ = np.zeros(self.n_classes_)
            for i, c in enumerate(self.classes_):
                count = 0
                for j in range(n_samples):
                    if y[j] == c:
                        count += 1
                self.priors_[i] = count / n_samples
        else:
            self.priors_ = np.array(self.priors)
            if len(self.priors_) != self.n_classes_:
                return self._response(0, content="priors长度必须与类别数一致")

        # 计算每个类别的均值
        self.means_ = np.zeros((self.n_classes_, n_features))
        for i, c in enumerate(self.classes_):
            # 收集属于当前类别的样本
            class_samples = []
            for j in range(n_samples):
                if y[j] == c:
                    class_samples.append(X[j])
            if not class_samples:
                return self._response(0, content=f"类别 {c} 没有样本")
            class_samples = np.array(class_samples)
            self.means_[i] = np.mean(class_samples, axis=0)

        # 计算全局均值
        self.X_mean_ = np.mean(X, axis=0)

        # 计算类内散布矩阵Sw
        Sw = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            # 收集属于当前类别的样本
            class_samples = []
            for j in range(n_samples):
                if y[j] == c:
                    class_samples.append(X[j])
            class_samples = np.array(class_samples)
            if len(class_samples) > 1:
                # 计算中心化样本
                centered_samples = class_samples - self.means_[i]
                # 计算协方差矩阵
                cov = np.dot(centered_samples.T, centered_samples) / len(class_samples)
                cov_shrunk = self._shrink_cov(cov)
                Sw += len(class_samples) * cov_shrunk

        # 计算类间散布矩阵Sb
        Sb = np.zeros((n_features, n_features))
        for i, c in enumerate(self.classes_):
            # 计算类别计数
            n_c = 0
            for j in range(n_samples):
                if y[j] == c:
                    n_c += 1
            # 计算均值差异
            mean_diff = self.means_[i] - self.X_mean_
            Sb += n_c * np.outer(mean_diff, mean_diff)

        # 存储协方差矩阵
        if self.store_covariance:
            self.covariance_ = Sw / n_samples

        # 计算特征向量
        if self.solver == 'eigen':
            # 计算广义特征值问题
            try:
                eig_vals, eig_vecs = eigh(Sb, Sw)
                # 按特征值降序排序
                idx = np.argsort(eig_vals)[::-1]
                eig_vecs = eig_vecs[:, idx]
                # 选择前n_components个特征向量
                self.coef_ = eig_vecs[:, :self.n_components]
            except Exception as e:
                return self._response(0, content=f"Eigen solver failed: {str(e)}")
        elif self.solver == 'svd':
            # 使用SVD求解
            try:
                # 对Sw进行SVD分解
                U, S, Vt = np.linalg.svd(Sw, hermitian=True)
                # 计算Sw的逆平方根
                Sw_inv_sqrt = U @ np.diag(1 / np.sqrt(S + self.tol)) @ Vt
                # 投影Sb到Sw的特征空间
                Sb_proj = Sw_inv_sqrt @ Sb @ Sw_inv_sqrt
                # 计算投影后的特征值和特征向量
                eig_vals, eig_vecs = eigh(Sb_proj)
                # 按特征值降序排序
                idx = np.argsort(eig_vals)[::-1]
                eig_vecs = eig_vecs[:, idx]
                # 转换回原始空间
                self.coef_ = Sw_inv_sqrt @ eig_vecs[:, :self.n_components]
            except Exception as e:
                return self._response(0, content=f"SVD solver failed: {str(e)}")

        return self._response(
            1,
            content={
                "coef_": self.coef_,
                "n_components": self.n_components,
                "n_classes_": self.n_classes_,
                "classes_": self.classes_,
                "priors_": self.priors_,
                "shrinkage": self.shrinkage,
                "solver": self.solver
            }
        )

    def transform(self, X):
        if self.coef_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        if X.ndim != 2:
            return self._response(0, content="X必须是二维数组")

        X_centered = X - self.X_mean_
        X_lda = X_centered @ self.coef_

        return self._response(1, content=X_lda)

    def predict(self, X):
        transform_resp = self.transform(X)
        if transform_resp["status"] == 0:
            return transform_resp
        X_lda = transform_resp["content"]

        means_lda = (self.means_ - self.X_mean_) @ self.coef_

        y_pred = []
        for x in X_lda:
            distances = [np.linalg.norm(x - m) for m in means_lda]
            pred_class = self.classes_[np.argmin(distances)]
            y_pred.append(pred_class)

        y_pred = np.array(y_pred)
        return self._response(1, content=y_pred)

    def predict_proba(self, X):
        transform_resp = self.transform(X)
        if transform_resp["status"] == 0:
            return transform_resp
        X_lda = transform_resp["content"]

        means_lda = (self.means_ - self.X_mean_) @ self.coef_

        y_prob = []
        for x in X_lda:
            distances = [np.linalg.norm(x - m) for m in means_lda]
            # 转换为概率（使用指数函数归一化）
            exp_dist = np.exp(-np.array(distances))
            prob = exp_dist / np.sum(exp_dist)
            y_prob.append(prob)

        y_prob = np.array(y_prob)
        return self._response(1, content=y_prob)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        y_true = np.array(y_true, dtype=np.int64)

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.int64)

        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        if y_true.shape[0] != y_pred.shape[0]:
            return self._response(status=0, content="y_true和y_pred样本数必须一致")
        if y_true.shape[1] != y_pred.shape[1]:
            return self._response(status=0, content="y_true和y_pred因变量数必须一致")
        n_outputs = y_true.shape[1]

        y_prob_resp = self.predict_proba(X)
        if y_prob_resp['status'] != 1:
            return y_prob_resp
        y_prob = y_prob_resp['content']
        if len(y_prob.shape) == 1:
            y_prob = y_prob.reshape(-1, 1)

        metrics_per_output = {}
        avg_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}

        for i in range(n_outputs):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            y_pba = y_prob[:, i] if (len(y_prob.shape) >= 2 and y_prob.shape[1] >= i + 1) else np.array([])

            f1_resp = self.f1(y_t, y_p)
            prec_resp = self.precision(y_t, y_p)
            rec_resp = self.recall(y_t, y_p)
            f1 = f1_resp['content']['f1'] if f1_resp['status'] == 1 else 0.0
            prec = prec_resp['content']['precision'] if prec_resp['status'] == 1 else 0.0
            rec = rec_resp['content']['recall'] if rec_resp['status'] == 1 else 0.0
            acc = np.mean(y_t == y_p)

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_t, y_p)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            roc_resp = self.roc(y_t, y_pba)
            roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
            auc = 0.0
            if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                    auc_val = auc_resp['content']['auc']
                    if not np.isnan(auc_val) and not np.isinf(auc_val):
                        auc = auc_val

            metrics_per_output[f"output_{i + 1}"] = {
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                "confusion_matrix": cm, "classes": classes
            }

            avg_metrics["auc"] += auc
            avg_metrics["accuracy"] += acc
            avg_metrics["precision"] += prec
            avg_metrics["recall"] += rec
            avg_metrics["f1"] += f1

        avg_metrics = {k: v / n_outputs for k, v in avg_metrics.items()}

        return self._response(
            status=1,
            content={
                "metrics_per_output": metrics_per_output,
                "average_metrics": avg_metrics,
                "n_outputs": n_outputs
            }
        )

# todo-2.7 build scikit-learn DecisionTreeClassifier
class DecisionTreeClassifier(Evaluation):
    def __init__(
            self,
            criterion='gini',
            splitter='best',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0
    ):
        super().__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

        self.tree_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))

    def _log_loss(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log(prob))

    def _calc_impurity(self, y):
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'log_loss':
            return self._log_loss(y)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= self.min_samples_split:
            return None, None

        best_impurity = self._calc_impurity(y)
        best_feature, best_threshold = None, None

        feature_indices = range(n_features)
        if self.max_features is not None:
            rng = np.random.RandomState(self.random_state)
            feature_indices = rng.choice(n_features, self.max_features, replace=False)

        for idx in feature_indices:
            X_col = X[:, idx]
            thresholds = np.unique(X_col)
            if self.splitter == 'random':
                rng = np.random.RandomState(self.random_state)
                thresholds = rng.choice(thresholds, min(10, len(thresholds)), replace=False)

            for threshold in thresholds:
                left_mask = X_col <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
                    continue

                impurity_left = self._calc_impurity(y[left_mask])
                impurity_right = self._calc_impurity(y[right_mask])
                impurity = (len(y[left_mask]) * impurity_left + len(y[right_mask]) * impurity_right) / n_samples

                if impurity < best_impurity - self.min_impurity_decrease:
                    best_impurity = impurity
                    best_feature = idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        unique_classes, counts = np.unique(y, return_counts=True)
        most_common = unique_classes[np.argmax(counts)]

        node = {
            'leaf': True,
            'class': most_common,
            'depth': depth
        }

        if (self.max_depth is not None and depth >= self.max_depth) or len(unique_classes) == 1:
            return node

        if n_samples < self.min_samples_split:
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return node

        X_col = X[:, feature]
        left_mask = X_col <= threshold
        right_mask = ~left_mask

        node['leaf'] = False
        node['feature'] = feature
        node['threshold'] = threshold
        node['left'] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.int64)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self.tree_ = self._build_tree(X, y)
        self.feature_importances_ = np.zeros(self.n_features_in_)

        return self._response(
            1,
            content={
                "n_classes_": self.n_classes_,
                "classes_": self.classes_,
                "n_features_in_": self.n_features_in_,
                "max_depth": self.max_depth,
                "criterion": self.criterion
            }
        )

    def _predict_sample(self, x, node):
        # 确保 node 是字典且包含必要的键
        if not isinstance(node, dict):
            return self.classes_[0] if hasattr(self, 'classes_') and len(self.classes_) > 0 else 0
        
        # 检查必要的键是否存在
        if 'leaf' not in node:
            return self.classes_[0] if hasattr(self, 'classes_') and len(self.classes_) > 0 else 0
        
        if node['leaf']:
            return node.get('class', 0)
        
        # 确保特征索引是整数
        feature = node.get('feature', 0)
        if not isinstance(feature, int):
            return self.classes_[0] if hasattr(self, 'classes_') and len(self.classes_) > 0 else 0
        
        # 确保特征索引在有效范围内
        if feature >= len(x):
            return self.classes_[0] if hasattr(self, 'classes_') and len(self.classes_) > 0 else 0
        
        threshold = node.get('threshold', 0)
        if x[feature] <= threshold:
            left_node = node.get('left', {'leaf': True, 'class': 0})
            return self._predict_sample(x, left_node)
        else:
            right_node = node.get('right', {'leaf': True, 'class': 0})
            return self._predict_sample(x, right_node)

    def predict(self, X):
        if self.tree_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        y_pred = [self._predict_sample(x, self.tree_) for x in X]
        y_pred = np.array(y_pred)

        return self._response(1, content=y_pred)

    def predict_proba(self, X):
        if self.tree_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 对于决策树，简单实现概率预测（基于叶节点的类别分布）
        y_pred = []
        for x in X:
            # 找到样本所在的叶节点
            node = self.tree_
            while True:
                # 确保 node 是字典且包含必要的键
                if not isinstance(node, dict) or 'leaf' not in node:
                    break
                if node['leaf']:
                    break
                # 确保特征索引是整数且在有效范围内
                feature = node.get('feature', 0)
                if not isinstance(feature, int) or feature >= len(x):
                    break
                threshold = node.get('threshold', 0)
                if x[feature] <= threshold:
                    node = node.get('left', {'leaf': True, 'class': 0})
                else:
                    node = node.get('right', {'leaf': True, 'class': 0})
            # 对于分类树，我们简单返回类别分布（这里简化为one-hot编码）
            prob = np.zeros(len(self.classes_))
            if isinstance(node, dict) and 'class' in node:
                class_idx = np.where(self.classes_ == node['class'])[0]
                if len(class_idx) > 0:
                    prob[class_idx[0]] = 1.0
            y_pred.append(prob)
        y_pred = np.array(y_pred)

        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        y_true = np.array(y_true, dtype=np.int64)

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.int64)

        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)

        if y_true.shape[0] != y_pred.shape[0]:
            return self._response(status=0, content="y_true和y_pred样本数必须一致")
        if y_true.shape[1] != y_pred.shape[1]:
            return self._response(status=0, content="y_true和y_pred因变量数必须一致")
        n_outputs = y_true.shape[1]

        y_prob_resp = self.predict_proba(X)
        if y_prob_resp['status'] != 1:
            return y_prob_resp
        y_prob = y_prob_resp['content']
        if len(y_prob.shape) == 1:
            y_prob = y_prob.reshape(-1, 1)

        metrics_per_output = {}
        avg_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0}

        for i in range(n_outputs):
            y_t = y_true[:, i]
            y_p = y_pred[:, i]
            y_pba = y_prob[:, i] if (len(y_prob.shape) >= 2 and y_prob.shape[1] >= i + 1) else np.array([])

            f1_resp = self.f1(y_t, y_p)
            prec_resp = self.precision(y_t, y_p)
            rec_resp = self.recall(y_t, y_p)
            f1 = f1_resp['content']['f1'] if f1_resp['status'] == 1 else 0.0
            prec = prec_resp['content']['precision'] if prec_resp['status'] == 1 else 0.0
            rec = rec_resp['content']['recall'] if rec_resp['status'] == 1 else 0.0
            acc = np.mean(y_t == y_p)

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_t, y_p)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            roc_resp = self.roc(y_t, y_pba)
            roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
            auc = 0.0
            if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                    auc_val = auc_resp['content']['auc']
                    if not np.isnan(auc_val) and not np.isinf(auc_val):
                        auc = auc_val

            metrics_per_output[f"output_{i + 1}"] = {
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc,
                "confusion_matrix": cm, "classes": classes
            }

            avg_metrics["auc"] += auc
            avg_metrics["accuracy"] += acc
            avg_metrics["precision"] += prec
            avg_metrics["recall"] += rec
            avg_metrics["f1"] += f1

        avg_metrics = {k: v / n_outputs for k, v in avg_metrics.items()}

        return self._response(
            status=1,
            content={
                "metrics_per_output": metrics_per_output,
                "average_metrics": avg_metrics,
                "n_outputs": n_outputs
            }
        )

# todo-2.8 build scikit-learn DecisionTreeRegressor
class DecisionTreeRegressor(Evaluation):
    def __init__(
            self,
            criterion='mse',
            splitter='best',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            ccp_alpha=0.0
    ):
        super().__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha

        self.tree_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.n_outputs_ = 1

    def _mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def _mae(self, y):
        return np.mean(np.abs(y - np.mean(y)))

    def _friedman_mse(self, y):
        mse = self._mse(y)
        return mse - np.var(y) / len(y)

    def _poisson(self, y):
        y_mean = np.mean(y)
        return 2 * (np.sum(y * np.log(y_mean / y + 1e-10)) - np.sum(y - y_mean)) / len(y)

    def _calc_impurity(self, y):
        if self.criterion == 'mse':
            return self._mse(y)
        elif self.criterion == 'mae':
            return self._mae(y)
        elif self.criterion == 'friedman_mse':
            return self._friedman_mse(y)
        elif self.criterion == 'poisson':
            return self._poisson(y)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        if n_samples <= self.min_samples_split:
            return None, None

        best_impurity = self._calc_impurity(y)
        best_feature, best_threshold = None, None

        feature_indices = range(n_features)
        if self.max_features is not None:
            rng = np.random.RandomState(self.random_state)
            feature_indices = rng.choice(n_features, self.max_features, replace=False)

        for idx in feature_indices:
            X_col = X[:, idx]
            thresholds = np.unique(X_col)
            if self.splitter == 'random':
                rng = np.random.RandomState(self.random_state)
                thresholds = rng.choice(thresholds, min(10, len(thresholds)), replace=False)

            for threshold in thresholds:
                left_mask = X_col <= threshold
                right_mask = ~left_mask

                if len(y[left_mask]) < self.min_samples_leaf or len(y[right_mask]) < self.min_samples_leaf:
                    continue

                impurity_left = self._calc_impurity(y[left_mask])
                impurity_right = self._calc_impurity(y[right_mask])
                impurity = (len(y[left_mask]) * impurity_left + len(y[right_mask]) * impurity_right) / n_samples

                if impurity < best_impurity - self.min_impurity_decrease:
                    best_impurity = impurity
                    best_feature = idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_samples = len(y)
        y_mean = np.mean(y)

        node = {
            'leaf': True,
            'value': y_mean,
            'depth': depth
        }

        if (self.max_depth is not None and depth >= self.max_depth):
            return node

        if n_samples < self.min_samples_split:
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return node

        X_col = X[:, feature]
        left_mask = X_col <= threshold
        right_mask = ~left_mask

        node['leaf'] = False
        node['feature'] = feature
        node['threshold'] = threshold
        node['left'] = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node['right'] = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return node

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        self.feature_importances_ = np.zeros(self.n_features_in_)

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "max_depth": self.max_depth,
                "criterion": self.criterion,
                "n_samples_trained": len(y)
            }
        )

    def _predict_sample(self, x, node):
        if node['leaf']:
            return node['value']
        # 确保特征索引在有效范围内
        feature = node['feature']
        if feature >= len(x):
            return node['value']
        if x[feature] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        if self.tree_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        y_pred = [self._predict_sample(x, self.tree_) for x in X]
        y_pred = np.array(y_pred)

        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).reshape(-1)

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).reshape(-1)

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        bias_resp = self.bias(y_true, y_pred)
        variance_resp = self.variance(y_pred)
        mase_resp = self.mase(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "bias": bias_resp["content"]["bias"],
            "variance": variance_resp["content"]["variance"],
            "mase": mase_resp["content"]["mase"],
            "n_samples": len(y_true)
        }

        return self._response(1, content=metrics)

# todo-2.9 build scikit-learn MLPClassifier
class MLPClassifier(Evaluation):
    def __init__(
            self,
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='sgd',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=10,
            use_ga_init=False  # 是否使用遗传算法初始化权重
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.use_ga_init = use_ga_init

        # 模型参数
        self.coefs_ = None
        self.intercepts_ = None
        self.n_features_in_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.n_iter_ = 0

    def _get_activation(self):
        """获取激活函数及导数（感知机/多层网络核心）"""
        if self.activation == 'sigmoid':
            return sigmoid, sigmoid_deriv
        elif self.activation == 'relu':
            return relu, relu_deriv
        elif self.activation == 'tanh':
            return tanh, tanh_deriv
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")

    def _forward(self, X, coefs=None, intercepts=None):
        """前向传播（感知机：单层；多层网络：多层）"""
        if coefs is None:
            coefs = self.coefs_
            intercepts = self.intercepts_
        activation, _ = self._get_activation()
        a = X
        for i in range(len(coefs) - 1):
            a = activation(a @ coefs[i] + intercepts[i])
        # 输出层softmax（多分类）
        logits = a @ coefs[-1] + intercepts[-1]
        a = softmax(logits)
        return a

    def _backward(self, X, y, y_pred):
        """反向传播（误差逆传播算法核心）"""
        _, activation_deriv = self._get_activation()
        n_samples = X.shape[0]
        grad_coefs = []
        grad_intercepts = []

        # 确保 y 是one-hot编码格式
        if y.ndim == 1:
            if self.n_classes_ > 1:
                # 转换为one-hot编码
                y = np.eye(self.n_classes_)[y]
            else:
                y = y.reshape(-1, 1)

        # 确保 y_pred 形状与 y 匹配
        if y_pred.shape != y.shape:
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            if y_pred.shape[1] == 1 and y.shape[1] > 1:
                # 对于二分类，扩展为两列
                y_pred = np.hstack([1 - y_pred, y_pred])
            # 确保 y_pred 形状与 y 完全匹配
            if y_pred.shape != y.shape:
                # 调整 y_pred 形状以匹配 y
                y_pred = np.zeros_like(y)

        # 输出层梯度
        delta = y_pred - y
        
        # 计算激活值
        a = X
        activations = [a]
        for i in range(len(self.coefs_) - 1):
            a = self._get_activation()[0](a @ self.coefs_[i] + self.intercepts_[i])
            activations.append(a)
        
        # 计算输出层权重梯度
        if a.ndim == 2 and delta.ndim == 2:
            grad_coef = (a.T @ delta) / n_samples + self.alpha * self.coefs_[-1] / n_samples
        else:
            grad_coef = np.zeros_like(self.coefs_[-1])
        grad_intercept = np.mean(delta, axis=0)
        grad_coefs.insert(0, grad_coef)
        grad_intercepts.insert(0, grad_intercept)

        # 隐藏层梯度
        for i in range(len(self.coefs_) - 2, -1, -1):
            # 确保 delta 和 coefs 形状匹配
            if delta.ndim == 2 and self.coefs_[i + 1].ndim == 2:
                delta = (delta @ self.coefs_[i + 1].T) * activation_deriv(activations[i + 1])
                if activations[i].ndim == 2 and delta.ndim == 2:
                    grad_coef = (activations[i].T @ delta) / n_samples + self.alpha * self.coefs_[i] / n_samples
                else:
                    grad_coef = np.zeros_like(self.coefs_[i])
            else:
                delta = np.zeros_like(activations[i + 1])
                grad_coef = np.zeros_like(self.coefs_[i])
            grad_intercept = np.mean(delta, axis=0)
            grad_coefs.insert(0, grad_coef)
            grad_intercepts.insert(0, grad_intercept)

        return grad_coefs, grad_intercepts

    def _init_weights(self, n_features, n_classes):
        """初始化权重（支持遗传算法和Xavier初始化）"""
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_classes]
        self.coefs_ = []
        self.intercepts_ = []
        weight_shapes = []
        for i in range(len(layer_sizes) - 1):
            shape = (layer_sizes[i], layer_sizes[i + 1])
            weight_shapes.append(shape)
            # 使用Xavier初始化，比随机正态分布更有效
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.coefs_.append(np.random.uniform(-limit, limit, shape))
            self.intercepts_.append(np.zeros(layer_sizes[i + 1]))

        # 遗传算法优化初始权重（可选）
        if self.use_ga_init and hasattr(self, 'X_train_') and hasattr(self, 'y_train_'):
            def forward_fn(X, individual):
                # 重构权重和偏置
                coefs = []
                intercepts = []
                idx = 0
                for i in range(len(self.coefs_)):
                    coefs.append(individual[idx])
                    idx += 1
                for i in range(len(self.intercepts_)):
                    intercepts.append(individual[idx])
                    idx += 1
                a = X
                activation, _ = self._get_activation()
                for i in range(len(coefs) - 1):
                    a = activation(a @ coefs[i] + intercepts[i])
                a = softmax(a @ coefs[-1] + intercepts[-1])
                return a

            # 合并coefs和intercepts为种群个体
            init_ind = self.coefs_ + self.intercepts_
            best_ind = genetic_algorithm(
                X=self.X_train_, y=self.y_train_,
                forward_fn=forward_fn,
                weight_shapes=[w.shape for w in init_ind],
                pop_size=5, generations=3  # 减少种群大小和代数，加快初始化速度
            )
            # 重构权重和偏置
            coefs = []
            intercepts = []
            idx = 0
            for i in range(len(self.coefs_)):
                coefs.append(best_ind[idx])
                idx += 1
            for i in range(len(self.intercepts_)):
                intercepts.append(best_ind[idx])
                idx += 1
            self.coefs_ = coefs
            self.intercepts_ = intercepts

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)
            if self.random_state is not None:
                np.random.seed(self.random_state)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            # 确保 X 是二维的
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y_onehot = np.eye(self.n_classes_)[y]

            # 早停拆分验证集
            if self.early_stopping:
                val_size = int(len(X) * self.validation_fraction)
                idx = np.random.permutation(len(X))
                self.X_train_ = X[idx[val_size:]]
                self.y_train_ = y_onehot[idx[val_size:]]
                self.X_val_ = X[idx[:val_size]]
                self.y_val_ = y_onehot[idx[:val_size]]
            else:
                self.X_train_ = X
                self.y_train_ = y_onehot

            # 初始化权重
            if not (self.warm_start and self.coefs_ is not None):
                self._init_weights(self.n_features_in_, self.n_classes_)

            # 批大小
            if self.batch_size == 'auto':
                self.batch_size = min(200, len(self.X_train_))

            # 优化器参数
            v_coefs = [np.zeros_like(c) for c in self.coefs_]
            v_intercepts = [np.zeros_like(i) for i in self.intercepts_]
            m_coefs = [np.zeros_like(c) for c in self.coefs_]
            m_intercepts = [np.zeros_like(i) for i in self.intercepts_]

            best_val_loss = np.inf
            no_improvement = 0

            for epoch in range(self.max_iter):
                self.n_iter_ = epoch + 1
                # 打乱数据
                if self.shuffle:
                    idx = np.random.permutation(len(self.X_train_))
                    X_shuffled = self.X_train_[idx]
                    y_shuffled = self.y_train_[idx]
                else:
                    X_shuffled = self.X_train_
                    y_shuffled = self.y_train_

                # 批处理
                for i in range(0, len(X_shuffled), self.batch_size):
                    X_batch = X_shuffled[i:i + self.batch_size]
                    y_batch = y_shuffled[i:i + self.batch_size]
                    y_pred = self._forward(X_batch)

                    # 反向传播
                    grad_coefs, grad_intercepts = self._backward(X_batch, y_batch, y_pred)

                    # 学习率衰减
                    if self.learning_rate == 'invscaling':
                        lr = self.learning_rate_init / (1 + epoch * self.power_t)
                    elif self.learning_rate == 'adaptive':
                        lr = self.learning_rate_init
                        if len(self.loss_history_) > 10 and np.mean(self.loss_history_[-10:]) > np.mean(
                                self.loss_history_[-20:-10]) - self.tol:
                            lr *= 0.5
                    else:
                        lr = self.learning_rate_init

                    # 优化器更新
                    if self.solver == 'sgd':
                        for i in range(len(self.coefs_)):
                            if self.nesterovs_momentum:
                                v_coefs[i] = self.momentum * v_coefs[i] - lr * grad_coefs[i]
                                self.coefs_[i] += v_coefs[i]
                                v_intercepts[i] = self.momentum * v_intercepts[i] - lr * grad_intercepts[i]
                                self.intercepts_[i] += v_intercepts[i]
                            else:
                                self.coefs_[i] -= lr * grad_coefs[i]
                                self.intercepts_[i] -= lr * grad_intercepts[i]
                    elif self.solver == 'adam':
                        for i in range(len(self.coefs_)):
                            m_coefs[i] = self.beta_1 * m_coefs[i] + (1 - self.beta_1) * grad_coefs[i]
                            v_coefs[i] = self.beta_2 * v_coefs[i] + (1 - self.beta_2) * (grad_coefs[i] ** 2)
                            m_hat = m_coefs[i] / (1 - self.beta_1 ** (epoch + 1))
                            v_hat = v_coefs[i] / (1 - self.beta_2 ** (epoch + 1))
                            self.coefs_[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                            m_intercepts[i] = self.beta_1 * m_intercepts[i] + (1 - self.beta_1) * grad_intercepts[i]
                            v_intercepts[i] = self.beta_2 * v_intercepts[i] + (1 - self.beta_2) * (grad_intercepts[i] ** 2)
                            m_hat = m_intercepts[i] / (1 - self.beta_1 ** (epoch + 1))
                            v_hat = v_intercepts[i] / (1 - self.beta_2 ** (epoch + 1))
                            self.intercepts_[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                # 记录损失 - 使用MSE损失，更适合回归问题
                y_pred_train = self._forward(self.X_train_)
                train_loss = np.mean((y_pred_train - self.y_train_) ** 2)
                self.loss_history_.append(train_loss)

                # 早停判断（缓解过拟合，辅助找全局最小）
                if self.early_stopping:
                    y_pred_val = self._forward(self.X_val_)
                    val_loss = np.mean((y_pred_val - self.y_val_) ** 2)
                    self.val_loss_history_.append(val_loss)
                    if val_loss < best_val_loss - self.tol:
                        best_val_loss = val_loss
                        no_improvement = 0
                        best_coefs = self.coefs_.copy()
                        best_intercepts = self.intercepts_.copy()
                    else:
                        no_improvement += 1
                        if no_improvement >= self.n_iter_no_change:
                            self.coefs_ = best_coefs
                            self.intercepts_ = best_intercepts
                            if self.verbose:
                                pass
                            break

                # 收敛判断
                if epoch > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                    if self.verbose:
                        # print(f"训练收敛于第{epoch + 1}轮")
                        pass
                    break

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "classes_": self.classes_,
                    "n_iter_": self.n_iter_,
                    "final_loss": self.loss_history_[-1] if self.loss_history_ else None
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def predict_proba(self, X):
        if self.coefs_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")
        X = np.array(X, dtype=np.float64)
        y_prob = self._forward(X)
        return self._response(1, content=y_prob)

    def predict(self, X):
        proba_resp = self.predict_proba(X)
        if proba_resp["status"] == 0:
            return proba_resp
        y_prob = proba_resp["content"]
        y_pred = self.classes_[np.argmax(y_prob, axis=1)]
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        y_true = np.array(y_true, dtype=np.int64)
        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.int64)

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        # 计算混淆矩阵
        cm_resp = self.confusion_matrix(y_true, y_pred)
        cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
        classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

        acc_resp = self.accuracy(y_true, y_pred)
        prec_resp = self.precision(y_true, y_pred)
        rec_resp = self.recall(y_true, y_pred)
        f1_resp = self.f1(y_true, y_pred)

        # 构建metrics_per_output
        metrics_per_output = {
            "output_1": {
                "accuracy": acc_resp["content"]["accuracy"],
                "precision": prec_resp["content"]["precision"],
                "recall": rec_resp["content"]["recall"],
                "f1": f1_resp["content"]["f1"],
                "confusion_matrix": cm,
                "classes": classes
            }
        }

        metrics = {
            "accuracy": acc_resp["content"]["accuracy"],
            "precision": prec_resp["content"]["precision"],
            "recall": rec_resp["content"]["recall"],
            "f1": f1_resp["content"]["f1"],
            "n_samples": len(y_true),
            "metrics_per_output": metrics_per_output
        }
        return self._response(1, content=metrics)

# todo-2.10 build scikit-learn MLPRegressor
class MLPRegressor(Evaluation):
    def __init__(
            self,
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='sgd',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            power_t=0.5,
            max_iter=200,
            shuffle=True,
            random_state=None,
            tol=1e-4,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=False,
            validation_fraction=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            n_iter_no_change=10
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change

        self.coefs_ = None
        self.intercepts_ = None
        self.n_features_in_ = None
        self.n_outputs_ = 1
        self.loss_history_ = []
        self.val_loss_history_ = []
        self.n_iter_ = 0

    def _get_activation(self):
        if self.activation == 'sigmoid':
            return sigmoid, sigmoid_deriv
        elif self.activation == 'relu':
            return relu, relu_deriv
        elif self.activation == 'tanh':
            return tanh, tanh_deriv
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")

    def _forward(self, X, coefs=None, intercepts=None):
        if coefs is None:
            coefs = self.coefs_
            intercepts = self.intercepts_
        activation, _ = self._get_activation()
        a = X
        for i in range(len(coefs) - 1):
            a = activation(a @ coefs[i] + intercepts[i])
        # 输出层线性激活（回归）
        a = linear(a @ coefs[-1] + intercepts[-1])
        return a

    def _backward(self, X, y, y_pred):
        _, activation_deriv = self._get_activation()
        n_samples = X.shape[0]
        grad_coefs = []
        grad_intercepts = []

        # 隐藏层梯度
        a = X
        activations = [a]
        for i in range(len(self.coefs_) - 1):
            a = self._get_activation()[0](a @ self.coefs_[i] + self.intercepts_[i])
            activations.append(a)

        # 输出层梯度（MSE损失）
        delta = (y_pred - y).reshape(-1, 1)
        # 使用最后一个隐藏层的激活值计算输出层梯度
        grad_coef = (activations[-1].T @ delta) / n_samples + self.alpha * self.coefs_[-1] / n_samples
        # 梯度裁剪，防止梯度爆炸
        grad_coef = np.clip(grad_coef, -1.0, 1.0)
        grad_intercept = np.mean(delta, axis=0)
        # 梯度裁剪，防止梯度爆炸
        grad_intercept = np.clip(grad_intercept, -1.0, 1.0)
        grad_coefs.insert(0, grad_coef)
        grad_intercepts.insert(0, grad_intercept)

        for i in range(len(self.coefs_) - 2, -1, -1):
            delta = (delta @ self.coefs_[i + 1].T) * activation_deriv(activations[i + 1])
            # 梯度裁剪，防止梯度爆炸
            delta = np.clip(delta, -1.0, 1.0)
            grad_coef = (activations[i].T @ delta) / n_samples + self.alpha * self.coefs_[i] / n_samples
            # 梯度裁剪，防止梯度爆炸
            grad_coef = np.clip(grad_coef, -1.0, 1.0)
            grad_intercept = np.mean(delta, axis=0)
            # 梯度裁剪，防止梯度爆炸
            grad_intercept = np.clip(grad_intercept, -1.0, 1.0)
            grad_coefs.insert(0, grad_coef)
            grad_intercepts.insert(0, grad_intercept)

        return grad_coefs, grad_intercepts

    def _init_weights(self, n_features, n_outputs):
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        self.coefs_ = []
        self.intercepts_ = []
        for i in range(len(layer_sizes) - 1):
            shape = (layer_sizes[i], layer_sizes[i + 1])
            # 使用Xavier初始化，比随机正态分布更有效
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.coefs_.append(np.random.uniform(-limit, limit, shape))
            self.intercepts_.append(np.zeros(layer_sizes[i + 1]))

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1

        # 早停拆分验证集
        if self.early_stopping:
            val_size = int(len(X) * self.validation_fraction)
            idx = np.random.permutation(len(X))
            self.X_train_ = X[idx[val_size:]]
            self.y_train_ = y[idx[val_size:]]
            self.X_val_ = X[idx[:val_size]]
            self.y_val_ = y[idx[:val_size]]
        else:
            self.X_train_ = X
            self.y_train_ = y
        
        # 确保y_train_和y_val_是二维数组
        if self.y_train_.ndim == 1:
            self.y_train_ = self.y_train_.reshape(-1, 1)
        if self.early_stopping and self.y_val_.ndim == 1:
            self.y_val_ = self.y_val_.reshape(-1, 1)

        # 初始化权重
        if not (self.warm_start and self.coefs_ is not None):
            self._init_weights(self.n_features_in_, self.n_outputs_)

        # 批大小
        if self.batch_size == 'auto':
            self.batch_size = min(200, len(self.X_train_))

        # 优化器参数
        v_coefs = [np.zeros_like(c) for c in self.coefs_]
        v_intercepts = [np.zeros_like(i) for i in self.intercepts_]
        m_coefs = [np.zeros_like(c) for c in self.coefs_]
        m_intercepts = [np.zeros_like(i) for i in self.intercepts_]

        best_val_loss = np.inf
        no_improvement = 0

        for epoch in range(self.max_iter):
            self.n_iter_ = epoch + 1
            # 打乱数据
            if self.shuffle:
                idx = np.random.permutation(len(self.X_train_))
                X_shuffled = self.X_train_[idx]
                y_shuffled = self.y_train_[idx]
            else:
                X_shuffled = self.X_train_
                y_shuffled = self.y_train_

            # 批处理
            for i in range(0, len(X_shuffled), self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                y_pred = self._forward(X_batch)

                # 反向传播
                grad_coefs, grad_intercepts = self._backward(X_batch, y_batch, y_pred)

                # 学习率衰减
                if self.learning_rate == 'invscaling':
                    lr = self.learning_rate_init / (1 + epoch * self.power_t)
                elif self.learning_rate == 'adaptive':
                    lr = self.learning_rate_init
                    if len(self.loss_history_) > 10 and np.mean(self.loss_history_[-10:]) > np.mean(
                            self.loss_history_[-20:-10]) - self.tol:
                        lr *= 0.5
                else:
                    lr = self.learning_rate_init

                # 优化器更新
                if self.solver == 'sgd':
                    for i in range(len(self.coefs_)):
                        if self.nesterovs_momentum:
                            v_coefs[i] = self.momentum * v_coefs[i] - lr * grad_coefs[i]
                            self.coefs_[i] += v_coefs[i]
                            v_intercepts[i] = self.momentum * v_intercepts[i] - lr * grad_intercepts[i]
                            self.intercepts_[i] += v_intercepts[i]
                        else:
                            self.coefs_[i] -= lr * grad_coefs[i]
                            self.intercepts_[i] -= lr * grad_intercepts[i]
                elif self.solver == 'adam':
                    for i in range(len(self.coefs_)):
                        m_coefs[i] = self.beta_1 * m_coefs[i] + (1 - self.beta_1) * grad_coefs[i]
                        v_coefs[i] = self.beta_2 * v_coefs[i] + (1 - self.beta_2) * (grad_coefs[i] ** 2)
                        m_hat = m_coefs[i] / (1 - self.beta_1 ** (epoch + 1))
                        v_hat = v_coefs[i] / (1 - self.beta_2 ** (epoch + 1))
                        self.coefs_[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

                        m_intercepts[i] = self.beta_1 * m_intercepts[i] + (1 - self.beta_1) * grad_intercepts[i]
                        v_intercepts[i] = self.beta_2 * v_intercepts[i] + (1 - self.beta_2) * (grad_intercepts[i] ** 2)
                        m_hat = m_intercepts[i] / (1 - self.beta_1 ** (epoch + 1))
                        v_hat = v_intercepts[i] / (1 - self.beta_2 ** (epoch + 1))
                        self.intercepts_[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # 记录损失
            train_loss = np.mean((self.y_train_ - self._forward(self.X_train_)) ** 2)
            self.loss_history_.append(train_loss)

            # 早停判断
            if self.early_stopping:
                val_loss = np.mean((self.y_val_ - self._forward(self.X_val_)) ** 2)
                self.val_loss_history_.append(val_loss)
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    no_improvement = 0
                    best_coefs = self.coefs_.copy()
                    best_intercepts = self.intercepts_.copy()
                else:
                    no_improvement += 1
                    if no_improvement >= self.n_iter_no_change:
                        self.coefs_ = best_coefs
                        self.intercepts_ = best_intercepts
                        if self.verbose:
                            # print(f"早停于第{epoch + 1}轮，验证损失不再下降")
                            pass
                        break

            # 收敛判断
            if epoch > 0 and abs(self.loss_history_[-2] - self.loss_history_[-1]) < self.tol:
                if self.verbose:
                    # print(f"训练收敛于第{epoch + 1}轮")
                    pass
                break

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "n_outputs_": self.n_outputs_,
                "n_iter_": self.n_iter_,
                "final_loss": self.loss_history_[-1] if self.loss_history_ else None
            }
        )

    def predict(self, X):
        if self.coefs_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")
        X = np.array(X, dtype=np.float64)
        y_pred = self._forward(X)
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()
        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        bias_resp = self.bias(y_true, y_pred)
        variance_resp = self.variance(y_pred)
        mase_resp = self.mase(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "bias": bias_resp["content"]["bias"],
            "variance": variance_resp["content"]["variance"],
            "mase": mase_resp["content"]["mase"],
            "n_samples": len(y_true)
        }
        return self._response(1, content=metrics)

# todo-2.11 build scikit-learn RBFClassifier
class RBFClassifier(Evaluation):
    def __init__(self,n_centers=50,sigma=1.0,learning_rate=0.01,max_iter=100,random_state=None):
        super().__init__()
        self.n_centers = n_centers
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.centers_ = None
        self.sigma_ = None
        self.W_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
    def fit(self,X,y):
        try:
            import time
            start_time = time.time()
            max_train_time = 30  # 最大训练时间（秒）
            
            X = np.array(X,dtype=np.float64)
            y = np.array(y,dtype=np.int64)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            if self.random_state is not None:
                np.random.seed(self.random_state)
            # 确保 X 是二维的
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            y_onehot = np.eye(self.n_classes_)[y]
            
            # 限制中心数量，避免计算过于复杂
            n_samples = X.shape[0]
            if self.n_centers > n_samples:
                self.n_centers = min(20, n_samples)  # 进一步减少中心数量
            
            # 使用K-means初始化中心，提高中心质量
            from sklearn.cluster import KMeans
            # 设置K-means的n_init参数为1，减少初始化次数
            kmeans = KMeans(n_clusters=self.n_centers, random_state=self.random_state, n_init=1)
            kmeans.fit(X)
            self.centers_ = kmeans.cluster_centers_
            
            # 自动计算sigma值
            if self.sigma <= 0:
                # 使用中心间的平均距离作为sigma
                dists = np.sqrt(np.sum((self.centers_[:, np.newaxis] - self.centers_) ** 2, axis=2))
                dists = dists[np.triu_indices(len(self.centers_), 1)]
                if len(dists) > 0:
                    self.sigma_ = np.mean(dists)
                else:
                    self.sigma_ = 1.0
            else:
                self.sigma_ = self.sigma
            
            self.W_ = np.random.randn(self.n_centers,self.n_classes_)*0.01
            
            # 添加早期停止条件
            prev_loss = float('inf')
            patience = 3  # 耐心值，连续多少轮没有改进就停止
            no_improvement = 0
            
            # 预计算RBF核矩阵，避免每次迭代重复计算
            # 直接实现rbf函数，避免导入问题
            def rbf_local(x, center, sigma):
                if x.ndim == 2 and center.ndim == 2:
                    return np.exp(-np.sum((x[:, np.newaxis, :] - center) ** 2, axis=2) / (2 * sigma ** 2 + 1e-10))
                else:
                    return np.exp(-np.sum((x - center) ** 2) / (2 * sigma ** 2 + 1e-10))
            
            phi = rbf_local(X, self.centers_, self.sigma_)
            # 确保 phi 是二维的
            if phi.ndim == 1:
                phi = phi.reshape(-1, 1)
            if phi.ndim != 2:
                phi = phi.reshape(-1, self.n_centers)
            
            for i in range(self.max_iter):
                # 检查训练时间
                if time.time() - start_time > max_train_time:
                    break
                
                y_pred = softmax(phi @ self.W_)
                grad = phi.T @ (y_pred - y_onehot)/len(X)
                
                # 学习率衰减
                current_lr = self.learning_rate * (0.95 ** (i // 50))
                self.W_ -= current_lr * grad
                
                # 计算损失，添加早期停止
                if i % 5 == 0:
                    loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-10), axis=1))
                    if abs(prev_loss - loss) < 1e-4:
                        no_improvement += 1
                        if no_improvement >= patience:
                            break
                    else:
                        no_improvement = 0
                    prev_loss = loss
            return self._response(1,content={"n_centers":self.n_centers,"sigma":self.sigma_})
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")
    def predict_proba(self,X):
        X = np.array(X,dtype=np.float64)
        # 确保 X 是二维的
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # 直接实现rbf函数，避免导入问题
        def rbf_local(x, center, sigma):
            if x.ndim == 2 and center.ndim == 2:
                return np.exp(-np.sum((x[:, np.newaxis, :] - center) ** 2, axis=2) / (2 * sigma ** 2 + 1e-10))
            else:
                return np.exp(-np.sum((x - center) ** 2) / (2 * sigma ** 2 + 1e-10))
        # 使用本地实现的rbf函数进行批量计算
        phi = rbf_local(X, self.centers_, self.sigma_)
        # 确保 phi 是二维的
        if phi.ndim == 1:
            phi = phi.reshape(-1, 1)
        # 确保 phi 和 W_ 形状匹配
        if phi.shape[1] != self.W_.shape[0]:
            return self._response(0, content=f"Feature dimension mismatch: phi shape {phi.shape}, W_ shape {self.W_.shape}")
        y_prob = softmax(phi @ self.W_)
        return self._response(1,content=y_prob)
    def predict(self,X):
        pr = self.predict_proba(X)
        if pr["status"]==0:return pr
        return self._response(1,content=self.classes_[np.argmax(pr["content"],axis=1)])
    def evaluate(self,X,y_true,y_pred=None,threshold=0.5):
        try:
            y_true = np.array(y_true,dtype=np.int64)
            if y_pred is None:
                y_pred = self.predict(X)["content"]
            
            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_true, y_pred)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []
            
            # 计算准确率
            acc_resp = self.accuracy(y_true, y_pred)
            acc = acc_resp['content']['accuracy'] if (acc_resp['status'] == 1) else 0.0
            
            # 计算精确率
            prec_resp = self.precision(y_true, y_pred)
            prec = prec_resp['content']['precision'] if (prec_resp['status'] == 1) else 0.0
            
            # 计算召回率
            rec_resp = self.recall(y_true, y_pred)
            rec = rec_resp['content']['recall'] if (rec_resp['status'] == 1) else 0.0
            
            # 计算F1值
            f1_resp = self.f1(y_true, y_pred)
            f1 = f1_resp['content']['f1'] if (f1_resp['status'] == 1) else 0.0
            
            # 计算AUC
            auc = 0.0
            try:
                y_prob_resp = self.predict_proba(X)
                y_prob = y_prob_resp['content'] if (y_prob_resp['status'] == 1) else None
                if y_prob is not None:
                    # 对于二分类，使用第二个概率值作为正类概率
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    roc_resp = self.roc(y_true, y_prob_pos)
                    roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
                    if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                        auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                        if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                            auc_val = auc_resp['content']['auc']
                            if not np.isnan(auc_val) and not np.isinf(auc_val):
                                auc = auc_val
            except Exception as e:
                # AUC计算失败不影响其他指标
                pass
        
            # 构建metrics_per_output
            metrics_per_output = {
                "output_1": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "confusion_matrix": cm,
                    "classes": classes
                }
            }
        
            # 构建平均指标
            average_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc
            }
        
            return self._response(1,content={
                "metrics_per_output": metrics_per_output,
                "average_metrics": average_metrics,
                "n_outputs": 1
            })
        except Exception as e:
            return self._response(0, content=f"评估失败: {str(e)}")

# todo-2.12 build scikit-learn SOM
class SOM(Evaluation):
    def __init__(self,map_size=(10,10),learning_rate=0.1,sigma=2.0,max_iter=1000,random_state=None):
        super().__init__()
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights_ = None
        self.n_features_in_ = None
    def fit(self,X):
        X = np.array(X,dtype=np.float64)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.weights_ = np.random.rand(self.map_size[0],self.map_size[1],self.n_features_in_)
        for i in range(self.max_iter):
            lr = self.learning_rate * (1-i/self.max_iter)
            s = self.sigma * (1-i/self.max_iter)
            for x in X:
                dist = euclidean_dist(x,self.weights_)
                bmu = np.unravel_index(np.argmin(dist),dist.shape)
                for r in range(self.map_size[0]):
                    for c in range(self.map_size[1]):
                        d = np.sqrt((r-bmu[0])**2 + (c-bmu[1])**2)
                        h = som_neighborhood(d,bmu,s)
                        self.weights_[r,c] += lr*h*(x-self.weights_[r,c])
        return self._response(1,content={"map_size":self.map_size})
    def predict(self,X):
        X = np.array(X,dtype=np.float64)
        labels = []
        for x in X:
            dist = euclidean_dist(x,self.weights_)
            bmu = np.unravel_index(np.argmin(dist),dist.shape)
            labels.append(bmu[0]*self.map_size[1]+bmu[1])
        return self._response(1,content=np.array(labels))
    def evaluate(self,X,y_true=None,y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)["content"]
        return self._response(1,content={"n_clusters":np.unique(y_pred).size})

# todo-2.12 build scikit-learn CascadeCorrelation
class CascadeCorrelation(Evaluation):
    def __init__(self,max_hidden=10,learning_rate=0.01,max_iter=100,random_state=None):
        super().__init__()
        self.max_hidden = max_hidden
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.input_weights_ = []
        self.cascade_weights_ = []
        self.output_weights_ = None
        self.n_features_in_ = None
        self.n_outputs_ = 1
    def fit(self,X,y):
        X = np.array(X,dtype=np.float64)
        y = np.array(y,dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1
        # 初始输出权重
        self.output_weights_ = np.random.rand(self.n_features_in_,1)*0.01
        # 构建级联网络
        current_features = X
        for _ in range(self.max_hidden):
            # 添加新的隐藏单元
            hidden_size = 1
            # 随机初始化输入权重
            input_weight = np.random.rand(current_features.shape[1], hidden_size)*0.01
            self.input_weights_.append(input_weight)
            # 计算隐藏单元输出
            hidden_output = relu(current_features @ input_weight)
            # 扩展特征向量
            current_features = np.hstack([current_features, hidden_output])
            # 更新输出权重
            new_output_weights = np.random.rand(current_features.shape[1], 1)*0.01
            # 训练新的输出权重
            for _ in range(self.max_iter):
                y_pred = current_features @ new_output_weights
                y_pred = y_pred.ravel()
                error = y_pred - y
                grad = current_features.T @ error.reshape(-1, 1) / len(y)
                # 梯度裁剪，防止梯度爆炸
                grad = np.clip(grad, -1.0, 1.0)
                new_output_weights -= self.learning_rate * grad
            self.output_weights_ = new_output_weights
        return self._response(1,content={"n_hidden":len(self.input_weights_)})
    def predict(self,X):
        X = np.array(X,dtype=np.float64)
        current_features = X
        for w in self.input_weights_:
            hidden_output = relu(current_features @ w)
            current_features = np.hstack([current_features, hidden_output])
        y_pred = current_features @ self.output_weights_
        return self._response(1,content=y_pred.ravel())
    def evaluate(self,X,y_true,y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()
        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()
        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")
        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)
        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_samples": len(y_true)
        }
        return self._response(1, content=metrics)

# todo-2.13 build scikit-learn ElmanNetwork
class ElmanNetwork(Evaluation):
    def __init__(self,hidden_size=50,learning_rate=0.01,max_iter=200,random_state=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.Wx_ = None
        self.Wh_ = None
        self.Wy_ = None
        self.h0_ = None
        self.n_features_in_ = None
        self.n_outputs_ = 1
    def fit(self,X,y):
        X = np.array(X,dtype=np.float64)
        y = np.array(y,dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1
        self.Wx_ = np.random.rand(self.n_features_in_,self.hidden_size)*0.01
        self.Wh_ = np.random.rand(self.hidden_size,self.hidden_size)*0.01
        self.Wy_ = np.random.rand(self.hidden_size,1)*0.01
        self.h0_ = np.zeros(self.hidden_size)
        for _ in range(self.max_iter):
            h = np.tanh(X @ self.Wx_ + self.h0_)
            yp = h @ self.Wy_  # 线性输出层
            yp = yp.ravel()
            # 均方误差损失
            error = yp - y
            # 反向传播
            dWy = h.T @ error.reshape(-1, 1) / len(X)
            # 梯度裁剪，防止梯度爆炸
            dWy = np.clip(dWy, -1.0, 1.0)
            self.Wy_ -= self.learning_rate * dWy
            # 更新隐藏层权重
            dh = error.reshape(-1, 1) @ self.Wy_.T * (1 - h**2)
            # 梯度裁剪，防止梯度爆炸
            dh = np.clip(dh, -1.0, 1.0)
            dWx = X.T @ dh / len(X)
            # 梯度裁剪，防止梯度爆炸
            dWx = np.clip(dWx, -1.0, 1.0)
            dWh = self.h0_.reshape(-1, 1) @ np.mean(dh, axis=0).reshape(1, -1)
            # 梯度裁剪，防止梯度爆炸
            dWh = np.clip(dWh, -1.0, 1.0)
            self.Wx_ -= self.learning_rate * dWx
            self.Wh_ -= self.learning_rate * dWh
            # 更新隐藏状态
            self.h0_ = np.mean(h, axis=0)
        return self._response(1,content={"hidden_size":self.hidden_size})
    def predict(self,X):
        X = np.array(X,dtype=np.float64)
        h = np.tanh(X @ self.Wx_ + self.h0_)
        yp = h @ self.Wy_
        return self._response(1,content=yp.ravel())
    def evaluate(self,X,y_true,y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()
        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()
        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")
        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)
        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_samples": len(y_true)
        }
        return self._response(1, content=metrics)

# todo-2.13 build scikit-learn BoltzmannMachine
class BoltzmannMachine(Evaluation):
    def __init__(self,hidden_size=64,learning_rate=0.01,max_iter=100,beta=1.0,random_state=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta = beta
        self.random_state = random_state
        self.W_ = None
        self.bv_ = None
        self.bh_ = None
        self.W_out_ = None
        self.b_out_ = None
        self.n_features_in_ = None
        self.n_outputs_ = 1
    def fit(self,X,y):
        X = np.array(X,dtype=np.float64)
        y = np.array(y,dtype=np.float64).ravel()
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = 1
        # 初始化RBM权重
        self.W_ = np.random.rand(self.n_features_in_,self.hidden_size)*0.01
        self.bv_ = np.zeros(self.n_features_in_)
        self.bh_ = np.zeros(self.hidden_size)
        # 初始化输出层权重
        self.W_out_ = np.random.rand(self.hidden_size,1)*0.01
        self.b_out_ = np.zeros(1)
        # 预训练RBM
        for _ in range(self.max_iter//2):
            ph = boltzmann_prob(X @ self.W_ + self.bh_,self.beta)
            h = np.random.binomial(1,ph)
            pv = boltzmann_prob(h @ self.W_.T + self.bv_,self.beta)
            self.W_ += self.learning_rate*(X.T @ ph - pv.T @ h)/len(X)
        # 训练输出层
        for _ in range(self.max_iter//2):
            h = boltzmann_prob(X @ self.W_ + self.bh_,self.beta)
            y_pred = h @ self.W_out_ + self.b_out_
            y_pred = y_pred.ravel()
            error = y_pred - y
            # 反向传播
            dW_out = h.T @ error.reshape(-1, 1) / len(y)
            db_out = np.mean(error) / len(y)
            self.W_out_ -= self.learning_rate * dW_out
            self.b_out_ -= self.learning_rate * db_out
        return self._response(1,content={"hidden_size":self.hidden_size})
    def predict(self,X):
        X = np.array(X,dtype=np.float64)
        h = boltzmann_prob(X @ self.W_ + self.bh_,self.beta)
        y_pred = h @ self.W_out_ + self.b_out_
        return self._response(1,content=y_pred.ravel())
    def evaluate(self,X,y_true,y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()
        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()
        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")
        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)
        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_samples": len(y_true)
        }
        return self._response(1, content=metrics)

# todo-2.14 build scikit-learn SVC
class SVC(Evaluation):
    def __init__(
            self,
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            tol=1e-3,
            max_iter=1000,
            random_state=None
    ):
        super().__init__()
        self.C = C  # 软间隔正则化参数（C越小，软间隔越宽松）
        self.kernel = kernel  # 核函数类型
        self.degree = degree  # 多项式核次数
        self.gamma = gamma  # RBF/多项式核系数
        self.coef0 = coef0  # 多项式核常数项
        self.tol = tol  # 收敛阈值
        self.max_iter = max_iter  # 最大迭代次数
        self.random_state = random_state

        # 模型参数（训练后赋值）
        self.sv_alpha_ = None  # 支持向量对应的拉格朗日乘子
        self.sv_X_ = None  # 支持向量
        self.sv_y_ = None  # 支持向量标签
        self.b_ = None  # 偏置项
        # 存储核函数参数，而不是核函数实例
        self.kernel_params_ = None  # 核函数参数
        self.classes_ = None  # 类别标签
        self.n_classes_ = None  # 类别数
        self.n_features_in_ = None  # 输入特征数

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)

            if self.random_state is not None:
                np.random.seed(self.random_state)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            # 二分类标签转换（-1/1）
            if self.n_classes_ == 2:
                y_train = np.where(y == self.classes_[0], -1, 1)
            else:
                return self._response(0, content="暂仅支持二分类")

            # 存储核函数参数
            self.kernel_params_ = {
                'kernel': self.kernel,
                'gamma': self.gamma,
                'degree': self.degree,
                'coef0': self.coef0
            }

            # 获取核函数实例用于训练
            kernel_func = get_kernel(self.kernel, self.gamma, self.degree, self.coef0)

            # SMO求解对偶问题，得到支持向量
            self.sv_alpha_, self.sv_X_, self.sv_y_, self.b_ = smo_solver(
                X, y_train, self.C, kernel_func, self.tol, self.max_iter
            )

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "n_support_vectors_": len(self.sv_X_),
                    "kernel": self.kernel,
                    "C": self.C
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def predict(self, X):
        try:
            if self.sv_X_ is None:
                return self._response(0, content="Model not trained yet - call fit() first")

            X = np.array(X, dtype=np.float64)
            # 确保 X 是二维的
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # 检查输入特征数
            if X.shape[1] != self.n_features_in_:
                return self._response(0, content=f"输入特征数必须为{self.n_features_in_}")
            # 动态获取核函数实例
            kernel_func = get_kernel(
                self.kernel_params_['kernel'],
                self.kernel_params_['gamma'],
                self.kernel_params_['degree'],
                self.kernel_params_['coef0']
            )
            # 核函数计算：预测样本与支持向量的核矩阵
            kernel_mat = kernel_func(X, self.sv_X_)
            # 决策函数：f(x) = Σ(α_i y_i K(x, x_i)) + b
            # 确保维度匹配
            alpha_y = self.sv_alpha_ * self.sv_y_
            # 确保 alpha_y 是一维的
            if alpha_y.ndim > 1:
                alpha_y = alpha_y.ravel()
            # 确保 kernel_mat 是二维的
            if kernel_mat.ndim == 1:
                kernel_mat = kernel_mat.reshape(-1, 1)
            # 确保 alpha_y 形状与 kernel_mat 列数匹配
            if len(alpha_y) != kernel_mat.shape[1]:
                # 调整 alpha_y 长度以匹配 kernel_mat 列数
                alpha_y = alpha_y[:kernel_mat.shape[1]]
            # 确保 kernel_mat 是二维的，且 alpha_y 是一维的
            if kernel_mat.ndim != 2:
                kernel_mat = kernel_mat.reshape(-1, 1)
            if alpha_y.ndim != 1:
                alpha_y = alpha_y.ravel()
            # 确保 alpha_y 长度与 kernel_mat 列数匹配
            if len(alpha_y) != kernel_mat.shape[1]:
                # 调整 alpha_y 长度以匹配 kernel_mat 列数
                alpha_y = alpha_y[:kernel_mat.shape[1]]
            decision = np.sum(kernel_mat * alpha_y, axis=1) + self.b_
            # 分类预测
            y_pred = np.where(decision > 0, self.classes_[1], self.classes_[0])

            return self._response(1, content=y_pred)
        except Exception as e:
            return self._response(0, content=f"预测失败: {str(e)}")

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        try:
            y_true = np.array(y_true, dtype=np.int64)

            if y_pred is None:
                pred_resp = self.predict(X)
                if pred_resp["status"] == 0:
                    return pred_resp
                y_pred = pred_resp["content"]
            else:
                y_pred = np.array(y_pred, dtype=np.int64)

            if len(y_true) != len(y_pred):
                return self._response(0, content="y_true和y_pred样本数必须一致")

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_true, y_pred)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            acc_resp = self.accuracy(y_true, y_pred)
            acc = acc_resp['content']['accuracy'] if (acc_resp['status'] == 1) else 0.0

            prec_resp = self.precision(y_true, y_pred)
            prec = prec_resp['content']['precision'] if (prec_resp['status'] == 1) else 0.0

            rec_resp = self.recall(y_true, y_pred)
            rec = rec_resp['content']['recall'] if (rec_resp['status'] == 1) else 0.0

            f1_resp = self.f1(y_true, y_pred)
            f1 = f1_resp['content']['f1'] if (f1_resp['status'] == 1) else 0.0

            # 计算AUC
            auc = 0.0
            try:
                # SVC没有predict_proba方法，我们使用decision_function来计算AUC
                # 这里简化处理，使用预测值作为概率近似
                roc_resp = self.roc(y_true, y_pred)
                roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
                if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                    auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                    if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                        auc_val = auc_resp['content']['auc']
                        if not np.isnan(auc_val) and not np.isinf(auc_val):
                            auc = auc_val
            except Exception as e:
                # AUC计算失败不影响其他指标
                pass

            # 构建metrics_per_output
            metrics_per_output = {
                "output_1": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "confusion_matrix": cm,
                    "classes": classes
                }
            }

            # 构建平均指标
            average_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc
            }

            metrics = {
                "metrics_per_output": metrics_per_output,
                "average_metrics": average_metrics,
                "n_outputs": 1,
                "n_support_vectors": len(self.sv_X_) if self.sv_X_ is not None else 0,
                "n_samples": len(y_true)
            }

            return self._response(1, content=metrics)
        except Exception as e:
            return self._response(0, content=f"评估失败: {str(e)}")

# todo-2.15 build scikit-learn SVR
class SVR(Evaluation):
    def __init__(
            self,
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            epsilon=0.1,
            tol=1e-3,
            max_iter=1000,
            random_state=None
    ):
        super().__init__()
        self.C = C  # 软间隔正则化参数
        self.kernel = kernel  # 核函数类型
        self.degree = degree  # 多项式核次数
        self.gamma = gamma  # RBF/多项式核系数
        self.coef0 = coef0  # 多项式核常数项
        self.epsilon = epsilon  # ε-不敏感损失参数
        self.tol = tol  # 收敛阈值
        self.max_iter = max_iter  # 最大迭代次数
        self.random_state = random_state

        # 模型参数（训练后赋值）
        self.sv_alpha_ = None  # 拉格朗日乘子（α - α*）
        self.sv_X_ = None  # 支持向量
        self.b_ = None  # 偏置项
        self.kernel_ = None  # 核函数实例
        self.n_features_in_ = None  # 输入特征数

    def _smo_solver_reg(self, X, y, C, epsilon, kernel, tol=1e-3, max_iter=1000):
        """SMO求解SVR对偶问题（ε-不敏感损失）"""
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        alpha_star = np.zeros(n_samples)
        b = 0.0
        E = np.zeros(n_samples)
        kernel_mat = kernel(X, X)

        iter_num = 0
        total_iter = 0
        
        # 优化：预计算误差的函数
        def compute_error(i):
            return np.sum((alpha - alpha_star) * kernel_mat[:, i]) + b - y[i]

        # 优化：添加误差阈值，当所有误差都小于阈值时提前停止
        def all_errors_below_tol():
            for i in range(n_samples):
                E[i] = compute_error(i)
                # SVR的KKT条件
                if (E[i] > epsilon + tol and alpha[i] < C) or \
                   (E[i] < -epsilon - tol and alpha_star[i] < C) or \
                   (abs(E[i]) <= epsilon + tol and (alpha[i] > 0 or alpha_star[i] > 0)):
                    return False
            return True

        while iter_num < max_iter and total_iter < max_iter * 10:
            # 优化：提前检查所有样本是否满足KKT条件
            if all_errors_below_tol():
                break
                
            alpha_changed = 0
            # 优化：优先选择违反KKT条件的样本
            non_bound_indices = np.where((alpha > 0) & (alpha < C) & (alpha_star > 0) & (alpha_star < C))[0]
            bound_indices = np.where(((alpha == 0) | (alpha == C)) | ((alpha_star == 0) | (alpha_star == C)))[0]
            # 先检查非边界样本，再检查边界样本
            check_indices = np.concatenate([non_bound_indices, bound_indices]) if len(non_bound_indices) > 0 else bound_indices
            
            for i in check_indices:
                E[i] = compute_error(i)

                # 违反KKT条件（SVR版）
                if not ((E[i] > epsilon + tol and alpha[i] < C) or \
                        (E[i] < -epsilon - tol and alpha_star[i] < C) or \
                        (abs(E[i]) <= epsilon + tol and (alpha[i] > 0 or alpha_star[i] > 0))):
                    continue

                # 优化：选择使|E_i - E_j|最大的j，加快收敛
                if len(non_bound_indices) > 0:
                    # 从非边界样本中选择
                    j = non_bound_indices[np.argmax(np.abs(E[non_bound_indices] - E[i]))]
                else:
                    # 随机选择
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                E[j] = compute_error(j)

                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()
                alpha_star_i_old = alpha_star[i].copy()
                alpha_star_j_old = alpha_star[j].copy()

                # 计算上下界（SVR的上下界计算）
                L = max(0, alpha_star[j] - alpha[i])
                H = min(C, C + alpha_star[j] - alpha[i])
                if L == H:
                    continue

                # 计算η
                eta = 2 * kernel_mat[i, j] - kernel_mat[i, i] - kernel_mat[j, j]
                if eta >= 0:
                    continue

                # 更新αj和α*j
                delta = (E[i] - E[j]) / eta
                alpha[j] -= delta
                alpha[j] = np.clip(alpha[j], L, H)
                alpha_star[j] += delta
                alpha_star[j] = np.clip(alpha_star[j], 0, C)

                if abs(alpha[j] - alpha_j_old) < tol:
                    continue

                # 更新αi和α*i
                alpha[i] += alpha_j_old - alpha[j]
                alpha_star[i] += alpha_star_j_old - alpha_star[j]

                # 更新b
                b1 = b - E[i] - (alpha[i] - alpha_i_old) * kernel_mat[i, i] - (alpha[j] - alpha_j_old) * kernel_mat[i, j]
                b2 = b - E[j] - (alpha[i] - alpha_i_old) * kernel_mat[i, j] - (alpha[j] - alpha_j_old) * kernel_mat[j, j]
                if 0 < alpha[i] < C and 0 < alpha_star[i] < C:
                    b = b1
                elif 0 < alpha[j] < C and 0 < alpha_star[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                alpha_changed += 1
                # 优化：每处理10个样本检查一次收敛，避免不必要的计算
                if alpha_changed >= 10:
                    break

            if alpha_changed == 0:
                iter_num += 1
            else:
                iter_num = 0
            total_iter += 1

        # 提取支持向量（α≠α*的样本）
        sv_idx = (alpha - alpha_star) != 0
        sv_alpha = alpha[sv_idx] - alpha_star[sv_idx]
        sv_X = X[sv_idx]

        return sv_alpha, sv_X, b

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.kernel_ = get_kernel(self.kernel, self.gamma, self.degree, self.coef0)

        # SMO求解SVR对偶问题
        self.sv_alpha_, self.sv_X_, self.b_ = self._smo_solver_reg(
            X, y, self.C, self.epsilon, self.kernel_, self.tol, self.max_iter
        )

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "n_support_vectors_": len(self.sv_X_),
                "kernel": self.kernel,
                "C": self.C,
                "epsilon": self.epsilon
            }
        )

    def predict(self, X):
        if self.sv_X_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        kernel_mat = self.kernel_(X, self.sv_X_)
        # SVR决策函数：f(x) = Σ((α_i - α*i) K(x, x_i)) + b
        y_pred = np.sum(self.sv_alpha_ * kernel_mat, axis=1) + self.b_

        return self._response(1, content=y_pred)

    def evaluate(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64).ravel()
        y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_support_vectors": len(self.sv_X_) if self.sv_X_ is not None else 0,
            "n_samples": len(y_true)
        }

        return self._response(1, content=metrics)

# todo-2.16 build scikit-learn XGBRegressor
class XGBRegressor(Evaluation):
    def __init__(
            self,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            gamma=0.0,
            min_child_weight=1.0,
            random_state=None,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=1e-4
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

        # 模型参数
        self.trees = []
        self.n_features_in_ = None
        self.base_score = None
        self.X_val = None
        self.y_val = None

    def _split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask

    def _compute_gain(self, grad, hess):
        sum_grad = np.sum(grad)
        sum_hess = np.sum(hess)
        if sum_hess <= 0:
            return 0
        return sum_grad ** 2 / (sum_hess + self.reg_lambda)

    def _find_best_split(self, X, y, grad, hess, feature_indices):
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature_idx in feature_indices:
            # 对特征值排序并去重
            unique_values = np.unique(X[:, feature_idx])
            if len(unique_values) <= 1:
                continue

            # 尝试不同的阈值
            for threshold in unique_values[:-1]:
                left_mask, right_mask = self._split(X, y, feature_idx, threshold)
                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)
                
                if left_count == 0 or right_count == 0:
                    continue

                # 计算左右子树的梯度和
                left_grad = grad[left_mask]
                left_hess = hess[left_mask]
                right_grad = grad[right_mask]
                right_hess = hess[right_mask]

                # 计算增益
                current_gain = self._compute_gain(left_grad, left_hess) + self._compute_gain(right_grad, right_hess)
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, grad, hess, depth=0):
        if depth >= self.max_depth or len(y) < 2:
            # 叶子节点
            sum_grad = np.sum(grad)
            sum_hess = np.sum(hess)
            if sum_hess <= 0:
                leaf_value = 0.0
            else:
                leaf_value = -sum_grad / (sum_hess + self.reg_lambda)
            return {'leaf': True, 'value': leaf_value}

        # 随机选择特征
        n_features = X.shape[1]
        n_subfeatures = int(self.colsample_bytree * n_features)
        feature_indices = np.random.choice(n_features, n_subfeatures, replace=False)

        # 寻找最佳分裂点
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, grad, hess, feature_indices)

        if best_feature is None:
            # 无法分裂，返回叶子节点
            sum_grad = np.sum(grad)
            sum_hess = np.sum(hess)
            if sum_hess <= 0:
                leaf_value = 0.0
            else:
                leaf_value = -sum_grad / (sum_hess + self.reg_lambda)
            return {'leaf': True, 'value': leaf_value}

        # 分裂数据
        left_mask, right_mask = self._split(X, y, best_feature, best_threshold)

        # 递归构建左右子树
        left_tree = self._build_tree(X[left_mask], y[left_mask], grad[left_mask], hess[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], grad[right_mask], hess[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _predict_tree(self, x, tree):
        if tree['leaf']:
            return tree['value']
        # 确保特征索引在有效范围内
        feature = tree['feature']
        if feature >= len(x):
            return tree['value']
        if x[feature] <= tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.base_score = np.mean(y)
        self.trees = []

        # 优化：进一步减少默认树的数量，加快训练速度
        if self.n_estimators > 50:
            self.n_estimators = 50

        # 早停拆分验证集
        if self.early_stopping:
            val_size = int(len(X) * self.validation_fraction)
            idx = np.random.permutation(len(X))
            X_train = X[idx[val_size:]]
            y_train = y[idx[val_size:]]
            self.X_val = X[idx[:val_size]]
            self.y_val = y[idx[:val_size]]
        else:
            X_train = X
            y_train = y

        # 初始化预测值
        y_pred = np.full_like(y_train, self.base_score)

        # 早期停止相关变量
        best_val_loss = np.inf
        no_improvement = 0
        best_trees = []

        for i in range(self.n_estimators):
            # 计算梯度和二阶导数（平方损失）
            grad = y_pred - y_train
            hess = np.ones_like(y_train)

            # 随机采样
            n_samples = X_train.shape[0]
            sample_indices = np.random.choice(n_samples, int(self.subsample * n_samples), replace=False)
            X_sub = X_train[sample_indices]
            y_sub = y_train[sample_indices]
            grad_sub = grad[sample_indices]
            hess_sub = hess[sample_indices]

            # 构建树
            tree = self._build_tree(X_sub, y_sub, grad_sub, hess_sub)
            self.trees.append(tree)

            # 优化：向量化更新预测值，减少循环
            tree_preds = np.array([self._predict_tree(x, tree) for x in X_train])
            y_pred -= self.learning_rate * tree_preds

            # 早期停止检查
            if self.early_stopping and i % 2 == 0:  # 每2棵树检查一次
                val_pred = self.predict(self.X_val)['content']
                val_loss = np.mean((self.y_val - val_pred) ** 2)
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    no_improvement = 0
                    best_trees = self.trees.copy()
                else:
                    no_improvement += 1
                    if no_improvement >= self.n_iter_no_change:
                        # 早停，使用最佳模型
                        self.trees = best_trees
                        break

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "n_estimators": len(self.trees),
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "early_stopping": self.early_stopping
            }
        )

    def predict(self, X):
        if not self.trees:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        y_pred = np.full(X.shape[0], self.base_score)

        # 向量化预测，减少循环嵌套
        for tree in self.trees:
            # 使用列表推导式替代内层循环
            tree_preds = np.array([self._predict_tree(x, tree) for x in X])
            y_pred -= self.learning_rate * tree_preds

        return self._response(1, content=y_pred)

    def evaluate(self, y_true, y_pred):
        y_true = np.array(y_true, dtype=np.float64).ravel()
        y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_estimators": self.n_estimators,
            "n_samples": len(y_true)
        }

        return self._response(1, content=metrics)

# todo-2.17 build scikit-learn RandomForestRegressor
class RandomForestRegressor(Evaluation):
    def __init__(
            self,
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='auto',
            bootstrap=True,
            random_state=None,
            n_jobs=1
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

        # 模型参数
        self.trees = []
        self.n_features_in_ = None

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=self.bootstrap)
        return X[indices], y[indices]

    def _get_max_features(self, n_features):
        if self.max_features == 'auto':
            return int(np.sqrt(n_features))
        elif self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split:
            # 叶子节点
            return {'leaf': True, 'value': np.mean(y)}

        # 随机选择特征
        max_features = self._get_max_features(n_features)
        feature_indices = np.random.choice(n_features, max_features, replace=False)

        # 寻找最佳分裂点
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        for feature_idx in feature_indices:
            unique_values = np.unique(X[:, feature_idx])
            if len(unique_values) <= 1:
                continue

            # 向量化计算每个阈值的MSE
            for threshold in unique_values[:-1]:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                left_count = np.sum(left_mask)
                right_count = np.sum(right_mask)
                
                if left_count < self.min_samples_leaf or right_count < self.min_samples_leaf:
                    continue

                # 计算左右子树的均值
                left_mean = np.mean(y[left_mask])
                right_mean = np.mean(y[right_mask])
                
                # 计算MSE
                mse_left = np.mean((y[left_mask] - left_mean) ** 2)
                mse_right = np.mean((y[right_mask] - right_mean) ** 2)
                mse = (left_count * mse_left + right_count * mse_right) / n_samples

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_feature is None:
            # 无法分裂，返回叶子节点
            return {'leaf': True, 'value': np.mean(y)}

        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # 递归构建左右子树
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _predict_tree(self, x, tree):
        if tree['leaf']:
            return tree['value']
        # 确保特征索引在有效范围内
        feature = tree['feature']
        if feature >= len(x):
            return tree['value']
        if x[feature] <= tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.trees = []

        # 优化：减少默认树的数量，加快训练速度
        if self.n_estimators > 100:
            self.n_estimators = 100

        # 并行构建树
        if self.n_jobs > 1:
            # 使用多进程并行构建树
            from multiprocessing import Pool
            import os
            
            # 限制进程数不超过CPU核心数
            n_processes = min(self.n_jobs, os.cpu_count())
            
            # 生成bootstrap样本的索引
            bootstrap_indices = []
            for _ in range(self.n_estimators):
                indices = np.random.choice(len(X), len(X), replace=self.bootstrap)
                bootstrap_indices.append(indices)
            
            # 定义构建树的函数
            def build_tree_wrapper(indices):
                X_sample = X[indices]
                y_sample = y[indices]
                return self._build_tree(X_sample, y_sample)
            
            # 使用进程池并行构建树
            with Pool(processes=n_processes) as pool:
                self.trees = pool.map(build_tree_wrapper, bootstrap_indices)
        else:
            # 串行构建树
            for _ in range(self.n_estimators):
                # bootstrap 采样
                X_sample, y_sample = self._bootstrap_sample(X, y)
                # 构建树
                tree = self._build_tree(X_sample, y_sample)
                self.trees.append(tree)

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "max_features": self.max_features,
                "n_jobs": self.n_jobs
            }
        )

    def predict(self, X):
        if not self.trees:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        y_pred = np.zeros(X.shape[0])

        # 向量化预测，减少循环嵌套
        for tree in self.trees:
            # 使用列表推导式替代内层循环
            tree_preds = np.array([self._predict_tree(x, tree) for x in X])
            y_pred += tree_preds

        y_pred /= self.n_estimators
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_estimators": self.n_estimators,
            "n_samples": len(y_true)
        }

        return self._response(1, content=metrics)

# todo-2.18 build scikit-learn AdaBoostRegressor
class AdaBoostRegressor(Evaluation):
    def __init__(
            self,
            n_estimators=50,
            learning_rate=1.0,
            loss='linear',
            random_state=None
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

        # 模型参数
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []
        self.n_features_in_ = None

    def _build_stump(self, X, y, sample_weight):
        """构建决策树桩（深度为1的决策树）"""
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_left_value = None
        best_right_value = None
        best_error = float('inf')

        for feature_idx in range(n_features):
            unique_values = np.unique(X[:, feature_idx])
            for threshold in unique_values[:-1]:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # 计算左右子节点的预测值（加权平均）
                if np.sum(sample_weight[left_mask]) > 0:
                    left_value = np.sum(y[left_mask] * sample_weight[left_mask]) / np.sum(sample_weight[left_mask])
                else:
                    left_value = np.mean(y)
                if np.sum(sample_weight[right_mask]) > 0:
                    right_value = np.sum(y[right_mask] * sample_weight[right_mask]) / np.sum(sample_weight[right_mask])
                else:
                    right_value = np.mean(y)

                # 计算误差
                y_pred = np.where(left_mask, left_value, right_value)
                if self.loss == 'linear':
                    error = np.sum(sample_weight * np.abs(y - y_pred))
                elif self.loss == 'square':
                    error = np.sum(sample_weight * (y - y_pred) ** 2)
                elif self.loss == 'exponential':
                    error = np.sum(sample_weight * np.exp(-y * y_pred))
                else:
                    error = np.sum(sample_weight * np.abs(y - y_pred))

                if error < best_error:
                    best_error = error
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_value = left_value
                    best_right_value = right_value

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': {'leaf': True, 'value': best_left_value},
            'right': {'leaf': True, 'value': best_right_value}
        }, best_error

    def _predict_stump(self, x, stump):
        if stump['leaf']:
            return stump['value']
        if x[stump['feature']] <= stump['threshold']:
            return self._predict_stump(x, stump['left'])
        else:
            return self._predict_stump(x, stump['right'])

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # 初始化样本权重
        sample_weight = np.ones(n_samples) / n_samples

        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []

        for _ in range(self.n_estimators):
            # 构建决策树桩
            stump, error = self._build_stump(X, y, sample_weight)
            self.estimators.append(stump)
            self.estimator_errors.append(error)

            # 计算估计器权重
            if error >= 0.5:
                # 如果误差太大，停止训练
                break
            alpha = self.learning_rate * (0.5 * np.log((1 - error) / (error + 1e-10)))
            self.estimator_weights.append(alpha)

            # 更新样本权重
            y_pred = np.array([self._predict_stump(x, stump) for x in X])
            if self.loss == 'linear':
                sample_weight *= np.exp(-alpha * np.sign(y - y_pred))
            elif self.loss == 'square':
                sample_weight *= np.exp(-alpha * (y - y_pred) ** 2)
            elif self.loss == 'exponential':
                sample_weight *= np.exp(-alpha * y * y_pred)
            else:
                sample_weight *= np.exp(-alpha * np.sign(y - y_pred))

            # 归一化样本权重
            sample_weight /= np.sum(sample_weight)

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "n_estimators": len(self.estimators),
                "learning_rate": self.learning_rate,
                "loss": self.loss
            }
        )

    def predict(self, X):
        if not self.estimators:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        y_pred = np.zeros(X.shape[0])

        for i, estimator in enumerate(self.estimators):
            alpha = self.estimator_weights[i]
            for j in range(X.shape[0]):
                y_pred[j] += alpha * self._predict_stump(X[j], estimator)

        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_estimators": len(self.estimators),
            "n_samples": len(y_true)
        }

        return self._response(1, content=metrics)

# todo-2.16 build scikit-learn GaussianNB
class GaussianNB(Evaluation):
    def __init__(self, priors=None, var_smoothing=1e-9):
        super().__init__()
        self.priors = priors
        self.var_smoothing = var_smoothing

        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.theta_ = None  # 每个类别各特征的均值
        self.var_ = None  # 每个类别各特征的方差
        self.class_prior_ = None  # 类别先验概率

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            # 确保 y 是一维的
            if y.ndim > 1:
                y = y.ravel()

            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            # 计算类别先验概率
            if self.priors is None:
                self.class_prior_ = np.array([np.sum(y == c) / len(y) for c in self.classes_])
            else:
                self.class_prior_ = np.array(self.priors)
                if len(self.class_prior_) != self.n_classes_:
                    return self._response(0, content="priors长度必须与类别数一致")

            # 计算每个类别下特征的均值和方差（加平滑避免方差为0）
            self.theta_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes_])
            self.var_ = np.array([np.var(X[y == c], axis=0) + self.var_smoothing for c in self.classes_])

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "classes_": self.classes_,
                    "var_smoothing": self.var_smoothing
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def _gaussian_prob(self, X, c_idx):
        """计算高斯分布概率密度"""
        mean = self.theta_[c_idx]
        var = self.var_[c_idx]
        exponent = -((X - mean) ** 2) / (2 * var)
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(exponent)

    def predict_proba(self, X):
        if self.theta_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 对数后验概率（避免数值下溢）：log(先验) + sum(log(似然))
        log_probs = []
        for c_idx in range(self.n_classes_):
            log_prior = np.log(self.class_prior_[c_idx])
            log_likelihood = np.sum(np.log(self._gaussian_prob(X, c_idx)), axis=1)
            log_probs.append(log_prior + log_likelihood)

        log_probs = np.array(log_probs).T
        # 归一化为概率
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return self._response(1, content=probs)

    def predict(self, X):
        proba_resp = self.predict_proba(X)
        if proba_resp["status"] == 0:
            return proba_resp

        probs = proba_resp["content"]
        y_pred = self.classes_[np.argmax(probs, axis=1)]
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        try:
            y_true = np.array(y_true, dtype=np.int64)

            if y_pred is None:
                pred_resp = self.predict(X)
                if pred_resp["status"] == 0:
                    return pred_resp
                y_pred = pred_resp["content"]
            else:
                y_pred = np.array(y_pred, dtype=np.int64)

            if len(y_true) != len(y_pred):
                return self._response(0, content="y_true和y_pred样本数必须一致")

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_true, y_pred)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            acc_resp = self.accuracy(y_true, y_pred)
            acc = acc_resp['content']['accuracy'] if (acc_resp['status'] == 1) else 0.0

            prec_resp = self.precision(y_true, y_pred)
            prec = prec_resp['content']['precision'] if (prec_resp['status'] == 1) else 0.0

            rec_resp = self.recall(y_true, y_pred)
            rec = rec_resp['content']['recall'] if (rec_resp['status'] == 1) else 0.0

            f1_resp = self.f1(y_true, y_pred)
            f1 = f1_resp['content']['f1'] if (f1_resp['status'] == 1) else 0.0

            # 计算AUC
            auc = 0.0
            try:
                y_prob_resp = self.predict_proba(X)
                y_prob = y_prob_resp['content'] if (y_prob_resp['status'] == 1) else None
                if y_prob is not None:
                    # 对于二分类，使用第二个概率值作为正类概率
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    roc_resp = self.roc(y_true, y_prob_pos)
                    roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
                    if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                        auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                        if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                            auc_val = auc_resp['content']['auc']
                            if not np.isnan(auc_val) and not np.isinf(auc_val):
                                auc = auc_val
            except Exception as e:
                # AUC计算失败不影响其他指标
                pass

            # 构建metrics_per_output
            metrics_per_output = {
                "output_1": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "confusion_matrix": cm,
                    "classes": classes
                }
            }

            # 构建平均指标
            average_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc
            }

            metrics = {
                "metrics_per_output": metrics_per_output,
                "average_metrics": average_metrics,
                "n_outputs": 1,
                "n_samples": len(y_true)
            }
            return self._response(1, content=metrics)
        except Exception as e:
            return self._response(0, content=f"评估失败: {str(e)}")

# todo-2.17 build scikit-learn MultinomialNB
class MultinomialNB(Evaluation):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super().__init__()
        self.alpha = alpha  # 平滑参数，避免0概率
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_count_ = None  # 每个类别各特征的总计数
        self.class_count_ = None  # 每个类别的样本数
        self.class_log_prior_ = None  # 类别对数先验
        self.feature_log_prob_ = None  # 特征对数概率

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            # 确保 X 是二维的
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # 确保 y 是一维的
            if y.ndim > 1:
                y = y.ravel()

            # 校验：多项式NB仅支持非负特征
            if np.any(X < 0):
                return self._response(0, content="MultinomialNB仅支持非负特征值（如词频、计数）")

            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            # 计算类别计数和特征计数
            self.class_count_ = np.array([np.sum(y == c) for c in self.classes_])
            self.feature_count_ = np.array([np.sum(X[y == c], axis=0) for c in self.classes_])

            # 计算类别对数先验
            if self.fit_prior:
                if self.class_prior is None:
                    self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
                else:
                    self.class_log_prior_ = np.log(np.array(self.class_prior))
                    if len(self.class_log_prior_) != self.n_classes_:
                        return self._response(0, content="class_prior长度必须与类别数一致")
            else:
                self.class_log_prior_ = np.full(self.n_classes_, -np.log(self.n_classes_))

            # 计算特征对数概率（加平滑）
            smoothed_fc = self.feature_count_ + self.alpha
            smoothed_cc = np.sum(smoothed_fc, axis=1, keepdims=True)
            self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "classes_": self.classes_,
                    "alpha": self.alpha
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def predict_proba(self, X):
        if self.feature_log_prob_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 确保 X 是二维的
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # 确保特征维度匹配
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            return self._response(0, content=f"Feature dimension mismatch: expected {self.feature_log_prob_.shape[1]}, got {X.shape[1]}")
        # 对数后验概率：log(先验) + X·log(特征概率)
        log_probs = self.class_log_prior_ + np.dot(X, self.feature_log_prob_.T)

        # 归一化
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return self._response(1, content=probs)

    def predict(self, X):
        proba_resp = self.predict_proba(X)
        if proba_resp["status"] == 0:
            return proba_resp

        probs = proba_resp["content"]
        y_pred = self.classes_[np.argmax(probs, axis=1)]
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        y_true = np.array(y_true, dtype=np.int64)

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.int64)

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        # 计算混淆矩阵
        cm_resp = self.confusion_matrix(y_true, y_pred)
        cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
        classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

        acc_resp = self.accuracy(y_true, y_pred)
        prec_resp = self.precision(y_true, y_pred)
        rec_resp = self.recall(y_true, y_pred)
        f1_resp = self.f1(y_true, y_pred)

        # 构建metrics_per_output
        metrics_per_output = {
            "output_1": {
                "accuracy": acc_resp["content"]["accuracy"],
                "precision": prec_resp["content"]["precision"],
                "recall": rec_resp["content"]["recall"],
                "f1": f1_resp["content"]["f1"],
                "confusion_matrix": cm,
                "classes": classes
            }
        }

        metrics = {
            "accuracy": acc_resp["content"]["accuracy"],
            "precision": prec_resp["content"]["precision"],
            "recall": rec_resp["content"]["recall"],
            "f1": f1_resp["content"]["f1"],
            "n_samples": len(y_true),
            "metrics_per_output": metrics_per_output
        }
        return self._response(1, content=metrics)

# todo-2.18 build scikit-learn BernoulliNB
class BernoulliNB(Evaluation):
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        super().__init__()
        self.alpha = alpha  # 平滑参数
        self.binarize = binarize  # 二值化阈值
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
        self.feature_count_ = None
        self.class_count_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_log_neg_prob_ = None  # 特征不出现的对数概率

    def _binarize(self, X):
        """特征二值化：X > binarize → 1，否则 → 0"""
        if self.binarize is not None:
            return np.where(X > self.binarize, 1, 0)
        return X

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            # 确保 X 是二维的
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            # 确保 y 是一维的
            if y.ndim > 1:
                y = y.ravel()

            # 特征二值化
            X_bin = self._binarize(X)
            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            # 计算类别计数和特征计数
            self.class_count_ = np.array([np.sum(y == c) for c in self.classes_])
            self.feature_count_ = np.array([np.sum(X_bin[y == c], axis=0) for c in self.classes_])

            # 计算类别对数先验
            if self.fit_prior:
                if self.class_prior is None:
                    self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
                else:
                    self.class_log_prior_ = np.log(np.array(self.class_prior))
                    if len(self.class_log_prior_) != self.n_classes_:
                        return self._response(0, content="class_prior长度必须与类别数一致")
            else:
                self.class_log_prior_ = np.full(self.n_classes_, -np.log(self.n_classes_))

            # 计算特征对数概率（考虑出现/不出现两种情况）
            smoothed_fc = self.feature_count_ + self.alpha
            smoothed_cc = self.class_count_[:, np.newaxis] + 2 * self.alpha  # 0/1两种情况的平滑
            self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
            self.feature_log_neg_prob_ = np.log((smoothed_cc - smoothed_fc) / smoothed_cc)

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "classes_": self.classes_,
                    "alpha": self.alpha,
                    "binarize": self.binarize
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def predict_proba(self, X):
        if self.feature_log_prob_ is None:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 确保 X 是二维的
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # 确保特征维度匹配
        if X.shape[1] != self.feature_log_prob_.shape[1]:
            return self._response(0, content=f"Feature dimension mismatch: expected {self.feature_log_prob_.shape[1]}, got {X.shape[1]}")
        X_bin = self._binarize(X)
        # 对数后验概率：log(先验) + sum(x*log(p) + (1-x)*log(1-p))
        log_probs = self.class_log_prior_ + \
                    np.dot(X_bin, self.feature_log_prob_.T) + \
                    np.dot((1 - X_bin), self.feature_log_neg_prob_.T)

        # 归一化
        probs = np.exp(log_probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return self._response(1, content=probs)

    def predict(self, X):
        proba_resp = self.predict_proba(X)
        if proba_resp["status"] == 0:
            return proba_resp

        probs = proba_resp["content"]
        y_pred = self.classes_[np.argmax(probs, axis=1)]
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        try:
            y_true = np.array(y_true, dtype=np.int64)

            if y_pred is None:
                pred_resp = self.predict(X)
                if pred_resp["status"] == 0:
                    return pred_resp
                y_pred = pred_resp["content"]
            else:
                y_pred = np.array(y_pred, dtype=np.int64)

            if len(y_true) != len(y_pred):
                return self._response(0, content="y_true和y_pred样本数必须一致")

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_true, y_pred)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            acc_resp = self.accuracy(y_true, y_pred)
            acc = acc_resp['content']['accuracy'] if (acc_resp['status'] == 1) else 0.0

            prec_resp = self.precision(y_true, y_pred)
            prec = prec_resp['content']['precision'] if (prec_resp['status'] == 1) else 0.0

            rec_resp = self.recall(y_true, y_pred)
            rec = rec_resp['content']['recall'] if (rec_resp['status'] == 1) else 0.0

            f1_resp = self.f1(y_true, y_pred)
            f1 = f1_resp['content']['f1'] if (f1_resp['status'] == 1) else 0.0

            # 计算AUC
            auc = 0.0
            try:
                y_prob_resp = self.predict_proba(X)
                y_prob = y_prob_resp['content'] if (y_prob_resp['status'] == 1) else None
                if y_prob is not None:
                    # 对于二分类，使用第二个概率值作为正类概率
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    roc_resp = self.roc(y_true, y_prob_pos)
                    roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
                    if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                        auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                        if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                            auc_val = auc_resp['content']['auc']
                            if not np.isnan(auc_val) and not np.isinf(auc_val):
                                auc = auc_val
            except Exception as e:
                # AUC计算失败不影响其他指标
                pass

            # 构建metrics_per_output
            metrics_per_output = {
                "output_1": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "confusion_matrix": cm,
                    "classes": classes
                }
            }

            # 构建平均指标
            average_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc
            }

            metrics = {
                "metrics_per_output": metrics_per_output,
                "average_metrics": average_metrics,
                "n_outputs": 1,
                "n_samples": len(y_true)
            }
            return self._response(1, content=metrics)
        except Exception as e:
            return self._response(0, content=f"评估失败: {str(e)}")

# todo-2.19 build scikit-learn RandomForestClassifier
class RandomForestClassifier(Evaluation):
    def __init__(
            self,
            n_estimators=100,
            criterion='gini',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=None,
            verbose=False,
            n_jobs=1,
            early_stopping=False,
            n_iter_no_change=5,
            validation_fraction=0.1
    ):
        super().__init__()
        self.n_estimators = n_estimators  # 决策树数量
        self.criterion = criterion  # 不纯度准则
        self.max_depth = max_depth  # 单树最大深度
        self.min_samples_split = min_samples_split  # 分割最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶节点最小样本数
        self.max_features = max_features  # 每棵树选择的最大特征数
        self.bootstrap = bootstrap  # 是否bootstrap采样
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs  # 并行训练的进程数
        self.early_stopping = early_stopping  # 早期停止
        self.n_iter_no_change = n_iter_no_change  # 无改进的迭代次数
        self.validation_fraction = validation_fraction  # 验证集比例

        # 模型参数
        self.trees_ = []  # 所有决策树模型
        self.classes_ = None  # 类别标签
        self.n_classes_ = None  # 类别数
        self.n_features_in_ = None  # 输入特征数

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            if self.random_state is not None:
                np.random.seed(self.random_state)
            rng = np.random.RandomState(self.random_state)

            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            n_samples = len(X)

            # 优化：进一步减少默认树的数量，加快训练速度
            if self.n_estimators > 50:
                self.n_estimators = 50

            # 确定每棵树的最大特征数
            if self.max_features == 'sqrt':
                max_feat = int(np.sqrt(self.n_features_in_))
            elif self.max_features == 'log2':
                max_feat = int(np.log2(self.n_features_in_))
            elif self.max_features is None:
                max_feat = self.n_features_in_
            else:
                max_feat = self.max_features

            # 分割验证集用于早期停止
            if self.early_stopping:
                val_size = int(n_samples * self.validation_fraction)
                idx = rng.permutation(n_samples)
                X_train = X[idx[val_size:]]
                y_train = y[idx[val_size:]]
                X_val = X[idx[:val_size]]
                y_val = y[idx[:val_size]]
                n_samples = len(X_train)
            else:
                X_train = X
                y_train = y

            # 训练多棵决策树
            best_accuracy = 0
            no_improvement = 0
            
            if self.n_jobs > 1:
                # 使用多进程并行构建树
                from multiprocessing import Pool
                import os
                
                # 限制进程数不超过CPU核心数
                n_processes = min(self.n_jobs, os.cpu_count())
                
                # 生成bootstrap样本的索引
                bootstrap_indices = []
                random_states = []
                for _ in range(self.n_estimators):
                    if self.bootstrap:
                        indices = rng.choice(n_samples, n_samples, replace=True)
                    else:
                        indices = np.arange(n_samples)
                    bootstrap_indices.append(indices)
                    random_states.append(rng.randint(0, 10000))
                
                # 定义构建树的函数
                def build_tree_wrapper(args):
                    indices, random_state = args
                    X_sample = X_train[indices]
                    y_sample = y_train[indices]
                    tree = DecisionTreeClassifier(
                        criterion=self.criterion,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=max_feat,
                        random_state=random_state
                    )
                    tree.fit(X_sample, y_sample)
                    return tree
                
                # 使用进程池并行构建树
                with Pool(processes=n_processes) as pool:
                    self.trees_ = pool.map(build_tree_wrapper, zip(bootstrap_indices, random_states))
            else:
                # 串行构建树
                for i in range(self.n_estimators):
                    if self.verbose and i % 10 == 0:
                        # print(f"Training tree {i + 1}/{self.n_estimators}")
                        pass

                    # 1. Bootstrap采样（有放回）
                    if self.bootstrap:
                        sample_idx = rng.choice(n_samples, n_samples, replace=True)
                        X_sample = X_train[sample_idx]
                        y_sample = y_train[sample_idx]
                    else:
                        X_sample = X_train
                        y_sample = y_train

                    # 2. 初始化单棵决策树（随机选特征）
                    tree = DecisionTreeClassifier(
                        criterion=self.criterion,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        max_features=max_feat,
                        random_state=rng.randint(0, 10000)
                    )
                    # 调用fit方法，但不存储响应对象
                    tree.fit(X_sample, y_sample)
                    # 确保tree是一个有效的决策树模型
                    if hasattr(tree, 'tree_'):
                        self.trees_.append(tree)
                        
                        # 早期停止检查
                        if self.early_stopping and len(self.trees_) % 5 == 0:
                            # 计算当前验证准确率
                            val_pred = self.predict(X_val)['content']
                            current_accuracy = np.mean(val_pred == y_val)
                            if current_accuracy > best_accuracy:
                                best_accuracy = current_accuracy
                                no_improvement = 0
                            else:
                                no_improvement += 1
                                if no_improvement >= self.n_iter_no_change:
                                    if self.verbose:
                                        pass
                                        # print(f"Early stopping after {len(self.trees_)} trees")
                                    break

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "n_estimators": len(self.trees_),
                    "max_depth": self.max_depth,
                    "bootstrap": self.bootstrap,
                    "n_jobs": self.n_jobs,
                    "early_stopping": self.early_stopping
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def predict_proba(self, X):
        if len(self.trees_) == 0:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 收集所有树的概率预测
        all_probs = []
        for tree in self.trees_:
            try:
                prob_resp = tree.predict_proba(X)
                if prob_resp and isinstance(prob_resp, dict) and "status" in prob_resp:
                    if prob_resp["status"] == 1:
                        prob_content = prob_resp.get("content", [])
                        # 确保概率内容是有效的数组
                        if isinstance(prob_content, np.ndarray) and len(prob_content) > 0:
                            all_probs.append(prob_content)
            except Exception as e:
                continue  # 跳过出错的树

        # 确保有有效的概率预测
        if not all_probs:
            # 如果没有有效的预测，返回默认概率
            n_samples = X.shape[0]
            default_probs = np.ones((n_samples, self.n_classes_)) / self.n_classes_
            return self._response(1, content=default_probs)

        # 概率平均（软投票）
        avg_probs = np.mean(np.array(all_probs), axis=0)
        return self._response(1, content=avg_probs)

    def predict(self, X):
        proba_resp = self.predict_proba(X)
        if proba_resp["status"] == 0:
            return proba_resp

        avg_probs = proba_resp["content"]
        y_pred = self.classes_[np.argmax(avg_probs, axis=1)]
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        try:
            y_true = np.array(y_true, dtype=np.int64)

            if y_pred is None:
                pred_resp = self.predict(X)
                if pred_resp["status"] == 0:
                    return pred_resp
                y_pred = pred_resp["content"]
            else:
                y_pred = np.array(y_pred, dtype=np.int64)

            if len(y_true) != len(y_pred):
                return self._response(0, content="y_true和y_pred样本数必须一致")

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_true, y_pred)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            acc_resp = self.accuracy(y_true, y_pred)
            acc = acc_resp['content']['accuracy'] if (acc_resp['status'] == 1) else 0.0

            prec_resp = self.precision(y_true, y_pred)
            prec = prec_resp['content']['precision'] if (prec_resp['status'] == 1) else 0.0

            rec_resp = self.recall(y_true, y_pred)
            rec = rec_resp['content']['recall'] if (rec_resp['status'] == 1) else 0.0

            f1_resp = self.f1(y_true, y_pred)
            f1 = f1_resp['content']['f1'] if (f1_resp['status'] == 1) else 0.0

            # 计算AUC
            auc = 0.0
            try:
                y_prob_resp = self.predict_proba(X)
                y_prob = y_prob_resp['content'] if (y_prob_resp['status'] == 1) else None
                if y_prob is not None:
                    # 对于二分类，使用第二个概率值作为正类概率
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    roc_resp = self.roc(y_true, y_prob_pos)
                    roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
                    if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                        auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                        if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                            auc_val = auc_resp['content']['auc']
                            if not np.isnan(auc_val) and not np.isinf(auc_val):
                                auc = auc_val
            except Exception as e:
                # AUC计算失败不影响其他指标
                pass

            # 构建metrics_per_output
            metrics_per_output = {
                "output_1": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "confusion_matrix": cm,
                    "classes": classes
                }
            }

            # 构建平均指标
            average_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc
            }

            metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc,
                "n_estimators": self.n_estimators,
                "n_samples": len(y_true),
                "metrics_per_output": metrics_per_output,
                "average_metrics": average_metrics,
                "n_outputs": 1
            }
            return self._response(1, content=metrics)
        except Exception as e:
            return self._response(0, content=f"评估失败: {str(e)}")

# todo-2.20 build scikit-learn RandomForestRegressor
class RandomForestRegressor(Evaluation):
    def __init__(
            self,
            n_estimators=100,
            criterion='mse',
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=None,
            verbose=False,
            n_jobs=1,
            early_stopping=False,
            n_iter_no_change=5,
            validation_fraction=0.1
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction

        self.trees_ = []
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        self.n_features_in_ = X.shape[1]
        n_samples = len(X)

        # 优化：进一步减少默认树的数量，加快训练速度
        if self.n_estimators > 50:
            self.n_estimators = 50

        # 确定每棵树的最大特征数
        if self.max_features == 'sqrt':
            max_feat = int(np.sqrt(self.n_features_in_))
        elif self.max_features == 'log2':
            max_feat = int(np.log2(self.n_features_in_))
        elif self.max_features is None:
            max_feat = self.n_features_in_
        else:
            max_feat = self.max_features

        # 分割验证集用于早期停止
        if self.early_stopping:
            val_size = int(n_samples * self.validation_fraction)
            idx = rng.permutation(n_samples)
            X_train = X[idx[val_size:]]
            y_train = y[idx[val_size:]]
            X_val = X[idx[:val_size]]
            y_val = y[idx[:val_size]]
            n_samples = len(X_train)
        else:
            X_train = X
            y_train = y

        # 训练多棵决策树
        best_mse = float('inf')
        no_improvement = 0
        
        if self.n_jobs > 1:
            # 使用多进程并行构建树
            from multiprocessing import Pool
            import os
            
            # 限制进程数不超过CPU核心数
            n_processes = min(self.n_jobs, os.cpu_count())
            
            # 生成bootstrap样本的索引
            bootstrap_indices = []
            random_states = []
            for _ in range(self.n_estimators):
                if self.bootstrap:
                    indices = rng.choice(n_samples, n_samples, replace=True)
                else:
                    indices = np.arange(n_samples)
                bootstrap_indices.append(indices)
                random_states.append(rng.randint(0, 10000))
            
            # 定义构建树的函数
            def build_tree_wrapper(args):
                indices, random_state = args
                X_sample = X_train[indices]
                y_sample = y_train[indices]
                tree = DecisionTreeRegressor(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=max_feat,
                    random_state=random_state
                )
                tree.fit(X_sample, y_sample)
                return tree
            
            # 使用进程池并行构建树
            with Pool(processes=n_processes) as pool:
                self.trees_ = pool.map(build_tree_wrapper, zip(bootstrap_indices, random_states))
        else:
            # 串行构建树
            for i in range(self.n_estimators):
                if self.verbose and i % 10 == 0:
                    # print(f"Training tree {i + 1}/{self.n_estimators}")
                    pass

                # Bootstrap采样
                if self.bootstrap:
                    sample_idx = rng.choice(n_samples, n_samples, replace=True)
                    X_sample = X_train[sample_idx]
                    y_sample = y_train[sample_idx]
                else:
                    X_sample = X_train
                    y_sample = y_train

                # 初始化单棵决策树
                tree = DecisionTreeRegressor(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=max_feat,
                    random_state=rng.randint(0, 10000)
                )
                tree.fit(X_sample, y_sample)
                self.trees_.append(tree)
                
                # 早期停止检查
                if self.early_stopping and len(self.trees_) % 5 == 0:
                    # 计算当前验证MSE
                    val_pred = self.predict(X_val)['content']
                    current_mse = np.mean((val_pred - y_val) ** 2)
                    if current_mse < best_mse:
                        best_mse = current_mse
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= self.n_iter_no_change:
                            if self.verbose:
                                pass
                                # print(f"Early stopping after {len(self.trees_)} trees")
                            break

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "n_estimators": len(self.trees_),
                "max_depth": self.max_depth,
                "bootstrap": self.bootstrap,
                "early_stopping": self.early_stopping
            }
        )

    def predict(self, X):
        if len(self.trees_) == 0:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 收集所有树的预测值
        all_preds = []
        for tree in self.trees_:
            pred_resp = tree.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            all_preds.append(pred_resp["content"])

        # 预测值平均
        avg_preds = np.mean(np.array(all_preds), axis=0)
        return self._response(1, content=avg_preds)

    def evaluate(self, X, y_true, y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_estimators": self.n_estimators,
            "n_samples": len(y_true)
        }
        return self._response(1, content=metrics)

# todo-2.21 build scikit-learn AdaBoostClassifier
class AdaBoostClassifier(Evaluation):
    def __init__(
            self,
            n_estimators=50,
            learning_rate=1.0,
            algorithm='SAMME.R',
            random_state=None
    ):
        super().__init__()
        self.n_estimators = n_estimators  # 弱学习器数量
        self.learning_rate = learning_rate  # 学习率（权重衰减）
        self.algorithm = algorithm  # 算法：SAMME/SAMME.R
        self.random_state = random_state

        # 模型参数
        self.estimators_ = []  # 弱学习器列表
        self.estimator_weights_ = []  # 弱学习器权重
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            if self.random_state is not None:
                np.random.seed(self.random_state)
            rng = np.random.RandomState(self.random_state)

            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)
            n_samples = len(X)

            # 1. 初始化样本权重（均匀分布）
            sample_weight = np.ones(n_samples) / n_samples

            # 2. 迭代训练弱学习器（默认用决策树桩：max_depth=1）
            for i in range(self.n_estimators):
                # 训练单棵弱学习器（带样本权重）
                estimator = DecisionTreeClassifier(
                    max_depth=1,  # 决策树桩（AdaBoost默认弱学习器）
                    random_state=rng.randint(0, 10000)
                )
                estimator.fit(X, y)
                y_pred = estimator.predict(X)["content"]

                # 3. 计算加权错误率
                incorrect = (y_pred != y)
                error = np.sum(sample_weight * incorrect) / np.sum(sample_weight)

                # 错误率过高则停止（弱学习器无价值）
                if error >= 1 - 1 / self.n_classes_:
                    break

                # 4. 计算弱学习器权重
                if self.algorithm == 'SAMME':
                    estimator_weight = self.learning_rate * np.log((1 - error) / error) + np.log(self.n_classes_ - 1)
                else:  # SAMME.R（概率版，收敛更快）
                    prob_pred = estimator.predict_proba(X)["content"]
                    # 修正概率避免log(0)
                    prob_pred = np.clip(prob_pred, 1e-10, 1 - 1e-10)
                    # 计算权重（基于对数概率）
                    log_prob = np.log(prob_pred[np.arange(n_samples), y])
                    estimator_weight = self.learning_rate * (log_prob / np.sum(log_prob))

                # 5. 更新样本权重
                if self.algorithm == 'SAMME':
                    sample_weight *= np.exp(estimator_weight * incorrect)
                else:
                    # SAMME.R 权重更新（基于概率）
                    sample_weight *= np.exp(-estimator_weight * np.log(prob_pred[np.arange(n_samples), y]))

                # 归一化样本权重
                sample_weight /= np.sum(sample_weight)

                # 保存弱学习器和权重
                self.estimators_.append(estimator)
                # 确保estimator_weight是标量
                if isinstance(estimator_weight, np.ndarray):
                    estimator_weight = np.mean(estimator_weight)
                self.estimator_weights_.append(estimator_weight)

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "n_estimators_used": len(self.estimators_),
                    "algorithm": self.algorithm,
                    "learning_rate": self.learning_rate
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def predict_proba(self, X):
        if len(self.estimators_) == 0:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 收集所有弱学习器的概率预测
        all_probs = []
        for estimator in self.estimators_:
            try:
                prob_resp = estimator.predict_proba(X)
                if prob_resp and isinstance(prob_resp, dict) and "status" in prob_resp:
                    if prob_resp["status"] == 0:
                        continue  # 跳过预测失败的弱学习器
                    prob_content = prob_resp.get("content", [])
                    if isinstance(prob_content, np.ndarray) and len(prob_content) > 0:
                        all_probs.append(prob_content)
            except Exception as e:
                continue  # 跳过出错的弱学习器

        # 确保有有效的概率预测
        if not all_probs:
            # 如果没有有效的预测，返回默认概率
            n_samples = X.shape[0]
            default_probs = np.ones((n_samples, self.n_classes_)) / self.n_classes_
            return self._response(1, content=default_probs)

        # 加权平均概率（SAMME.R）
        if self.algorithm == 'SAMME.R':
            # 确保权重是一维的
            weights = np.array(self.estimator_weights_)
            if weights.ndim > 1:
                weights = weights.ravel()
            # 确保权重长度与学习器数量匹配
            if len(weights) != len(all_probs):
                weights = np.ones(len(all_probs)) / len(all_probs)
            avg_probs = np.average(np.array(all_probs), axis=0, weights=weights)
        else:  # SAMME（硬投票加权）
            all_preds = []
            for estimator in self.estimators_:
                try:
                    pred_resp = estimator.predict(X)
                    if pred_resp and isinstance(pred_resp, dict) and "status" in pred_resp:
                        if pred_resp["status"] == 1:
                            pred_content = pred_resp.get("content", [])
                            if isinstance(pred_content, np.ndarray) and len(pred_content) > 0:
                                all_preds.append(np.eye(self.n_classes_)[pred_content])
                except Exception as e:
                    continue  # 跳过出错的弱学习器
            
            # 确保有有效的预测
            if not all_preds:
                # 如果没有有效的预测，返回默认概率
                n_samples = X.shape[0]
                default_probs = np.ones((n_samples, self.n_classes_)) / self.n_classes_
                return self._response(1, content=default_probs)
                
            # 确保权重是一维的
            weights = np.array(self.estimator_weights_)
            if weights.ndim > 1:
                weights = weights.ravel()
            # 确保权重长度与学习器数量匹配
            if len(weights) != len(all_preds):
                weights = np.ones(len(all_preds)) / len(all_preds)
            avg_probs = np.average(np.array(all_preds), axis=0, weights=weights)

        # 归一化概率
        avg_probs /= np.sum(avg_probs, axis=1, keepdims=True)
        return self._response(1, content=avg_probs)

    def predict(self, X):
        proba_resp = self.predict_proba(X)
        if proba_resp["status"] == 0:
            return proba_resp

        avg_probs = proba_resp["content"]
        y_pred = self.classes_[np.argmax(avg_probs, axis=1)]
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        try:
            y_true = np.array(y_true, dtype=np.int64)

            if y_pred is None:
                pred_resp = self.predict(X)
                if pred_resp["status"] == 0:
                    return pred_resp
                y_pred = pred_resp["content"]
            else:
                y_pred = np.array(y_pred, dtype=np.int64)

            if len(y_true) != len(y_pred):
                return self._response(0, content="y_true和y_pred样本数必须一致")

            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_true, y_pred)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            acc_resp = self.accuracy(y_true, y_pred)
            acc = acc_resp['content']['accuracy'] if (acc_resp['status'] == 1) else 0.0

            prec_resp = self.precision(y_true, y_pred)
            prec = prec_resp['content']['precision'] if (prec_resp['status'] == 1) else 0.0

            rec_resp = self.recall(y_true, y_pred)
            rec = rec_resp['content']['recall'] if (rec_resp['status'] == 1) else 0.0

            f1_resp = self.f1(y_true, y_pred)
            f1 = f1_resp['content']['f1'] if (f1_resp['status'] == 1) else 0.0

            # 计算AUC
            auc = 0.0
            try:
                y_prob_resp = self.predict_proba(X)
                y_prob = y_prob_resp['content'] if (y_prob_resp['status'] == 1) else None
                if y_prob is not None:
                    # 对于二分类，使用第二个概率值作为正类概率
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                        y_prob_pos = y_prob[:, 1]
                    else:
                        y_prob_pos = y_prob
                    roc_resp = self.roc(y_true, y_prob_pos)
                    roc_res = roc_resp['content']['roc'] if (roc_resp['status'] == 1 and 'roc' in roc_resp['content']) else {}
                    if roc_res and len(roc_res.get('fpr', [])) > 0 and len(roc_res.get('tpr', [])) > 0:
                        auc_resp = self.auc(roc_res['fpr'], roc_res['tpr'])
                        if auc_resp['status'] == 1 and 'auc' in auc_resp['content']:
                            auc_val = auc_resp['content']['auc']
                            if not np.isnan(auc_val) and not np.isinf(auc_val):
                                auc = auc_val
            except Exception as e:
                # AUC计算失败不影响其他指标
                pass

            # 构建metrics_per_output
            metrics_per_output = {
                "output_1": {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "confusion_matrix": cm,
                    "classes": classes
                }
            }

            # 构建平均指标
            average_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc
            }

            metrics = {
                "metrics_per_output": metrics_per_output,
                "average_metrics": average_metrics,
                "n_outputs": 1,
                "n_estimators_used": len(self.estimators_),
                "n_samples": len(y_true)
            }
            return self._response(1, content=metrics)
        except Exception as e:
            return self._response(0, content=f"评估失败: {str(e)}")

# todo-2.22 build scikit-learn AdaBoostRegressor
class AdaBoostRegressor(Evaluation):
    def __init__(
            self,
            n_estimators=50,
            learning_rate=1.0,
            loss='linear',
            random_state=None
    ):
        super().__init__()
        self.n_estimators = n_estimators  # 弱学习器数量
        self.learning_rate = learning_rate  # 学习率
        self.loss = loss  # 损失函数：linear/square/exponential
        self.random_state = random_state

        self.estimators_ = []  # 弱学习器列表
        self.estimator_weights_ = []  # 弱学习器权重
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        self.n_features_in_ = X.shape[1]
        n_samples = len(X)

        # 初始化样本权重
        sample_weight = np.ones(n_samples) / n_samples
        y_pred_total = np.zeros(n_samples)

        for i in range(self.n_estimators):
            # 训练单棵决策树桩
            estimator = DecisionTreeRegressor(
                max_depth=1,
                random_state=rng.randint(0, 10000)
            )
            estimator.fit(X, y)
            y_pred = estimator.predict(X)["content"]

            # 计算加权误差
            residual = np.abs(y - y_pred)
            weighted_error = np.sum(sample_weight * residual) / np.sum(sample_weight)

            # 计算学习器权重
            if weighted_error >= 0.5:
                estimator_weight = self.learning_rate * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            else:
                estimator_weight = self.learning_rate

            # 更新样本权重
            if self.loss == 'linear':
                sample_weight *= np.exp(estimator_weight * residual)
            elif self.loss == 'square':
                sample_weight *= np.exp(estimator_weight * residual ** 2)
            elif self.loss == 'exponential':
                sample_weight *= np.exp(estimator_weight * np.abs(residual))

            # 归一化
            sample_weight /= np.sum(sample_weight)

            # 保存模型
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            y_pred_total += estimator_weight * y_pred

        return self._response(1, content={
            "n_features_in_": self.n_features_in_,
            "n_estimators_used": len(self.estimators_),
            "loss": self.loss,
            "learning_rate": self.learning_rate
        })

    def predict(self, X):
        if len(self.estimators_) == 0:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 加权求和所有弱学习器的预测
        y_pred_total = 0.0
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            pred_resp = estimator.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred_total += weight * pred_resp["content"]

        return self._response(1, content=y_pred_total)

    def evaluate(self, X, y_true, y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_estimators_used": len(self.estimators_),
            "n_samples": len(y_true)
        }
        return self._response(1, content=metrics)

# todo-2.23 build scikit-learn XGBoostTree
class XGBoostTree:
    """XGBoost 单棵回归树（带正则+预剪枝）"""
    def __init__(self, max_depth=3, learning_rate=0.1, reg_lambda=1.0, reg_alpha=0.0):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda  # L2 正则
        self.reg_alpha = reg_alpha    # L1 正则
        self.tree = {}  # 存储树结构
        self.leaf_vals = {}  # 叶节点值（带正则的最优值）

    def _calc_leaf_val(self, grad, hess):
        """计算叶节点最优值（带L1/L2正则）"""
        sum_grad = np.sum(grad)
        sum_hess = np.sum(hess)
        # XGBoost 叶节点值公式：-G/(H+λ)，带L1正则时需调整
        if sum_hess < 1e-10:
            return 0.0
        return - (sum_grad + self.reg_alpha * np.sign(sum_grad)) / (sum_hess + self.reg_lambda)

    def _split_gain(self, grad, hess, left_grad, left_hess, right_grad, right_hess):
        """计算分裂增益（XGBoost 核心：衡量分裂是否有价值）"""
        def gain(g, h):
            return (g**2) / (h + self.reg_lambda)
        total_gain = gain(np.sum(grad), np.sum(hess))
        left_gain = gain(np.sum(left_grad), np.sum(left_hess))
        right_gain = gain(np.sum(right_grad), np.sum(right_hess))
        return left_gain + right_gain - total_gain

    def _best_split(self, X, grad, hess):
        """找最优分裂特征和阈值（带预剪枝）"""
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = -1
        best_threshold = None

        for feat in range(n_features):
            # 去重+排序，减少分裂阈值数量
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                # 划分左右子树
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask

                left_grad = grad[left_mask]
                left_hess = hess[left_mask]
                right_grad = grad[right_mask]
                right_hess = hess[right_mask]

                # 计算分裂增益
                gain = self._split_gain(grad, hess, left_grad, left_hess, right_grad, right_hess)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thresh

        # 预剪枝：增益不足则不分裂
        if best_gain < 1e-6:
            return None, None
        return best_feature, best_threshold

    def _build_tree(self, X, grad, hess, depth, node_id):
        """递归构建树（带最大深度限制）"""
        # 终止条件：达到最大深度 或 样本数过少 或 无分裂增益
        if depth >= self.max_depth or len(X) < 2:
            self.leaf_vals[node_id] = self._calc_leaf_val(grad, hess)
            return

        # 找最优分裂
        best_feat, best_thresh = self._best_split(X, grad, hess)
        if best_feat is None:
            self.leaf_vals[node_id] = self._calc_leaf_val(grad, hess)
            return

        # 存储分裂节点信息
        self.tree[node_id] = (best_feat, best_thresh)

        # 划分左右子树
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        # 递归构建左右子树
        self._build_tree(X[left_mask], grad[left_mask], hess[left_mask], depth+1, node_id*2)
        self._build_tree(X[right_mask], grad[right_mask], hess[right_mask], depth+1, node_id*2+1)

    def fit(self, X, grad, hess):
        """训练单棵XGBoost树"""
        # 检查长度是否匹配
        if len(X) != len(grad) or len(X) != len(hess):
            raise ValueError(f"Length mismatch: X={len(X)}, grad={len(grad)}, hess={len(hess)}")
        self._build_tree(X, grad, hess, depth=0, node_id=1)
        return self

    def _predict_sample(self, x, node_id):
        """单样本预测"""
        if node_id in self.leaf_vals:
            return self.leaf_vals[node_id] * self.learning_rate
        # 非叶节点，继续遍历
        feat, thresh = self.tree[node_id]
        if x[feat] <= thresh:
            return self._predict_sample(x, node_id*2)
        else:
            return self._predict_sample(x, node_id*2+1)

    def predict(self, X):
        """批量预测"""
        return np.array([self._predict_sample(x, node_id=1) for x in X])

# todo-2.24 build scikit-learn XGBClassifier
class XGBClassifier(Evaluation):
    def __init__(
            self,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective='binary:logistic',
            random_state=None,
            verbose=False
    ):
        super().__init__()
        self.n_estimators = n_estimators  # 树的数量
        self.max_depth = max_depth  # 单树最大深度
        self.learning_rate = learning_rate  # 学习率（步长）
        self.reg_lambda = reg_lambda  # L2 正则（权重衰减）
        self.reg_alpha = reg_alpha  # L1 正则（特征选择）
        self.objective = objective  # 目标函数
        self.random_state = random_state
        self.verbose = verbose

        # 模型参数
        self.trees_ = []  # 所有XGBoost树
        self.classes_ = None  # 类别标签
        self.n_classes_ = None  # 类别数
        self.n_features_in_ = None  # 输入特征数

    def fit(self, X, y):
        try:
            X = np.array(X, dtype=np.float64)
            y = np.array(y, dtype=np.int64)

            # 检查输入数据
            if len(X.shape) != 2:
                return self._response(0, content="X必须是二维数组")
            # 确保y是一维数组
            if len(y.shape) == 2 and y.shape[1] == 1:
                y = y.reshape(-1)
            if len(y.shape) != 1:
                return self._response(0, content="y必须是一维数组")
            if X.shape[0] != y.shape[0]:
                return self._response(0, content="X和y的样本数必须一致")

            if self.random_state is not None:
                np.random.seed(self.random_state)

            self.n_features_in_ = X.shape[1]
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            # 处理目标函数和标签
            if self.objective == 'binary:logistic' and self.n_classes_ == 2:
                # 二分类：标签转0/1
                y_train = np.where(y == self.classes_[0], 0, 1)
                loss_type = 'binary'
            elif self.objective == 'multi:softmax' and self.n_classes_ > 2:
                # 多分类：one-hot编码
                y_train = np.eye(self.n_classes_)[y]
                loss_type = 'multiclass'
            else:
                # 默认使用二分类
                y_train = np.where(y == self.classes_[0], 0, 1)
                loss_type = 'binary'

            # 初始化预测值（全0）
            y_pred = np.zeros_like(y_train, dtype=np.float64)

            # 串行训练每棵树
            for i in range(self.n_estimators):
                if self.verbose and i % 10 == 0:
                    # print(f"Training XGBoost tree {i + 1}/{self.n_estimators}")
                    pass

                # 1. 计算一阶/二阶梯度（泰勒展开核心）
                grad, hess = xgb_loss_gradient(y_train, y_pred, loss_type)
                
                # 2. 训练单棵XGBoost树
                tree = XGBoostTree(
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    reg_lambda=self.reg_lambda,
                    reg_alpha=self.reg_alpha
                )
                
                # 3. 处理多分类情况
                if len(grad.shape) == 2:  # 多分类
                    # 为每个类别训练一棵子树
                    tree_pred = np.zeros_like(y_pred)
                    for cls in range(self.n_classes_):
                        # 提取当前类别的梯度和海森矩阵
                        cls_grad = grad[:, cls]
                        cls_hess = hess[:, cls]
                        # 训练树
                        tree.fit(X, cls_grad, cls_hess)
                        # 预测并累加到对应类别
                        tree_pred[:, cls] = tree.predict(X)
                        # 保存树
                        self.trees_.append(tree)
                else:  # 二分类
                    # 直接训练树
                    tree.fit(X, grad, hess)
                    self.trees_.append(tree)
                    # 预测
                    tree_pred = tree.predict(X)
                
                # 4. 更新预测值（累加当前树的预测）
                y_pred += tree_pred

            return self._response(
                1,
                content={
                    "n_features_in_": self.n_features_in_,
                    "n_classes_": self.n_classes_,
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "reg_lambda": self.reg_lambda,
                    "reg_alpha": self.reg_alpha
                }
            )
        except Exception as e:
            return self._response(0, content=f"训练失败: {str(e)}")

    def predict_proba(self, X):
        if len(self.trees_) == 0:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 确保 X 是二维的
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = len(X)
        # 累加所有树的预测值
        if self.n_classes_ > 2:  # 多分类
            y_pred = np.zeros((n_samples, self.n_classes_))
            # 为每个类别单独累加预测值
            tree_idx = 0
            for i in range(self.n_estimators):
                for cls in range(self.n_classes_):
                    if tree_idx < len(self.trees_):
                        tree = self.trees_[tree_idx]
                        tree_pred = tree.predict(X)
                        # 确保tree_pred是标量或一维数组
                        if isinstance(tree_pred, dict) and 'content' in tree_pred:
                            tree_pred = tree_pred['content']
                        if isinstance(tree_pred, np.ndarray):
                            if tree_pred.ndim == 1 and len(tree_pred) == n_samples:
                                y_pred[:, cls] += tree_pred
                            elif tree_pred.ndim == 0:
                                y_pred[:, cls] += tree_pred
                        tree_idx += 1
        else:  # 二分类
            y_pred = np.zeros(n_samples)
            # 直接累加所有树的预测值
            for tree in self.trees_:
                tree_pred = tree.predict(X)
                # 确保tree_pred是标量或一维数组
                if isinstance(tree_pred, dict) and 'content' in tree_pred:
                    tree_pred = tree_pred['content']
                if isinstance(tree_pred, np.ndarray):
                    if tree_pred.ndim == 1 and len(tree_pred) == n_samples:
                        y_pred += tree_pred
                    elif tree_pred.ndim == 0:
                        y_pred += tree_pred

        # 转换为概率
        if self.n_classes_ == 2:
            probs = sigmoid(y_pred)
            # 二分类返回 [1-p, p] 格式
            if len(probs.shape) == 1:
                probs = np.vstack([1 - probs, probs]).T
            else:
                # 确保 probs 是二维的
                probs = probs.reshape(-1, 1)
                probs = np.hstack([1 - probs, probs])
        elif self.n_classes_ > 2:
            probs = softmax(y_pred)
        else:
            # 处理单类别情况
            probs = np.ones((n_samples, 1))

        return self._response(1, content=probs)

    def predict(self, X):
        proba_resp = self.predict_proba(X)
        if proba_resp["status"] == 0:
            return proba_resp

        probs = proba_resp["content"]
        y_pred = self.classes_[np.argmax(probs, axis=1)]
        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None, threshold=0.5):
        y_true = np.array(y_true, dtype=np.int64)

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.int64)

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        try:
            # 计算混淆矩阵
            cm_resp = self.confusion_matrix(y_true, y_pred)
            cm = cm_resp['content']['confusion_matrix'] if (cm_resp['status'] == 1 and 'confusion_matrix' in cm_resp['content']) else []
            classes = cm_resp['content']['classes'] if (cm_resp['status'] == 1 and 'classes' in cm_resp['content']) else []

            acc_resp = self.accuracy(y_true, y_pred)
            if acc_resp["status"] == 0:
                return acc_resp
            prec_resp = self.precision(y_true, y_pred)
            if prec_resp["status"] == 0:
                return prec_resp
            rec_resp = self.recall(y_true, y_pred)
            if rec_resp["status"] == 0:
                return rec_resp
            f1_resp = self.f1(y_true, y_pred)
            if f1_resp["status"] == 0:
                return f1_resp

            # 构建metrics_per_output
            metrics_per_output = {
                "output_1": {
                    "accuracy": acc_resp["content"]["accuracy"],
                    "precision": prec_resp["content"]["precision"],
                    "recall": rec_resp["content"]["recall"],
                    "f1": f1_resp["content"]["f1"],
                    "confusion_matrix": cm,
                    "classes": classes
                }
            }

            metrics = {
                "accuracy": acc_resp["content"]["accuracy"],
                "precision": prec_resp["content"]["precision"],
                "recall": rec_resp["content"]["recall"],
                "f1": f1_resp["content"]["f1"],
                "n_estimators": self.n_estimators,
                "n_samples": len(y_true),
                "metrics_per_output": metrics_per_output
            }
            return self._response(1, content=metrics)
        except Exception as e:
            return self._response(0, content=f"评估失败: {str(e)}")

# todo-2.25 build scikit-learn XGBRegressor
class XGBRegressor(Evaluation):
    def __init__(
            self,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            objective='reg:squarederror',
            random_state=None,
            verbose=False
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.objective = objective
        self.random_state = random_state
        self.verbose = verbose

        self.trees_ = []
        self.n_features_in_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        loss_type = 'regression'

        # 初始化预测值（全0）
        y_pred = np.zeros_like(y, dtype=np.float64)

        # 串行训练每棵树
        for i in range(self.n_estimators):
            if self.verbose and i % 10 == 0:
                # print(f"Training XGBoost tree {i + 1}/{self.n_estimators}")
                pass

            # 计算一阶/二阶梯度
            grad, hess = xgb_loss_gradient(y, y_pred, loss_type)

            # 训练单棵XGBoost树
            tree = XGBoostTree(
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha
            )
            tree.fit(X, grad, hess)
            self.trees_.append(tree)

            # 更新预测值
            y_pred += tree.predict(X)

        return self._response(
            1,
            content={
                "n_features_in_": self.n_features_in_,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "reg_lambda": self.reg_lambda,
                "reg_alpha": self.reg_alpha
            }
        )

    def predict(self, X):
        if len(self.trees_) == 0:
            return self._response(0, content="Model not trained yet - call fit() first")

        X = np.array(X, dtype=np.float64)
        # 累加所有树的预测值
        y_pred = np.zeros(len(X))
        for tree in self.trees_:
            y_pred += tree.predict(X)

        return self._response(1, content=y_pred)

    def evaluate(self, X, y_true, y_pred=None):
        y_true = np.array(y_true, dtype=np.float64).ravel()

        if y_pred is None:
            pred_resp = self.predict(X)
            if pred_resp["status"] == 0:
                return pred_resp
            y_pred = pred_resp["content"]
        else:
            y_pred = np.array(y_pred, dtype=np.float64).ravel()

        if len(y_true) != len(y_pred):
            return self._response(0, content="y_true和y_pred样本数必须一致")

        mse_resp = self.mse(y_pred, y_true)
        r2_resp = self.r2(y_true, y_pred)
        mae_resp = self.mae(y_pred, y_true)

        metrics = {
            "mse": mse_resp["content"]["mse"],
            "r2": r2_resp["content"]["r2"],
            "mae": mae_resp["content"]["mae"],
            "n_estimators": self.n_estimators,
            "n_samples": len(y_true)
        }
        return self._response(1, content=metrics)

# todo-2.26 build scikit-learn KMeans
class KMeans(Evaluation):
    def __init__(self, n_clusters=8, init='k-means++', max_iter=300, tol=1e-4, metric='euclidean', random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_features_in_ = None
        self.dist_fn = None

    def _init_centers(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        if self.init == 'random':
            return X[rng.choice(n_samples, self.n_clusters, replace=False)]
        elif self.init == 'k-means++':
            ct = [X[rng.choice(n_samples)]]
            for _ in range(1, self.n_clusters):
                d = np.min(self.dist_fn(X, np.array(ct)), axis=1)
                p = d / np.sum(d)
                ct.append(X[rng.choice(n_samples, p=p)])
            return np.array(ct)

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        if self.random_state: np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.dist_fn = get_distance_fn(self.metric)
        n_samples = X.shape[0]
        ct = self._init_centers(X)
        lb = np.zeros(n_samples, dtype=int)
        for it in range(self.max_iter):
            d_mat = self.dist_fn(X, ct)
            new_lb = np.argmin(d_mat, axis=1)
            ine = np.sum(np.min(d_mat, axis=1)**2)
            if np.all(new_lb==lb) or (self.inertia_ and abs(ine-self.inertia_)<self.tol):
                break
            lb = new_lb
            self.inertia_ = ine
            new_ct = np.zeros_like(ct)
            for c in range(self.n_clusters):
                clu = X[lb==c]
                new_ct[c] = np.mean(clu, axis=0) if len(clu) else X[np.random.choice(n_samples)]
            ct = new_ct
        self.cluster_centers_ = ct
        self.labels_ = lb
        return self._response(1, {"n_features_in_":self.n_features_in_, "n_clusters":self.n_clusters, "inertia_":self.inertia_, "n_iter_":it+1})

    def predict(self, X):
        if self.cluster_centers_ is None:
            return self._response(0, "Model not trained yet")
        X = np.array(X, dtype=np.float64)
        return self._response(1, np.argmin(self.dist_fn(X, self.cluster_centers_), axis=1))

    def evaluate(self, X, y_true=None, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)["content"]
        X = np.array(X, dtype=np.float64)
        m = {
            "silhouette": self.silhouette_score(X, y_pred),
            "ch_score": self.calinski_harabasz_score(X, y_pred),
            "db_score": self.davies_bouldin_score(X, y_pred),
            "n_clusters": len(np.unique(y_pred)),
            "n_samples": len(y_pred)
        }
        if y_true is not None:
            m["ari"] = self.adjusted_rand_score(y_true, y_pred)
        return self._response(1, m)

# todo-2.27 build scikit-learn LVQ
class LVQ(Evaluation):
    def __init__(self, n_prototypes=8, learning_rate=0.01, max_iter=300, metric='euclidean', random_state=None):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.metric = metric
        self.random_state = random_state
        self.prototypes_ = None
        self.labels_ = None
        self.n_features_in_ = None
        self.dist_fn = None

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        if self.random_state: np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.dist_fn = get_distance_fn(self.metric)
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, self.n_prototypes, replace=False)
        self.prototypes_ = X[idx].copy()
        lr = self.learning_rate
        for ep in range(self.max_iter):
            for i in np.random.permutation(n_samples):
                x = X[i].reshape(1,-1)
                best = np.argmin(self.dist_fn(x, self.prototypes_))
                self.prototypes_[best] += lr * (x - self.prototypes_[best])
            lr *= (1 - ep/self.max_iter)
        self.labels_ = np.argmin(self.dist_fn(X, self.prototypes_), axis=1)
        return self._response(1, {"n_features_in_":self.n_features_in_, "n_prototypes":self.n_prototypes, "n_iter_":self.max_iter})

    def predict(self, X):
        if self.prototypes_ is None:
            return self._response(0, "Model not trained yet")
        X = np.array(X, dtype=np.float64)
        return self._response(1, np.argmin(self.dist_fn(X, self.prototypes_), axis=1))

    def evaluate(self, X, y_true=None, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)["content"]
        X = np.array(X, dtype=np.float64)
        m = {
            "silhouette": self.silhouette_score(X, y_pred),
            "ch_score": self.calinski_harabasz_score(X, y_pred),
            "db_score": self.davies_bouldin_score(X, y_pred),
            "n_prototypes": self.n_prototypes,
            "n_samples": len(y_pred)
        }
        if y_true is not None:
            m["ari"] = self.adjusted_rand_score(y_true, y_pred)
        return self._response(1, m)

# todo-2.28 build scikit-learn GaussianMixture
class GaussianMixture(Evaluation):
    def __init__(self, n_components=8, covariance_type='full', max_iter=100, tol=1e-3, random_state=None):
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.labels_ = None
        self.n_features_in_ = None

    def _gaussian(self, X, m, c):
        n, f = X.shape
        det = np.linalg.det(c) if np.linalg.det(c) > 1e-10 else 1e-10
        inv = np.linalg.inv(c)
        dif = X - m
        exp = -0.5 * np.sum(dif @ inv * dif, axis=1)
        return np.exp(exp) / np.sqrt((2*np.pi)**f * det)

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        if self.random_state: np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]
        idx = np.random.choice(n, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        self.covariances_ = [np.eye(self.n_features_in_) for _ in range(self.n_components)]
        self.weights_ = np.ones(self.n_components)/self.n_components
        ll = -np.inf
        for it in range(self.max_iter):
            resp = np.zeros((n, self.n_components))
            for c in range(self.n_components):
                resp[:,c] = self.weights_[c] * self._gaussian(X, self.means_[c], self.covariances_[c])
            resp /= np.sum(resp, axis=1, keepdims=True)
            self.weights_ = np.mean(resp, axis=0)
            for c in range(self.n_components):
                self.means_[c] = np.sum(resp[:,c,None]*X, axis=0)/np.sum(resp[:,c])
                dif = X - self.means_[c]
                self.covariances_[c] = (dif.T @ (resp[:,c,None]*dif)) / np.sum(resp[:,c])
            new_ll = np.sum(np.log(np.sum(resp*self.weights_, axis=1)))
            if abs(new_ll - ll) < self.tol: break
            ll = new_ll
        self.labels_ = np.argmax(resp, axis=1)
        return self._response(1, {"n_features_in_":self.n_features_in_, "n_components":self.n_components, "cov_type":self.covariance_type, "n_iter_":it+1})

    def predict(self, X):
        if self.means_ is None:
            return self._response(0, "Model not trained yet")
        X = np.array(X, dtype=np.float64)
        resp = np.zeros((len(X), self.n_components))
        for c in range(self.n_components):
            resp[:,c] = self.weights_[c] * self._gaussian(X, self.means_[c], self.covariances_[c])
        return self._response(1, np.argmax(resp, axis=1))

    def evaluate(self, X, y_true=None, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)["content"]
        X = np.array(X, dtype=np.float64)
        m = {
            "silhouette": self.silhouette_score(X, y_pred),
            "ch_score": self.calinski_harabasz_score(X, y_pred),
            "db_score": self.davies_bouldin_score(X, y_pred),
            "n_components": self.n_components,
            "n_samples": len(y_pred)
        }
        if y_true is not None:
            m["ari"] = self.adjusted_rand_score(y_true, y_pred)
        return self._response(1, m)

# todo-2.29 build scikit-learn DBSCAN
class DBSCAN(Evaluation):
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', random_state=None):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.random_state = random_state
        self.labels_ = None
        self.core_indices_ = None
        self.n_features_in_ = None
        self.dist_fn = None

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        if self.random_state: np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.dist_fn = get_distance_fn(self.metric)
        n = X.shape[0]
        lb = -np.ones(n, dtype=int)
        cid = 0
        core = []
        for i in range(n):
            if lb[i] != -1: continue
            nei = np.where(self.dist_fn(X[i,None], X)[0] <= self.eps)[0]
            if len(nei) < self.min_samples: continue
            core.append(i)
            lb[i] = cid
            q = list(nei)
            while q:
                j = q.pop(0)
                if lb[j] == -1:
                    lb[j] = cid
                    jn = np.where(self.dist_fn(X[j,None], X)[0] <= self.eps)[0]
                    if len(jn) >= self.min_samples: q += jn.tolist()
            cid +=1
        self.labels_ = lb
        self.core_indices_ = np.array(core)
        return self._response(1, {"n_features_in_":self.n_features_in_, "n_clusters":cid, "noise":np.sum(lb==-1), "eps":self.eps, "min_samples":self.min_samples})

    def predict(self, X):
        if self.labels_ is None or len(X)!=len(self.labels_):
            return self._response(0, "DBSCAN only supports training set prediction")
        return self._response(1, self.labels_)

    def evaluate(self, X, y_true=None, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)["content"]
        v = y_pred != -1
        Xv, yv = X[v], y_pred[v]
        m = {"n_clusters":len(np.unique(yv)), "noise":np.sum(y_pred==-1), "n_samples":len(y_pred), "n_valid":len(yv)}
        if len(yv)>=2:
            m["silhouette"] = self.silhouette_score(Xv, yv)
            m["ch_score"] = self.calinski_harabasz_score(Xv, yv)
            m["db_score"] = self.davies_bouldin_score(Xv, yv)
        if y_true is not None:
            m["ari"] = self.adjusted_rand_score(y_true[v], yv)
        return self._response(1, m)

# todo-2.30 build scikit-learn AgglomerativeClustering
class AgglomerativeClustering(Evaluation):
    def __init__(self, n_clusters=2, linkage='ward', metric='euclidean', random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.random_state = random_state
        self.labels_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        if self.random_state: np.random.seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        link = scipy_linkage(X, method=self.linkage, metric=self.metric)
        self.labels_ = fcluster(link, self.n_clusters, criterion='maxclust') - 1
        return self._response(1, {"n_features_in_":self.n_features_in_, "n_clusters":self.n_clusters, "linkage":self.linkage})

    def predict(self, X):
        if self.labels_ is None or len(X)!=len(self.labels_):
            return self._response(0, "Agglomerative only supports training set prediction")
        return self._response(1, self.labels_)

    def evaluate(self, X, y_true=None, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)["content"]
        X = np.array(X, dtype=np.float64)
        m = {
            "silhouette": self.silhouette_score(X, y_pred),
            "ch_score": self.calinski_harabasz_score(X, y_pred),
            "db_score": self.davies_bouldin_score(X, y_pred),
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "n_samples": len(y_pred)
        }
        if y_true is not None:
            m["ari"] = self.adjusted_rand_score(y_true, y_pred)
        return self._response(1, m)

# todo-2.3n k-nearest Neighbor


# todo-2.nn save and load model
class ModelPersistence(SkLearn):
    def save_model(self, model, filepath, format="pickle"):
        if not isinstance(filepath, str):
            return self._response(0, "filepath必须为字符串类型")

        # 确保目录存在（贴合你代码的工程化风格）
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except Exception as e:
                return self._response(0, f"创建目录失败：{str(e)}")

        try:
            if format == "pickle":
                # 保存numpy数组友好的pickle格式（贴合你代码的数值计算场景）
                with open(filepath, "wb") as f:
                    pickle.dump(model, f, protocol=4)  # 兼容Python3.4+
            elif format == "json":
                # 手动处理numpy数组（贴合你代码的ndarray使用场景）
                def serialize(obj):
                    if isinstance(obj, np.ndarray):
                        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
                    elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                        return float(obj)  # 统一转为float避免json序列化问题
                    elif hasattr(obj, "__dict__"):
                        # 序列化自定义类（贴合你代码的类实例风格）
                        return {k: serialize(v) for k, v in obj.__dict__.items()}
                    else:
                        return obj

                model_serialized = serialize(model)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(model_serialized, f, ensure_ascii=False, indent=2)
            else:
                return self._response(0, "不支持的格式：仅支持pickle/json")

            return self._response(1, f"模型已保存至：{filepath}")
        except Exception as e:
            return self._response(0, f"模型保存失败：{str(e)}")

    def load_model(self, filepath, format="pickle"):
        if not os.path.exists(filepath):
            return self._response(0, f"模型文件不存在：{filepath}")
        try:
            if format == "pickle":
                with open(filepath, "rb") as f:
                    model = pickle.load(f)
            elif format == "json":
                def deserialize(obj):
                    if isinstance(obj, dict) and "__ndarray__" in obj:
                        return np.array(obj["__ndarray__"], dtype=obj["dtype"])
                    elif isinstance(obj, dict):
                        return {k: deserialize(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [deserialize(i) for i in obj]
                    else:
                        return obj

                with open(filepath, "r", encoding="utf-8") as f:
                    model_serialized = json.load(f)
                model = deserialize(model_serialized)
            else:
                return self._response(0, "不支持的格式：仅支持pickle/json")

            return self._response(1, model)
        except Exception as e:
            return self._response(0, f"模型加载失败：{str(e)}")


# todo-3 test
if __name__ == '__main__':
    pass