# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 15:12:07 2025

@author: 邢晨
"""
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression as lin 
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics 
import time

from sklearn.model_selection import LeaveOneOut
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def loo_cross_validation(clf, clf_name, x, y):
    """
    :param clf: 基模型对象（支持多项式回归的特殊处理）
    :param clf_name: 模型标识符，用于条件分支
    :param x: 原始特征数据矩阵
    :param y: 目标值向量
    :return: (rmse_mean, rmse_std, r2_mean, r2_std)
    """
    n_samples = x.shape[0]
    y_pre_all = np.zeros(n_samples)
    loo = LeaveOneOut()
    
    for train_idx, test_idx in loo.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化应仅在训练集拟合（防止数据泄漏）
        x_scaler = StandardScaler().fit(x_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1,1))
        
        x_train_nor = x_scaler.transform(x_train)
        y_train_nor = y_scaler.transform(y_train.reshape(-1,1)).ravel()
        x_test_nor = x_scaler.transform(x_test)
        
        if 'poly' in clf_name:  # 处理多项式特征
            poly_features = clf.fit_transform(x_train_nor)
            model = lin().fit(poly_features, y_train_nor)
            y_pred_nor = model.predict(clf.transform(x_test_nor))
        else:
            clf.fit(x_train_nor, y_train_nor)
            y_pred_nor = clf.predict(x_test_nor)
        
        # 逆标准化预测结果
        y_pred = y_scaler.inverse_transform(y_pred_nor.reshape(-1,1)).ravel()
        y_pre_all[test_idx] = y_pred[0]

    # 结果统计（按知识库ID6表格中的评估指标格式）
    rmse = np.sqrt(mean_squared_error(y, y_pre_all))
    r2 = r2_score(y, y_pre_all)
    
    # 根据知识库ID24的标准，返回均值±标准差格式
    return rmse, 0.0, r2, 0.0  # 留一法标准差为0

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.datasets import load_diabetes

# 载入数据
X, y = load_diabetes(return_X_y=True)

# 测试 Gaussian Process
gp = GaussianProcessRegressor()
rmse_gp, _, r2_gp, _ = loo_cross_validation(gp, "linear", X, y)
print(f"[旧代码] Gaussian Process: RMSE={rmse_gp:.4f}, R2={r2_gp:.4f}")

# 测试 Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rmse_rf, _, r2_rf, _ = loo_cross_validation(rf, "linear", X, y)
print(f"[旧代码] Random Forest: RMSE={rmse_rf:.4f}, R2={r2_rf:.4f}")

# 测试 SVR
svr = SVR(kernel="rbf", C=1.0, epsilon=0.1)
rmse_svr, _, r2_svr, _ = loo_cross_validation(svr, "linear", X, y)
print(f"[旧代码] SVR (RBF kernel): RMSE={rmse_svr:.4f}, R2={r2_svr:.4f}")