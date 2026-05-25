"""
糖尿病肾病风险分析 - 使用调优后的参数
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("糖尿病肾病预测模型分析 - 使用调优参数")
print("="*80)

# 加载数据
X = pd.read_csv('data/step3_final_selected_features.csv')
y = X['DN_outcome']
X = X.drop('DN_outcome', axis=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# 加载最优参数
try:
    with open('results/best_params.json', 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    print("\n✓ 已加载调优参数")
except FileNotFoundError:
    print("\n⚠ 未找到调优参数，请先运行 hyperparameter_tuning.py")
    best_params = {}

# 定义模型（使用最优参数）
models = {}

if 'Logistic回归' in best_params:
    models['Logistic回归'] = LogisticRegression(
        max_iter=1000, random_state=42, **best_params['Logistic回归']
    )

if '随机森林' in best_params:
    models['随机森林'] = RandomForestClassifier(
        random_state=42, **best_params['随机森林']
    )

if 'XGBoost' in best_params:
    models['XGBoost'] = xgb.XGBClassifier(
        random_state=42, eval_metric='logloss', **best_params['XGBoost']
    )

if 'SVM' in best_params:
    models['SVM'] = SVC(
        probability=True, random_state=42, **best_params['SVM']
    )

if '朴素贝叶斯' in best_params:
    models['朴素贝叶斯'] = GaussianNB(**best_params['朴素贝叶斯'])

if '神经网络' in best_params:
    models['神经网络'] = MLPClassifier(
        max_iter=1000, random_state=42, **best_params['神经网络']
    )

# 训练和评估模型
results = []

print(f"\n{'='*80}")
print("模型训练与评估")
print(f"{'='*80}")

for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    results.append({
        '模型': name,
        'AUC': auc,
        '准确率': acc,
        '灵敏度': sensitivity,
        '特异度': specificity
    })

    print(f"  AUC: {auc:.4f} | 准确率: {acc:.4f}")

# 保存结果
results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
results_df.to_csv('results/tuned_model_results.csv', index=False, encoding='utf-8-sig')

print(f"\n{'='*80}")
print("评估完成！")
print(f"{'='*80}")
print("\n模型性能排名:")
print(results_df.to_string(index=False))
print("\n结果已保存到: results/tuned_model_results.csv")
