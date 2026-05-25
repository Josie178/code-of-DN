"""
超参数调优脚本 - 使用网格搜索优化模型参数
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 重定向输出到日志文件
log_file = open('results/hyperparameter_tuning.log', 'w', encoding='utf-8')
sys.stdout = log_file

print("="*80)
print("超参数调优 - 网格搜索")
print("="*80)

# 加载预处理后的数据
X = pd.read_csv('data/step3_final_selected_features.csv')
y = X['DN_outcome']
X = X.drop('DN_outcome', axis=1)

print(f"\n数据加载完成: {X.shape[0]} 样本, {X.shape[1]} 特征")

# 定义交叉验证策略
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义参数网格
param_grids = {
    'Logistic回归': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    '随机森林': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', device='cpu'),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    },
    '朴素贝叶斯': {
        'model': GaussianNB(),
        'params': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    },
    '神经网络': {
        'model': MLPClassifier(max_iter=1000, random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01]
        }
    }
}

# 存储最优参数
best_params_dict = {}
results_summary = []

# 对每个模型进行网格搜索
for model_name, config in param_grids.items():
    print(f"\n{'='*80}")
    print(f"调优模型: {model_name}")
    print(f"{'='*80}")

    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    print(f"\n最优参数: {grid_search.best_params_}")
    print(f"最优AUC: {grid_search.best_score_:.4f}")

    best_params_dict[model_name] = grid_search.best_params_
    results_summary.append({
        '模型': model_name,
        '最优AUC': grid_search.best_score_,
        '最优参数': str(grid_search.best_params_)
    })

    # 保存详细的CV结果
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(f'results/cv_results_{model_name}.csv', index=False, encoding='utf-8-sig')

# 保存最优参数
with open('results/best_params.json', 'w', encoding='utf-8') as f:
    json.dump(best_params_dict, f, indent=4, ensure_ascii=False)

# 保存汇总结果
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values('最优AUC', ascending=False)
summary_df.to_csv('results/hyperparameter_tuning_summary.csv', index=False, encoding='utf-8-sig')

# 生成超参数性能统计表
print(f"\n{'='*80}")
print("生成超参数性能统计表")
print(f"{'='*80}")

for model_name in param_grids.keys():
    cv_results = pd.read_csv(f'results/cv_results_{model_name}.csv')

    # 提取参数列和性能列
    params_cols = [col for col in cv_results.columns if col.startswith('param_')]

    # 创建性能统计表
    performance_table = cv_results[params_cols + ['mean_test_score', 'std_test_score', 'rank_test_score']].copy()
    performance_table.columns = [col.replace('param_', '') for col in params_cols] + ['平均AUC', '标准差', '排名']

    # 按AUC降序排列
    performance_table = performance_table.sort_values('平均AUC', ascending=False)
    performance_table['平均AUC'] = performance_table['平均AUC'].round(4)
    performance_table['标准差'] = performance_table['标准差'].round(4)

    # 保存为CSV
    performance_table.to_csv(f'results/hyperparameter_table_{model_name}.csv', index=False, encoding='utf-8-sig')
    print(f"\n{model_name} - 前10组参数:")
    print(performance_table.head(10).to_string(index=False))
    print(f"  已保存: results/hyperparameter_table_{model_name}.csv")

print(f"\n{'='*80}")
print("调优完成！")
print(f"{'='*80}")
print("\n最优参数已保存到: results/best_params.json")
print("汇总结果已保存到: results/hyperparameter_tuning_summary.csv")
print("性能对比图已保存到: figures/hyperparameter_performance_*.png")
print("\n模型排名:")
print(summary_df.to_string(index=False))

# 关闭日志文件
log_file.close()
