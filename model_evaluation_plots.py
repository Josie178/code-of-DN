"""
模型评估四联图：ROC、DCA、PR曲线、校准曲线
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

# ============================================================================
# 加载模型和数据
# ============================================================================

print("加载模型和数据...")

model_path = 'results/trained_models.pkl'
data_path  = 'results/train_test_data.pkl'

if not os.path.exists(model_path) or not os.path.exists(data_path):
    raise FileNotFoundError("请先运行 main_analysis.py 生成模型和数据文件。")

trained_models = joblib.load(model_path)
data = joblib.load(data_path)
X_test, y_test = data['X_test'], data['y_test']

# 同时加载逻辑回归（已单独保存）
lr_model = joblib.load('results/lr_model.pkl')
trained_models['Logistic回归'] = lr_model

# 英文名映射
name_map = {
    'Logistic回归': 'Logistic Regression',
    '随机森林':     'Random Forest',
    'XGBoost':      'XGBoost',
    '支持向量机':   'SVM',
    '朴素贝叶斯':   'Naive Bayes',
}

# 颜色列表
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

os.makedirs('figures', exist_ok=True)

# ============================================================================
# 计算各模型预测概率
# ============================================================================

model_probs = {}
for name, model in trained_models.items():
    model_probs[name] = model.predict_proba(X_test)[:, 1]

# ============================================================================
# 绘制四联图
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Model Evaluation", fontsize=15, y=1.01)

# ── (A) ROC 曲线 ──────────────────────────────────────────────────────────────
ax = axes[0, 0]
for (name, probs), color in zip(model_probs.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = auc(fpr, tpr)
    label = f"{name_map.get(name, name)} (AUC={auc_val:.3f})"
    ax.plot(fpr, tpr, color=color, lw=1.8, label=label)

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel("1 - Specificity (FPR)", fontsize=11)
ax.set_ylabel("Sensitivity (TPR)", fontsize=11)
ax.set_title("(A) ROC Curve", fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

# ── (B) DCA 决策曲线 ──────────────────────────────────────────────────────────
ax = axes[0, 1]
thresholds = np.linspace(0.01, 0.99, 200)
n = len(y_test)
prevalence = y_test.mean()

# Treat all / Treat none 基准线
net_benefit_all  = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
net_benefit_none = np.zeros_like(thresholds)

ax.plot(thresholds, net_benefit_all,  'k--', lw=1.2, label='Treat All',  alpha=0.6)
ax.plot(thresholds, net_benefit_none, 'k-',  lw=1.2, label='Treat None', alpha=0.6)

for (name, probs), color in zip(model_probs.items(), colors):
    nb = []
    for pt in thresholds:
        pred_pos = probs >= pt
        tp = ((pred_pos == 1) & (y_test == 1)).sum()
        fp = ((pred_pos == 1) & (y_test == 0)).sum()
        nb.append(tp / n - fp / n * pt / (1 - pt))
    nb = np.array(nb)
    nb = np.clip(nb, -0.05, None)
    ax.plot(thresholds, nb, color=color, lw=1.8, label=name_map.get(name, name))

ax.set_xlabel("Threshold Probability", fontsize=11)
ax.set_ylabel("Net Benefit", fontsize=11)
ax.set_title("(B) Decision Curve Analysis", fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([0, 1]); ax.set_ylim([-0.05, prevalence + 0.05])

# ── (C) PR 曲线 ───────────────────────────────────────────────────────────────
ax = axes[1, 0]
for (name, probs), color in zip(model_probs.items(), colors):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    label = f"{name_map.get(name, name)} (AP={ap:.3f})"
    ax.plot(recall, precision, color=color, lw=1.8, label=label)

ax.axhline(prevalence, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Baseline (prev={prevalence:.2f})')
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("(C) Precision-Recall Curve", fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

# ── (D) 校准曲线 ──────────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect Calibration')

for (name, probs), color in zip(model_probs.items(), colors):
    fraction_pos, mean_pred = calibration_curve(y_test, probs, n_bins=5, strategy='quantile')
    ax.plot(mean_pred, fraction_pos, 'o-', color=color, lw=1.8,
            markersize=4, label=name_map.get(name, name))

ax.set_xlabel("Mean Predicted Probability", fontsize=11)
ax.set_ylabel("Fraction of Positives", fontsize=11)
ax.set_title("(D) Calibration Curve", fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

plt.tight_layout()
plt.savefig('figures/model_evaluation_4panel.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> figures/model_evaluation_4panel.png")
print("完成。")
