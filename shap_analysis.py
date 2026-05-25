"""
SHAP 分析脚本 - 基于 Logistic 回归模型
分析各特征对糖尿病肾病预测的贡献
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)

# ============================================================================
# 1. 数据加载（复用 main_analysis.py 的处理逻辑）
# ============================================================================

print("加载数据...")
df = pd.read_excel('data/rawdata_体检数据截止到0602.xlsx')
df_dm = df[df['有无糖尿病病史'] == '有'].copy()

df_dm['UACR'] = pd.to_numeric(df_dm['尿微量白蛋白尿肌酐'], errors='coerce')
df_dm['血肌酐'] = df_dm['肌酐CR女'].fillna(df_dm['肌酐CR男'])
df_dm['血肌酐'] = pd.to_numeric(df_dm['血肌酐'], errors='coerce')


def calculate_egfr(scr, age, sex):
    scr_mgdl = scr / 88.4
    if sex == '女':
        kappa, alpha = 0.7, -0.329
        egfr = 141 * (min(scr_mgdl / kappa, 1) ** alpha) * (max(scr_mgdl / kappa, 1) ** -1.209) * (0.993 ** age) * 1.018
    else:
        kappa, alpha = 0.9, -0.411
        egfr = 141 * (min(scr_mgdl / kappa, 1) ** alpha) * (max(scr_mgdl / kappa, 1) ** -1.209) * (0.993 ** age)
    return egfr


df_dm['eGFR'] = df_dm.apply(
    lambda row: calculate_egfr(row['血肌酐'], row['年龄'], row['性别'])
    if pd.notna(row['血肌酐']) and pd.notna(row['年龄']) else np.nan, axis=1
)

df_dm['收缩压_num'] = pd.to_numeric(df_dm['收缩压'], errors='coerce')
df_dm['舒张压_num'] = pd.to_numeric(df_dm['舒张压'], errors='coerce')
df_dm['has_hypertension'] = (
    (df_dm['收缩压_num'] >= 140) |
    (df_dm['舒张压_num'] >= 90) |
    (df_dm['有无高血压病史'] == '有')
).astype(int)


def classify_hypertension_pattern(row):
    sbp, dbp = row['收缩压_num'], row['舒张压_num']
    if pd.isna(sbp) or pd.isna(dbp):
        return np.nan
    if sbp >= 140 and dbp < 90:
        return 'ISH_单纯收缩期高血压'
    elif sbp < 140 and dbp >= 90:
        return 'IDH_单纯舒张期高血压'
    elif sbp >= 140 and dbp >= 90:
        return 'SDH_收缩舒张期高血压'
    else:
        return '正常血压'


df_dm['hypertension_pattern'] = df_dm.apply(classify_hypertension_pattern, axis=1)
df_dm['DN_outcome'] = ((df_dm['UACR'] >= 30) | (df_dm['eGFR'] < 60)).astype(int)

df_analysis = df_dm[
    df_dm['UACR'].notna() &
    df_dm['eGFR'].notna() &
    df_dm['DN_outcome'].notna() &
    df_dm['hypertension_pattern'].notna()
].copy()

print(f"分析样本: {len(df_analysis)} 例，肾病患者: {df_analysis['DN_outcome'].sum()} 例")

# ============================================================================
# 2. 特征工程（与 main_analysis.py 保持一致）
# ============================================================================

demographic_features = ['年龄', '性别']
physical_features = ['BMI指数', '心脏心率']
lab_features = [
    '空腹血糖', '糖化血红蛋白',
    '总胆固醇TC', '甘油三酯TG', '高密度脂蛋白胆固醇', '低密度脂蛋白',
    '血红蛋白HGB', '平均血红蛋白浓度MCHC',
    '白细胞总数', '血小板计数PLT'
]
feature_cols = demographic_features + physical_features + lab_features

hypertension_dummies = pd.get_dummies(df_analysis['hypertension_pattern'], prefix='HTN', drop_first=False)

X_base = df_analysis[feature_cols].copy()
y = df_analysis['DN_outcome'].copy()

for col in X_base.columns:
    if col != '性别':
        X_base[col] = pd.to_numeric(X_base[col], errors='coerce')

# 性别编码
le = LabelEncoder()
X_base['性别_encoded'] = le.fit_transform(X_base['性别'])
X_base = X_base.drop('性别', axis=1)

# 缺失值处理
missing_pct = (X_base.isnull().sum() / len(X_base) * 100)
cols_to_drop = missing_pct[missing_pct > 5.0].index.tolist()
if cols_to_drop:
    X_base = X_base.drop(cols_to_drop, axis=1)

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_base), columns=X_base.columns, index=X_base.index)

# 标准化
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)

# 添加高血压哑变量
hypertension_dummies_aligned = hypertension_dummies.loc[X_scaled.index]
if 'HTN_正常血压' in hypertension_dummies_aligned.columns:
    hypertension_dummies_aligned = hypertension_dummies_aligned.drop('HTN_正常血压', axis=1)

X_final_with_htn = pd.concat([X_scaled, hypertension_dummies_aligned], axis=1)

# VIF 过滤
selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(X_scaled)
selected_features_var = X_scaled.columns[selector_var.get_support()].tolist()
X_vif = pd.DataFrame(X_var, columns=selected_features_var, index=X_scaled.index)

vif_data = pd.DataFrame({
    "特征": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
while vif_data['VIF'].max() > 10:
    max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), '特征']
    X_vif = X_vif.drop(max_vif_feature, axis=1)
    vif_data = pd.DataFrame({
        "特征": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    })

X_final = pd.concat([X_vif, hypertension_dummies_aligned], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
print(f"训练集: {X_train.shape[0]} 例，测试集: {X_test.shape[0]} 例，特征数: {X_train.shape[1]}")

# ============================================================================
# 3. 加载 Logistic 回归模型
# ============================================================================

import joblib

model_path = 'results/lr_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"找不到模型文件 {model_path}，请先运行 main_analysis.py 生成模型。"
    )

print(f"\n加载模型: {model_path}")
lr_model = joblib.load(model_path)
print("模型加载完成")

# ============================================================================
# 4. SHAP 分析
# ============================================================================

print("\n计算 SHAP 值...")
# Logistic 回归使用 LinearExplainer，背景数据用训练集
explainer = shap.LinearExplainer(lr_model, X_train, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_test)

# 特征名映射（中文显示）
feature_name_map = {
    '年龄': 'Age',
    '性别_encoded': 'Sex',
    'BMI指数': 'BMI',
    '心脏心率': 'Heart Rate',
    '空腹血糖': 'FBG',
    '糖化血红蛋白': 'HbA1c',
    '总胆固醇TC': 'TC',
    '甘油三酯TG': 'TG',
    '高密度脂蛋白胆固醇': 'HDL-C',
    '低密度脂蛋白': 'LDL-C',
    '血红蛋白HGB': 'HGB',
    '平均血红蛋白浓度MCHC': 'MCHC',
    '白细胞总数': 'WBC',
    '血小板计数PLT': 'PLT',
    'HTN_ISH_单纯收缩期高血压': 'HTN-ISH',
    'HTN_IDH_单纯舒张期高血压': 'HTN-IDH',
    'HTN_SDH_收缩舒张期高血压': 'HTN-SDH',
}
display_names = [feature_name_map.get(c, c) for c in X_test.columns]

os.makedirs('figures', exist_ok=True)

# 预测概率（用于选取典型患者）
pred_proba = lr_model.predict_proba(X_test)[:, 1]

# 构建 Explanation 对象
explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_test.values,
    feature_names=display_names
)

# ============================================================================
# 图1：3个典型患者的 Force Plot 拼图
# 高风险：DN=1 且预测概率最高
# 低风险：DN=0 且预测概率最低
# 中间：预测概率最接近 0.5
# ============================================================================

print("绘制 Force Plot（3个典型患者）...")

pos_mask = y_test.values == 1
neg_mask = y_test.values == 0

idx_high = np.where(pos_mask)[0][np.argmax(pred_proba[pos_mask])]
idx_low  = np.where(neg_mask)[0][np.argmin(pred_proba[neg_mask])]
idx_mid  = np.argmin(np.abs(pred_proba - 0.5))

patients = [
    (idx_high, f"High Risk  (pred={pred_proba[idx_high]:.2f}, true=DN+)"),
    (idx_mid,  f"Borderline (pred={pred_proba[idx_mid]:.2f}, true={'DN+' if y_test.values[idx_mid]==1 else 'DN-'})"),
    (idx_low,  f"Low Risk   (pred={pred_proba[idx_low]:.2f}, true=DN-)"),
]

fig, axes = plt.subplots(3, 1, figsize=(16, 9))
titles = [p[1] for p in patients]

for ax, (idx, title) in zip(axes, patients):
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X_test.iloc[idx].astype(float).round(4),
        feature_names=display_names,
        matplotlib=True,
        show=False,
        figsize=(16, 3),
        text_rotation=0,
        contribution_threshold=0.05,
    )
    # force_plot 会自己创建 figure，把它贴到我们的 axes 上
    src_fig = plt.gcf()
    src_fig.canvas.draw()
    import io
    buf = io.BytesIO()
    src_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(src_fig)
    buf.seek(0)
    from PIL import Image
    img = Image.open(buf)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=4, loc='left')

plt.tight_layout(h_pad=1.5)
plt.savefig('figures/shap_waterfall_3patients.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> figures/shap_waterfall_3patients.png")

# ============================================================================
# 图2：所有特征的 Dependence Plot 拼成大图
# ============================================================================

print("绘制 Dependence Plot（所有特征）...")

n_features = len(display_names)
ncols = 3
nrows = (n_features + ncols - 1) // ncols

# 预先计算每个特征的最佳交互特征（用 pandas corr 避免 numpy 2.0 bug）
corr_matrix = X_test.corr().values.copy()
np.fill_diagonal(corr_matrix, 0)
best_interaction = np.argmax(np.abs(corr_matrix), axis=1)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes_flat = axes.flatten()

for i, feat_name in enumerate(display_names):
    ax = axes_flat[i]
    interact_idx = int(best_interaction[i])
    shap.dependence_plot(
        i,
        shap_values,
        X_test,
        feature_names=display_names,
        interaction_index=interact_idx,
        ax=ax,
        show=False,
        dot_size=15,
        alpha=0.7
    )
    ax.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.8)
    ax.set_title(feat_name, fontsize=11)
    ax.set_xlabel(feat_name, fontsize=9)
    ax.set_ylabel("SHAP value", fontsize=9)

# 隐藏多余的子图
for j in range(n_features, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("SHAP Dependence Plots — All Features", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('figures/shap_dependence_all.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> figures/shap_dependence_all.png")

# ============================================================================
# 5. 保存 SHAP 值到 CSV
# ============================================================================

shap_df = pd.DataFrame(shap_values, columns=display_names)
shap_df['expected_value'] = explainer.expected_value
shap_df['true_label'] = y_test.values
shap_df.to_csv('results/shap_values.csv', index=False, encoding='utf-8-sig')
print("\nSHAP 值已保存到: results/shap_values.csv")

print("\n全部完成。")
