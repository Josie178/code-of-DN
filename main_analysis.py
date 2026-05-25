"""
糖尿病肾病风险分析 - 主分析脚本
研究目标：探讨糖尿病患者合并不同类型高血压时，肾病发生风险的差异
研究设计：按高血压模式分组，比较各组肾病发生率
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

print("="*80)
print("糖尿病肾病预测模型分析")
print("="*80)

# ============================================================================
# 第一部分：数据加载和变量创建
# ============================================================================

print("\n第一部分：数据加载和变量创建")
print("-"*80)

# 1. 读取数据
df = pd.read_excel('data/rawdata_体检数据截止到0602.xlsx')
print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

# 2. 筛选糖尿病患者
df_dm = df[df['有无糖尿病病史'] == '有'].copy()
print(f"糖尿病患者: {len(df_dm)} 例")

# 3. 使用原始数据中的UACR（尿微量白蛋白尿肌酐）
df_dm['UACR'] = pd.to_numeric(df_dm['尿微量白蛋白尿肌酐'], errors='coerce')
print(f"UACR数据读取完成: {df_dm['UACR'].notna().sum()} 例有效值")


# 4. 计算eGFR
df_dm['血肌酐'] = df_dm['肌酐CR女'].fillna(df_dm['肌酐CR男'])
df_dm['血肌酐'] = pd.to_numeric(df_dm['血肌酐'], errors='coerce')

def calculate_egfr(scr, age, sex):
    """
    eGFR计算公式
    eGFR = 141 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^-1.209 × 0.993^年龄 × (女性×1.018)
    """
    # 单位转换：μmol/L → mg/dL (1 mg/dL = 88.4 μmol/L)
    scr_mgdl = scr / 88.4

    if sex == '女':
        kappa = 0.7
        alpha = -0.329
        min_term = min(scr_mgdl / kappa, 1) ** alpha
        max_term = max(scr_mgdl / kappa, 1) ** (-1.209)
        egfr = 141 * min_term * max_term * (0.993 ** age) * 1.018
    else:  # 男性
        kappa = 0.9
        alpha = -0.411
        min_term = min(scr_mgdl / kappa, 1) ** alpha
        max_term = max(scr_mgdl / kappa, 1) ** (-1.209)
        egfr = 141 * min_term * max_term * (0.993 ** age)

    return egfr

df_dm['eGFR'] = df_dm.apply(
    lambda row: calculate_egfr(row['血肌酐'], row['年龄'], row['性别'])
    if pd.notna(row['血肌酐']) and pd.notna(row['年龄']) else np.nan, axis=1
)
print(f"eGFR计算完成: {df_dm['eGFR'].notna().sum()} 例有效值")


# 5. 定义高血压模式
df_dm['收缩压_num'] = pd.to_numeric(df_dm['收缩压'], errors='coerce')
df_dm['舒张压_num'] = pd.to_numeric(df_dm['舒张压'], errors='coerce')

# 高血压定义：收缩压≥140 或 舒张压≥90 或 有高血压病史
df_dm['has_hypertension'] = (
    (df_dm['收缩压_num'] >= 140) | 
    (df_dm['舒张压_num'] >= 90) | 
    (df_dm['有无高血压病史'] == '有')
).astype(int)

# 高血压模式分类（核心分组变量）
def classify_hypertension_pattern(row):
    """
    按照研究设计分类高血压模式：
    - 单纯收缩期高血压(ISH): SBP≥140 且 DBP<90
    - 单纯舒张期高血压(IDH): SBP<140 且 DBP≥90
    - 收缩舒张期高血压(SDH): SBP≥140 且 DBP≥90
    - 正常血压: SBP<140 且 DBP<90
    """
    sbp = row['收缩压_num']
    dbp = row['舒张压_num']
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
print(f"\n高血压模式分布（核心分组变量）:")
print(df_dm['hypertension_pattern'].value_counts())
print(f"缺失值: {df_dm['hypertension_pattern'].isna().sum()} 例")


# 6. 定义结局变量（糖尿病肾病）
# 诊断标准：UACR≥30 mg/g 或 eGFR<60 ml/min/1.73m²
df_dm['DN_outcome'] = (
    (df_dm['UACR'] >= 30) | 
    (df_dm['eGFR'] < 60)
).astype(int)

print(f"\n结局变量统计:")
print(f"糖尿病肾病患者: {df_dm['DN_outcome'].sum()} 例 ({df_dm['DN_outcome'].mean()*100:.1f}%)")
print(f"- UACR≥30: {(df_dm['UACR'] >= 30).sum()} 例")
print(f"- eGFR<60: {(df_dm['eGFR'] < 60).sum()} 例")

# 筛选有完整数据的样本（必须包含高血压模式分类）
df_analysis = df_dm[
    df_dm['UACR'].notna() &
    df_dm['eGFR'].notna() &
    df_dm['DN_outcome'].notna() &
    df_dm['hypertension_pattern'].notna()
].copy()

print(f"\n可用于分析的样本: {len(df_analysis)} 例")
print(f"其中肾病患者: {df_analysis['DN_outcome'].sum()} 例 ({df_analysis['DN_outcome'].mean()*100:.1f}%)")

# 保存处理后的糖尿病患者数据（包含UACR、eGFR、高血压模式、肾病诊断等）
df_analysis.to_csv('data/step1_processed_diabetes_data.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 步骤1：处理后的糖尿病患者数据已保存到: step1_processed_diabetes_data.csv")

# 按高血压模式分组统计肾病发生率
print(f"\n按高血压模式分组的肾病发生率:")
print("-"*60)
for pattern in ['正常血压', 'ISH_单纯收缩期高血压', 'IDH_单纯舒张期高血压', 'SDH_收缩舒张期高血压']:
    if pattern in df_analysis['hypertension_pattern'].values:
        group_data = df_analysis[df_analysis['hypertension_pattern'] == pattern]
        n_total = len(group_data)
        n_dn = group_data['DN_outcome'].sum()
        rate = n_dn / n_total * 100 if n_total > 0 else 0
        print(f"{pattern:20s}: {n_dn:3d}/{n_total:4d} ({rate:5.2f}%)")


# ============================================================================
# 第二部分：统计分析 - 按高血压模式分组比较
# ============================================================================

print("\n" + "="*80)
print("第二部分：统计分析 - 按高血压模式分组比较")
print("-"*80)

# 进行卡方检验，比较不同高血压模式组的肾病发生率
from scipy.stats import chi2_contingency

# 创建列联表
contingency_table = pd.crosstab(
    df_analysis['hypertension_pattern'],
    df_analysis['DN_outcome'],
    margins=True
)
print("\n列联表（高血压模式 vs 肾病发生）:")
print(contingency_table)

# 卡方检验
chi2, chi2_p_value, dof, expected = chi2_contingency(
    pd.crosstab(df_analysis['hypertension_pattern'], df_analysis['DN_outcome'])
)
print(f"\n卡方检验结果:")
print(f"  χ² = {chi2:.4f}")
print(f"  p-value = {chi2_p_value:.4f}")
print(f"  自由度 = {dof}")
if chi2_p_value < 0.05:
    print(f"  结论: 不同高血压模式组的肾病发生率存在显著差异 (p < 0.05)")
else:
    print(f"  结论: 不同高血压模式组的肾病发生率无显著差异 (p ≥ 0.05)")


# ============================================================================
# 第三部分：准备特征变量（用于多因素分析）
# ============================================================================

print("\n" + "="*80)
print("第三部分：准备特征变量（用于多因素分析）")
print("-"*80)

# 注意：高血压模式将作为分组变量，不作为预测特征
# 我们将构建模型来预测肾病风险，同时控制其他混杂因素

# 1. 人口学特征
demographic_features = ['年龄', '性别']

# 2. 体检指标（不包括血压值，因为已经用于分组）
physical_features = ['BMI指数', '心脏心率']

# 3. 实验室指标
lab_features = [
    '空腹血糖', '糖化血红蛋白',
    '总胆固醇TC', '甘油三酯TG', '高密度脂蛋白胆固醇', '低密度脂蛋白',
    '血红蛋白HGB', '平均血红蛋白浓度MCHC',
    '白细胞总数', '血小板计数PLT'
]

# 4. 高血压模式（作为分类变量）
# 将高血压模式编码为哑变量（以"正常血压"为参照组）
hypertension_dummies = pd.get_dummies(
    df_analysis['hypertension_pattern'],
    prefix='HTN',
    drop_first=False  # 保留所有类别，后续手动选择参照组
)

# 合并所有特征
feature_cols = demographic_features + physical_features + lab_features

print(f"\n基础特征数量: {len(feature_cols)}")
print("基础特征列表:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

print(f"\n高血压模式哑变量:")
for i, col in enumerate(hypertension_dummies.columns, 1):
    print(f"  {i}. {col}")

# 提取特征和目标变量
X_base = df_analysis[feature_cols].copy()
y = df_analysis['DN_outcome'].copy()

print(f"\n基础特征矩阵: {X_base.shape}")
print(f"目标变量: {y.shape}, 阳性率: {y.mean()*100:.1f}%")


# 转换所有特征列为数值型
for col in X_base.columns:
    if col != '性别':
        X_base[col] = pd.to_numeric(X_base[col], errors='coerce')

# 检查缺失值
print(f"\n特征缺失值统计:")
missing_pct = (X_base.isnull().sum() / len(X_base) * 100).sort_values(ascending=False)
print(missing_pct[missing_pct > 0])


# ============================================================================
# 第四部分：数据预处理
# ============================================================================

print("\n" + "="*80)
print("第四部分：数据预处理")
print("-"*80)

# 1. 处理性别变量（转换为数值）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_base['性别_encoded'] = le.fit_transform(X_base['性别'])
X_base = X_base.drop('性别', axis=1)

print("\n1. 性别编码完成（女=0, 男=1）")


# 2. 缺失值处理
print("\n2. 缺失值处理")

# 删除缺失率>5%的变量
missing_threshold = 5.0
cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
if cols_to_drop:
    print(f"   删除缺失率>{missing_threshold}%的变量: {cols_to_drop}")
    X_base = X_base.drop(cols_to_drop, axis=1)

# 对剩余变量填补缺失值（使用中位数）
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X_base),
    columns=X_base.columns,
    index=X_base.index
)

print(f"   填补后的特征数量: {X_imputed.shape[1]}")
print(f"   剩余缺失值: {X_imputed.isnull().sum().sum()}")


# 3. 数据标准化
print("\n3. 数据标准化")
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_imputed),
    columns=X_imputed.columns,
    index=X_imputed.index
)
print(f"   标准化完成，特征数量: {X_scaled.shape[1]}")

# 4. 添加高血压模式哑变量（不标准化）
print("\n4. 添加高血压模式哑变量")
# 以"正常血压"为参照组，删除该列
hypertension_dummies_aligned = hypertension_dummies.loc[X_scaled.index]
if 'HTN_正常血压' in hypertension_dummies_aligned.columns:
    hypertension_dummies_aligned = hypertension_dummies_aligned.drop('HTN_正常血压', axis=1)
    print(f"   参照组: 正常血压")

# 合并特征
X_final_with_htn = pd.concat([X_scaled, hypertension_dummies_aligned], axis=1)
print(f"   合并后特征数量: {X_final_with_htn.shape[1]}")
print(f"   高血压模式变量: {list(hypertension_dummies_aligned.columns)}")

# 保存预处理后的特征数据（标准化+哑变量）
X_final_with_htn_save = X_final_with_htn.copy()
X_final_with_htn_save['DN_outcome'] = y
X_final_with_htn_save.to_csv('data/step2_preprocessed_features.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 步骤2：预处理后的特征数据已保存到: step2_preprocessed_features.csv")


# ============================================================================
# 第五部分：特征选择（仅对基础特征，不包括高血压模式）
# ============================================================================

print("\n" + "="*80)
print("第五部分：特征选择")
print("-"*80)

# 注意：特征选择仅针对基础特征，高血压模式作为核心研究变量必须保留

# 1. 低方差过滤
print("\n1. 低方差过滤（方差阈值=0.01）")
from sklearn.feature_selection import VarianceThreshold
selector_var = VarianceThreshold(threshold=0.01)
X_var = selector_var.fit_transform(X_scaled)
selected_features_var = X_scaled.columns[selector_var.get_support()].tolist()
print(f"   保留特征数量: {len(selected_features_var)}")


# 2. VIF多重共线性检查
print("\n2. VIF多重共线性检查（VIF阈值=10）")
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = pd.DataFrame(X_var, columns=selected_features_var, index=X_scaled.index)
vif_data = pd.DataFrame()
vif_data["特征"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))

# 迭代删除VIF>10的特征
while vif_data['VIF'].max() > 10:
    max_vif_feature = vif_data.loc[vif_data['VIF'].idxmax(), '特征']
    print(f"   删除VIF最高的特征: {max_vif_feature} (VIF={vif_data['VIF'].max():.2f})")
    X_vif = X_vif.drop(max_vif_feature, axis=1)
    vif_data = pd.DataFrame()
    vif_data["特征"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

print(f"   VIF检查后保留特征数量: {X_vif.shape[1]}")

# 3. 添加高血压模式变量到最终特征集
print("\n3. 添加高血压模式变量")
X_final = pd.concat([X_vif, hypertension_dummies_aligned], axis=1)
print(f"   最终特征数量: {X_final.shape[1]}")
print(f"   - 基础特征: {X_vif.shape[1]}")
print(f"   - 高血压模式变量: {hypertension_dummies_aligned.shape[1]}")

# 保存特征选择后的最终特征数据
X_final_save = X_final.copy()
X_final_save['DN_outcome'] = y
X_final_save.to_csv('data/step3_final_selected_features.csv', index=False, encoding='utf-8-sig')
print(f"\n✓ 步骤3：特征选择后的最终数据已保存到: step3_final_selected_features.csv")


# ============================================================================
# 第六部分：划分训练集和测试集
# ============================================================================

print("\n" + "="*80)
print("第六部分：划分训练集和测试集")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n训练集: {X_train.shape[0]} 例 (阳性: {y_train.sum()}, {y_train.mean()*100:.1f}%)")
print(f"测试集: {X_test.shape[0]} 例 (阳性: {y_test.sum()}, {y_test.mean()*100:.1f}%)")
print(f"特征数量: {X_train.shape[1]} (包含{hypertension_dummies_aligned.shape[1]}个高血压模式变量)")


# ============================================================================
# 第七部分：构建Logistic回归模型（多因素分析）
# ============================================================================

print("\n" + "="*80)
print("第七部分：构建Logistic回归模型（多因素分析）")
print("-"*80)
print("\n注意：本研究重点关注高血压模式对肾病风险的影响")
print("      Logistic回归可以提供各高血压模式的OR值和95%CI")

# 1. 训练Logistic回归模型
print("\n1. 训练Logistic回归模型")
# 尝试加载最优参数
try:
    with open('results/best_params.json', 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    if 'Logistic回归' in best_params:
        lr_model = LogisticRegression(random_state=42, max_iter=1000, **best_params['Logistic回归'])
        print("   ✓ 使用调优后的参数")
    else:
        lr_model = LogisticRegression(random_state=42, max_iter=1000, penalty='l2', C=1.0)
        print("   ⚠ 使用默认参数")
except:
    lr_model = LogisticRegression(random_state=42, max_iter=1000, penalty='l2', C=1.0)
    print("   ⚠ 使用默认参数")
lr_model.fit(X_train, y_train)
print("   ✓ Logistic回归模型训练完成")

# 2. 计算OR值和95%置信区间
print("\n2. 计算OR值和95%置信区间")
print("   (针对高血压模式变量，参照组=正常血压)")

# 获取系数和标准误
coefficients = lr_model.coef_[0]
feature_names = X_train.columns.tolist()

# 计算OR值
or_values = np.exp(coefficients)

# 使用Bootstrap方法计算95%CI
from scipy import stats
n_bootstrap = 1000
or_bootstrap = []

print(f"   正在进行Bootstrap重采样 ({n_bootstrap}次)...")
for i in range(n_bootstrap):
    # 重采样
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_boot = X_train.iloc[indices]
    y_boot = y_train.iloc[indices]

    # 训练模型
    try:
        if 'Logistic回归' in best_params:
            lr_boot = LogisticRegression(random_state=42, max_iter=1000, **best_params['Logistic回归'])
        else:
            lr_boot = LogisticRegression(random_state=42, max_iter=1000, penalty='l2', C=1.0)
    except:
        lr_boot = LogisticRegression(random_state=42, max_iter=1000, penalty='l2', C=1.0)
    lr_boot.fit(X_boot, y_boot)

    # 保存OR值
    or_bootstrap.append(np.exp(lr_boot.coef_[0]))

or_bootstrap = np.array(or_bootstrap)

# 计算95%CI
or_results = []
for i, feature in enumerate(feature_names):
    or_val = or_values[i]
    ci_lower = np.percentile(or_bootstrap[:, i], 2.5)
    ci_upper = np.percentile(or_bootstrap[:, i], 97.5)
    p_value = stats.norm.sf(abs(coefficients[i] / np.std(or_bootstrap[:, i]))) * 2

    or_results.append({
        '变量': feature,
        '系数': coefficients[i],
        'OR': or_val,
        '95%CI下限': ci_lower,
        '95%CI上限': ci_upper,
        'P值': p_value
    })

or_df = pd.DataFrame(or_results)

# 筛选高血压模式变量
htn_or_df = or_df[or_df['变量'].str.contains('HTN_')]
other_or_df = or_df[~or_df['变量'].str.contains('HTN_')]

print("\n" + "="*80)
print("高血压模式对肾病风险的影响（多因素Logistic回归）")
print("="*80)
print("\n参照组：正常血压")
print("-"*80)
print(htn_or_df.to_string(index=False))

print("\n\n其他协变量:")
print("-"*80)
print(other_or_df.to_string(index=False))

# 保存结果
or_df.to_csv('results/logistic_regression_OR_results.csv', index=False, encoding='utf-8-sig')
print("\n✓ OR值结果已保存到: logistic_regression_OR_results.csv")

# 保存模型
import joblib
joblib.dump(lr_model, 'results/lr_model.pkl')
print("✓ Logistic回归模型已保存到: results/lr_model.pkl")


# ============================================================================
# 第七部分B：多模型对比分析
# ============================================================================

print("\n" + "="*80)
print("第七部分B：多模型对比分析")
print("-"*80)
print("\n训练多个模型并对比性能...")

# 尝试加载最优参数
try:
    with open('results/best_params.json', 'r', encoding='utf-8') as f:
        best_params = json.load(f)
    print("✓ 已加载调优后的最优参数")
    use_tuned = True
except FileNotFoundError:
    print("⚠ 未找到调优参数，使用默认参数")
    best_params = {}
    use_tuned = False

# 定义多个模型（使用最优参数或默认参数）
if use_tuned and 'Logistic回归' in best_params:
    models = {
        'Logistic回归': LogisticRegression(random_state=42, max_iter=1000, **best_params['Logistic回归']),
        '随机森林': RandomForestClassifier(random_state=42, **best_params.get('随机森林', {})),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', device='cpu', **best_params.get('XGBoost', {})),
        '支持向量机': SVC(random_state=42, probability=True, **best_params.get('SVM', {})),
        '朴素贝叶斯': GaussianNB(**best_params.get('朴素贝叶斯', {}))
    }
else:
    models = {
        'Logistic回归': LogisticRegression(random_state=42, max_iter=1000, penalty='l2', C=1.0),
        '随机森林': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss', device='cpu'),
        '支持向量机': SVC(random_state=42, probability=True, kernel='rbf', C=1.0),
        '朴素贝叶斯': GaussianNB()
    }

# 存储所有模型的结果
all_models_results = []

print("\n训练各模型...")
for model_name, model in models.items():
    print(f"\n{model_name}:")

    # 训练模型
    model.fit(X_train, y_train)
    print(f"   ✓ 训练完成")

    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 计算性能指标
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_curve, auc)

    auc_score = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    sensitivity = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    ppv = precision_score(y_test, y_pred) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # 保存结果
    all_models_results.append({
        '模型': model_name,
        'AUC': auc_score,
        'Brier_Score': brier,
        '准确率': accuracy,
        '灵敏度': sensitivity,
        '特异度': specificity,
        'PPV': ppv,
        'NPV': npv,
        'F1_score': f1
    })

    print(f"   AUC: {auc_score:.4f}, 准确率: {accuracy:.4f}, F1: {f1:.4f}")

# 创建对比表格
comparison_df = pd.DataFrame(all_models_results)
comparison_df = comparison_df.sort_values('AUC', ascending=False)

print("\n" + "="*80)
print("模型性能对比（按AUC排序）")
print("="*80)
print(comparison_df.to_string(index=False))

# 保存对比结果
comparison_df.to_csv('results/model_comparison_results.csv', index=False, encoding='utf-8-sig')
print("\n✓ 模型对比结果已保存到: model_comparison_results.csv")

# 保存训练好的模型（用于后续分析）
trained_models = {}
for model_name, model in models.items():
    trained_models[model_name] = model

# 保存所有模型和测试集
joblib.dump(trained_models, 'results/trained_models.pkl')
joblib.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test},
            'results/train_test_data.pkl')
print("✓ 所有模型已保存到: results/trained_models.pkl")
print("✓ 训练/测试集已保存到: results/train_test_data.pkl")


# ============================================================================
# 第八部分：模型评估（详细评估Logistic回归）
# ============================================================================

print("\n" + "="*80)
print("第八部分：详细评估Logistic回归模型")
print("-"*80)
print("\n注：Logistic回归是医学研究的标准方法，可提供OR值")

# 评估Logistic回归模型性能
print("\n评估Logistic回归模型性能...")

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

# 预测
y_pred = lr_model.predict(X_test)
y_pred_proba = lr_model.predict_proba(X_test)[:, 1]

# 计算指标
auc_score = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)
sensitivity = recall_score(y_test, y_pred)  # 灵敏度=召回率
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)  # 特异度
ppv = precision_score(y_test, y_pred) if (tp + fp) > 0 else 0  # PPV=精确率
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # NPV
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n模型性能指标:")
print(f"  AUC: {auc_score:.4f}")
print(f"  Brier Score: {brier:.4f}")
print(f"  准确率: {accuracy:.4f}")
print(f"  灵敏度: {sensitivity:.4f}")
print(f"  特异度: {specificity:.4f}")
print(f"  PPV: {ppv:.4f}")
print(f"  NPV: {npv:.4f}")
print(f"  F1-score: {f1:.4f}")

# 保存评估结果
eval_results = {
    'AUC': auc_score,
    'Brier_Score': brier,
    '准确率': accuracy,
    '灵敏度': sensitivity,
    '特异度': specificity,
    'PPV': ppv,
    'NPV': npv,
    'F1_score': f1
}
eval_df = pd.DataFrame([eval_results])
eval_df.to_csv('results/model_evaluation_results.csv', index=False, encoding='utf-8-sig')
print("\n✓ 评估结果已保存到: model_evaluation_results.csv")


# ============================================================================
# 第九部分：绘制ROC曲线（多模型对比）
# ============================================================================

print("\n" + "="*80)
print("第九部分：绘制ROC曲线（多模型对比）")
print("-"*80)

# 1. 绘制多模型ROC曲线对比图
plt.figure(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (model_name, model) in enumerate(trained_models.items()):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})',
             linewidth=2, color=colors[i])

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison - Diabetic Nephropathy Risk Models', fontsize=14)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ 多模型ROC曲线对比图已保存到: roc_curves_comparison.png")
plt.close()

# 2. 单独保存Logistic回归的ROC曲线（用于报告）
plt.figure(figsize=(8, 6))
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_lr)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Diabetic Nephropathy Risk Model', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curve.png', dpi=300, bbox_inches='tight')
print("\n✓ ROC曲线已保存到: roc_curve.png")
plt.close()


# ============================================================================
# 第十部分：生成最终分析报告
# ============================================================================

print("\n" + "="*80)
print("第十部分：生成最终分析报告")
print("-"*80)

# 创建报告
report = []
report.append("="*80)
report.append("糖尿病患者高血压模式与肾病风险关联分析 - 最终报告")
report.append("="*80)
report.append("")

# 1. 研究目标
report.append("一、研究目标")
report.append("-"*80)
report.append("探讨糖尿病患者合并不同类型高血压时，肾病发生风险的差异")
report.append("")
report.append("高血压模式分类:")
report.append("  - 单纯收缩期高血压(ISH): SBP≥140 且 DBP<90")
report.append("  - 单纯舒张期高血压(IDH): SBP<140 且 DBP≥90")
report.append("  - 收缩舒张期高血压(SDH): SBP≥140 且 DBP≥90")
report.append("  - 正常血压: SBP<140 且 DBP<90")
report.append("")
report.append("结局变量: 糖尿病肾病 (UACR≥30 mg/g 或 eGFR<60 ml/min/1.73m²)")
report.append("")

# 2. 数据概况
report.append("二、数据概况")
report.append("-"*80)
report.append(f"总样本数: {len(df_analysis)} 例")
report.append(f"糖尿病肾病患者: {df_analysis['DN_outcome'].sum()} 例 ({df_analysis['DN_outcome'].mean()*100:.1f}%)")
report.append("")
report.append("按高血压模式分组:")
for pattern in ['正常血压', 'ISH_单纯收缩期高血压', 'IDH_单纯舒张期高血压', 'SDH_收缩舒张期高血压']:
    if pattern in df_analysis['hypertension_pattern'].values:
        group_data = df_analysis[df_analysis['hypertension_pattern'] == pattern]
        n_total = len(group_data)
        n_dn = group_data['DN_outcome'].sum()
        rate = n_dn / n_total * 100 if n_total > 0 else 0
        report.append(f"  {pattern:25s}: {n_dn:3d}/{n_total:4d} ({rate:5.2f}%)")
report.append("")

# 3. 统计分析结果
report.append("三、统计分析结果")
report.append("-"*80)
report.append(f"卡方检验: χ²={chi2:.4f}, p={chi2_p_value:.4f}")
if chi2_p_value < 0.05:
    report.append("结论: 不同高血压模式组的肾病发生率存在显著差异 (p<0.05)")
else:
    report.append("结论: 不同高血压模式组的肾病发生率无显著差异 (p≥0.05)")
report.append("")

# 4. 多因素Logistic回归结果
report.append("四、多因素Logistic回归分析（OR值和95%CI）")
report.append("-"*80)
report.append("参照组: 正常血压")
report.append("")
report.append(htn_or_df.to_string(index=False))
report.append("")

# 5. 模型性能
report.append("五、模型预测性能")
report.append("-"*80)
report.append(f"AUC: {auc_score:.4f}")
report.append(f"灵敏度: {sensitivity:.4f}")
report.append(f"特异度: {specificity:.4f}")
report.append(f"准确率: {accuracy:.4f}")
report.append("")

# 6. 结论与建议
report.append("六、结论与临床意义")
report.append("-"*80)
report.append("1. 主要发现:")
report.append("   - 分析了不同高血压模式对糖尿病肾病风险的影响")
report.append("   - 通过多因素Logistic回归控制了混杂因素")
report.append("   - 提供了各高血压模式的OR值和95%置信区间")
report.append("")
report.append("2. 临床意义:")
report.append("   - 识别高危高血压模式，指导临床干预")
report.append("   - 为糖尿病患者的血压管理提供依据")
report.append("   - 建议对高风险模式患者加强肾功能监测")
report.append("")

# 保存报告
report.append("="*80)
report.append("报告生成时间: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
report.append("="*80)

report_text = '\n'.join(report)
print("\n" + report_text)

# 保存到文件
with open('results/final_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print("\n✓ 最终报告已保存到: final_report.txt")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n生成的文件:")
print("  1. step1_processed_diabetes_data.csv - 处理后的糖尿病患者数据")
print("  2. step2_preprocessed_features.csv - 预处理后的特征数据")
print("  3. step3_final_selected_features.csv - 特征选择后的最终数据")
print("  4. model_comparison_results.csv - 多模型性能对比")
print("  5. logistic_regression_OR_results.csv - Logistic回归OR值详细结果")
print("  6. model_evaluation_results.csv - Logistic回归性能指标")
print("  7. roc_curves_comparison.png - 多模型ROC曲线对比图")
print("  8. roc_curve.png - Logistic回归ROC曲线图")
print("  9. final_report.txt - 最终分析报告")
print("\n" + "="*80)

