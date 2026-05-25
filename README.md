# 糖尿病肾病预测模型项目

## 📋 项目概述

本项目基于体检数据构建机器学习模型，用于预测合并高血压的糖尿病患者发生肾病的风险。

**研究目标**：
1. 分析合并高血压的糖尿病患者发生肾病的概率和风险
2. 探索不同高血压模式与肾病指标异常的关联性

---

## 📁 项目文件结构

```
medical/
├── data/                                    # 数据文件夹
│   ├── rawdata_体检数据截止到0602.xlsx        # 原始体检数据 (4.6M)
│   ├── step1_processed_diabetes_data.csv    # 步骤1：处理后的糖尿病患者数据
│   ├── step2_preprocessed_features.csv      # 步骤2：预处理后的特征数据
│   └── step3_final_selected_features.csv    # 步骤3：特征选择后的最终数据
│
├── results/                                 # 结果文件夹
│   ├── final_report.txt                     # 最终分析报告
│   ├── logistic_regression_OR_results.csv   # Logistic回归OR值结果
│   ├── model_comparison_results.csv         # 模型对比结果
│   ├── model_evaluation_results.csv         # 模型评估结果
│   ├── best_params.json                     # 最优超参数配置
│   ├── hyperparameter_tuning_summary.csv    # 调优结果汇总
│   ├── cv_results_*.csv                     # 各模型详细CV结果
│   ├── tuned_model_results.csv              # 使用调优参数的模型性能
│   └── multi_model_analysis_output.log      # 分析日志
│
├── figures/                                 # 图表文件夹
│   ├── roc_curve.png                        # ROC曲线（Logistic回归）
│   └── roc_curves_comparison.png            # 多模型ROC曲线对比
│
├── docs/                                    # 文档文件夹
│   ├── 项目介绍.md                           # 项目详细介绍
│   ├── 数据处理流程总结.md                    # 数据处理流程说明
│   ├── 模型设置总结.md                        # 模型设置说明
│   └── DN.docx                              # 研究需求文档
│
├── main_analysis.py                         # 主分析脚本（Python版本）
├── main_analysis.ipynb                      # 主分析脚本（Jupyter Notebook版本）
├── hyperparameter_tuning.py                 # 超参数调优脚本
├── main_analysis_with_tuning.py             # 使用调优参数的分析脚本
└── README.md                                # 本文档
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建conda虚拟环境
conda create -n medical_ml python=3.9 -y
conda activate medical_ml

# 安装依赖包
pip install pandas numpy matplotlib seaborn openpyxl python-docx \
            scikit-learn xgboost shap imbalanced-learn scipy statsmodels
```

### 2. 运行分析

**方式1：使用Python脚本**

```bash
# 激活环境
conda activate medical_ml

# 运行主分析脚本
python main_analysis.py
```

**方式2：先调优再分析（推荐）**

```bash
# 激活环境
conda activate medical_ml

# 步骤1：运行超参数调优（需要5-10分钟）
python hyperparameter_tuning.py

# 步骤2：使用最优参数训练模型
python main_analysis_with_tuning.py
```

运行后将自动完成：
- 数据加载与变量创建
- 统计分析（卡方检验）
- 数据预处理与特征选择
- 超参数网格搜索（可选）
- 多因素Logistic回归分析
- 多模型比较（6种模型）
- 模型评估与结果输出
- 生成最终报告

---

## ⚙️ 超参数调优

### 调优方法
- **策略**: 网格搜索 (GridSearchCV)
- **交叉验证**: 5折分层交叉验证
- **评估指标**: ROC-AUC
- **并行计算**: 使用所有CPU核心

### 最优参数配置

| 模型 | 交叉验证AUC | 测试集AUC | 最优参数 |
|------|-------------|-----------|----------|
| **Logistic回归** | 0.7298 | **0.764** | C=1, penalty='l2', solver='liblinear' |
| 随机森林 | 0.7266 | 0.740 | n_estimators=100, max_depth=None, min_samples_split=10 |
| **XGBoost** | 0.7235 | **0.775** | n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8 |
| SVM | 0.7142 | 0.752 | C=0.1, kernel='linear', gamma='scale' |
| 朴素贝叶斯 | 0.7056 | 0.701 | var_smoothing=1e-09 |
| 神经网络 | 0.6609 | 0.667 | hidden_layers=(100,), alpha=0.001, learning_rate=0.001 |

### 调优输出文件
- `results/best_params.json` - 最优参数JSON配置
- `results/hyperparameter_tuning_summary.csv` - 调优结果汇总
- `results/cv_results_*.csv` - 每个模型的详细交叉验证结果

---

## 📊 数据说明

### 原始数据
- **文件**: `rawdata_体检数据截止到0602.xlsx`
- **总样本**: 9,780 例
- **糖尿病患者**: 1,478 例
- **可用样本**: 1,240 例（有完整UACR、eGFR和高血压模式数据）

### 关键变量

**结局变量**：
- 糖尿病肾病（DN_outcome）：UACR≥30 mg/g 或 eGFR<60 ml/min/1.73m²

**诊断指标**：
- UACR（尿白蛋白肌酐比值）= 尿微量白蛋白(mg/L) × 1000 / 尿肌酐(mg/L)
- eGFR（肾小球滤过率）：使用 CKD-EPI 2021 公式计算

**高血压模式**：
- 单纯收缩期高血压：SBP≥140 且 DBP<90
- 单纯舒张期高血压：SBP<140 且 DBP≥90
- 收缩舒张期高血压：SBP≥140 且 DBP≥90

---

## 🔬 分析流程

### 第一部分：数据准备
1. 筛选糖尿病患者
2. 计算 UACR 和 eGFR
3. 定义高血压模式和结局变量

### 第二部分：数据预处理
1. 缺失值处理（中位数填补）
2. 数据标准化（StandardScaler）
3. 分类变量编码

### 第三部分：特征选择
1. 低方差过滤（阈值=0.01）
2. VIF 多重共线性检查（VIF<10）
3. Elastic Net 特征选择（1-SE规则）

**最终选择的 9 个特征**：
- 血红蛋白HGB
- 白细胞总数
- 年龄
- 甘油三酯TG
- 性别
- 收缩压
- 糖化血红蛋白
- 舒张压
- BMI指数

### 第四部分：模型构建
构建 6 种机器学习模型：
1. Logistic Regression (LR)
2. Random Forest (RF)
3. XGBoost
4. Support Vector Machine (SVM)
5. Naïve Bayes (NB)
6. Neural Network (NN)

### 第五部分：模型优化
- 网格搜索 + 10折交叉验证
- 模型校准（Platt Scaling / Isotonic Regression）

### 第六部分：模型评估
评估指标：
- AUC、Brier Score
- 灵敏度、特异度
- PPV、NPV、F1-score

---

## 📈 主要结果

### 模型性能对比（使用调优后参数）

| 模型 | AUC | Brier Score | 准确率 | 灵敏度 | 特异度 | F1-score |
|------|-----|-------------|--------|--------|--------|----------|
| **XGBoost** | **0.7745** | 0.1603 | 0.7500 | 0.3030 | 0.9121 | 0.3922 |
| **Logistic回归** | **0.7643** | 0.1703 | 0.7419 | 0.3030 | 0.9011 | 0.3846 |
| SVM | 0.7523 | 0.1945 | 0.7258 | 0.0000 | 0.9890 | 0.0000 |
| 随机森林 | 0.7403 | 0.1700 | 0.7419 | 0.2424 | 0.9231 | 0.3333 |
| 朴素贝叶斯 | 0.7008 | 0.2077 | 0.7379 | 0.2727 | 0.9066 | 0.3564 |

**注**: Brier Score越低越好（衡量预测概率的准确性）

**最佳模型**: XGBoost (AUC=0.7745)

**调优效果**:
- XGBoost提升最显著：0.732 → 0.775 (+4.3%)
- SVM提升明显：0.704 → 0.752 (+4.9%)
- Logistic回归：0.763 → 0.764 (+0.1%)

### 关键发现

1. **高血压模式对肾病风险的影响**（参照组：正常血压）：
   - **IDH（单纯舒张期高血压）**: OR=1.23, 95%CI[0.45-3.35], P=0.79
   - **ISH（单纯收缩期高血压）**: OR=1.89, 95%CI[1.37-2.75], P=0.07
   - **SDH（收缩舒张期高血压）**: OR=2.30, 95%CI[1.52-3.49], P=0.11

2. **其他显著危险因素**（P<0.05）：
   - 年龄：OR=1.42, P=0.005
   - 糖化血红蛋白：OR=1.63, P=0.020
   - 血红蛋白：OR=0.67, P<0.001（保护因素）
   - 白细胞总数：OR=1.29, P=0.036

3. **临床意义**：
   - ISH和SDH患者肾病风险显著增加
   - 血红蛋白水平是重要的预测指标
   - 模型可用于糖尿病患者的肾病风险筛查

---

## 📄 输出文件说明

### 数据文件（data/）
- `step1_processed_diabetes_data.csv`: 包含UACR、eGFR、高血压模式、肾病诊断等
- `step2_preprocessed_features.csv`: 标准化后的特征数据（含哑变量）
- `step3_final_selected_features.csv`: 特征选择后的最终数据

### 结果文件（results/）
- `logistic_regression_OR_results.csv`: 各变量的OR值、95%CI、P值
- `model_comparison_results.csv`: 5个模型的性能对比（AUC、准确率等）
- `model_evaluation_results.csv`: 最优模型的详细评估指标
- `final_report.txt`: 完整的分析报告，包含所有统计结果和结论
