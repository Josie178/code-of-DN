from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "step1_processed_diabetes_data.csv"
RESULTS_DIR = BASE_DIR / "results"

GROUP_COL = "DN_outcome"
GROUP_LABELS = {1: "DN", 0: "Non-DN"}

CORE_VARIABLES = [
    "年龄",
    "性别",
    "BMI指数",
    "心脏心率",
    "收缩压",
    "舒张压",
    "有无高血压病史",
    "hypertension_pattern",
    "空腹血糖",
    "糖化血红蛋白",
    "总胆固醇TC",
    "甘油三酯TG",
    "高密度脂蛋白胆固醇",
    "低密度脂蛋白",
    "血红蛋白HGB",
    "平均血红蛋白浓度MCHC",
    "白细胞总数",
    "血小板计数PLT",
    "尿微量白蛋白",
    "UACR",
    "血肌酐",
    "eGFR",
]

EXCLUDE_EXACT = {
    GROUP_COL,
    "体检编号",
    "体检日期",
    "出生日期",
    "住址",
    "居委",
    "现象描述",
    "诊断",
    "尿微量白蛋白尿肌酐",
    "收缩压_num",
    "舒张压_num",
    "has_hypertension",
}

EXCLUDE_KEYWORDS = [
    "编号",
    "日期",
    "住址",
    "居委",
    "备注",
    "描述",
    "时间",
    "分期",
    "验光",
    "视力",
    "球镜",
    "柱镜",
    "轴位",
    "曲率",
    "瞳距",
    "胸片",
]

MISSING_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "nat",
    "na",
    "n/a",
}

YES_NO_MAP = {
    "有": "Yes",
    "无": "No",
    "是": "Yes",
    "否": "No",
    "阳性": "Yes",
    "阴性": "No",
    "男": "Male",
    "女": "Female",
    "1": "Yes",
    "0": "No",
    1: "Yes",
    0: "No",
    True: "Yes",
    False: "No",
}


def should_exclude(column: str) -> bool:
    if column in EXCLUDE_EXACT:
        return True
    if "?" in column:
        return True
    if any(keyword in column for keyword in EXCLUDE_KEYWORDS):
        return True
    return False


def load_analysis_dataframe() -> tuple[pd.DataFrame, str]:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH), f"processed:{DATA_PATH.name}"

    raise FileNotFoundError("未找到可用于生成基线表的数据源")


def clean_categorical(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.where(~cleaned.isna(), np.nan)
    cleaned = cleaned.astype("object")
    cleaned = cleaned.map(lambda value: value.strip() if isinstance(value, str) else value)
    cleaned = cleaned.map(
        lambda value: np.nan
        if isinstance(value, str) and value.lower() in MISSING_TOKENS
        else value
    )
    return cleaned


def looks_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True

    non_missing = clean_categorical(series).dropna()
    if non_missing.empty:
        return False

    converted = pd.to_numeric(non_missing, errors="coerce")
    success_rate = converted.notna().mean()
    return success_rate >= 0.9


def classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    continuous: list[str] = []
    categorical: list[str] = []

    for column in df.columns:
        if should_exclude(column):
            continue

        series = df[column]
        non_missing = clean_categorical(series).dropna()
        if len(non_missing) < 20:
            continue

        if looks_numeric(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            unique_count = numeric.nunique()
            if unique_count <= 1:
                continue
            if unique_count == 2:
                categorical.append(column)
            elif unique_count >= 5:
                continuous.append(column)
            continue

        unique_count = non_missing.nunique()
        if unique_count <= 1:
            continue
        if unique_count <= 5:
            categorical.append(column)

    return continuous, categorical


def format_p_value(p_value: float | None) -> str:
    if p_value is None or pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def format_mean_ci(series: pd.Series) -> tuple[str, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return "", 0

    mean = values.mean()
    if n == 1:
        return f"{mean:.2f} (NA, NA)", 1

    sem = stats.sem(values, nan_policy="omit")
    if pd.isna(sem):
        return f"{mean:.2f} (NA, NA)", n

    ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    return f"{mean:.2f} ({ci_low:.2f}, {ci_high:.2f})", n


def continuous_p_value(group_a: pd.Series, group_b: pd.Series) -> float | None:
    values_a = pd.to_numeric(group_a, errors="coerce").dropna()
    values_b = pd.to_numeric(group_b, errors="coerce").dropna()
    if len(values_a) < 2 or len(values_b) < 2:
        return None

    _, p_value = stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
    return float(p_value)


def normalize_level(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        return YES_NO_MAP.get(normalized, normalized)
    return str(YES_NO_MAP.get(value, value))


def categorical_summary(
    series: pd.Series,
    dn_mask: pd.Series,
    non_dn_mask: pd.Series,
) -> list[dict[str, object]]:
    cleaned = clean_categorical(series)
    levels = [level for level in cleaned.dropna().unique()]
    if not levels:
        return []

    contingency = pd.crosstab(cleaned, dn_mask.map({True: "DN", False: "Non-DN"}))
    contingency = contingency.reindex(index=levels, fill_value=0)
    contingency = contingency.reindex(columns=["DN", "Non-DN"], fill_value=0)

    p_value = None
    if contingency.shape[0] >= 2 and contingency.values.sum() > 0:
        if contingency.shape == (2, 2):
            _, p_value_tmp, _, expected = stats.chi2_contingency(contingency)
            if (expected < 5).any():
                _, p_value = stats.fisher_exact(contingency.values)
            else:
                p_value = p_value_tmp
        else:
            _, p_value, _, _ = stats.chi2_contingency(contingency)

    dn_non_missing = int(cleaned[dn_mask].notna().sum())
    non_dn_non_missing = int(cleaned[non_dn_mask].notna().sum())

    rows: list[dict[str, object]] = []
    for index, level in enumerate(levels):
        dn_count = int(((cleaned == level) & dn_mask).sum())
        non_dn_count = int(((cleaned == level) & non_dn_mask).sum())

        dn_text = (
            f"{dn_count} ({dn_count / dn_non_missing * 100:.2f}%)"
            if dn_non_missing
            else ""
        )
        non_dn_text = (
            f"{non_dn_count} ({non_dn_count / non_dn_non_missing * 100:.2f}%)"
            if non_dn_non_missing
            else ""
        )

        rows.append(
            {
                "Level": normalize_level(level),
                "DN": dn_text,
                "Non-DN": non_dn_text,
                "P-value": format_p_value(p_value) if index == 0 else "",
                "DN_nonmissing": dn_non_missing,
                "Non-DN_nonmissing": non_dn_non_missing,
            }
        )

    return rows


def build_table(
    df: pd.DataFrame,
    variables: list[str],
    output_name: str,
) -> pd.DataFrame:
    dn_mask = df[GROUP_COL] == 1
    non_dn_mask = df[GROUP_COL] == 0
    dn_header = f"DN (n={int(dn_mask.sum())})"
    non_dn_header = f"Non-DN (n={int(non_dn_mask.sum())})"

    rows: list[dict[str, object]] = []
    for variable in variables:
        if variable not in df.columns:
            continue

        series = df[variable]
        if looks_numeric(series) and pd.to_numeric(series, errors="coerce").dropna().nunique() >= 5:
            dn_text, dn_n = format_mean_ci(series[dn_mask])
            non_dn_text, non_dn_n = format_mean_ci(series[non_dn_mask])
            p_value = continuous_p_value(series[dn_mask], series[non_dn_mask])

            rows.append(
                {
                    "Variable": variable,
                    "Type": "continuous",
                    "Level": "",
                    dn_header: dn_text,
                    non_dn_header: non_dn_text,
                    "P-value": format_p_value(p_value),
                    "DN_nonmissing": dn_n,
                    "Non-DN_nonmissing": non_dn_n,
                }
            )
            continue

        summaries = categorical_summary(series, dn_mask, non_dn_mask)
        for summary in summaries:
            rows.append(
                {
                    "Variable": variable,
                    "Type": "categorical",
                    "Level": summary["Level"],
                    dn_header: summary["DN"],
                    non_dn_header: summary["Non-DN"],
                    "P-value": summary["P-value"],
                    "DN_nonmissing": summary["DN_nonmissing"],
                    "Non-DN_nonmissing": summary["Non-DN_nonmissing"],
                }
            )

    table = pd.DataFrame(rows)
    output_path = RESULTS_DIR / output_name
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return table


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df, source = load_analysis_dataframe()

    if GROUP_COL not in df.columns:
        raise KeyError(f"缺少分组字段: {GROUP_COL}")

    continuous_vars, categorical_vars = classify_columns(df)
    all_variables = [*continuous_vars, *categorical_vars]

    all_table = build_table(
        df=df,
        variables=all_variables,
        output_name="dn_baseline_table_all_variables.csv",
    )
    core_table = build_table(
        df=df,
        variables=CORE_VARIABLES,
        output_name="dn_baseline_table_core_variables.csv",
    )

    with pd.ExcelWriter(RESULTS_DIR / "dn_baseline_table.xlsx") as writer:
        all_table.to_excel(writer, sheet_name="all_variables", index=False)
        core_table.to_excel(writer, sheet_name="core_variables", index=False)

    df.to_csv(RESULTS_DIR / "dn_baseline_analysis_dataset.csv", index=False, encoding="utf-8-sig")

    print(f"数据来源: {source}")
    print(f"DN组样本量: {(df[GROUP_COL] == 1).sum()}")
    print(f"Non-DN组样本量: {(df[GROUP_COL] == 0).sum()}")
    print(f"分析数据快照: {RESULTS_DIR / 'dn_baseline_analysis_dataset.csv'}")
    print(f"全量变量表: {RESULTS_DIR / 'dn_baseline_table_all_variables.csv'}")
    print(f"核心变量表: {RESULTS_DIR / 'dn_baseline_table_core_variables.csv'}")
    print(f"Excel汇总: {RESULTS_DIR / 'dn_baseline_table.xlsx'}")


if __name__ == "__main__":
    main()
