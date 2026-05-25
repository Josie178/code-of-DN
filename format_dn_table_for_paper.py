from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SOURCE_PATH = RESULTS_DIR / "dn_baseline_table_core_variables.csv"
ANALYSIS_DATASET_PATH = RESULTS_DIR / "dn_baseline_analysis_dataset.csv"

VARIABLE_LABELS = {
    "hypertension_pattern": "高血压模式",
}

LEVEL_LABELS = {
    "Female": "女",
    "Male": "男",
    "Yes": "是",
    "No": "否",
    "ISH_单纯收缩期高血压": "ISH_单纯收缩期高血压",
    "IDH_单纯舒张期高血压": "IDH_单纯舒张期高血压",
    "SDH_收缩舒张期高血压": "SDH_收缩舒张期高血压",
    "正常血压": "正常血压",
}

RAW_LEVEL_TO_DISPLAY = {
    "女": "女",
    "男": "男",
    "有": "是",
    "无": "否",
    "是": "是",
    "否": "否",
    "ISH_单纯收缩期高血压": "ISH_单纯收缩期高血压",
    "IDH_单纯舒张期高血压": "IDH_单纯舒张期高血压",
    "SDH_收缩舒张期高血压": "SDH_收缩舒张期高血压",
    "正常血压": "正常血压",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return LEVEL_LABELS.get(text, text)


def format_mean_ci(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return ""
    mean = values.mean()
    if n == 1:
        return f"{mean:.2f} (NA, NA)"
    sem = stats.sem(values, nan_policy="omit")
    if pd.isna(sem):
        return f"{mean:.2f} (NA, NA)"
    ci_low, ci_high = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)
    return f"{mean:.2f} ({ci_low:.2f}, {ci_high:.2f})"


def parse_count(text: object) -> int:
    if pd.isna(text):
        return 0
    text_str = str(text).strip()
    if not text_str:
        return 0
    return int(text_str.split(" ", 1)[0])


def format_count_pct(count: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    return f"{count} ({count / denominator * 100:.2f}%)"


def main() -> None:
    source_df = pd.read_csv(SOURCE_PATH)
    analysis_df = pd.read_csv(ANALYSIS_DATASET_PATH)

    dn_col = next(column for column in source_df.columns if column.startswith("DN "))
    non_dn_col = next(column for column in source_df.columns if column.startswith("Non-DN "))
    dn_n = int(dn_col.split("n=")[1].rstrip(")"))
    non_dn_n = int(non_dn_col.split("n=")[1].rstrip(")"))
    total_n = dn_n + non_dn_n
    total_col = f"Total sample (n={total_n})"

    rows: list[dict[str, str]] = []
    for variable, group in source_df.groupby("Variable", sort=False):
        variable_label = VARIABLE_LABELS.get(variable, variable)
        variable_type = group["Type"].iloc[0]

        if variable_type == "continuous":
            total_text = format_mean_ci(analysis_df[variable])
            row = group.iloc[0]
            rows.append(
                {
                    "Variable": variable_label,
                    total_col: total_text,
                    dn_col: clean_text(row[dn_col]),
                    non_dn_col: clean_text(row[non_dn_col]),
                    "P-value": clean_text(row["P-value"]),
                }
            )
            continue

        rows.append(
            {
                "Variable": variable_label,
                total_col: "",
                dn_col: "",
                non_dn_col: "",
                "P-value": clean_text(group["P-value"].iloc[0]),
            }
        )

        dn_nonmissing = int(group["DN_nonmissing"].iloc[0])
        non_dn_nonmissing = int(group["Non-DN_nonmissing"].iloc[0])
        total_nonmissing = dn_nonmissing + non_dn_nonmissing

        for _, row in group.iterrows():
            level_display = clean_text(row["Level"])
            dn_count = parse_count(row[dn_col])
            non_dn_count = parse_count(row[non_dn_col])
            total_count = dn_count + non_dn_count
            total_text = format_count_pct(total_count, total_nonmissing)

            # Prefer counts reconstructed from the source table so the total column
            # stays exactly aligned with the DN/Non-DN rows already shown.
            if variable in analysis_df.columns:
                raw_counts = (
                    analysis_df[variable]
                    .astype("object")
                    .map(lambda value: RAW_LEVEL_TO_DISPLAY.get(str(value).strip(), str(value).strip()) if pd.notna(value) else value)
                )
                observed_total = int((raw_counts == level_display).sum())
                if observed_total == total_count:
                    total_text = format_count_pct(observed_total, total_nonmissing)

            rows.append(
                {
                    "Variable": f"  {level_display}",
                    total_col: total_text,
                    dn_col: clean_text(row[dn_col]),
                    non_dn_col: clean_text(row[non_dn_col]),
                    "P-value": "",
                }
            )

    formatted = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "table1_dn_non_dn_formatted.csv"
    xlsx_path = RESULTS_DIR / "table1_dn_non_dn_formatted.xlsx"
    md_path = RESULTS_DIR / "table1_dn_non_dn_formatted.md"

    formatted.to_csv(csv_path, index=False, encoding="utf-8-sig")
    formatted.to_excel(xlsx_path, index=False)

    markdown_lines = [
        "# Table 1. DN组与Non-DN组变量分布",
        "",
        "连续变量为均值 (95%CI)，分类变量为人数 (百分比)。",
        "",
        formatted.to_markdown(index=False),
        "",
    ]
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print(csv_path)
    print(xlsx_path)
    print(md_path)


if __name__ == "__main__":
    main()
