"""
Add latex code below to your document:
usepackage{longtable}
"""

import os
import re
import pandas as pd

NAMES_MAP = {
    "ALPHA": r"$\\alpha$",
    "BETA": r"$\\beta$",
    "ACCURACY": "Acc",
    "BALANCED": "Bal",
}

def replace_column_names(column_name: str) -> str:
    formatted = column_name.replace('_', ' ').title()

    for key, value in NAMES_MAP.items():
        # Find and replace case-insensitively
        formatted = re.sub(re.escape(key), value, formatted, flags=re.IGNORECASE)

    return formatted


def sort_result_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort result rows by the following order (descending):
    balanced_accuracy -> accuracy -> balanced_accuracy_std -> accuracy_std
    """
    sort_by_cols = [
        'balanced_accuracy',
        'accuracy',
        'balanced_accuracy_std',
        'accuracy_std'
    ]

    sorting_cols = [col for col in sort_by_cols if col in df.columns]

    if sorting_cols:
        return df.sort_values(by=sorting_cols, ascending=False).reset_index(drop=True)

    return df

def save_ds_result_to_latex(dataset_results, dataset_name: str, output_dir="results_latex", filename=None):
    """
    Generates table of results for single dataset
    :param filename: output filename (without extension). If None, it will be generated from dataset_name
    """

    df = pd.DataFrame(dataset_results)

    if df.empty:
        return

    os.makedirs(output_dir, exist_ok=True)

    if 'dataset' in df.columns:
        df = df.drop(columns=['dataset'])

    df = sort_result_rows(df)

    df = replace_dataframe_values(df, columns_to_process=['order'])

    df.columns = [replace_column_names(str(col)) for col in df.columns]

    styler = df.style.format(
        precision=4,
        na_rep="-"
    ).hide(axis="index")

    dataset_name = dataset_name.removesuffix(".csv")

    if filename is None:
        filename = dataset_name.replace(" ", "_").lower()

    latex_code = styler.to_latex(
        environment="longtable",
        hrules=True,
        column_format="l" + "c" * (len(df.columns) - 1),
        caption=f"Wyniki klasyfikacji dla zbioru danych: \\texttt{{{dataset_name}}} ",        label=f"tab:results_{filename}"
    )

    file_path = os.path.join(output_dir, f"{filename}_results.tex")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    print(f"Zapisano tabelę LaTeX do: {file_path}")


def replace_dataframe_values(df: pd.DataFrame, columns_to_process: list = None) -> pd.DataFrame:
    """
    Replace text values in DataFrame using NAMES_MAP.
    If columns_to_process is None, processes all columns with object dtype.
    If columns_to_process is provided, processes only those columns.
    """
    df_copy = df.copy()

    if columns_to_process is None:
        # Process all object/string type columns
        cols_to_check = df_copy.columns[df_copy.dtypes == 'object'].tolist()
    else:
        # Process only specified columns
        cols_to_check = [col for col in columns_to_process if col in df_copy.columns]

    for col in cols_to_check:
        for key, value in NAMES_MAP.items():
            # Case-insensitive replacement in string columns
            df_copy[col] = df_copy[col].astype(str).str.replace(
                key, value, case=False, regex=False
            )

    return df_copy
