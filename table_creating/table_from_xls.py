import pandas as pd

from generalized_borda_count.data_provider import DatasetName
from generalized_borda_count.utils.latex_exporter import save_ds_result_to_latex

def _save_table_to_latex(dataset_name, filepath, output_dir="latex_tables"):

    df = pd.read_excel(filepath)

    save_ds_result_to_latex(
        dataset_results=df,
        dataset_name=dataset_name,
        output_dir=output_dir
    )

def _save_best_worst_5_and_simple_borda_table(dataset_name, filepath, output_dir="latex_tables"):
    df = pd.read_excel(filepath)

    df = df[df['dataset'] == dataset_name]


    df_simple = df[df['method'] == 'Simple Borda']
    df_interval = df[df['method'] == 'Interval Borda']
    df_interval_sorted = df_interval.sort_values(by='balanced_accuracy', ascending=False)

    top_5 = df_interval_sorted.head(5)
    bottom_5 = df_interval_sorted.tail(5)

    df_final = pd.concat([top_5, bottom_5, df_simple]).drop_duplicates()
    df_final = df_final.reset_index(drop=True)

    if df_final.empty:
        print("Brak danych po filtrowaniu!")
    else:
        save_ds_result_to_latex(
            dataset_results=df_final,
            dataset_name=dataset_name,
            output_dir=output_dir
        )


# put xlsx file in table_creating dir and run this code to get summary tables for each dataset
filepath = "results_xlsx/no_tie_res/4_hm_max_borda_results_no_tie.xlsx"
interval_type = "hm-max"
tie_resolving = "no-tie-res"  # tie-res or no-tie-res

# tieresolving + interval creating way
subdir = tie_resolving + "_" + interval_type
output_dir = "latex_tables/" + subdir

for dataset in DatasetName:
    dataset_name = dataset.value
    _save_best_worst_5_and_simple_borda_table(
        dataset_name,
        filepath,
        output_dir=output_dir
    )

    # dataset_name.replace(" ", "_").lower()