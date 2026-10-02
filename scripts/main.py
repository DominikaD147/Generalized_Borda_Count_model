import pandas as pd
from config import DEFAULT_MODEL_GROUPS, RNG, ORDERS, LOWER_AGG_FUNC, UPPER_AGG_FUNC
from generalized_borda_count.data_provider import get_dataset, DatasetName
from generalized_borda_count.enums.interval_orders import IntervalOrder
from generalized_borda_count.evaluation import run_kfold_validation
from generalized_borda_count.utils.excel_exporter import save_results_to_xls
from generalized_borda_count.utils.latex_exporter import save_ds_result_to_latex

def print_results(results):
    df = pd.DataFrame([results])
    print(df.round(3))
    print("\n")


def _get_row(
        dataset: DatasetName,
        method,
        order: IntervalOrder | None,
        results,
        alpha=None,
        beta=None
):
    order = order.value if order else None

    row = {
        'method': method,
        'dataset': dataset.value,
        'order': order,
        'alpha': alpha,
        'beta': beta,
        **results
    }

    return row


def main():
    model_groups = DEFAULT_MODEL_GROUPS.copy()
    all_results = []
    sets = DatasetName

    for dataset_name in sets:
        dataset_results = []

        n_splits = 3 if dataset_name == DatasetName.ZOO else 5
        print(dataset_name.value)

        X, y = get_dataset(dataset_name)

        for order, alpha, beta in ORDERS:
            print("\n--- Order:", order.value)
            if alpha:
                print(f"alpha = {alpha}")
                print(f"beta = {beta}")


            # Generalized Borda Count evaluation
            results = run_kfold_validation(
                X, y,
                model_groups,
                order=order,
                alpha=alpha,
                beta=beta,
                n_splits=n_splits,
                use_imputation=True,
                use_scaling=True,
                lower_agg_func=LOWER_AGG_FUNC,
                upper_agg_func=UPPER_AGG_FUNC
            )

            # print_results(results)
            row = _get_row(
                dataset_name,
                'Interval Borda',
                order,
                results,
                alpha=alpha,
                beta=beta
            )
            dataset_results.append(row)
            all_results.append(row)

        # Simple Borda Count evaluation
        print(f"Simple Borda count for {dataset_name.value}:")

        results = run_kfold_validation(
            X, y,
            model_groups,
            use_imputation=True,
            use_scaling=True,
            simple_borda=True,
            n_splits=n_splits
        )

        # print_results(results)
        row = _get_row(
            dataset_name,
            'Simple Borda',
            None,
            results
        )
        all_results.append(row)
        dataset_results.append(row)

        save_ds_result_to_latex(dataset_results, dataset_name.value)

    if all_results:
        save_results_to_xls(all_results)


if __name__ == '__main__':
    main()