from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import RNG
from generalized_borda_count.classifiers.generalized_borda_model import GeneralizedBordaModel
from generalized_borda_count.classifiers.simple_borda_count_model import SimpleBordaCountModel
from generalized_borda_count.enums.interval_orders import IntervalOrder


def run_kfold_validation(
        X,
        y,
        model_groups: dict,
        order: IntervalOrder | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        n_splits = 5,
        use_imputation = False,
        use_scaling = False,
        simple_borda: bool = False,
        lower_agg_func: Callable = None,
        upper_agg_func: Callable = None
):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    accuracy_scores = []
    balanced_accuracy_scores = []


    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if simple_borda:
            clf = SimpleBordaCountModel(
                flatten_model_dict(model_groups),
                use_imputation=use_imputation,
                use_scaling=use_scaling
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf = GeneralizedBordaModel(
                model_groups,
                use_imputation=use_imputation,
                use_scaling=use_scaling,
                lower_agg_func=lower_agg_func,
                upper_agg_func=upper_agg_func
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test, order, alpha=alpha, beta=beta)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        accuracy_scores.append(accuracy)
        balanced_accuracy_scores.append(balanced_accuracy)

    # standard deviations
    acc_std = np.std(accuracy_scores)
    bal_acc_std = np.std(balanced_accuracy_scores)


    mean_accuracy = np.mean(accuracy_scores)
    mean_balanced_accuracy = np.mean(balanced_accuracy_scores)

    return {
        'accuracy': mean_accuracy,
        'accuracy_std': acc_std,
        'balanced_accuracy': mean_balanced_accuracy,
        'balanced_accuracy_std': bal_acc_std,
    }

def flatten_model_dict(models: dict) -> list:
    all_classifiers = [
        clf
        for classifiers in models.values()
        for clf in classifiers
    ]

    return all_classifiers