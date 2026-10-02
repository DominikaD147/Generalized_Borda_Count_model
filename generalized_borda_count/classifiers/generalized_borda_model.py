from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RESOLVE_TIES
from generalized_borda_count.enums.interval_orders import IntervalOrder
from generalized_borda_count.enums.model_types import ModelType
from generalized_borda_count.interval import Interval
from generalized_borda_count import orders

GroupedModels = dict[ModelType, list[BaseEstimator]]
GroupedPredictionIntervals = dict[ModelType, list[list[Interval]]]  # modelType -> sample x class = interval


class GeneralizedBordaModel:
    def __init__(
            self,
            models: GroupedModels,
            use_imputation: bool = False,
            use_scaling: bool = False,
            resolve_ties: bool = RESOLVE_TIES,
            lower_agg_func: Callable = min,
            upper_agg_func: Callable = max
    ):
        self.models = models
        self.y_labels = None
        self.trained_models: GroupedModels = None
        self.use_imputation = use_imputation
        self.use_scaling = use_scaling
        self.resolve_ties = resolve_ties
        self.lower_agg_func = lower_agg_func
        self.upper_agg_func = upper_agg_func


    def fit(self, X_train, y_train):
        self.y_labels = np.unique(y_train)

        trained: GroupedModels = {}

        for model_type, model_group in self.models.items():
            trained[model_type] = []
            for model in model_group:
                wrapped_model = self._wrap_model_in_pipeline(model)
                wrapped_model.fit(X_train, y_train)
                trained[model_type].append(wrapped_model)

        self.trained_models = trained


    def predict(self, X_test, order: IntervalOrder, alpha=None, beta=None):
        prediction_intervals: GroupedPredictionIntervals = {}

        # 1. Create intervals for every group
        for model_type, models in self.trained_models.items():
            prediction_intervals[model_type] = []

            # Collect predictions of every model in group
            group_probas = [model.predict_proba(X_test) for model in models]  # 3D array -> models x samples x classes
            group_probas = np.stack(group_probas, axis=0)

            n_samples = X_test.shape[0]

            # Create interval for every possible class of x
            for i in range(n_samples):
                sample_intervals: list[tuple[int, Interval]] = self._create_intervals_for_sample(group_probas, i)
                sorted_intervals = orders.sort_intervals(
                    sample_intervals,
                    order,
                    alpha=alpha,
                    beta=beta
                )
                prediction_intervals[model_type].append(sorted_intervals)

        final_predictions = self._borda_count_aggregation(prediction_intervals, n_samples)
        return final_predictions

    def _wrap_model_in_pipeline(self, model: BaseEstimator):
        """
        wrap model in pipeline with preprocessing if needed or return model

        :param model:
        :return: wrapped model
        """

        steps = []

        if self.use_imputation:
            steps.append(('imputer', SimpleImputer(strategy='mean')))

        if self.use_scaling:
            steps.append(('scaler', StandardScaler()))

        if len(steps) > 0:
            steps.append(('classifier', model))
            return Pipeline(steps)
        else:
            return model


    def _create_intervals_for_sample(self, group_probas, sample_index) -> list[tuple[int, Interval]]:
        """
        For every possible class of sample computes min and max probas and creates interval of it

        :param group_probas: 3D array -> models x samples x classes
        :param sample_index: index of sample of X_test
        :return: list of intervals
        """
        n_classes = group_probas.shape[2]

        sample_intervals: list[Interval] = []
        for class_idx in range(n_classes):
            elems = group_probas[:, sample_index, class_idx]  # take all models probas for j class for i sample

            start = self.lower_agg_func(elems)
            end = self.upper_agg_func(elems)
            interval = Interval(start, end)

            sample_intervals.append((class_idx, interval))  # tuple contains class idx and interval of probas

        return sample_intervals

    def _borda_count_aggregation(self, prediction_intervals: GroupedPredictionIntervals, n_samples: int):
        """
        Aggregate ranks of every group using Borda Count

        : param prediction_intervals: dictionary of ranks of every sample sorted for every group
        : param n_samples: number of samples in set

        :return: list of final predictions (class labels)
        """
        n_classes = len(self.y_labels)

        total_borda_scores = np.zeros((n_samples, n_classes))
        wins_count = np.zeros((n_samples, n_classes))

        for model_type, rankings_per_sample in prediction_intervals.items():
            for sample_idx, sorted_intervals in enumerate(rankings_per_sample):
                for idx, (class_idx, interval) in enumerate(sorted_intervals):
                    total_borda_scores[sample_idx, class_idx] += idx

                    # if class wins
                    if idx == 0:
                        wins_count[sample_idx, class_idx] += 1

        return self._resolve_predictions(n_samples, total_borda_scores, wins_count)
        # final decision - class of minimum points
        # predicted_classes_idx = np.argmin(total_borda_scores, axis=1)
        #
        # final_predictions = self.y_labels[predicted_classes_idx]
        #
        # return final_predictions.tolist()

    def _resolve_predictions(self, n_samples, total_borda_scores, wins_count):
        predicted_classes_idx = []

        for i in range(n_samples):
            min_score = np.min(total_borda_scores[i])
            best_classes = np.where(total_borda_scores[i] == min_score)[0]

            if len(best_classes) == 1 or not self.resolve_ties:
                predicted_classes_idx.append(best_classes[0])
            else:  # there is a tie resolve it by number of wins
                tie_classes_wins_count = wins_count[i, best_classes]
                winner_idx = np.argmax(tie_classes_wins_count)
                predicted_classes_idx.append(best_classes[winner_idx])


        final_preds = self.y_labels[predicted_classes_idx]
        return final_preds.tolist()
