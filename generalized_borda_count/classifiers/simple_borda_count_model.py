import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SimpleBordaCountModel:
    def __init__(
            self,
            models: list,
            use_imputation: bool = False,
            use_scaling: bool = False,
    ):
        self.models = models
        self.y_labels = None
        self.n_classes = None
        self.trained_models: list = None
        self.use_imputation = use_imputation
        self.use_scaling = use_scaling


    def fit(self, X_train, y_train):
        self.y_labels = np.unique(y_train)
        self.n_classes = len(self.y_labels)

        trained = []

        for model in self.models:
            wrapped_model = self._wrap_model_in_pipeline(model)
            wrapped_model.fit(X_train, y_train)
            trained.append(wrapped_model)

        self.trained_models = trained


    def predict(self, X):
        n_samples = X.shape[0]
        total_borda_scores = np.zeros((n_samples, self.n_classes))

        for model in self.trained_models:
            probs = model.predict_proba(X)

            for sample in range(n_samples):
                ranking = np.argsort(-probs[sample])  # array of idx to sort (minus for desc)
                for rank, class_idx in enumerate(ranking):
                    total_borda_scores[sample, class_idx] += rank

        predicted_classes_idx = np.argmin(total_borda_scores, axis=1)
        final_predictions =  self.y_labels[predicted_classes_idx]
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

