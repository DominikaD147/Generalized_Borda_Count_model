import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from generalized_borda_count import aggregation
from generalized_borda_count.enums.aggregation_type import AggregationType
from generalized_borda_count.enums.interval_orders import IntervalOrder
from generalized_borda_count.enums.model_types import ModelType

RNG = 42

DEFAULT_MODEL_GROUPS = {
    ModelType.KNN: (
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsClassifier(n_neighbors=5),
        KNeighborsClassifier(n_neighbors=9),
        KNeighborsClassifier(n_neighbors=4),
    ),
    ModelType.SVM: (
        SVC(kernel='rbf', C=1.0, probability=True, random_state=RNG),
        SVC(kernel='poly', C=1.0, probability=True, random_state=RNG),
        SVC(kernel='linear', C=1.0, probability=True, random_state=RNG),
    ),
    ModelType.DT: (
        DecisionTreeClassifier(max_depth=5, random_state=RNG),
        DecisionTreeClassifier(max_depth=10, random_state=RNG),
        DecisionTreeClassifier(max_depth=15, random_state=RNG),
        DecisionTreeClassifier(max_depth=30, random_state=RNG)
    ),
    ModelType.RF: (
        RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RNG),
        RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RNG),
        RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RNG),
        RandomForestClassifier(n_estimators=500, max_depth=5, random_state=RNG),
    ),
    # ModelType.MLP: (
    #     MLPClassifier(
    #         hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs',
    #         max_iter=2000, early_stopping=True, random_state=RNG
    #     ),
    #     MLPClassifier(
    #         hidden_layer_sizes=(100, 50, 25), activation='relu', solver='lbfgs',
    #         max_iter=2000, early_stopping=True, random_state=RNG
    #     ),
    #     MLPClassifier(
    #         hidden_layer_sizes=(100, 50, 25), activation='tanh', solver='lbfgs',
    #         max_iter=2000, early_stopping=True, random_state=RNG
    #     ),
    # )
}

base_orders = [
    (IntervalOrder.LEX1, None, None),
    (IntervalOrder.LEX2, None, None),
    (IntervalOrder.XU_YAGER, None, None),
]

alpha_beta_values = np.around(np.arange(0.1, 1.0, 0.1), 1)

alpha_beta_orders = [
    (IntervalOrder.ALPHA_BETA, a, b)
    for a in alpha_beta_values
    for b in alpha_beta_values
    if a != b
]

ORDERS = base_orders + alpha_beta_orders

RESOLVE_TIES = True

LOWER_AGG_FUNC = aggregation.get_aggregation(AggregationType.HARMONIC)
UPPER_AGG_FUNC = aggregation.get_aggregation(AggregationType.MAX)