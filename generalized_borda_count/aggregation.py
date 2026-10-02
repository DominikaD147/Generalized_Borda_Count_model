from statistics import harmonic_mean
from typing import Callable
from aggregationslib.aggregations import A_hm, A_ex
from generalized_borda_count.enums.aggregation_type import AggregationType
import numpy as np


def safe_harmonic_mean(predictions, epsilon=1e-7):
    arr = np.asarray(predictions)
    arr_safe = np.where(arr == 0, epsilon, arr)
    return len(arr_safe) / np.sum(1.0 / arr_safe, axis=0)


def zero_aware_harmonic_mean(predictions):
    '''
    if any of predictions is zero, return zero, otherwise return harmonic mean
    '''
    arr = np.asarray(predictions)

    if np.any(arr == 0):
        return 0.0

    return len(arr) / np.sum(1.0 / arr, axis=0)

AGGREGATIONS_REGISTRY = {
    AggregationType.MIN: min,
    AggregationType.MAX: max,

    # AggregationType.HARMONIC: A_hm(),
    # AggregationType.HARMONIC: harmonic_mean,
    # AggregationType.HARMONIC: safe_harmonic_mean,
    AggregationType.HARMONIC: zero_aware_harmonic_mean,
    AggregationType.EX: A_ex(r=5),
}

def get_aggregation(agg: AggregationType) -> Callable:
    if agg not in AGGREGATIONS_REGISTRY:
        raise ValueError(f"Unsupported aggregation type: {agg}")

    return AGGREGATIONS_REGISTRY[agg]


# lista = [1,3,6,2,4,5, 0]
# agg = get_aggregation(AggregationType.HARMONIC)
# print(agg(lista))
