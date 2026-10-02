from functools import cmp_to_key

from generalized_borda_count.enums.interval_orders import IntervalOrder
from generalized_borda_count.interval import Interval

def xu_yager_compare(interval1: Interval, interval2: Interval):
    """
    Compares two intervals using XuYager order:
    compares middles of intervals, then their lengths

    :params interval1, interval2: Interval objects


    :return: 1 if interval1 is greater than interval2,
            -1 if interval1 is smaller than interval2,
            0 otherwise
    """
    if interval1.__eq__(interval2):
        return 0

    if interval1.start + interval1.end < interval2.start + interval2.end:
        return -1

    if (interval1.start + interval1.end == interval2.start + interval2.end
            and interval1.length() <= interval2.length()):
        return -1

    # if ((interval1.start + interval1.end == interval2.start + interval2.end)
    #         and interval1.length == interval2.length):
    #     return 0

    return 1


def lex1_compare(interval1: Interval, interval2: Interval):
    """
    Compares two intervals using first Lexicographical order:
    comparing intervals based on its start, then end values

    :params interval1, interval2: Interval objects

    :return: 1 if interval1 is greater than interval2,
            -1 if interval1 is smaller than interval2,
            0 otherwise
    """
    if interval1.__eq__(interval2):
        return 0

    if interval1.start < interval2.start:
        return -1

    if interval1.start == interval2.start and interval1.end <= interval2.end:
        return -1

    return 1

def lex2_compare(interval1: Interval, interval2: Interval):
    """
    Compares two intervals using second Lexicographical order:
    comparing intervals based on its end, then start values

    :params interval1, interval2: Interval objects

    :return: 1 if interval1 is greater than interval2,
            -1 if interval1 is smaller than interval2,
            0 otherwise
    """
    if interval1.__eq__(interval2):
        return 0

    if interval1.end < interval2.end:
        return -1

    if interval1.end == interval2.end and interval1.start <= interval2.start:
        return -1

    return 1

def count_k_alpha(interval: Interval, alpha):
    return alpha * interval.start + (1 - alpha) * interval.end

def alpha_beta_compare(interval1: Interval, interval2: Interval, alpha: float, beta: float):
    """
    Compares two intervals using Alpha Beta linear order:

    :param interval1: Interval object
    :param interval2 Interval object
    :param alpha: float value of [0, 1]
    :param beta: float value of [0, 1]

    :return: 1 if interval1 is greater than interval2,
            -1 if interval1 is smaller than interval2
    """
    if interval1.__eq__(interval2):
        return 0

    if alpha < 0 or alpha > 1:
        raise ValueError(f'Invalid value for alpha {alpha}')

    if alpha == beta:
        raise ValueError(f'Alpha cannot be equal beta alpha = {alpha}, beta = {beta}')

    k_alpha_1 = count_k_alpha(interval1, alpha)
    k_alpha_2 = count_k_alpha(interval2, alpha)

    if k_alpha_1 < k_alpha_2:
        return -1

    if k_alpha_1 == k_alpha_2:
        k_beta_1 = count_k_alpha(interval1, beta)
        k_beta_2 = count_k_alpha(interval2, beta)
        if k_beta_1 <= k_beta_2:
            return -1

    return 1


def sort_intervals(
        labels_intervals: list[tuple[int, Interval]],
        order: IntervalOrder,
        reverse = True,
        alpha: float = 0.1,
        beta: float = 0.2
):
    """
    Sorts the intervals

    :param reverse: if True: descending, if False: ascending
    :param order: order type to sort intervals (enum IntervalOrder)
    :param labels_intervals: A list of tuples of class index and Interval object
    :param alpha: put if you use alpha beta order
    :param beta: put if you use alpha beta order

    :return: A list of sorted Interval objects
    """

    match order:
        case IntervalOrder.XU_YAGER:
            key_func = lambda x: cmp_to_key(xu_yager_compare)(x[1])
        case IntervalOrder.LEX1:
            key_func = lambda x: cmp_to_key(lex1_compare)(x[1])
        case IntervalOrder.LEX2:
            key_func = lambda x: cmp_to_key(lex2_compare)(x[1])
        case IntervalOrder.ALPHA_BETA:
            def custom_compare(interval1: Interval, interval2: Interval):
                return alpha_beta_compare(interval1, interval2, alpha=alpha, beta=beta)
            key_func = lambda x: cmp_to_key(custom_compare)(x[1])
        case _:
            raise ValueError(f'Unknown order type {order}')

    return sorted(labels_intervals, key=key_func, reverse=reverse)
