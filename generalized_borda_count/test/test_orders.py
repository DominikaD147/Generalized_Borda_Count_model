from unittest import TestCase

from generalized_borda_count.enums.interval_orders import IntervalOrder
from generalized_borda_count.interval import Interval
from generalized_borda_count.orders import xu_yager_compare, sort_intervals


class Test(TestCase):
    def test_xu_yager_compare(self):
        interval1 = Interval(0, 6)  # greater
        interval2 = Interval(1, 5)
        interval3 = Interval(1, 6)

        self.assertEqual(xu_yager_compare(interval1, interval2), 1)
        self.assertEqual(xu_yager_compare(interval2, interval1), -1)
        self.assertEqual(xu_yager_compare(interval1, interval3), -1)
        self.assertEqual(xu_yager_compare(interval3, interval2), 1)


    def test_sort_intervals(self):
        labels_intervals = [
            (1, Interval(1, 2)),
            (2, Interval(1, 3)),
            (3, Interval(3, 4)),
            (4, Interval(1, 2)),
            (5, Interval(6, 7))
        ]

        expected_sorted = [
            (5, Interval(6, 7)),
            (3, Interval(3, 4)),
            (2, Interval(1, 3)),
            (1, Interval(1, 2)),
            (4, Interval(1, 2)),
        ]

        sorted_label_intervals = sort_intervals(labels_intervals, IntervalOrder.XU_YAGER)

        self.assertEqual(len(sorted_label_intervals), len(expected_sorted))

        for i in range(len(expected_sorted)):
            self.assertEqual(
                sorted_label_intervals[i][1].start,
                expected_sorted[i][1].start,
                f"Error in {i} position: start values are different"
            )
            self.assertEqual(
                sorted_label_intervals[i][1].end,
                expected_sorted[i][1].end,
                f"Error in {i} position: end values are different"
            )

        # self.assertEqual(sorted_label_intervals, expected_sorted)
