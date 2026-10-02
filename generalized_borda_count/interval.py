class Interval:
    def __init__(self, start=None, end=None, elems: list[float] | None = None):
        """
        :params: pass start and end of the interval or list of elements
        :param start: left side of the interval
        :param end: right side of the interval
        :param elems: elements of the interval
        """
        if elems is not None:
            self.start = min(elems)
            self.end = max(elems)
        else:
            if start is None or end is None:
                raise ValueError('start and end cannot be None when the interval is empty')
            if start > end:
                raise ValueError("Start must not be greater than end")
            self.start = start
            self.end = end

    def length(self):
        return self.end - self.start

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end

    def __str__(self):
        return f'[{self.start}, {self.end}]'
