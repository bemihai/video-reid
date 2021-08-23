from collections import Counter


# ------------------------------------------------------------------------------------------------

class ModeFilter:
    """
    A mode filter, which returns the most frequent value among the recent ones.
    The window size (which is configurable in the constructor) indicates how many
    past elements are used when evaluating the mode.
    """

    def __init__(self, window_size=10):
        self._window_size = window_size
        self._old_values = []


# ------------------------------------------------------------------------------------------------

    def update(self, value):
        self._old_values.append(value)
        if len(self._old_values) > self._window_size:
            self._old_values = self._old_values[1:]

        c = Counter(self._old_values)
        return c.most_common(1)[0][0]


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    f = ModeFilter()
    print(f.update('a'))
    print(f.update('a'))
    print(f.update('a'))
    print(f.update(1))
    print(f.update(1))
    print(f.update(1))
    print(f.update(2))
    print(f.update(2))
    print(f.update(2))
    print(f.update(2))
    print(f.update(2))


# ------------------------------------------------------------------------------------------------
