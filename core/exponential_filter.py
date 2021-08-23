class ExponentialFilter:
    """
    An exponential filter which might be used for
    smoothing a sequence of numbers.
    """

    def __init__(self, alpha=0.9):
        self._alpha = alpha
        self._old_value = None


# ------------------------------------------------------------------------------------------------

    def update(self, value):
        if self._old_value is None:
            self._old_value = value
        else:
            self._old_value = value * self._alpha + self._old_value * (1 - self._alpha)

        return self._old_value


# ------------------------------------------------------------------------------------------------

