"""
General mathematical helper functions.
"""


# ------------------------------------------------------------------------------------------------

def clamp(val, low, high):
    """
    Clamp a value such that the result is in the [low, high] interval.
    """
    return low if val < low else high if val > high else val


# ------------------------------------------------------------------------------------------------
