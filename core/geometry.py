"""
General geometry helper functions.
"""

import numpy as np
from core.math import clamp


# ------------------------------------------------------------------------------------------------

def distance(point1, point2):
    """
    Compute the euclidean distance between two points.
    :param point1: the (x, y) coordinates of the first point.
    :param point2: the (x, y) coordinates of the second point.
    """

    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


# ------------------------------------------------------------------------------------------------

def distance2(point1, point2):
    """
    Compute the squared euclidean distance between two points.
    :param point1: the (x, y) coordinates of the first point.
    :param point2: the (x, y) coordinates of the second point.
    """

    x1, y1 = point1
    x2, y2 = point2
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)


# ------------------------------------------------------------------------------------------------

def point_to_line_distance(point, line):
    """
    Compute the perpendicular distance from a point to a line.
    :param point: the (x, y) point coordinates.
    :param line: the (a, b, c) parameter of the line equation, ax + by + c = 0.
    """

    x, y = point
    a = line[0]
    b = line[1]
    c = line[2]
    dist = np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
    return dist


# ------------------------------------------------------------------------------------------------

def enlarge_bbox(bbox, img_w, img_h, enlarge_factor=1.1, enlarge_delta=10):
    """
    Enlarge a rectangle (bounding box) such that its
    center position remains fixed. Also clamp it such that the
    resulting box does not exceed image dimensions W, H.
    """
    min_x, min_y, max_x, max_y = bbox
    w = max_x - min_x
    h = max_y - min_y

    w2 = w * enlarge_factor + enlarge_delta
    h2 = h * enlarge_factor + enlarge_delta

    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    min_x2 = cx - w2 // 2
    max_x2 = cx + w2 // 2
    min_y2 = cy - h2 // 2
    max_y2 = cy + h2 // 2

    min_x2 = clamp(min_x2, 0, img_w)
    max_x2 = clamp(max_x2, 0, img_w)
    min_y2 = clamp(min_y2, 0, img_h)
    max_y2 = clamp(max_y2, 0, img_h)

    if min_x2 == max_x2:
        min_x2 = clamp(min_x2 - 2, 0, img_w)
        max_x2 = clamp(max_x2 + 2, 0, img_w)

    if min_y2 == max_y2:
        min_y2 = clamp(min_y2 - 2, 0, img_h)
        max_y2 = clamp(max_y2 + 2, 0, img_h)

    return int(min_x2), int(min_y2), int(max_x2), int(max_y2)


# ------------------------------------------------------------------------------------------------

def intersect_lines(line1, line2):
    """
    Compute the intersection point of two lines.
    :param line1: the first line.
    :param line2: the second line.
    :return: the (x, y) coordinates of the intersection point,
             or None, if the lines are parallel.
    """

    (p1, p2) = line1
    (x1, y1) = p1
    (x2, y2) = p2

    (p3, p4) = line2
    (x3, y3) = p3
    (x4, y4) = p4

    denominator = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

    if denominator == 0:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return px, py


# ------------------------------------------------------------------------------------------------
