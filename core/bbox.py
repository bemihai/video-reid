"""
Bounding box helper functions. The convention is that a bounding
box should be specified as [x_min, y_min, x_max, y_max].
"""

# from core.math import clamp


# ------------------------------------------------------------------------------------------------

def area(bbox):
    """
    Computes the area of a bounding box.
    """

    x1, y1, x2, y2 = bbox
    return (y2 - y1) * (x2 - x1)


# ------------------------------------------------------------------------------------------------

def compute_iou(bbox1, bbox2):
    """
    Calculates intersection over union (IoU, overlap) of the given boxes.
    """

    # calculate the intersection bounding box
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 >= x2 or y1 >= y2:
        return 0

    area_intersection = area((x1, y1, x2, y2))
    area_union = area(bbox1) + area(bbox2) - area_intersection
    iou = area_intersection / area_union

    return iou


# ------------------------------------------------------------------------------------------------

def compute_intersection_bbox(bbox1, bbox2):
    """
    Returns the intersection bounding box.
    """

    # calculate the intersection bounding box
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 > x2 or y1 > y2:
        # no intersection
        return None

    return [x1, y1, x2, y2]


# ------------------------------------------------------------------------------------------------

def scale(bbox, fx, fy):
    """
    Scale a bounding box according to different scaling factors
    on the x and y axes.
    """

    return [bbox[0] * fx, bbox[1] * fy, bbox[2] * fx, bbox[3] * fy]


# ------------------------------------------------------------------------------------------------

# def enlarge_bbox(bbox, W, H, enlarge_factor=1.1, enlarge_delta=10):
#     """
#     Enlarge a rectangle (bounding box) such that its
#     center position remains fixed. Also clamp it such that the
#     resulting box does not exceed image dimensions W, H.
#     """
#     min_x, min_y, max_x, max_y = bbox
#     w = max_x - min_x
#     h = max_y - min_y
#
#     w2 = w * enlarge_factor + enlarge_delta
#     h2 = h * enlarge_factor + enlarge_delta
#
#     cx = (min_x + max_x) // 2
#     cy = (min_y + max_y) // 2
#
#     min_x2 = cx - w2 // 2
#     max_x2 = cx + w2 // 2
#     min_y2 = cy - h2 // 2
#     max_y2 = cy + h2 // 2
#
#     min_x2 = clamp(min_x2, 0, W)
#     max_x2 = clamp(max_x2, 0, W)
#     min_y2 = clamp(min_y2, 0, H)
#     max_y2 = clamp(max_y2, 0, H)
#
#     if min_x2 == max_x2:
#         min_x2 = clamp(min_x2 - 2, 0, W)
#         max_x2 = clamp(max_x2 + 2, 0, W)
#
#     if min_y2 == max_y2:
#         min_y2 = clamp(min_y2 - 2, 0, H)
#         max_y2 = clamp(max_y2 + 2, 0, H)
#
#     return int(min_x2), int(min_y2), int(max_x2), int(max_y2)

# ------------------------------------------------------------------------------------------------
