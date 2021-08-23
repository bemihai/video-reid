import numpy as np


def bbox_to_rectangle(bbox):
    # convert from (x1, y1, x2, y2) -- the bounding box format, to the rectangle format used
    # by the tracker (x1, y1, w, h)
    x1, y1, x2, y2 = bbox
    return x1, y1, x2 - x1, y2 - y1


def rectangle_to_bbox(rectangle):
    # convert from (x1, y1, w, h)-- the rectangle format used by the tracker, to the bounding box
    # format (x1, y1, x2, y2)
    x1, y1, w, h = rectangle
    return int(x1), int(y1), int(x1 + w), int(y1 + h)


# ------------------------------------------------------------------------------------------------

def to_bbox(pose):
    """
    Convert a pose into a bounding box with format (x1, y1, x2, y2).
    """

    min_x = None
    min_y = None
    max_x = None
    max_y = None
    for i in range(pose.shape[0]):
        x = pose[i][0]
        y = pose[i][1]
        if x != 0 and y != 0:
            if min_x is None or x < min_x:
                min_x = x
            if max_x is None or x > max_x:
                max_x = x
            if min_y is None or y < min_y:
                min_y = y
            if max_y is None or y > max_y:
                max_y = y

    return min_x, min_y, max_x, max_y


# ------------------------------------------------------------------------------------------------

def is_valid(pose, joint):
    """
    Check if a specific joint was detected.
    """

    return pose[joint][0] != 0 or pose[joint][1] != 0


# ------------------------------------------------------------------------------------------------

def filter_poses(poses, min_avg_confidence, min_n_detected_joints):
    """
    Filter poses by average joint confidence and by number of detected joints.
    """

    keep_idxs = np.zeros(poses.shape[0], dtype=bool)
    for i in range(poses.shape[0]):
        keep_idxs[i] = np.mean(poses[i, poses[i, :, 1] > 0, 2]) > min_avg_confidence and \
                       np.sum(poses[i, :, 1] > 0) > min_n_detected_joints

    return poses[keep_idxs, :, :], keep_idxs


# ------------------------------------------------------------------------------------------------

def scale_poses(poses, fx, fy):
    """
    Scale all the poses according to separated
    factors on the x and y axes.
    """

    poses[:, :, 0] = (poses[:, :, 0] * fx).astype(poses.dtype)
    poses[:, :, 1] = (poses[:, :, 1] * fy).astype(poses.dtype)

    return poses


# ------------------------------------------------------------------------------------------------
