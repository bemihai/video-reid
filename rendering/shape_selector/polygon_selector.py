import cv2
import numpy as np

points = []


# ------------------------------------------------------------------------------------------------

def _select_points(event, x, y, flags, param):
    global points

    win_name, frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        points.append(point)

        frame_clone = np.copy(frame)

        # draw the poly
        for i in range(1, len(points)):
            point1 = points[i - 1]
            point2 = points[i]

            cv2.line(frame_clone, point1, point2, (0, 255, 0), 2)

        cv2.imshow(win_name, frame_clone)
        cv2.waitKey(1)


# ------------------------------------------------------------------------------------------------

def select_polygon(frame, win_name='Select area'):
    """
    Remark: returns the points in the selected order.
    """
    global points
    points = []

    print('press any key to close and accept the current polygon')
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _select_points, param=(win_name, frame))
    cv2.imshow(win_name, frame)
    cv2.waitKey(-1)
    cv2.destroyWindow(win_name)

    return points


# ------------------------------------------------------------------------------------------------
