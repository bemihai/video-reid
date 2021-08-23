import cv2
import numpy as np

point1 = point2 = None


# ------------------------------------------------------------------------------------------------

def select_points(event, x, y, flags, param):
    global point1, point2

    win_name, frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        frame_clone = np.copy(frame)
        cv2.line(frame_clone, point1, point2, (255, 0, 0), 2)
        cv2.imshow(win_name, frame_clone)
        cv2.waitKey(1)


# ------------------------------------------------------------------------------------------------

def select_line(frame, win_name='Select line'):
    """
    Remark: returns the points in the selected order.
    """
    print('press any key to accept the current line')
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points, param=(win_name, frame))
    cv2.imshow(win_name, frame)
    cv2.waitKey(-1)
    cv2.destroyWindow(win_name)

    return point1, point2


# ------------------------------------------------------------------------------------------------
