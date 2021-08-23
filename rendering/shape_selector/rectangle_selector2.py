import cv2
import numpy as np


# ------------------------------------------------------------------------------------------------

point1 = None
point2 = None


# ------------------------------------------------------------------------------------------------

def _select_points(event, x, y, flags, param):
    global point1, point2

    win_name, frame = param

    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        point2 = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        point1 = None
        point2 = None
        cv2.imshow(win_name, frame)
    elif event == cv2.EVENT_LBUTTONUP:
        if point1 is not None:
            point2 = (x, y)
            frame_clone = np.copy(frame)
            cv2.rectangle(frame_clone, point1, point2, (0, 255, 0), 2)
            cv2.imshow(win_name, frame_clone)
            cv2.waitKey(1)

    if point1 is not None:
        if point2 is None:
            frame_clone = np.copy(frame)
            cv2.rectangle(frame_clone, point1, (x, y), (0, 255, 0), 2)
            cv2.imshow(win_name, frame_clone)


# ------------------------------------------------------------------------------------------------

def select_rectangle(frame, win_name='Select rectangle (press any key to quit)'):
    """
    Remark: returns the points in the selected order.
    Select first top-left, then bottom-right.
    """
    global point1, point2
    point1 = None
    point2 = None

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _select_points, param=(win_name, frame))
    cv2.imshow(win_name, frame)

    while True:
        k = cv2.waitKey(10)
        if k > 0:
            cv2.destroyWindow(win_name)
            return None, k
        if point1 is not None and point2 is not None:
            cv2.destroyWindow(win_name)
            return (point1, point2), -1


# ------------------------------------------------------------------------------------------------

def _test():
    img = cv2.imread(r'C:\work\multi-camera-tracking\resources\frame1_651.png')
    points, key = select_rectangle(img)
    print(points, key)


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    _test()

# ------------------------------------------------------------------------------------------------
