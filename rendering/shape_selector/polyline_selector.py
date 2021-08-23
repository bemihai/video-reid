import cv2
import numpy as np
points = []


# -------------------------------------------------------------------------------------------------

def _display_info():
    print('+' + '-' * 60 + '+')
    print('| Click at multiple locations to draw the polyline and then  |')
    print('| press any key to close and accept the current polyline.    |')
    print('+' + '-' * 60 + '+')


# -------------------------------------------------------------------------------------------------

def _select_points(event, x, y, unused, param):
    global points
    del unused

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


# -------------------------------------------------------------------------------------------------

def select_polyline(frame, win_name='Select area'):
    """
    Remark: returns the points in the selected order.
    """
    global points
    points = []

    _display_info()

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, _select_points, param=(win_name, frame))
    cv2.imshow(win_name, frame)
    cv2.waitKey(-1)
    cv2.destroyWindow(win_name)

    return points


# -------------------------------------------------------------------------------------------------

def _test():
    # a black frame
    frame = np.zeros((512, 512, 3), dtype=np.uint8)

    polyline_points = select_polyline(frame)
    print(polyline_points)


# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    _test()


# -------------------------------------------------------------------------------------------------
