import numpy as np
import cv2
import winding
from core.geometry import intersect_lines


# ------------------------------------------------------------------------------------------------

def _point_between(point, point1, point2):
    return (point1[0] - point[0]) * (point[0] - point2[0]) >= 0 and \
           (point1[1] - point[1]) * (point[1] - point2[1]) >= 0


# ------------------------------------------------------------------------------------------------

class Polyline:
    def __init__(self, points):
        """
        Creates a polyline from a list of points.
        """

        self._points = points

        self._enclosing_polygon = None
        self._enclosed_above = None


# ------------------------------------------------------------------------------------------------

    def _init_side_checks(self):
        """
        Private method that performs preparations for self.side() calls. The polyline will
        be intersected and enclosed into a large square, in order to define a closed polygon
        and resolve side() calls using the winding algorithm.
        """

        self._enclosing_polygon = []
        self._enclosing_polygon += self._points

        # TODO: need to ensure we always have a large enough value
        inf = 1e6
        inf_square = [(-inf, +inf),     # bottom-left
                      (+inf, +inf),     # bottom-right
                      (+inf, -inf),     # top-right
                      (-inf, -inf)]     # top-left

        line = (self._points[-1], self._points[-2])
        p, nxt_i = None, None

        for i in range(4):
            p = intersect_lines(line, (inf_square[i], inf_square[(i + 1) % 4]))
            if _point_between(self._points[-1], p, self._points[-2]) and \
               _point_between(p, inf_square[i], inf_square[(i + 1) % 4]):
                nxt_i = i + 1
                p = (int(p[0]), int(p[1]))
                break

        if p is None:
            assert False

        line = (self._points[0], self._points[1])
        crt_point = p
        for j in range(nxt_i, nxt_i + 5):
            nxt_point = inf_square[j % 4]
            self._enclosing_polygon.append(crt_point)

            p = intersect_lines(line, (crt_point, nxt_point))
            if _point_between(p, crt_point, nxt_point) and \
               _point_between(self._points[0], p, self._points[1]):
                p = (int(p[0]), int(p[1]))
                self._enclosing_polygon.append(p)
                break

            crt_point = nxt_point


# ------------------------------------------------------------------------------------------------

    def intersect_line_segment(self, line):
        """
        Intersects this polyline with a line.
        :param line: the line in the format (point1, point2),
                     where each point is specified as (x, y).
        :return: the (x, y) intersection point.
        """

        for i in range(len(self._points) - 1):
            p1 = self._points[i]
            p2 = self._points[i + 1]

            p = intersect_lines(line, (p1, p2))
            if _point_between(p, p1, p2) and _point_between(p, line[0], line[1]):
                return p

        return None


# ------------------------------------------------------------------------------------------------

    def side(self, point):
        """
        Returns the side of the polyline that the given point lies on.
        :param point: the (x, y) coordinates of the point to check.
        :return: True/False, depending on the side the point is on.
        """

        if self._enclosing_polygon is None:
            self._init_side_checks()

        return winding.wn_PnPoly(point, self._enclosing_polygon) != 0


# ------------------------------------------------------------------------------------------------

    def render(self, frame):
        pass


# ------------------------------------------------------------------------------------------------

def _test():
    points = [(10, 10), (60, 50), (20, 120), (60, 150), (200, 20)]
    polyline = Polyline(points)

    frame = np.zeros((512, 512, 3), dtype=np.uint8)

    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], color=(255, 0, 0), thickness=2)

    def callback(event, x, y, flags, param):
        del event, flags, param
        print('\r' + str(polyline.side((x, y))), end='')

    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', callback)
    cv2.waitKey(-1)


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    _test()


# ------------------------------------------------------------------------------------------------
