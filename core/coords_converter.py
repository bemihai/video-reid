class CoordinatesConverter:
    """
    Converts 3d coordinates from one reference system to another.
    """

    def __init__(self, origin=(0, 0, 0),
                 x_axis=(1, 0, 0),
                 y_axis=(0, 1, 0),
                 z_axis=(0, 0, 1)):
        """
        :param origin: the position of the origin of the new coordinate system,
                       expressed in terms of the old coordinate system.
        :param x_axis: the orientation of the x axis of the new coordinate system,
                       expressed in terms of the old coordinate system.
        :param y_axis: the orientation of the y axis of the new coordinate system,
                       expressed in terms of the old coordinate system.
        :param z_axis: the orientation of the z axis of the new coordinate system,
                       expressed in terms of the old coordinate system.
        :remark under the default values, the old and new systems are identical.
        """

        self._origin = origin
        self._x_axis = x_axis
        self._y_axis = y_axis
        self._z_axis = z_axis


# ------------------------------------------------------------------------------------------------

    def convert(self, x, y, z):
        dx = x - self._origin[0]
        dy = y - self._origin[1]
        dz = z - self._origin[2]

        new_x = dx * self._x_axis[0] + dy * self._x_axis[1] + dz * self._x_axis[2]
        new_y = dx * self._y_axis[0] + dy * self._y_axis[1] + dz * self._y_axis[2]
        new_z = dx * self._z_axis[0] + dy * self._z_axis[1] + dz * self._z_axis[2]

        return new_x, new_y, new_z


# ------------------------------------------------------------------------------------------------
