import numpy as np
import cv2
from enum import Enum


# ------------------------------------------------------------------------------------------------

class ClickType(Enum):
    Left = 1
    Right = 2


# ------------------------------------------------------------------------------------------------

def _mouse_handler(event, x, y, flags, self):
    if self._click_handler is not None:
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            row = (y - self._margins[0]) // self._row_height
            col = (x - self._margins[2]) // self._col_width
            if 0 <= row < self._n_rows and 0 <= col < self._n_cols:
                click_type = ClickType.Left if event == cv2.EVENT_LBUTTONDOWN \
                    else ClickType.Right
                self._click_handler(row, col, click_type, self._click_user_data)


# ------------------------------------------------------------------------------------------------

class ImageTable:
    """
    A table that can display images.
    """

    def __init__(self, n_rows, n_cols,
                 cell_size,
                 img_size=None,
                 win_title='Images',
                 bg_color=(255, 255, 255),
                 margins=(10, 10, 10, 10),
                 show_grid=True,
                 custom_grid=None,
                 grid_thickness=1,
                 grid_color=(255, 0, 0),
                 highlight_color=(255, 0, 0)):
        """
        :param n_rows: the number of rows in the table.
        :param n_cols: the number of columns in the table.
        :param cell_size: the size (w, h) of a cell in the table.
        :param img_size: size (w, h) to which images are resized.
                         If not specified, images are not re-sized.
        :param win_title: the title of the window containing the table.
        :param bg_color: background color of the table.
        :param margins: (top, bottom, left, right) margins for the table.
        :param highlight_color: color around a highlighted cell.
        :param show_grid: indicates whether to display a grid of lines.
        :param custom_grid: allows to specify which lines of the grid to
                            be drawn. It should be a pair (h_lines, v_lines),
                            where h_lines is the list of (row, col) indicating
                            the cells for which top border is drawn, and
                            v_lines is a list of (row, col) indicating the
                            cells for which left border is drawn.
        :param grid_color: color of the grid lines.
        :param grid_thickness: thickness of the grid lines.
        :param click_handler: method to be called when the user clicks the table.
        :param click_user_data: custom data to be passed to the click handler.
        """

        self._shown = False
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._col_width = cell_size[0]
        self._row_height = cell_size[1]
        self._img_size = img_size
        self._win_title = win_title
        self._bg_color = bg_color
        self._margins = margins
        self._highlight_color = highlight_color
        self._show_grid = show_grid
        self._custom_grid = custom_grid
        self._grid_color = grid_color
        self._grid_thickness = grid_thickness

        self._img, self._img_width, self._img_height = self._init_image()


# ------------------------------------------------------------------------------------------------

    def _init_image(self):
        img_height = self._margins[0] + self._margins[1] + \
                     self._n_rows * self._row_height

        img_width = self._margins[2] + self._margins[3] + \
                    self._n_cols * self._col_width

        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img[:, :, 0] = self._bg_color[0]
        img[:, :, 1] = self._bg_color[1]
        img[:, :, 2] = self._bg_color[2]

        return img, img_width, img_height


# ------------------------------------------------------------------------------------------------

    def _top_pixel_y(self, row):
        return self._margins[0] + row * self._row_height


# ------------------------------------------------------------------------------------------------

    def _bottom_pixel_y(self, row):
        return self._margins[0] + (row + 1) * self._row_height


# ------------------------------------------------------------------------------------------------

    def _left_pixel_x(self, col):
        return self._margins[2] + col * self._col_width


# ------------------------------------------------------------------------------------------------

    def _right_pixel_x(self, col):
        return self._margins[2] + (col + 1) * self._col_width


# ------------------------------------------------------------------------------------------------

    def _center_pixel_xy(self, row, col):
        return ((self._left_pixel_x(col) + self._right_pixel_x(col)) // 2,
                (self._top_pixel_y(row) + self._bottom_pixel_y(row)) // 2)


# ------------------------------------------------------------------------------------------------

    def _render_grid(self):
        if self._custom_grid is not None:
            self._render_custom_grid()
            return

        # horizontal lines
        crt_y = self._margins[0]
        for i in range(0, self._n_rows + 1):
            point1 = (self._margins[2], crt_y)
            point2 = (self._img_width - self._margins[3], crt_y)
            cv2.line(self._img, point1, point2, self._grid_color, self._grid_thickness)
            crt_y += self._row_height

        # vertical lines
        crt_x = self._margins[2]
        for i in range(0, self._n_cols + 1):
            point1 = (crt_x, self._margins[0])
            point2 = (crt_x, self._img_height - self._margins[1])
            cv2.line(self._img, point1, point2, self._grid_color, self._grid_thickness)
            crt_x += self._col_width


# ------------------------------------------------------------------------------------------------

    def _render_custom_grid(self):
        assert self._custom_grid is not None
        h_lines, v_lines = self._custom_grid

        # horizontal lines
        for row, col in h_lines:
            point1 = self._left_pixel_x(col), self._top_pixel_y(row)
            point2 = self._right_pixel_x(col), self._top_pixel_y(row)
            cv2.line(self._img, point1, point2, self._grid_color, self._grid_thickness)

        # vertical lines
        for row, col in v_lines:
            point1 = self._left_pixel_x(col), self._top_pixel_y(row)
            point2 = self._left_pixel_x(col), self._bottom_pixel_y(row)
            cv2.line(self._img, point1, point2, self._grid_color, self._grid_thickness)


# ------------------------------------------------------------------------------------------------

    def _render_image(self, row, col, img):
        img_h, img_w = img.shape[0], img.shape[1]
        cx, cy = self._center_pixel_xy(row, col)
        x0 = cx - img_w // 2
        x1 = x0 + img_w
        y0 = cy - img_h // 2
        y1 = y0 + img_h

        self._img[y0: y1, x0: x1, :] = img


# ------------------------------------------------------------------------------------------------

    def _render_text(self, row, col, text):
        pass


# ------------------------------------------------------------------------------------------------

    def _fill_cell_uniform_color(self, row, col, bg_color):
        y0, y1 = self._top_pixel_y(row), self._bottom_pixel_y(row)
        x0, x1 = self._left_pixel_x(col), self._right_pixel_x(col)
        self._img[y0:y1, x0:x1, 0] = bg_color[0]
        self._img[y0:y1, x0:x1, 1] = bg_color[1]
        self._img[y0:y1, x0:x1, 2] = bg_color[2]


# ------------------------------------------------------------------------------------------------

    def show(self):
        """
        Shows the window containing the table.
        Needs to be called only once.
        """

        if self._render_grid():
            self._render_grid()

        cv2.imshow(self._win_title, self._img)

        self._shown = True


# ------------------------------------------------------------------------------------------------

    def set_click_handler(self, click_handler, user_data):
        """
        Sets a function that will be called when a cell is clicked.
        The handler will receive the row and column of the clicked cell.
        """
        cv2.namedWindow(self._win_title)
        cv2.setMouseCallback(self._win_title, _mouse_handler, param=self)
        self._click_handler = click_handler
        self._click_user_data = user_data


# ------------------------------------------------------------------------------------------------

    def set_cell_image(self, row, col, image, highlight=False):
        assert 0 <= row < self._n_rows
        assert 0 <= col < self._n_cols

        if highlight:
            self._fill_cell_uniform_color(row, col, self._highlight_color)
        else:
            self._fill_cell_uniform_color(row, col, self._bg_color)

        if self._img_size is not None:
            image = cv2.resize(image, self._img_size)

        if self._show_grid:
            self._render_grid()
        self._render_image(row, col, image)

        cv2.imshow(self._win_title, self._img)


# ------------------------------------------------------------------------------------------------

    def set_cell_text(self, row, col, text):
        assert False, 'Not implemented'

        self._fill_cell_uniform_color(row, col)
        if self._show_grid:
            self._render_grid()
        self._render_text(row, col, text)

        cv2.imshow(self._win_title, self._img)


# ------------------------------------------------------------------------------------------------

    def clear_cell(self, row, col):
        self._fill_cell_uniform_color(row, col, self._bg_color)
        if self._show_grid:
            self._render_grid()

        cv2.imshow(self._win_title, self._img)


# ------------------------------------------------------------------------------------------------

    def set_custom_grid(self, custom_grid):
        self._custom_grid = custom_grid


# ------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    import os
    max_n_identities = 7
    img_top = cv2.imread(os.path.join(r'C:\work\multi-camera-tracking\resources\tracking\person.png'))
    img_info = cv2.imread(os.path.join(r'C:\work\multi-camera-tracking\resources\tracking\info.png'))
    # img_bottom = cv2.imread(os.path.join('data', 'no_match.png'))

    custom_grid_h_lines = []
    for r in range(3):
        for c in range(max_n_identities):
            custom_grid_h_lines.append((r, c))

    custom_grid_v_lines = [(1, 0), (1, max_n_identities)]
    for c in range(max_n_identities + 1):
        custom_grid_v_lines.append((0, c))

    custom_grid = (custom_grid_h_lines, custom_grid_v_lines)

    identity_table = ImageTable(2, max_n_identities,
                                cell_size=(110, 150),
                                img_size=None,
                                win_title='Tracked Identities',
                                custom_grid=custom_grid,
                                grid_thickness=1,
                                grid_color=(175, 0, 0),
                                margins=(20, 20, 20, 20))

    img_size = (100, 140)
    img_top = cv2.resize(img_top, img_size)
    for c in range(max_n_identities):
        identity_table.set_cell_image(0, c, img_top)

    img_size = (max_n_identities * 100, 140)
    #img_info = cv2.resize(img_info, img_size)
    identity_table.set_cell_image(1, max_n_identities // 2, img_info)

    identity_table.show()

    while True:
        k = cv2.waitKey(100) & 0xFF
        if k == ord('q'):
            break

    # import glob
    #
    # def table_click_handler(r, c):
    #     print(r, c)
    #
    # table = ImageTable(5, 2, cell_size=(110, 150), img_size=(100, 140),
    #                    win_title='Identities',
    #                    grid_thickness=1,
    #                    grid_color=(175, 0, 0))
    # table.set_click_handler(table_click_handler)
    #
    # table.show()
    # table.clear_cell(1, 1)
    #
    # all_imgs = glob.glob(r'D:\reid\DukeMTMC-reID\bounding_box_train\*.jpg')
    #
    # for i in range(5):
    #     img = cv2.imread(all_imgs[i * 72])
    #     table.set_cell_image(i, 0, img)
    #
    #     img = cv2.imread(all_imgs[i * 137])
    #     table.set_cell_image(i, 1, img)
    #
    # img = cv2.imread(all_imgs[7 * 72])
    # table.set_cell_image(3, 0, img, highlight=True)
    #
    # # table.clear_cell(1, 1)
    # # table.clear_cell(0, 4)
    #
    # s = 0
    # i = 0
    # while True:
    #     table.set_cell_image(3, 0, cv2.imread(all_imgs[i]))
    #     i += 1
    #
    #     k = cv2.waitKey(100) & 0xFF
    #     if k == ord('q'):
    #         break


# ------------------------------------------------------------------------------------------------
