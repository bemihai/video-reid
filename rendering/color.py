import numpy as np


# ------------------------------------------------------------------------------------------------

def int_to_color(value):
    """
    Creates an RGB color from an integer value.
    Nearby integer values generate perceptually distinct colors.
    """
    return 23 * value % 256, 71 * value % 256, 97 * value % 256


# ------------------------------------------------------------------------------------------------

def labels_img_to_color_img(labels_img):
    """
    Creates an RGB image from a label (single channel) image.
    Nearby integer labels generate perceptually distinct colors.
    """

    color_img = np.zeros((labels_img.shape[0], labels_img.shape[1], 3), dtype=np.uint8)
    color_img[:, :, 0] = (labels_img * 23) % 256
    color_img[:, :, 1] = (labels_img * 71) % 256
    color_img[:, :, 2] = (labels_img * 97) % 256

    return color_img

# ------------------------------------------------------------------------------------------------
