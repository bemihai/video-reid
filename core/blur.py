import cv2


# ------------------------------------------------------------------------------------------------

def blur_bbox(img, bbox, kernel_size=(50, 50)):
    """
    Blur a portion of the image.
    :param img: the input image to be blurred.
    :param bbox: the bounding box (x1, y1, x2, y2) where the blur is applied.
    :param kernel_size: the blur kernel size.
    :return:
    """

    h = img.shape[0]
    w = img.shape[1]

    left, top, right, bottom = bbox
    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)

    left = max(left, 0)
    right = min(right, w - 1)
    top = max(top, 0)
    bottom = min(bottom, h - 1)

    sub_img = img[top: bottom, left: right, :]
    if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
        sub_img = cv2.blur(sub_img, kernel_size)
    img[top: bottom, left: right, :] = sub_img

    return img


# ------------------------------------------------------------------------------------------------
