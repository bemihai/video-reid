import os
import json
import errno
from PIL import Image


def read_lines(f_path):
    names = []
    with open(f_path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            names.append(new_line)
    return names


def read_image_from_file(path, to_RGB=True):
    """
    Read image from a file using ``PIL.Image``.
    Args:
        path (str): path to the file.
        to_RGB (boolean): convert image to RGB or not. Default is True.
    """
    image = None
    read_image = False

    if not os.path.exists(path):
        raise IOError('"{}" does not exist'.format(path))

    while not read_image:
        try:
            image = Image.open(path)
            if to_RGB:
                image = image.convert('RGB')
            read_image = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo.'.format(path))
    return image


def cv2pil(img):
    """ Transform a ``cv2.imread()`` array into a ``PIL.Image`` array. """
    img = img[..., ::-1]  # change image from BGR to RGB
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


def check_files(required_files):
    if isinstance(required_files, str):
        required_files = [required_files]

    for f_path in required_files:
        if not os.path.exists(f_path):
            raise RuntimeError('"{}" not found'.format(f_path))


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_json(f_path):
    """Reads json file from a path."""
    with open(f_path, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, f_path):
    """Writes to a json file."""
    mkdir_if_missing(os.path.dirname(f_path))
    with open(f_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))