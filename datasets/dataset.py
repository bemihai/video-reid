import numpy as np
import torch
from torchvision import transforms
from datasets.tools import read_image_from_file


class Dataset(object):
    """
    A class implementing a (video type) dataset. Train, query and gallery contain tuples of the form
    ``(img_paths, metadata)``, where ``img_paths`` are the actual paths to the image files and
    ``metadata`` is a dictionary containing tracklet information like person ID, camera ID and coordinates,
    initial image dimension, etc.
    Each item of the dataset is a tracklet, i.e. a series of images of the same person, from the same camera.

    Args:
        train (list): contains tuples of (img_paths, metadata).
        query (list): contains tuples of (img_paths, metadata).
        gallery (list): contains tuples of (img_paths, metadata).
        transform: transform to add on images (e.g. resize). Must be a function from ``torchvision.transforms``.
        mode (str): 'train', 'query' or 'gallery'.
        track_len (int): number of images in a tracklet.
        track_sampler (str): how to sample images in a tracklet ('random' or 'evenly').
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(
            self,
            train,
            query,
            gallery,
            transform=None,
            mode='train',
            track_len=1,
            track_sampler='evenly',
            **kwargs
    ):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transforms = transform
        self.mode = mode
        self.track_len = track_len
        self.track_sampler = track_sampler

        self.num_train_pids = self.get_num_pids()
        self.num_train_cams = self.get_num_cams()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Expected one of [train | query | gallery]'.format(self.mode))

    def __getitem__(self, i):
        img_paths, metadata = self.data[i]
        num_imgs = len(img_paths)
        to_tensor = transforms.ToTensor()

        if self.track_sampler == 'random':
            # randomly samples track_len images from a sequence of length num_imgs
            # if num_imgs is smaller than track_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.track_len else True
            indices = np.random.choice(indices, size=self.track_len, replace=replace)
            indices = np.sort(indices)

        elif self.track_sampler == 'evenly':
            # evenly samples track_len images from a tracklet
            # if num_imgs is smaller than track_len, simply replicate the last image until the track_len
            # requirement is satisfied
            if num_imgs >= self.track_len:
                num_imgs -= num_imgs % self.track_len
                indices = np.arange(0, num_imgs, num_imgs / self.track_len)
            else:
                indices = np.arange(0, num_imgs)
                num_pads = self.track_len - num_imgs
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num_imgs - 1)])
            assert len(indices) == self.track_len

        else:
            raise ValueError('Unknown sample method: {}'.format(self.track_sampler))

        imgs = []
        for i in indices:
            img_path = img_paths[int(i)]
            img = read_image_from_file(img_path)
            if self.transforms is not None:
                img = self.transforms(img)
            if not isinstance(img, torch.Tensor):
                img = to_tensor(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        # imgs is a tensor of shape (track_len, C, H, W)
        imgs = torch.cat(imgs, dim=0)

        return imgs, metadata

    def __len__(self):
        return len(self.data)

    def _parse(self):
        """ Parse train data list and returns the number of person IDs and the number of camera views. """
        pids = set()
        cams = set()
        for _, metadata in self.train:
            pids.add(metadata['pid'])
            cams.add(metadata['cam_id'])
        return len(pids), len(cams)

    def get_num_pids(self):
        """ Return the number of training person identities. """
        return self._parse()[0]

    def get_num_cams(self):
        """ Return the number of training cameras. """
        return self._parse()[1]


