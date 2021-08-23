import os
import torch
from torch.utils.data import DataLoader

from datasets import init_dataset
from datasets.transforms import build_transforms
from datasets.samplers import build_train_sampler


class DataManager(object):
    """
    A class implementing a (video type) DataManager. Provides a train loader and test loader (query and gallery),
    which are ``torch.utils.data.DataLoader`` objects.

    Args:
        root (str): path to the data folder.
        source (str): source dataset name.
        target (str, optional): target dataset name. If not given, equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use ImageNet mean).
        norm_std (list or None, optional): data std. Default is None (use ImageNet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        batch_size_train (int, optional): number of tracklets in a training batch. Default is 3.
        batch_size_test (int, optional): number of tracklets in a test batch. Default is 3.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        train_sampler (str, optional): sampler. Default is RandomSampler.
        track_len (int, optional): how many images to sample in a tracklet. Default is 15.
        track_sampler (str, optional): how to sample images in a tracklet. Default is "evenly".
            Choices are ["evenly", "random"].
    """

    def __init__(
            self,
            root=None,
            source=None,
            target=None,
            height=256,
            width=128,
            transforms='random_flip',
            norm_mean=None,
            norm_std=None,
            use_gpu=False,
            batch_size_train=3,
            batch_size_test=3,
            workers=4,
            num_instances=4,
            train_sampler='RandomSampler',
            track_len=1,
            track_sampler='evenly',
            split_id=0,
            **kwargs
    ):
        self.root = root
        self.source = source
        self.target = target
        self.height = height
        self.width = width
        self.train_sampler = train_sampler

        if self.source is None:
            raise ValueError('Source must not be None.')

        if self.target is None:
            self.target = self.source

        self.transform_tr, self.transform_te = build_transforms(
            self.height,
            self.width,
            transform=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std
        )
        self.use_gpu = (torch.cuda.is_available() and use_gpu)

        train_set = init_dataset(
            self.source,
            dataset_dir=os.path.join(self.root, self.source),
            transform=self.transform_tr,
            mode='train',
            track_len=track_len,
            sample_method=track_sampler,
            split_id=split_id
        )

        self._num_train_pids = train_set.num_train_pids
        self._num_train_cams = train_set.num_train_cams

        train_sampler = build_train_sampler(
            train_set.train,
            self.train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        self.train_loader = DataLoader(
            train_set,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        self.test_loader = {'query': None, 'gallery': None}
        self.test_set = {'query': None, 'gallery': None}

        query_set = init_dataset(
            self.target,
            dataset_dir=os.path.join(self.root, self.target),
            transform=self.transform_te,
            mode='query',
            track_len=track_len,
            sample_method=track_sampler,
            split_id=split_id
        )

        self.test_loader['query'] = DataLoader(
            query_set,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False
        )

        gallery_set = init_dataset(
            self.target,
            dataset_dir=os.path.join(self.root, self.target),
            transform=self.transform_te,
            mode='gallery',
            track_len=track_len,
            sample_method=track_sampler,
            split_id=split_id,
        )

        self.test_loader['gallery'] = DataLoader(
            gallery_set,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=False
        )

        self.test_set['query'] = query_set.query
        self.test_set['gallery'] = gallery_set.gallery

    @property
    def num_train_pids(self):
        """ Return the number of training person identities. """
        return self._num_train_pids

    @property
    def num_train_cams(self):
        """ Return the number of training cameras. """
        return self._num_train_cams

    def preprocess_pil_img(self, img):
        """ Transform a PIL image to torch tensor for testing. """
        return self.transform_te(img)


def test():

    data = DataManager(
        root='D:/Projects/person-reid/data',
        source='ilidsvid',
        track_len=5,
        batch_size_train=7
    )

    print(data.num_train_cams)
    print(data.num_train_pids)
    print(type(data.train_loader))
    print(type(data.test_loader['query']))

    for batch in data.train_loader:
        print(batch[0].shape)
        break


if __name__ == '__main__':
    test()
