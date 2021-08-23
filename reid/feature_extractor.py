import cv2
import torch
import numpy as np
from torchvision import transforms
from torchreid import models
from PIL import Image

from datasets.tools import cv2pil, read_image_from_file
from config.default_config import get_default_config
from tracking.person_detector import PersonDetector


def f_to_array(features, normalize=False):
    """
    Transforms a batch of features given as a ``torch.Tensor`` into a ``np.ndarray``.
    Args:
        features (torch.Tensor): tensor with shape (batch size, feature dim).
        normalize (bool, optional): normalize the feature vectors using L2-norm.
            Default is False.
    """
    features = features.cpu()
    features = features.numpy()

    if normalize:
        features = features / np.linalg.norm(features, axis=1).reshape(features.shape[0], -1)

    return features


class ReidFeatureExtractor:
    """
    ReID feature extractor built on top of ``torchreid.``
    Go to ``torchreid`` model zoo to get config files for pre-trained models.

    Args:
        model_name (str): model name as in ``torchreid`` model zoo.
        cfg_filename (str): path to the .yaml config file.
        options (list, optional): additional config options.
    """

    def __init__(self, model_name, cfg_filename, options=None):
        self.model_name = model_name
        self.cfg = cfg_filename
        self.opts = options
        self.model = None  # pretrained model from cfg file
        self.use_gpu = True  # from cfg file
        self.transform = None  # resize and normalize params from cfg file

    def _setup_config(self):
        cfg = get_default_config()
        cfg.merge_from_file(self.cfg)
        if self.opts:
            cfg.merge_from_list(self.opts)
        cfg.freeze()
        return cfg

    def build_extractor(self):
        cfg = self._setup_config()
        assert self.model_name == cfg.model.name

        self.use_gpu = cfg.model.use_gpu

        self.transform = transforms.Compose([
            transforms.Resize((cfg.data.height, cfg.data.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.data.norm_mean, std=cfg.data.norm_std),
        ])

        self.model = models.build_model(
            name=cfg.model.name,
            num_classes=1000,  # not used for feature extraction
            loss=cfg.loss.name,
            use_gpu=cfg.model.use_gpu,
            pretrained=cfg.model.pretrained,
        )

        if self.use_gpu:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        self.model.eval()

    def extract_from_batch(self, imgs):
        """
        Extracts features from a batch of crops containing detected pedestrians.
        Args:
            imgs (np.ndarray or PIL.Image or list): image array or list of arrays.
        Returns a tensor with shape (batch size, feature dim).
        """
        assert self.transform is not None
        if not isinstance(imgs, list):
            imgs = [imgs]
        pil_imgs = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                img = cv2pil(img)
            assert isinstance(img, Image.Image)
            img = self.transform(img)
            pil_imgs.append(img)

        if len(pil_imgs) > 0:
            batch = torch.stack(pil_imgs, 0)

            if self.use_gpu:
                batch = batch.cuda()

            with torch.no_grad():
                features = self.model(batch)

            return features

    def extract_from_frame(self, frame, bboxes):
        imgs = [frame[int(p[1]):int(p[3]), int(p[0]):int(p[2])] for p in bboxes]
        return self.extract_from_batch(imgs)

    def extract_from_tracklets(self, tracklets, pooling='avg'):
        """
        Extract features from a batch of tracklets (sequences of images with the same identity).
        Args:
            tracklets (torch.Tensor): batch of tracklets.
                Shape of ``tracklets`` must be (batch size, seq length, C, H, W).
            pooling (str): pooling method for features ('max' or 'avg). Default is 'avg'.
        Returns a tensor with shape (batch size, feature dim).
        """
        assert self.transform is not None
        b, s, c, h, w = tracklets.size()
        batch = tracklets.view(b * s, c, h, w)

        if self.use_gpu:
            batch = batch.cuda()

        with torch.no_grad():
            features = self.model(batch)

        features = features.view(b, s, -1)
        if pooling == 'avg':
            features = torch.mean(features, 1)
        elif pooling == 'max':
            features = torch.max(features, 1)[0]

        return features


def test():
    reid_extractor = ReidFeatureExtractor(
        model_name='osnet_x1_0',
        cfg_filename='../config/configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml',
    )
    reid_extractor.build_extractor()

    detect_args = {
        'cfg': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'model': 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
        'opts': ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.7]
    }

    detector = PersonDetector(detect_args)

    frame = cv2.imread('../data/images/cam2.jpg')
    frm = Image.open('../data/images/straw.jpg').convert('RGB')
    frm1 = read_image_from_file('../data/images/straw.jpg')

    # TODO: what type is input of detector ? works with PIL ?
    persons, scores = detector.detect(frame)
    # detector.visualize(frame)
    features = reid_extractor.extract_from_frame(frame, persons)

    print(len(persons))
    print(features.shape)


if __name__ == '__main__':
    test()
