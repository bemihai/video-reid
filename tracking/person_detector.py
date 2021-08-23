import cv2
import os
import pickle
import numpy as np
from rendering.renderer import render_bboxes
from processing.video_file_reader import VideoFileReader

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

class PersonDetector:
    """
    Detectron2 person detector (faster R-CNN).

    Args is a dictionary of parameters:
        cfg: configuration file for a Detectron2 model (yaml),
        model: model file with trained weights (pkl),
        opts (optional): additional config options.
    """

    def __init__(self, args):
        cfg = get_cfg()
        cfg_file = model_zoo.get_config_file(args["cfg"])
        cfg.merge_from_file(cfg_file)
        if args["opts"]:
            cfg.merge_from_list(args["opts"])
        cfg.MODEL.WEIGHTS = args["model"]
        cfg.freeze()
        self.cfg = cfg
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.predictor = DefaultPredictor(cfg)

    def detect(self, img):
        """ Returns an array with bounding boxes of persons detected in the image."""
        outputs = self.predictor(img)
        fields = outputs["instances"].get_fields()
        objects = fields["pred_classes"].cpu().numpy()
        bboxes = fields["pred_boxes"].tensor.cpu().numpy()
        scores = fields["scores"].cpu().numpy()
        return bboxes[objects == 0], scores[objects == 0]

    def visualize(self, img):
        persons, _ = self.detect(img)
        v = Visualizer(img[:, :, ::-1],
                       self.metadata,
                       scale=0.7,
                       instance_mode=ColorMode.IMAGE_BW)
        # outputs = self.predictor(img)
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        for person in persons:
            v.draw_box(person, edge_color='red')

        vis_image = v.get_output()
        cv2.imshow('Person detector', vis_image.get_image())
        cv2.waitKey(0) & 0xFF

class CachedPersonDetector:
    """
    A (person detector + feature extractor) whose results for a specific video file are cached on the disk.
    """
    def __init__(self, video_filename, extract_features=False):
        self.video_filename = video_filename
        self.filename, _ = os.path.splitext(self.video_filename)
        self.features = extract_features
        self.detections_cache_file = self.filename + '_detections.pkl'
        self.features_cache_file = self.filename + '_features.pkl'
        self.cached_frames = None

        self._all_detections = None  # detections = (bboxes, scores)
        self._all_features = None
        self._next_frame_idx_det = 0
        self._next_frame_idx_feat = 0

    def compute(self, detector, extractor=None, normalized=False, overwrite=False, max_frames=None):
        """
        Computes the detector results for all frames of the input video.
        :param detector: the raw detector whose results are cached.
        :param extractor: feature extractor to use in case ``extract_features=True``
        :param normalized: if True, normalize feature vectors.
        :param overwrite: if set to True, the cache is overwritten.
        :param max_frames max number of frames to cache.
        """
        if os.path.exists(self.detections_cache_file) and not overwrite:
            return

        cap = cv2.VideoCapture(self.video_filename)
        frame_idx = -1
        all_detections = []
        all_features = []

        if self.features:
            assert extractor is not None

        if max_frames is None:
            max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_idx < max_frames:
            frame_idx += 1

            success, frame = cap.read()
            if not success or frame is None:
                break

            detections = detector.detect(frame)
            all_detections.append(detections)

            if self.features:
                feats = extractor.extract_from_frame(frame, detections[0], normalize=normalized)
                all_features.append(feats)

            # print('frame {} - detections {} - features {} '.format(frame_idx, len(detections[0]), len(feats)))
            print(1 + frame_idx, '/', max_frames)

        with open(self.detections_cache_file, 'wb') as f:
            pickle.dump(all_detections, f)

        if self.features:
            assert len(all_detections) == len(all_features)
            with open(self.features_cache_file, 'wb') as f:
                pickle.dump(all_features, f)


    def is_cached(self):
        """
        Checks whether the cache files exist.
        """
        value = os.path.exists(self.detections_cache_file)
        if self.features:
            value = value and os.path.exists(self.features_cache_file)
        return value

    def load(self):
        """
        Load the cached results.
        """
        with open(self.detections_cache_file, 'rb') as f:
            self._all_detections = pickle.load(f)
        self._next_frame_idx_det = 0

        if self.features:
            with open(self.features_cache_file, 'rb') as f:
                self._all_features = pickle.load(f)
            self._next_frame_idx_feat = 0

        self.cached_frames = len(self._all_detections)

    def get_detections(self):
        """
        Returns the detections for the next frame.
        Requires ``self.load()`` to be called before any call to this function.
        """
        assert (self._all_detections is not None)

        if self._next_frame_idx_det < len(self._all_detections):
            detections = self._all_detections[self._next_frame_idx_det]
            self._next_frame_idx_det += 1
            return detections

        return None

    def get_features(self):
        """
        Returns the features for the next frame.
        Requires ``self.load()`` to be called before any call to this function.
        """
        assert self.features
        assert self._all_features is not None

        if self._next_frame_idx_feat < len(self._all_features):
            features = self._all_features[self._next_frame_idx_feat]
            self._next_frame_idx_feat += 1
            return features
        return None

    def is_eos(self):
        """
        Check whether the end of the stream was reached.
        """
        return self._next_frame_idx_det >= len(self._all_detections)

    def seek(self, frame_no):
        """
        Jumps to a specific frame number.
        Requires ``self.load()`` to be called before any call to this function.
        """
        assert self._all_detections is not None
        assert frame_no < len(self._all_detections)
        self._next_frame_idx_det = frame_no

        if self.features:
            assert self._all_features is not None
            assert frame_no < len(self._all_features)
            self._next_frame_idx_feat = frame_no


    def close(self):
        """
        Release from memory the cached data required for a replay.
        """
        self._all_detections = None
        if self.features:
            self._all_features = None

    def detect(self, _):
        """
        The detect() method is added mostly for interface compatibility with live Detector classes.
        """
        return self.get_detections()

    def extract_from_frame(self, frame, persons, normalized=True):
        """
        The extract_from_frame() method is added mostly for interface compatibility with live Features
         Extractor classes.
        """
        assert self.features
        return self.get_features()


def test():
    cap = cv2.VideoCapture('../data/duke videos/camera3.mp4')
    detector = CachedPersonDetector('../data/duke videos/camera3.mp4', extract_features=True)
    detector.load()

    frame_idx = 0

    while not detector.is_eos():

        check, frame = cap.read()
        if not check or frame is None:
            break

        bboxes, scores = detector.detect(frame)
        features = detector.extract_from_frame(frame, bboxes)
        render_bboxes(frame, bboxes)
        if features is not None:
            print(features.shape, bboxes.shape)

        cv2.imshow('Detection', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    test()









