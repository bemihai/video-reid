import cv2
from core.geometry import enlarge_bbox
import numpy as np


class Tracklet:
    """
    A tracklet represents the trajectory of a person within a single camera, tracked during
    the time span of a single appearance within that camera (i.e. it does not account for the
    re-entering of the same person inside the same camera later -- distinct tracklets will
    be created in this case).
    Besides the trajectory, the tracklet also contains features useful for re-identification.
    """

    # the default size at which all the tracklet images are saved
    default_img_size = (180, 240)

    def __init__(self, track_id, cam_id, color):
        self.id = track_id
        self.cam_id = cam_id
        self.identity = None
        self.bbox = None
        self.n_detections = 0
        self.last_detection = 0
        self.feats_sum = None
        self.feats_avg = None
        self.n_feats = 0
        self.img = None
        self.color = color
        self.img_score = None

    def update_bbox(self, bbox, score, frame, frame_no, img_size=default_img_size):
        """
        Add a new person detection (bbox) to the tracklet.
        """
        self.bbox = bbox
        self.n_detections += 1
        self.last_detection = frame_no

        # decide whether the image crop from this detection should be used (confidence should be above a threshold)
        img_score_threshold = 1.0e7
        use_img = False
        if self.img_score is None or score > self.img_score:
            use_img = True
            self.img_score = score
        if score > img_score_threshold:
            self.img_score = score
            use_img = True

        # update the image for this tracklet
        if use_img:
            bbox = enlarge_bbox(bbox, frame.shape[1], frame.shape[0])
            self.img = frame[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
            self.img = cv2.resize(self.img, img_size)

    def update_features(self, features):
        """
        Update the tracklet re-identification features using the ones just extracted.
        """
        if self.feats_sum is None:
            self.feats_sum = features
            self.n_feats = 1
            self.feats_avg = features
        else:
            self.feats_sum += features
            self.n_feats += 1
            self.feats_avg = self.feats_sum / self.n_feats

        if self.identity is not None:
            self.identity.features = self.feats_avg



