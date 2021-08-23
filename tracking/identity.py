import numpy as np


class Identity:
    """
    An identity is a set of tracklets (from the same or from different cameras),
    associated together as belonging to the same actual person, based on
    re-identification feature descriptors.
    """

    def __init__(self, tracklet):
        self.initial_tracklet = tracklet
        self.person_id = self.initial_tracklet.id
        self.initial_tracklet_img = None
        self.matched_tracklet = None
        self.nearest_match_distance = None
        self.all_matched_tracklets = {}
        self.features = None
        self.color = tracklet.color

    def set_nearest_match(self, tracklet, dist):
        self.matched_tracklet = tracklet
        self.nearest_match_distance = dist

        if tracklet not in self.all_matched_tracklets:
            self.all_matched_tracklets[tracklet] = 0
        else:
            self.all_matched_tracklets[tracklet] += 1


def find_identity(tracklet, identities, same_cam_thr, diff_cam_thr, frame_no):
    min_distance = None
    nearest_identity = None
    for identity in identities:
        if identity.initial_tracklet != tracklet:
            same_camera = identity.initial_tracklet.cam_id == tracklet.cam_id
            threshold = same_cam_thr if same_camera else diff_cam_thr
            max_track_age = 5
            initial_tracklet_visible = frame_no - identity.initial_tracklet.last_detection < max_track_age
            if same_camera and initial_tracklet_visible:
                threshold = 0

            if identity.features is not None and tracklet.feats_avg is not None:
                dist = np.linalg.norm(identity.features - tracklet.feats_avg)
                if dist < threshold:
                    if min_distance is None or dist < min_distance:
                        min_distance = dist
                        nearest_identity = identity

    return nearest_identity, min_distance
