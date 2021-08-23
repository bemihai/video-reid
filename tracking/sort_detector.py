import numpy as np
from external.sort_tracker import Sort
from core.utils import to_bbox

class Tracker:
    """
    Performs tracking of poses or bounding boxes.
    Relies on the Sort tracking algorithm and provides a different
    interface compared to the other Tracker class. The outputs
    of this class consists of track ids (integer values) and gives
    the caller full control and responsibility to perform operations
    when tracks are created, destroyed, etc.
    """

    def __init__(self, max_age=25, min_hits=1, iou_threshold=0.1):
        """
        Args:
            max_age: maximum number of updates where the tracks is not matched for it to be deleted.
            min_hits: minimum number of hits for the track to be published.
            iou_threshold: minimum IoU between detection and track for which association is allowed.
        """
        self._tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self._max_age = max_age


    def update(self, persons, scores):
        """
        The update method should be called after each frame (even when no person is detected).
        It performs the actual tracking (i.e. each person is matched against the set of
        current tracks, matched tracks are updated, old tracks which were unmatched for
        a long time are deleted).
        :param persons: a numpy array with shape (num_persons, num_bboxes).
        :param scores: a numpy array with shape (num_persons, num_scores).
        :return: a tuple consisting of a list of matched track ids, and the ids of the deleted tracks.
        """
        num_objects = len(persons)
        # prepare the input for the Sort tracker
        # the bbox format is (xmin, ymin, xmax, ymax, confidence score)
        scored_bboxes = np.zeros((0, 5))
        for i in range(num_objects):
            # the Sort tracker requires a score as the last component of the detection
            bbox = [persons[i][0], persons[i][1], persons[i][2], persons[i][3], scores[i]]
            scored_bboxes = np.concatenate((scored_bboxes, np.array(bbox).reshape(1, -1)))

        tmp_tracks, matched_track_ids, dropped = self._tracker.update(scored_bboxes)

        assert num_objects == len(matched_track_ids)
        matches = [-1] * num_objects
        for detection_id, track_id in matched_track_ids:
            matches[detection_id] = track_id

        return matches, dropped

# ------------------------------------------------------------------------------------------------
