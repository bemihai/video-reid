import cv2
import collections

from rendering.renderer import render_bbox, label_bbox
from rendering.color import int_to_color
from core.geometry import enlarge_bbox
from tracking.sort_detector import Tracker
from tracking.person_detector import PersonDetector, CachedPersonDetector
from rendering.image_table import ImageTable
from reid.feature_extractor import ReidFeatureExtractor
from tracking.tracklet import Tracklet
from tracking.identity import Identity, find_identity


def read_video(video_filename, start_frame):
    cap = cv2.VideoCapture(video_filename)
    frame_idx = 0
    while frame_idx < start_frame:
        _, _ = cap.read()
        frame_idx += 1
    return cap


def main():
    # initialize the person detector
    args = {
        'cfg': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'model': 'reid/models/faster_rcnn_R50_fpn3x_model_final_280758.pkl',
        'opts': ['MODEL.ROI_HEADS.SCORE_THRESH_TEST', 0.7]
    }

    # initialize reid feature extractor
    feature_extractor = ReidFeatureExtractor(
        model_name='osnet_x1_0',
        cfg_filename='config/configs/im_osnet_x1_0_softmax_256x128_amsgrad.yaml',
    )
    feature_extractor.build_extractor()

    person_detector = PersonDetector(args)
    cam2_detector = CachedPersonDetector('data/duke videos/camera2.mp4', extract_features=True)
    cam3_detector = CachedPersonDetector('data/duke videos/camera3.mp4', extract_features=True)
    detectors = [cam3_detector, cam2_detector]

    for detector in detectors:
        if not detector.is_cached():
            print('Please wait, caching person detections and features for "{}"'.format(detector.video_filename))
            detector.compute(person_detector, feature_extractor, normalized=True, overwrite=True, max_frames=2000)
        detector.load()
        print('Person detections loaded from "{}"'.format(detector.detections_cache_file))
        print('Features loaded from "{}"'.format(detector.features_cache_file))

    detectors[1].seek(frame_no=280)

    cameras = [2, 3]  # camera ids
    max_n_identities = 50  # max number of identities tracked at the same time
    # TODO: adjust thresholds to reid algorithm
    id_match_thr = 0.90  # identity match in different cameras
    id_match_thr_same_camera = 0.70  # identity match in the same camera
    all_tracklets = []  # save all tracklets from all cameras
    identities = collections.deque([], maxlen=max_n_identities)  # save all tracked identities

    # initialize trackers for each camera
    trackers = []
    for _ in cameras:
        trackers.append(Tracker(max_age=0))
        all_tracklets.append(dict())

    # load video streams from all cameras
    cam3_stream = read_video('data/duke videos/camera3.mp4', start_frame=0)
    cam2_stream = read_video('data/duke videos/camera2.mp4', start_frame=280)
    video_streams = [cam3_stream, cam2_stream]
    frame_no = 0

    # show image table to display video streams
    table_img_size = (820, 460)  # image size
    table_cell_size = (840, 460)  # image cell size
    image_table = ImageTable(n_rows=1,
                             n_cols=2,
                             cell_size=table_cell_size,
                             img_size=None,
                             win_title='Person ReID',
                             grid_thickness=1,
                             grid_color=(175, 0, 0))
    image_table.show()

    while True:
        frame_no += 1
        print('Frame {}'.format(frame_no))

        # read the current frame from each camera
        frames = []
        for cap in video_streams:
            check, frame = cap.read()
            if not check:
                print('Could not read frame from camera')
                frame = None
            frames.append(frame)

        # for each frame at a time
        for cam_id in range(len(cameras)):

            frame = frames[cam_id]
            if frame is None:
                continue

            # detect persons in the frame and extract reid features
            detector = detectors[cam_id]
            person_bboxes, person_scores = detector.detect(frame)
            features = detector.extract_from_frame(frame, person_bboxes)

            # update the corresponding tracker
            matched_tracks, dropped_tracks = trackers[cam_id].update(person_bboxes, person_scores)
            tracklets = all_tracklets[cam_id]

            # drop unmatched tracks
            for track_id in dropped_tracks:
                del tracklets[track_id]

            # for each matched track
            for i, track_id in enumerate(matched_tracks):
                # if track not seen before
                if track_id not in tracklets.keys():
                    # create a new tracklet
                    tracklet = Tracklet(track_id, cam_id, int_to_color(track_id))
                    tracklets[track_id] = tracklet
                else:
                    tracklet = tracklets[track_id]

                # update tracklet's detection (bbox)
                tracklet.update_bbox(person_bboxes[i], person_scores[i], frame, frame_no, table_img_size)
                # update tracklet's reid features
                if features[i] is not None:
                    tracklet.update_features(features[i])

                # assign identity to tracklet
                if tracklet.identity is None:
                    nearest_identity, min_distance = find_identity(tracklet, identities, id_match_thr_same_camera,
                                                                   id_match_thr, frame_no)
                    if nearest_identity is not None:
                        tracklet.color = nearest_identity.color
                        nearest_identity.set_nearest_match(tracklet, min_distance)
                        tracklet.identity = nearest_identity
                    else:
                        identity = Identity(tracklet)
                        tracklet.identity = identity
                        identities.append(identity)
                        # bbox = enlarge_bbox(person_bboxes[i], frame.shape[1], frame.shape[0])
                        # img = frame[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
                        # tracklet_img = np.copy(img)
                        # identity.initial_tracklet_img = cv2.resize(tracklet_img, table_img_size)

                render_bbox(frame, person_bboxes[i], thickness=2, color=tracklet.identity.color)
                label_bbox(frame,
                           'id ' + str(tracklet.identity.person_id),
                           person_bboxes[i],
                           color=tracklet.identity.color)

            image_out = cv2.resize(frame, table_img_size)
            image_table.set_cell_image(0, cam_id, image_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
