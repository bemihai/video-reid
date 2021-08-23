import math
import cv2
# import core.rendering.constants as pose_constants
import numpy as np


# ------------------------------------------------------------------------------------------------

def render_circle(frame, center, radius, color=(0, 0, 0)):
    """
    The center is in the (x, y) format.
    """
    cv2.circle(frame, center, radius, color, -1)


# ------------------------------------------------------------------------------------------------

# def render_pose(frame, pose):
#     for i in range(pose_constants.n_joints):
#         joint = (pose[i][0], pose[i][1])
#         if joint[0] != 0 and joint[1] != 0:
#             cv2.circle(frame, (pose[i][0], pose[i][1]), 5, (0, 0, 255), -1)

#     for pair in pose_constants.connections:
#         joint1 = (pose[pair[0]][0], pose[pair[0]][1])
#         joint2 = (pose[pair[1]][0], pose[pair[1]][1])

#         if joint1[0] != 0 and joint1[1] != 0 and joint2[0] != 0 and joint2[1] != 0:
#             cv2.line(frame, joint1, joint2, (255, 0, 0), 2)

#     return frame


# ------------------------------------------------------------------------------------------------

def render_poses(frame, poses, highlighted_joint=1,
                 highlighted_joint_color=(0, 255, 0),
                 joint_thickness=5, limb_thickness=2,
                 pose_color=(0, 0, 255)):
    """
    Render a list of poses on the input frame.
    :param frame: the input frame where poses are rendered.
    :param poses: the poses to be rendered.
    :param highlighted_joint: the index of the highlighted joint.
    :param highlighted_joint_color: the color of the highlighted joint.
                                    It can also be a list, in which case each pose
                                    will have its separated highlighted color.
    :param joint_thickness: the tickness of a joint points.
    :param limb_thickness: the tickenss of the limb lines.
    :param pose_color: the color of the limbs and non-highlighted joints
                       It can also be a list, in which case each pose
                       will have its separated highlighted color.
    """

    if type(pose_color) == tuple:
        pose_color = [pose_color] * len(poses)
    if type(highlighted_joint_color) == tuple:
        highlighted_joint_color = [highlighted_joint_color] * len(poses)
    for i in range(len(poses)):
        render_pose2(frame, poses[i],
                     highlighted_joint=highlighted_joint,
                     highlighted_joint_color=highlighted_joint_color[i],
                     joint_thickness=joint_thickness,
                     limb_thickness=limb_thickness,
                     color=pose_color[i])

    return frame


# ------------------------------------------------------------------------------------------------

# def render_pose2(frame, pose, color=(0, 0, 255),
#                  highlighted_joint=None,
#                  highlighted_joint_color=(0, 0, 255),
#                  joint_thickness=5,
#                  limb_thickness=2):
#     for i in range(pose_constants.n_joints):
#         joint = (pose[i][0], pose[i][1])
#         if joint[0] != 0 and joint[1] != 0:
#             sz = joint_thickness
#             crt_color = color
#             if i == highlighted_joint:
#                 sz *= 2
#                 crt_color = highlighted_joint_color
#             cv2.circle(frame, (pose[i][0], pose[i][1]), sz, crt_color, -1)

#     for pair in pose_constants.connections:
#         joint1 = (pose[pair[0]][0], pose[pair[0]][1])
#         joint2 = (pose[pair[1]][0], pose[pair[1]][1])

#         if joint1[0] != 0 and joint1[1] != 0 and joint2[0] != 0 and joint2[1] != 0:
#             cv2.line(frame, joint1, joint2, color, limb_thickness)

#     return frame


# ------------------------------------------------------------------------------------------------

def render_poly(img, poly, thickness=1, color=(255, 0, 0), close=False,
                text=None, text_color=(0, 255, 255)):
    for i in range(1, len(poly)):
        start_point = poly[i - 1]
        end_point = poly[i]
        start_point = int(start_point[0]), int(start_point[1])
        end_point = int(end_point[0]), int(end_point[1])
        cv2.line(img, start_point, end_point, color=color,
                 thickness=thickness)

    if close:
        start_point = poly[0]
        end_point = poly[-1]
        start_point = int(start_point[0]), int(start_point[1])
        end_point = int(end_point[0]), int(end_point[1])
        cv2.line(img, start_point, end_point, color=color,
                 thickness=thickness)

    if text is not None:
        cv2.putText(img, str(text), (int(poly[-1][0]), int(poly[-1][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)


# ------------------------------------------------------------------------------------------------

def render_recognition(img, bboxes, labels, all_scores, threshold=0.5,
                       bbox_thickness=2, bbox_color=(255, 0, 0),
                       recognition_color=(10, 255, 10),
                       line_spacing=30):
    """
    Render the results of object/face recognition on an image.
    @:param img: the input image where the recognition was performed.
    @:param bboxes: the detected bounding boxes for which recognition was performed.
    @:param labels: a list of all the distinct class labels.
    @:param all_scores: the scores, for each bounding box, associated to each label,
                        i.e. all_scores[i][j] is the confidence that the i`th bounding box
                        is associated with the j`th label.
    @:param threshold: the score threshold for which labels are displayed for a detection.
    @:param thickness: bounding box line thickness.
    @:param color: bounding box color.
    """

    for bbox_idx, bbox in enumerate(bboxes):
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        n_identities = 0
        for i in range(len(labels)):
            if all_scores[bbox_idx, i] > threshold:
                n_identities += 1
                cv2.putText(img,
                            "{}: {:.2f}".format(labels[i], all_scores[bbox_idx, i]),
                            (int(x1), int(y1) - line_spacing * n_identities),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, recognition_color, 2)

        if n_identities == 0:
            color = bbox_color
        else:
            color = recognition_color

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      color, bbox_thickness)


# ------------------------------------------------------------------------------------------------

def render_bbox(img, bbox, thickness=2, color=(255, 0, 0), label=None):
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)


def label_bbox(img, label, bbox, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.75, color=(0, 0, 255), thickness=2):
    cv2.putText(img, str(label), (int(bbox[0]), int(10+bbox[1])), font, font_scale, color, thickness)


# ------------------------------------------------------------------------------------------------

def render_bboxes(img, bboxes, thickness=2, color=(255, 0, 0)):
    """
    Render a list of bounding boxes specified [x_min, y_min, x_max, y_max].
    :param img: the image where the bounding boxes are rendered.
    :param bboxes: the list of bounding boxes to be rendered.
    :param thickness: the thickness of the bounding box line.
    :param color: the color of the bounding boxes. This parameter can be a list,
                  in which case it specifies an individual color for each bbox
                  (and it should have the same number of elements as bboxes).
    """

    if type(color) == list:
        assert len(color) == len(bboxes)
        for i, bbox in enumerate(bboxes):
            render_bbox(img, bbox, thickness, color[i])
    else:
        for i, bbox in enumerate(bboxes):
            render_bbox(img, bbox, thickness, color)


def label_bboxes(img, labels, bboxes):
    """
    Label a list of bounding boxes specified [x_min, y_min, x_max, y_max]. Label is added in the top left corner.
    :param img: the image where the bounding boxes are rendered.
    :param bboxes: the list of bounding boxes to be labeled.
    :param labels: a list of labels to add in the top left corner of the frame.
                   Must have the same number of elements as bboxes.
    """

    assert len(bboxes) == len(labels)
    for i, bbox in enumerate(bboxes):
        label_bbox(img, labels[i], bbox)


# ------------------------------------------------------------------------------------------------

def set_transparency(frame, transparency=0.3, canvas_color=(255, 255, 255)):
    # get image type
    # normalize data between 0 - 1
    # scale by 255

    # info_data_type = np.iinfo(frame.dtype)
    # empty_frame = np.ones((frame.shape[0], frame.shape[1], 3)) * canvas_color
    # empty_frame = empty_frame / info_data_type.max
    # empty_frame = 255 * empty_frame
    # empty_frame = empty_frame.astype(np.uint8)
    # processed_frame = (1-transparency) * frame + transparency * empty_frame
    # return processed_frame.astype(np.uint8)

    processed_frame = (np.ones_like(frame) * 255 * transparency + frame * (1 - transparency)).astype(frame.dtype)
    return processed_frame


# ------------------------------------------------------------------------------------------------

# def render_pose_text(frame, pose, text, color=(0, 0, 0), font_scale=1, thickness=3,
#                      delta_x=0, delta_y=-30):
#     cv2.putText(frame, str(text),
#                 (int(pose[pose_constants.Neck][0] + delta_x),
#                  int(pose[pose_constants.Neck][1] + delta_y)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness)

#     return frame


# ------------------------------------------------------------------------------------------------

def render_bbox_text(frame, bbox, text, color=(0, 0, 0), font_scale=1, thickness=3):
    cv2.putText(frame, str(text),
                (int(bbox[0]), int(bbox[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)

    return frame


# ------------------------------------------------------------------------------------------------

def render_covariance(img, covariance, center, color=(0, 255, 255), thickness=3):
    """
    Renders the ellipse associated to a covariance matrix.
    Only handles axis-aligned covariance ellipses.
    :param img: the image on which the covariance is rendered.
    :param covariance: the 2x2 position covariance matrix.
    :param center: the ellipse center.
    :param color: the color of the covariance ellipse.
    :param thickness: the thickness of the covariance ellipse.
    :return:
    """

    std_x = math.sqrt(covariance[0][0])
    std_y = math.sqrt(covariance[1][1])
    cv2.ellipse(img, (int(center[0]), int(center[1])), (int(std_x), int(std_y)),
                0, 0, 360, color=color, thickness=thickness)


# ------------------------------------------------------------------------------------------------

def render_arrow(img, start_point, end_point, color=(0, 255, 255), thickness=3):
    cv2.arrowedLine(img, (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])), color=color,
                    thickness=thickness)


# ------------------------------------------------------------------------------------------------

def render_velocity(img, position, velocity, factor=1, color=(0, 255, 255), thickness=3):
    """
    Renders velocity arrow from an object at a specific position.
    :param img: the image on which the arrow is rendered.
    :param position: the (x, y) position of the object.
    :param velocity: the (vx, vy) velocity of the object.
    :param factor: the factor to multiply velocity with before rendering.
    :param color: the color of the arrow.
    :param thickness: the thickness of the arrow.
    :return:
    """
    cv2.arrowedLine(img, (int(position[0]), int(position[1])),
                    (int(position[0] + velocity[0] * factor),
                     int(position[1] + velocity[1] * factor)),
                    color=color, thickness=thickness)


# ------------------------------------------------------------------------------------------------

# def render_pose_anomaly(img, pose, delta_x=0, delta_y=-60):
#     cv2.circle(img, (int(pose[pose_constants.Neck][0] + delta_x),
#                      int(pose[pose_constants.Neck][1] + delta_y)),
#                radius=10, color=(0, 0, 255), thickness=20)

# ------------------------------------------------------------------------------------------------
