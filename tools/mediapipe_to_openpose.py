"""Generate OpenPose-format keypoint JSONs from MediaPipe Holistic Landmarker.

Fills the DATA_FOLDER/keypoints directory expected by smplifyx/main.py
(BODY_25 body + 21-pt hands + 70-pt face) without needing an OpenPose build.

Usage:
    python tools/mediapipe_to_openpose.py \
        --img_folder DATA_FOLDER/images \
        --out_folder DATA_FOLDER/keypoints \
        --model smplifyx/mp_models/holistic_landmarker.task

Face mapping (mediapipe 468-mesh -> dlib-68 layout) ported from
https://huggingface.co/spaces/Akjava/mediapipe-68-points-facial-landmark
(mp_constants.py / draw_landmarks68.py), which smplifyx's read_keypoints()
expects (face_keypoints_2d[17:17+51]).
"""
import argparse
import json
import math
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- BODY_25 <- MediaPipe Pose (33 landmarks) ------------------------------
# index into mp pose landmarks, or ('mid', a, b) to average two indices
BODY25_FROM_MP_POSE = [
    0,               # 0 Nose
    ('mid', 11, 12), # 1 Neck
    12,              # 2 RShoulder
    14,              # 3 RElbow
    16,              # 4 RWrist
    11,              # 5 LShoulder
    13,              # 6 LElbow
    15,              # 7 LWrist
    ('mid', 23, 24), # 8 MidHip
    24,              # 9 RHip
    26,              # 10 RKnee
    28,              # 11 RAnkle
    23,              # 12 LHip
    25,              # 13 LKnee
    27,              # 14 LAnkle
    5,               # 15 REye
    2,               # 16 LEye
    8,               # 17 REar
    7,               # 18 LEar
    31,              # 19 LBigToe
    31,              # 20 LSmallToe (mediapipe has no separate small toe)
    29,              # 21 LHeel
    32,              # 22 RBigToe
    32,              # 23 RSmallToe
    30,              # 24 RHeel
]

# --- face (dlib-68 layout) <- MediaPipe FaceMesh (478 landmarks) ----------
RIGHT_EYEBROW = [[70, 46], [63, 53], [105, 52], [66, 65], [107, 55]]
LEFT_EYEBROW = [[336, 285], [296, 295], [334, 282], [293, 283], [300, 276]]

NOSE_VERTICAL_GROUPS = [[8, 168], [6], [197, 195], [5, 4]]
NOSE_HORIZONTAL = [98, 97, 2, 326, 327]

RIGHT_EYE_UPPER_MIXED = [
    [33, 33, 247], [246, 246, 247], [161, 161, 30], [160, 160, 29],
    [159, 159, 27], [158, 158, 28], [173], [173, 133],
]
RIGHT_EYE_LOWER_MIXED = [
    [173, 133, 155], [154], 153, 145, 144, 163, 7, [33, 247],
]
LEFT_EYE_UPPER_MIXED = [
    [362, 362, 398, 382], [398, 398, 414], [384, 384, 384, 286],
    [385, 385, 258], [386, 386, 257], [387, 387, 259], [466], [467, 263],
]
LEFT_EYE_LOWER_MIXED = [
    [467, 263], [390], 373, 374, 380, 381, 382, [362, 398, 382],
]

UPPER_OUTER_LIP = [[61], [40, 39], [37], [0], [267], [270, 269], [291]]
LOWER_OUTER_LIP = [[375, 321], [405, 314], [17], [84, 181], [146, 91]]
UPPER_INNER_LIP = [[78], [81], [13], [311], [308]]
LOWER_INNER_LIP = [[402], [14], [178]]


def _mean_point(landmarks, group, width, height):
    if isinstance(group, int):
        group = [group]
    xs, ys = [], []
    for idx in group:
        xs.append(landmarks[idx].x * width)
        ys.append(landmarks[idx].y * height)
    return [sum(xs) / len(xs), sum(ys) / len(ys)]


def _dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _divide_line_to_points(points, divided):
    """Resample a polyline (list of [x,y]) into divided+1 arc-length-even points."""
    seg_lengths = [_dist(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total = sum(seg_lengths)
    step = total / divided if total > 0 else 0
    out = [points[0]]
    for i in range(1, divided):
        need = step * i
        acc = 0
        for j, seg_len in enumerate(seg_lengths):
            acc += seg_len
            if acc > need or j == len(seg_lengths) - 1:
                diff = acc - need
                lerp = 1.0 - (diff / seg_len) if seg_len > 0 else 0
                p1, p2 = points[j], points[j + 1]
                out.append([p1[0] + (p2[0] - p1[0]) * lerp,
                           p1[1] + (p2[1] - p1[1]) * lerp])
                break
    out.append(points[-1])
    return out


def _mean_points(landmarks, groups, width, height):
    return [_mean_point(landmarks, g, width, height) for g in groups]


def _divided_points(landmarks, groups, width, height, divided=3):
    pts = _mean_points(landmarks, groups, width, height)
    return _divide_line_to_points(pts, divided)


def face_68_51(face_landmarks, width, height):
    """Returns the 51 face points smplifyx expects (dlib-68 indices 17-67)."""
    pts = []
    pts += _mean_points(face_landmarks, RIGHT_EYEBROW, width, height)
    pts += _mean_points(face_landmarks, LEFT_EYEBROW, width, height)

    nose_v = _divided_points(face_landmarks, NOSE_VERTICAL_GROUPS, width, height, 3)
    pts += nose_v
    pts += _mean_points(face_landmarks, [[i] for i in NOSE_HORIZONTAL], width, height)

    r_upper = _divided_points(face_landmarks, RIGHT_EYE_UPPER_MIXED, width, height, 3)
    r_lower = _divided_points(face_landmarks, RIGHT_EYE_LOWER_MIXED, width, height, 3)
    pts += r_upper + r_lower[1:-1]

    l_upper = _divided_points(face_landmarks, LEFT_EYE_UPPER_MIXED, width, height, 3)
    l_lower = _divided_points(face_landmarks, LEFT_EYE_LOWER_MIXED, width, height, 3)
    pts += l_upper + l_lower[1:-1]

    upper_outer = _mean_points(face_landmarks, UPPER_OUTER_LIP, width, height)
    lower_outer = _mean_points(face_landmarks, LOWER_OUTER_LIP, width, height)
    pts += upper_outer + lower_outer

    upper_inner = _mean_points(face_landmarks, UPPER_INNER_LIP, width, height)
    lower_inner = _mean_points(face_landmarks, LOWER_INNER_LIP, width, height)
    pts += upper_inner + lower_inner

    assert len(pts) == 51, len(pts)
    return pts


def body25_from_pose(pose_landmarks, width, height):
    out = []
    for entry in BODY25_FROM_MP_POSE:
        if isinstance(entry, tuple):
            _, a, b = entry
            x = (pose_landmarks[a].x + pose_landmarks[b].x) / 2 * width
            y = (pose_landmarks[a].y + pose_landmarks[b].y) / 2 * height
            v = min(pose_landmarks[a].visibility, pose_landmarks[b].visibility)
        else:
            lm = pose_landmarks[entry]
            x, y, v = lm.x * width, lm.y * height, lm.visibility
        out += [x, y, v]
    return out


def hand21_flat(hand_landmarks, width, height):
    if not hand_landmarks:
        return [0.0] * (21 * 3)
    out = []
    for lm in hand_landmarks:
        out += [lm.x * width, lm.y * height, 1.0]
    return out


def face70_flat(face_landmarks, width, height):
    if not face_landmarks:
        return [0.0] * (70 * 3)
    contour = [[0.0, 0.0]] * 17
    core51 = face_68_51(face_landmarks, width, height)
    pupils = [[0.0, 0.0]] * 2
    out = []
    for x, y in contour + core51 + pupils:
        out += [x, y, 1.0]
    return out


def process_image(landmarker, img_path):
    mp_img = mp.Image.create_from_file(img_path)
    width, height = mp_img.width, mp_img.height
    result = landmarker.detect(mp_img)

    if not result.pose_landmarks:
        return {'people': []}

    person = {
        'pose_keypoints_2d': body25_from_pose(result.pose_landmarks, width, height),
        'face_keypoints_2d': face70_flat(
            result.face_landmarks if result.face_landmarks else None,
            width, height),
        'hand_left_keypoints_2d': hand21_flat(
            result.left_hand_landmarks, width, height),
        'hand_right_keypoints_2d': hand21_flat(
            result.right_hand_landmarks, width, height),
    }
    return {'people': [person]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True)
    parser.add_argument('--out_folder', required=True)
    parser.add_argument('--model', default='smplifyx/mp_models/holistic_landmarker.task')
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    base_options = mp_python.BaseOptions(model_asset_path=args.model)
    options = vision.HolisticLandmarkerOptions(base_options=base_options)
    landmarker = vision.HolisticLandmarker.create_from_options(options)

    img_files = sorted(f for f in os.listdir(args.img_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    for fn in img_files:
        img_path = os.path.join(args.img_folder, fn)
        data = process_image(landmarker, img_path)
        out_name = os.path.splitext(fn)[0] + '_keypoints.json'
        out_path = os.path.join(args.out_folder, out_name)
        with open(out_path, 'w') as f:
            json.dump(data, f)
        n_people = len(data['people'])
        print(f'{fn}: {n_people} person(s) detected -> {out_name}')


if __name__ == '__main__':
    main()
