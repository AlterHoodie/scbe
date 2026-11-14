"""Run RTMPose on squat videos and write the side-view measurement record.

Usage:
    uv run python -m src.pose
    uv run python -m src.pose ../dataset/squat_10.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.logger import configure, get_logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "output"
DEFAULT_DATASET = REPO_ROOT / "dataset"

# COCO-17 indices used by RTMPose. That set stops at the wrist, so the hand is the wrist point.
COCO_NOSE = 0
COCO_LEFT_EAR, COCO_RIGHT_EAR = 3, 4
COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER = 5, 6
COCO_LEFT_ELBOW, COCO_RIGHT_ELBOW = 7, 8
COCO_LEFT_WRIST, COCO_RIGHT_WRIST = 9, 10
COCO_LEFT_HIP, COCO_RIGHT_HIP = 11, 12
COCO_LEFT_KNEE, COCO_RIGHT_KNEE = 13, 14
COCO_LEFT_ANKLE, COCO_RIGHT_ANKLE = 15, 16

COCO_EDGES = [
    (COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER),
    (COCO_LEFT_SHOULDER, COCO_LEFT_ELBOW),
    (COCO_LEFT_ELBOW, COCO_LEFT_WRIST),
    (COCO_RIGHT_SHOULDER, COCO_RIGHT_ELBOW),
    (COCO_RIGHT_ELBOW, COCO_RIGHT_WRIST),
    (COCO_LEFT_SHOULDER, COCO_LEFT_HIP),
    (COCO_RIGHT_SHOULDER, COCO_RIGHT_HIP),
    (COCO_LEFT_HIP, COCO_RIGHT_HIP),
    (COCO_LEFT_HIP, COCO_LEFT_KNEE),
    (COCO_LEFT_KNEE, COCO_LEFT_ANKLE),
    (COCO_RIGHT_HIP, COCO_RIGHT_KNEE),
    (COCO_RIGHT_KNEE, COCO_RIGHT_ANKLE),
]

COCO_HIGHLIGHTS = {
    COCO_LEFT_HIP,
    COCO_RIGHT_HIP,
    COCO_LEFT_KNEE,
    COCO_RIGHT_KNEE,
    COCO_LEFT_ANKLE,
    COCO_RIGHT_ANKLE,
}

POINT_MIN = 0.35
BOTTOM_BAND_DEG = 10.0
SAMPLE_FPS = 10.0

@dataclass
class FrameMeasure:
    detected: bool
    knee_angle: float | None
    back_angle: float | None
    hip_below_knee: bool | None
    hip_y: float | None
    confidence: float = 0.0
    hip_angle: float | None = None
    hip_knee_gap: float | None = None
    knee_forward: float | None = None
    nose_below_ear: float | None = None
    side: str | None = None


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def angle_from_vertical(upper: np.ndarray, lower: np.ndarray) -> float:
    vector = lower - upper
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return float("nan")
    cosine = float(np.clip(vector[1] / norm, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _finite(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _span_ratio(numer: float, start: np.ndarray, end: np.ndarray) -> float | None:
    length = float(np.linalg.norm(end - start))
    if length < 1e-6:
        return None
    return float(numer / length)


def blank_side(side: str) -> FrameMeasure:
    return FrameMeasure(True, None, None, None, None, 0.0, side=side)


def measure_side(
    points: np.ndarray,
    scores: np.ndarray,
    hip: int,
    knee: int,
    ankle: int,
    shoulder: int,
    nose: int,
    ear: int,
    side: str,
) -> FrameMeasure:
    needed = (hip, knee, ankle, shoulder)
    confidence = float(np.min(scores[list(needed)]))
    if confidence < POINT_MIN:
        return blank_side(side)
    hip_pt = points[hip]
    knee_pt = points[knee]
    ankle_pt = points[ankle]
    shoulder_pt = points[shoulder]
    knee_angle = _finite(angle_at(hip_pt, knee_pt, ankle_pt))
    hip_angle = _finite(angle_at(shoulder_pt, hip_pt, knee_pt))
    back_angle = _finite(angle_from_vertical(shoulder_pt, hip_pt))
    # Image y grows downward, so a larger hip y means a lower hip.
    hip_below = bool(hip_pt[1] > knee_pt[1])
    hip_knee_gap = _span_ratio(float(hip_pt[1] - knee_pt[1]), hip_pt, knee_pt)
    knee_forward = _span_ratio(float(knee_pt[0] - ankle_pt[0]), knee_pt, ankle_pt)
    nose_below_ear = None
    if float(scores[nose]) >= POINT_MIN and float(scores[ear]) >= POINT_MIN:
        nose_below_ear = _span_ratio(float(points[nose][1] - points[ear][1]), shoulder_pt, hip_pt)
    return FrameMeasure(
        True,
        knee_angle,
        back_angle,
        hip_below,
        float(hip_pt[1]),
        confidence,
        hip_angle,
        hip_knee_gap,
        knee_forward,
        nose_below_ear,
        side,
    )


def better_side(left: FrameMeasure, right: FrameMeasure) -> FrameMeasure:
    if left.confidence <= 0 and right.confidence <= 0:
        return FrameMeasure(True, None, None, None, None, 0.0)
    return left if left.confidence >= right.confidence else right


def draw_pose(
    frame: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    canvas = frame.copy()
    for start, end in edges:
        if scores[start] < POINT_MIN or scores[end] < POINT_MIN:
            continue
        p1 = tuple(np.round(points[start]).astype(int))
        p2 = tuple(np.round(points[end]).astype(int))
        cv2.line(canvas, p1, p2, (80, 220, 80), 2, cv2.LINE_AA)
    visible = {index for edge in edges for index in edge}
    for index, (point, score) in enumerate(zip(points, scores)):
        if index not in visible or score < POINT_MIN:
            continue
        center = tuple(np.round(point).astype(int))
        color = (40, 180, 255) if index in COCO_HIGHLIGHTS else (255, 255, 255)
        cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)
    return canvas


class RTMPoseRunner:
    name = "rtmpose"

    def __init__(self) -> None:
        from rtmlib import Body

        get_logger("pose").info("Loading pose model", device="cpu")
        self._body = Body(mode="balanced", backend="onnxruntime", device="cpu")

    def reset(self) -> None:
        return None

    def close(self) -> None:
        return None

    def joints(self, side: str) -> dict[str, int]:
        if side == "left":
            return {
                "shoulder": COCO_LEFT_SHOULDER,
                "hip": COCO_LEFT_HIP,
                "knee": COCO_LEFT_KNEE,
                "ankle": COCO_LEFT_ANKLE,
            }
        return {
            "shoulder": COCO_RIGHT_SHOULDER,
            "hip": COCO_RIGHT_HIP,
            "knee": COCO_RIGHT_KNEE,
            "ankle": COCO_RIGHT_ANKLE,
        }

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple:
        del timestamp_ms
        boxes = self._body.det_model(frame_bgr)
        if boxes is None or len(boxes) == 0:
            return None, None, FrameMeasure(False, None, None, None, None), blank_side("left"), blank_side("right")
        keypoints, scores = self._body.pose_model(frame_bgr, bboxes=boxes)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.size == 0:
            return None, None, FrameMeasure(False, None, None, None, None), blank_side("left"), blank_side("right")
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if scores.ndim == 1:
            scores = scores[None, ...]
        scores = np.squeeze(scores, axis=-1) if scores.ndim == 3 else scores
        index = _largest_person(keypoints, scores)
        points = keypoints[index].astype(np.float32)
        person_scores = scores[index].astype(np.float32)
        left = measure_side(
            points,
            person_scores,
            COCO_LEFT_HIP,
            COCO_LEFT_KNEE,
            COCO_LEFT_ANKLE,
            COCO_LEFT_SHOULDER,
            COCO_NOSE,
            COCO_LEFT_EAR,
            "left",
        )
        right = measure_side(
            points,
            person_scores,
            COCO_RIGHT_HIP,
            COCO_RIGHT_KNEE,
            COCO_RIGHT_ANKLE,
            COCO_RIGHT_SHOULDER,
            COCO_NOSE,
            COCO_RIGHT_EAR,
            "right",
        )
        return points, person_scores, better_side(left, right), left, right

    def draw(self, frame, points, scores) -> np.ndarray:
        if points is None:
            return draw_pose(frame, np.zeros((1, 2)), np.zeros(1), [])
        return draw_pose(frame, points, scores, COCO_EDGES)


def _largest_person(keypoints: np.ndarray, scores: np.ndarray) -> int:
    best_index = 0
    best_area = -1.0
    for index, (points, person_scores) in enumerate(zip(keypoints, scores)):
        keep = person_scores > POINT_MIN
        if int(np.count_nonzero(keep)) < 4:
            continue
        xs = points[keep, 0]
        ys = points[keep, 1]
        area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
        if area > best_area:
            best_area = area
            best_index = index
    return best_index


def summarize(measures: list[FrameMeasure]) -> dict:
    detected = [item for item in measures if item.detected]
    angled = [item for item in measures if item.knee_angle is not None and item.hip_y is not None]
    bottom = max(angled, key=lambda item: item.hip_y) if angled else None
    return {
        "frames": len(measures),
        "detected_frames": len(detected),
        "detection_rate": round(len(detected) / len(measures), 3) if measures else 0,
        "bottom_knee_deg": None if bottom is None else round(bottom.knee_angle, 1),
        "bottom_back_from_vertical_deg": None if bottom is None or bottom.back_angle is None else round(bottom.back_angle, 1),
        "bottom_hip_below_knee": None if bottom is None else bottom.hip_below_knee,
    }


class ClipWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.raw_path = path.with_suffix(".avi")
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = size
        self._writer = cv2.VideoWriter(
            str(self.raw_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open a video writer for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
        ffmpeg = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(self.raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.path),
            ],
            check=False,
        )
        self.raw_path.unlink(missing_ok=True)
        if ffmpeg.returncode != 0 or not self.path.exists():
            raise RuntimeError(f"ffmpeg failed while writing {self.path}")


def _num(value: float | None, digits: int = 1) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def side_payload(item: FrameMeasure) -> dict:
    return {
        "confidence": round(item.confidence, 3),
        "knee_deg": _num(item.knee_angle),
        "hip_deg": _num(item.hip_angle),
        "back_from_vertical_deg": _num(item.back_angle),
        "hip_below_knee": item.hip_below_knee,
        "hip_knee_gap": _num(item.hip_knee_gap, 3),
        "knee_forward": _num(item.knee_forward, 3),
        "nose_below_ear": _num(item.nose_below_ear, 3),
    }


def _snapshot(records: list[dict], frame: int, side: str) -> dict:
    item = records[frame]
    measure = item[side]
    payload = side_payload(measure)
    return {
        "frame": frame,
        "t_ms": item["t_ms"],
        "confidence": payload["confidence"],
        "knee_deg": payload["knee_deg"],
        "hip_deg": payload["hip_deg"],
        "back_from_vertical_deg": payload["back_from_vertical_deg"],
        "hip_below_knee": payload["hip_below_knee"],
        "hip_knee_gap": payload["hip_knee_gap"],
        "knee_forward": payload["knee_forward"],
        "nose_below_ear": payload["nose_below_ear"],
    }


def _smooth_angles(values: list[float | None], window: int) -> list[float | None]:
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed: list[float | None] = []
    for index in range(len(values)):
        chunk = [
            values[other]
            for other in range(max(0, index - half), min(len(values), index + half + 1))
            if values[other] is not None
        ]
        smoothed.append(sum(chunk) / len(chunk) if chunk else None)
    return smoothed


def _measured(smooth: list[float | None], knees: list[float | None], start: int, end: int) -> list[int]:
    return [
        offset
        for offset in range(start, end)
        if smooth[offset] is not None and knees[offset] is not None
    ]


def _finish_frame(
    smooth: list[float | None],
    knees: list[float | None],
    bottom: int,
    next_standing: int | None,
) -> int | None:
    if next_standing is not None:
        return next_standing
    window = _measured(smooth, knees, bottom, len(smooth))
    if not window:
        return None
    return max(window, key=lambda offset: smooth[offset])


def _drive_frame(
    smooth: list[float | None],
    knees: list[float | None],
    bottom: int,
    finish: int | None,
) -> int | None:
    if finish is None or smooth[bottom] is None or smooth[finish] is None:
        return None
    target = (smooth[bottom] + smooth[finish]) / 2
    window = _measured(smooth, knees, bottom, finish + 1)
    if not window:
        return None
    return min(window, key=lambda offset: (abs(smooth[offset] - target), offset))


def _bottom_span(smooth: list[float | None], standing: int, bottom: int, finish: int | None) -> tuple[int, int]:
    deepest = smooth[bottom]
    if deepest is None:
        return bottom, bottom
    limit = deepest + BOTTOM_BAND_DEG
    start = bottom
    index = bottom
    while index > standing:
        previous = smooth[index - 1]
        if previous is None or previous > limit:
            break
        index -= 1
        start = index
    end = bottom
    last = bottom if finish is None else finish
    index = bottom
    while index < last:
        following = smooth[index + 1]
        if following is None or following > limit:
            break
        index += 1
        end = index
    return start, end


def video_side(records: list[dict]) -> str | None:
    counts = {"left": 0, "right": 0}
    weight = {"left": 0.0, "right": 0.0}
    for item in records:
        chosen = item["chosen"]
        if chosen.side not in counts or chosen.confidence < POINT_MIN:
            continue
        counts[chosen.side] += 1
        weight[chosen.side] += chosen.confidence
    if counts["left"] == 0 and counts["right"] == 0:
        return None
    if counts["left"] != counts["right"]:
        return "left" if counts["left"] > counts["right"] else "right"
    return "left" if weight["left"] >= weight["right"] else "right"


def segment_reps(records: list[dict], fps: float, side: str) -> tuple[list[dict], str | None]:
    knees = [
        None if item[side].knee_angle is None else float(item[side].knee_angle)
        for item in records
    ]
    window = max(3, int(round(fps * 0.15)))
    smooth = _smooth_angles(knees, window)
    valid = [value for value in smooth if value is not None]
    if len(valid) < max(5, int(fps * 0.5)):
        return [], "not enough confident knee angles to segment reps"

    radius = max(2, int(round(fps * 0.25)))
    lookback = max(radius + 1, int(round(fps * 2)))
    candidates: list[int] = []
    for index, value in enumerate(smooth):
        if value is None or knees[index] is None:
            continue
        neighborhood = [
            smooth[other]
            for other in range(max(0, index - radius), min(len(smooth), index + radius + 1))
            if smooth[other] is not None
        ]
        if not neighborhood or value > min(neighborhood) + 1.0:
            continue
        earlier = [
            smooth[other]
            for other in range(max(0, index - lookback), index)
            if smooth[other] is not None
        ]
        if not earlier or max(earlier) - value < 25:
            continue
        candidates.append(index)

    bottoms: list[int] = []
    gap = max(1, int(round(fps * 0.5)))
    for index in candidates:
        if bottoms and index - bottoms[-1] < gap:
            if smooth[index] < smooth[bottoms[-1]]:
                bottoms[-1] = index
            continue
        bottoms.append(index)

    reps = []
    previous = 0
    for bottom in bottoms:
        segment = _measured(smooth, knees, previous, bottom + 1)
        if not segment:
            previous = bottom
            continue
        standing = max(segment, key=lambda offset: smooth[offset])
        standing_angle = smooth[standing]
        bottom_angle = smooth[bottom]
        if (
            bottom_angle is None
            or standing_angle is None
            or standing_angle - bottom_angle < 25
            or bottom - standing < max(2, int(fps * 0.25))
        ):
            previous = bottom
            continue
        target = (standing_angle + bottom_angle) / 2
        descent = _measured(smooth, knees, standing, bottom + 1)
        midway = min(descent, key=lambda offset: abs(smooth[offset] - target))
        reps.append({"standing": standing, "midway": midway, "bottom": bottom})
        previous = bottom
    if not reps:
        return [], "no knee-angle valley deep enough to mark a squat"
    for index, rep in enumerate(reps):
        next_standing = reps[index + 1]["standing"] if index + 1 < len(reps) else None
        finish = _finish_frame(smooth, knees, rep["bottom"], next_standing)
        rep["finish"] = finish
        rep["drive"] = _drive_frame(smooth, knees, rep["bottom"], finish)
        rep["bottom_start"], rep["bottom_end"] = _bottom_span(
            smooth, rep["standing"], rep["bottom"], finish
        )
    return reps, None


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        current = 0
        ok, frame = capture.read()
        while ok and current < frame_index:
            current += 1
            ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def _draw_chain(
    canvas: np.ndarray,
    points: np.ndarray,
    scores: np.ndarray,
    indices: list[int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    usable = []
    for index in indices:
        if index >= len(scores) or scores[index] < POINT_MIN:
            usable.append(None)
            continue
        usable.append(tuple(np.round(points[index]).astype(int)))
    for start, end in zip(usable, usable[1:]):
        if start is None or end is None:
            continue
        cv2.line(canvas, start, end, color, thickness, cv2.LINE_AA)
    for point in usable:
        if point is None:
            continue
        cv2.circle(canvas, point, thickness + 2, color, -1, cv2.LINE_AA)


def _label(canvas: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


_STILL_MAX_HEIGHT = 480

# (record key, filename suffix, caption drawn on the still)
_EVENT_STILLS = (
    ("standing", "standing", "Standing"),
    ("midway", "down", "Down"),
    ("bottom", "seated", "Seated"),
    ("drive", "drive", "Drive up"),
)


def _fit_height(image: np.ndarray, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= max_height:
        return image
    scale = max_height / height
    return cv2.resize(image, (max(1, int(width * scale)), max_height), interpolation=cv2.INTER_AREA)


def write_event_still(
    video_path: Path,
    model: str,
    rep_number: int,
    records: list[dict],
    frame: int,
    joints: dict[str, int],
    suffix: str,
    caption: str,
    output_dir: Path,
) -> Path | None:
    record = records[frame]
    background = _read_frame(video_path, record["source_frame"])
    if background is None:
        return None
    points = record["points"]
    scores = record["scores"]
    if points is not None and scores is not None:
        color = (80, 220, 80)
        _draw_chain(
            background,
            points,
            scores,
            [joints["shoulder"], joints["hip"], joints["knee"]],
            color,
            2,
        )
        _draw_chain(
            background,
            points,
            scores,
            [joints["hip"], joints["knee"], joints["ankle"]],
            color,
            2,
        )
    _label(background, caption, (16, 32))
    image = _fit_height(background, _STILL_MAX_HEIGHT)
    path = output_dir / f"{video_path.stem}_{model}_rep{rep_number}_{suffix}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")
    return path


def _rep_payload(records: list[dict], rep: dict, number: int, side: str) -> dict:
    standing = rep["standing"]
    bottom = rep["bottom"]
    finish = rep["finish"]
    payload = {
        "rep": number,
        "down_ms": records[bottom]["t_ms"] - records[standing]["t_ms"],
        "bottom_ms": records[rep["bottom_end"]]["t_ms"] - records[rep["bottom_start"]]["t_ms"],
        "up_ms": None if finish is None else records[finish]["t_ms"] - records[bottom]["t_ms"],
        "standing": _snapshot(records, standing, side),
        "bottom": _snapshot(records, bottom, side),
        "drive": None if rep["drive"] is None else _snapshot(records, rep["drive"], side),
        "finish": None if finish is None else _snapshot(records, finish, side),
    }
    return payload


def write_context(
    video_path: Path,
    model: str,
    fps: float,
    records: list[dict],
    runner: RTMPoseRunner,
    output_dir: Path,
) -> tuple[list[dict], Path]:
    detected = sum(1 for item in records if item["chosen"].detected)
    side = video_side(records)
    if side is None:
        reps, note = [], "not enough confident knee angles to segment reps"
    else:
        reps, note = segment_reps(records, fps, side)
    rep_payloads = []
    for number, rep in enumerate(reps, start=1):
        joints = runner.joints(side or "left")
        for key, suffix, caption in _EVENT_STILLS:
            frame = rep.get(key)
            if frame is None:
                continue
            write_event_still(
                video_path, model, number, records, frame, joints, suffix, caption, output_dir
            )
        rep_payloads.append(_rep_payload(records, rep, number, side or "left"))
    reason = "success" if rep_payloads else (note or "not enough confident knee angles to segment reps")
    payload = {
        "video": video_path.name,
        "model": model,
        "fps": round(fps, 3),
        "side": side,
        "frame_count": len(records),
        "detected_frames": detected,
        "detection_rate": round(detected / len(records), 3) if records else 0,
        "reason": reason,
        "frames": [
            {
                "frame": index,
                "t_ms": item["t_ms"],
                "detected": item["chosen"].detected,
                "side": item["chosen"].side,
                "left": side_payload(item["left"]),
                "right": side_payload(item["right"]),
            }
            for index, item in enumerate(records)
        ],
        "reps": rep_payloads,
    }
    path = output_dir / f"{video_path.stem}_{model}_context.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    agent = {
        "video": payload["video"],
        "model": payload["model"],
        "fps": payload["fps"],
        "side": payload["side"],
        "reason": reason,
        "reps": rep_payloads,
    }
    agent_path = output_dir / f"{video_path.stem}_{model}_reps.json"
    agent_path.write_text(json.dumps(agent, indent=2), encoding="utf-8")
    return rep_payloads, agent_path


def process_video(
    video_path: Path,
    runner: RTMPoseRunner,
    output_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    output_dir = output_dir or DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / SAMPLE_FPS)))
    sampled_fps = fps / stride
    runner.reset()

    writer = ClipWriter(output_dir / f"{video_path.stem}_{runner.name}.mp4", fps, (width, height))
    measures: list[FrameMeasure] = []
    tracks: list[dict] = []

    source = 0
    last_points = None
    last_scores = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source % stride == 0:
                timestamp_ms = int(round(source * 1000.0 / fps))
                points, scores, measure, left, right = runner.infer(frame, timestamp_ms)
                last_points = None if points is None else points.copy()
                last_scores = None if scores is None else scores.copy()
                measures.append(measure)
                tracks.append(
                    {
                        "t_ms": timestamp_ms,
                        "source_frame": source,
                        "points": last_points,
                        "scores": last_scores,
                        "chosen": measure,
                        "left": left,
                        "right": right,
                    }
                )
            writer.write(runner.draw(frame, last_points, last_scores))
            if on_progress is not None and source % 40 == 0:
                on_progress(source + 1, frame_count)
            source += 1
    finally:
        capture.release()
        writer.close()

    if source == 0:
        raise RuntimeError(f"No frames in {video_path}")
    if on_progress is not None:
        on_progress(source, frame_count)
    summary = {"video": video_path.name, "model": runner.name, **summarize(measures)}
    write_context(video_path, runner.name, sampled_fps, tracks, runner, output_dir)
    return summary


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description="Overlay RTMPose on squat videos.")
    parser.add_argument("videos", nargs="*", type=Path, help="Video files. Defaults to dataset/*.mp4")
    args = parser.parse_args()
    videos = args.videos or sorted(DEFAULT_DATASET.glob("*.mp4"))
    if not videos:
        raise SystemExit("No videos found. Pass a path or put mp4 files in dataset/.")

    runner = RTMPoseRunner()
    all_summaries: list[dict] = []
    try:
        for video in videos:
            all_summaries.append(process_video(video, runner, DEFAULT_OUTPUT))
    finally:
        runner.close()

    summary_path = DEFAULT_OUTPUT / "summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print("\nBottom of the rep, using the lower hip point:")
    for row in all_summaries:
        hip = row["bottom_hip_below_knee"]
        hip_text = "n/a" if hip is None else ("below knee" if hip else "above knee")
        print(
            f"  {row['video']:16} {row['model']:10} "
            f"detected {row['detection_rate']:.0%}  "
            f"knee {row['bottom_knee_deg']} deg  "
            f"back {row['bottom_back_from_vertical_deg']} deg  "
            f"hip {hip_text}"
        )
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
