"""Verify distinct foreground faces before rendering stacked panels."""

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import tempfile
import threading
import urllib.request

import cv2
import numpy as np

from .detect import _attribute_faces_to_persons
from .types import BBox

log = logging.getLogger("ave.face_identity")
_REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
_MODELS = (
    ("face_detection_yunet", "face_detection_yunet_2023mar.onnx",
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    ("face_recognition_sface", "face_recognition_sface_2021dec.onnx",
     "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
)
_models = {}
_lock = threading.Lock()


def _sha256(path):
    """Hash a cached model without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_model(directory, folder, filename, digest):
    """Atomically cache an immutable, checksum-verified OpenCV model and license."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    if not target.exists() or _sha256(target) != digest:
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{_REVISION}/models/{folder}/{filename}"
        log.info("Downloading local face model: %s", filename)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url, timeout=30) as response:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
            if _sha256(temporary) != digest:
                raise ValueError(f"Face model checksum mismatch: {filename}")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    license_path = directory / f"{folder}.LICENSE"
    if not license_path.exists():
        url = f"https://raw.githubusercontent.com/opencv/opencv_zoo/{_REVISION}/models/{folder}/LICENSE"
        with urllib.request.urlopen(url, timeout=30) as response:
            license_path.write_bytes(response.read())
    return target


@dataclass
class FaceSubject:
    """One foreground face, associated body box, and transient recognition features."""

    person: BBox
    face: BBox
    landmarks: np.ndarray
    feature: np.ndarray | None = None


class FaceIdentityGuard:
    """Allow stacking only when distinct faces remain visible in both crops."""

    def __init__(self, cfg):
        self.settings = cfg.crop.face_identity
        self.detector = self.recognizer = None
        directory = Path(self.settings.model_dir).expanduser() if self.settings.model_dir else Path.home() / ".cache/podclipper/models"
        try:
            with _lock:
                key = (str(directory.resolve()), self.settings.detection_score)
                if key not in _models:
                    detector, recognizer = [_download_model(directory, *model) for model in _MODELS]
                    _models[key] = (
                        cv2.FaceDetectorYN.create(str(detector), "", (640, 640), self.settings.detection_score),
                        cv2.FaceRecognizerSF.create(str(recognizer), ""),
                    )
                self.detector, self.recognizer = _models[key]
        except Exception as exc:
            log.warning("Face verification unavailable (%s); unverified panels will not be stacked", type(exc).__name__)
        self.error_logged = False

    def subjects(self, frame, persons):
        """Find foreground faces; exclude small background people before framing."""
        if self.detector is None:
            return []
        height, width = frame.shape[:2]
        scale = min(1.0, 960 / max(width, height))
        reduced = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
        try:
            self.detector.setInputSize((reduced.shape[1], reduced.shape[0]))
            _, rows = self.detector.detect(reduced)
            if rows is None:
                return []
            rows = rows.copy()
            rows[:, :14] /= scale
            largest = float(max(row[3] for row in rows))
            rows = [row for row in rows if row[3] >= height * self.settings.min_face_height_frac
                    and row[3] >= largest * self.settings.min_relative_face_size]
            faces = [BBox(*map(float, row[:4]), confidence=float(row[14])) for row in rows]
            attributed = _attribute_faces_to_persons(persons, faces)
            result = []
            for row, face in zip(rows, faces):
                person = next((p for p, assigned in zip(persons, attributed) if assigned is face), None)
                if person is None:
                    person = BBox(face.x - face.w * .5, face.y - face.h * .2, face.w * 2,
                                  min(height, face.h * 3.5), face.confidence)
                person = BBox(face.x_center - person.w / 2, person.y, person.w, person.h, person.confidence)
                result.append(FaceSubject(person, face, row))
            return result
        except (cv2.error, ValueError, TypeError) as exc:
            if not self.error_logged:
                log.warning("Face verification failed (%s); retaining a full source view", type(exc).__name__)
                self.error_logged = True
            return []

    def distinct(self, frame, first, second):
        """Compare local face features without naming people or persisting embeddings."""
        if first is second or first.face.iou(second.face) > 0.3:
            return False
        try:
            for subject in (first, second):
                if subject.feature is None:
                    aligned = self.recognizer.alignCrop(frame, subject.landmarks)
                    subject.feature = self.recognizer.feature(aligned).copy()
                if not np.isfinite(subject.feature).all() or np.linalg.norm(subject.feature) == 0:
                    return False
            score = self.recognizer.match(first.feature, second.feature, cv2.FaceRecognizerSF_FR_COSINE)
            return np.isfinite(score) and score < self.settings.similarity_threshold
        except cv2.error:
            return False

    def visible(self, subject, box):
        """Require nearly the entire intended face to remain inside its output crop."""
        if box is None:
            return False
        x, y, width, height = box
        face = subject.face
        area = max(0, min(x + width, face.x + face.w) - max(x, face.x)) * max(
            0, min(y + height, face.y + face.h) - max(y, face.y))
        return area / max(1, face.area) >= self.settings.min_face_coverage

    def valid_stack(self, frame, subjects, first, second, first_box, second_box):
        """Reject duplicate identities, clipped faces, and overlapping views of both people."""
        left = next((subject for subject in subjects if subject.person is first), None)
        right = next((subject for subject in subjects if subject.person is second), None)
        if left is None or right is None or not self.visible(left, first_box) or not self.visible(right, second_box):
            return False
        if self.visible(left, second_box) or self.visible(right, first_box):
            return False
        return self.distinct(frame, left, right)
