"""Exercise the renderer without model downloads or external encoders."""
import io
from types import SimpleNamespace
import numpy as np
import pytest
import podclipper.crop as crop
from podclipper.config import load_default_config
from podclipper.types import BBox


def render(monkeypatch, tmp_path, frames, persons, wide):
    cfg = load_default_config()
    cfg.crop.target_width, cfg.crop.target_height = 108, 192
    cfg.crop.scene_cuts.crossfade_seconds = 0
    remaining = iter(frames)
    cap = SimpleNamespace(isOpened=lambda: True,
        get=lambda prop: {crop.cv2.CAP_PROP_FPS: 25, crop.cv2.CAP_PROP_FRAME_WIDTH: 600,
                          crop.cv2.CAP_PROP_FRAME_HEIGHT: 180}.get(prop, 0),
        release=lambda: None)
    def read():
        frame = next(remaining, None)
        return frame is not None, frame
    cap.read = read
    monkeypatch.setattr(crop.cv2, 'VideoCapture', lambda _: cap)
    encoded, boxes, poses = [], [], []
    encoder = SimpleNamespace(stdin=SimpleNamespace(write=encoded.append, close=lambda: None),
                              stderr=io.BytesIO(), wait=lambda **_: 0)
    monkeypatch.setattr(crop, '_open_ffmpeg_pipe', lambda *args: encoder)
    monkeypatch.setattr(crop, '_mux_audio', lambda *args: None)
    monkeypatch.setattr(crop, '_get_pose_landmarker', lambda _: object())
    def pose(frame, person, model, panel=None):
        poses.append(person)
        return {'nose': (person.x_center, 65), 'shoulder_mid': (person.x_center, 95)}
    monkeypatch.setattr(crop, '_pose_anchors_for_person', pose)
    original = crop._render_from_bbox
    def capture(frame, box, width, height):
        boxes.append(box)
        return original(frame, box, width, height)
    monkeypatch.setattr(crop, '_render_from_bbox', capture)
    crop.smart_crop_916_stacked(tmp_path/'input.mp4', persons, np.array(wide), tmp_path/'out.mp4', cfg)
    assert len(encoded) == len(frames)
    assert all(len(frame) == 108 * 192 * 3 for frame in encoded)
    return boxes, poses


def test_stale_wide_flag_cannot_duplicate_single_person(monkeypatch, tmp_path):
    frame = np.full((180, 600, 3), 100, dtype=np.uint8)
    persons = [[BBox(200, 10, 150, 160), BBox(10, 120, 20, 25)]]
    boxes, _ = render(monkeypatch, tmp_path, [frame], persons, [True])
    assert len(boxes) == 1


def test_scene_cut_refreshes_pose_even_with_unchanged_person_boxes(monkeypatch, tmp_path):
    frames = [np.full((180, 600, 3), value, dtype=np.uint8) for value in (70, 180)]
    people = [BBox(30, 10, 140, 160), BBox(400, 10, 140, 160)]
    _, poses = render(monkeypatch, tmp_path, frames, [people, people], [True, True])
    assert len(poses) == 4


def test_three_columns_keep_two_chosen_views_when_sizes_change(monkeypatch, tmp_path):
    frame = np.full((180, 600, 3), (80, 100, 140), dtype=np.uint8)
    frame[:, 198:202] = frame[:, 398:402] = 255
    people = [BBox(20, 20, 80, 120), BBox(215, 10, 160, 160), BBox(420, 10, 150, 150)]
    changed = [BBox(0, 0, 190, 180), people[1], people[2]]
    boxes, _ = render(monkeypatch, tmp_path, [frame, frame.copy()], [people, changed], [True, True])
    assert len(boxes) == 4
    for x, y, w, h in boxes:
        assert (204 <= x and x+w <= 396) or (404 <= x and x+w <= 600)
        assert 0 <= y < y+h <= 180


@pytest.mark.parametrize('duration', [0, 1, 3])
def test_crossfade_finishes_on_time_and_preserves_dimensions(duration):
    transition = crop._CutTransition(duration)
    old = np.zeros((24, 12, 3), dtype=np.uint8)
    new = np.full_like(old, 200)
    assert np.array_equal(transition.apply(old, False), old)
    output = [transition.apply(new, i == 0) for i in range(4)]
    assert all(frame.shape == new.shape for frame in output)
    assert np.array_equal(output[max(0, duration-1)], new)
    if duration > 1:
        assert 0 < output[0].mean() < output[1].mean() < 200
    else:
        assert np.array_equal(output[0], new)
