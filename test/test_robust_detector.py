"""Simulated-scene tests for the --robust detector (no camera or LEDs needed)."""

import math

import cv2
import numpy as np
import pytest

from marimapper import robust_detector as rd
from marimapper.detector import find_led_in_image
from marimapper.led import LEDInfo
from marimapper.queues import DetectionControlEnum
from marimapper.timeout_controller import TimeoutController

H, W = 240, 320


def _stars(seed=1, count=40):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    for _ in range(count):
        cv2.circle(
            img,
            (int(rng.integers(5, W - 5)), int(rng.integers(5, H - 5))),
            1,
            (255, 255, 255),
            -1,
        )
    return img


class Scene:
    """Fake camera: draws whatever the fake backend has lit, one frame late, plus distractions."""

    def __init__(
        self,
        positions,
        base=None,
        hidden=(),
        cars=False,
        flashlight=False,
        reflections=False,
    ):
        self.positions, self.hidden = positions, set(hidden)
        self.base = np.zeros((H, W, 3), np.uint8) if base is None else base
        self.cars, self.flashlight, self.reflections = cars, flashlight, reflections
        self.lit, self.shown, self.frames = {}, {}, 0

    def read(self):
        self.frames += 1
        img = self.base.copy()
        shown, self.shown = self.shown, dict(self.lit)  # one frame of latency
        if self.cars:  # headlights sweeping across, brighter than any LED
            x = 10 + (self.frames * 7) % (W - 40)
            cv2.circle(img, (x, H - 20), 8, (255, 255, 255), -1)
            cv2.circle(img, (x + 25, H - 20), 8, (255, 255, 255), -1)
        if self.flashlight and self.frames % 7 < 3:
            cv2.circle(img, (W - 30, 25), 10, (210, 230, 255), -1)
        for i, (r, g, b) in shown.items():
            if i in self.hidden or not (r or g or b):
                continue
            x, y = self.positions[i]
            cv2.circle(img, (x, y), 5, (b, g, r), -1)  # coloured glow
            cv2.circle(img, (x, y), 2, (255, 255, 255), -1)  # saturated core
            if self.reflections:
                cv2.ellipse(
                    img,
                    (x, min(H - 3, y + 30)),
                    (30, 7),
                    0,
                    0,
                    360,
                    (b // 4, g // 4, r // 4),
                    -1,
                )
        return img


class Backend:
    def __init__(self, scene, count, colour=True):
        self.scene, self.count = scene, count
        if colour:
            self.set_leds = self._set_leds

    def get_led_count(self):
        return self.count

    def _set_leds(self, buffer):
        self.scene.lit = {i: tuple(c) for i, c in enumerate(buffer) if any(c)}

    def set_led(self, i, on):
        if on:
            self.scene.lit[i] = (255, 255, 255)
        else:
            self.scene.lit.pop(i, None)


class Queue:
    def __init__(self):
        self.items = []

    def put(self, control, data):
        self.items.append((control, data))


def _px(point):
    return point.u() * W, point.v() * W - (W - H) / 2


def _scan(scene, n, settings, colour=True):
    queue = Queue()
    leds = rd.detect_leds(
        0,
        n,
        scene,
        Backend(scene, n, colour),
        0,
        # fixed: the fake camera answers in microseconds, so an adapted timeout would be flaky
        TimeoutController(default_timeout_sec=0.3, sample_size_min=10**9),
        40,
        False,
        [queue],
        settings,
    )
    return leds, queue


def _errors(leds, positions):
    return [math.dist(_px(led.point), positions[led.led_id]) for led in leds]


def _curve(n):
    return {
        i: (int(30 + 260 * i / n), int(120 + 70 * math.sin(i / n * 2 * math.pi)))
        for i in range(n)
    }


def test_ignores_stars_that_the_classic_detector_picks():
    positions = {0: (250, 60)}
    scene = Scene(positions, base=_stars())
    scene.lit = {0: (255, 255, 255)}
    classic = find_led_in_image(scene.read(), 128)
    assert math.dist(_px(classic), positions[0]) > 10  # classic picks a star

    leds, _ = _scan(
        Scene(positions, base=_stars()), 1, rd.DetectionSettings(robust=True)
    )
    assert len(leds) == 1 and _errors(leds, positions)[0] < 2


def test_rejects_cars_flashlights_and_reflections():
    positions = _curve(12)
    scene = Scene(
        positions, base=_stars(), cars=True, flashlight=True, reflections=True
    )
    leds, _ = _scan(scene, 12, rd.DetectionSettings(robust=True))
    assert len(leds) == 12
    assert max(_errors(leds, positions)) < 3


def test_centre_is_core_not_lopsided_glow():
    bg = np.zeros((H, W, 3), np.uint8)
    img = bg.copy()
    cv2.ellipse(
        img, (212, 100), (14, 7), 0, 0, 360, (0, 0, 255), -1
    )  # red glow pushed 12 px right
    cv2.circle(
        img, (200, 100), 3, (255, 255, 255), -1
    )  # saturated core at the real LED
    p = rd.locate(img, bg, 2, 40, rd.DetectionSettings())
    assert math.dist(_px(p), (200, 100)) < 1


def test_batching_is_accurate_and_three_times_cheaper():
    positions = _curve(30)
    batched_scene, single_scene = Scene(positions), Scene(positions)
    batched, q = _scan(batched_scene, 30, rd.DetectionSettings(robust=True, batch=3))
    single, _ = _scan(single_scene, 30, rd.DetectionSettings(robust=True, batch=1))
    assert len(batched) == len(single) == 30
    assert max(_errors(batched, positions)) < 3
    assert batched_scene.frames * 2.5 < single_scene.frames
    assert len(q.items) == 30 and [led.led_id for led in batched] == list(range(30))


def test_overlapping_leds_fall_back_to_single():
    n = 30
    positions = {
        i: (int(30 + 260 * (i % 10) / 10), 120) for i in range(n)
    }  # three passes, same spots
    leds, _ = _scan(Scene(positions), n, rd.DetectionSettings(robust=True))
    assert len(leds) == n
    assert max(_errors(leds, positions)) < 3


def test_hidden_leds_are_not_reported_and_not_retried_forever():
    n = 60
    positions = {
        i: (int(160 + 80 * math.sin(i / 20 * 6.28)), int(220 - 3 * i)) for i in range(n)
    }
    hidden = {i for i in range(n) if i % 20 >= 5}  # tree: a quarter faces the camera
    leds, q = _scan(
        Scene(positions, hidden=hidden), n, rd.DetectionSettings(robust=True)
    )
    found = {led.led_id for led in leds}
    assert found == set(range(n)) - hidden
    assert len(q.items) == n


def test_step_scans_every_nth_led_and_completes_progress():
    positions = _curve(24)
    leds, q = _scan(Scene(positions), 24, rd.DetectionSettings(robust=True, step=4))
    assert [led.led_id for led in leds] == [0, 4, 8, 12, 16, 20]
    assert len(q.items) == 24
    assert sum(c == DetectionControlEnum.SKIP for c, _ in q.items) == 18


def test_white_only_backend_still_works():
    positions = _curve(6)
    leds, _ = _scan(
        Scene(positions, base=_stars()),
        6,
        rd.DetectionSettings(robust=True),
        colour=False,
    )
    assert len(leds) == 6
    assert max(_errors(leds, positions)) < 3


@pytest.mark.parametrize("step", [1, 3])
def test_preview_and_coverage_buffers(step):
    settings = rd.DetectionSettings(robust=True, step=step, status_brightness=0.5)
    preview = rd.preview_buffer(12, range(3, 9), settings)
    assert [i for i, c in enumerate(preview) if any(c)] == list(range(3, 9, step))

    coverage = rd.coverage_buffer(
        {3: LEDInfo.RECONSTRUCTED, 6: LEDInfo.DETECTED}, 12, range(3, 9), settings
    )
    assert coverage[3] == [0, 127, 0]  # green, in 3D
    assert coverage[6] == [127, 82, 0]  # orange, seen but not in 3D
    unseen = [i for i in range(3, 9, step) if i not in (3, 6)]
    assert all(coverage[i] == [127, 0, 127] for i in unseen)  # magenta, never seen
    assert not any(any(coverage[i]) for i in range(12) if i not in range(3, 9))
