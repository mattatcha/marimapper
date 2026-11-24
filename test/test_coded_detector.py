"""Simulated-scene tests for --coded (no camera or LEDs needed)."""

import cv2
import numpy as np
import pytest

from marimapper import coded_detector as cd
from marimapper import robust_detector as rd
from marimapper.queues import DetectionControlEnum
from test_robust_detector import H, W, Scene, _errors, _scan, _stars


def _grid(n, cols=10, spacing=28):
    return {
        i: (20 + spacing * (i % cols), 20 + spacing * (i // cols)) for i in range(n)
    }


CODED = rd.DetectionSettings(robust=True, coded=True)


@pytest.mark.parametrize("n", [1, 2, 7, 60, 300])
def test_codebook_is_distinct_and_never_one_colour(n):
    codes = cd.codebook(n)
    assert len(codes) == n and len(set(codes)) == n
    assert all(len(set(c)) >= 2 for c in codes)
    for a in range(n):
        for b in range(a + 1, n):
            assert sum(x != y for x, y in zip(codes[a], codes[b])) >= cd.CODE_DISTANCE


def test_codes_stay_short():
    assert len(cd.codebook(300)[0]) <= 9


def test_finds_every_led_accurately_with_far_fewer_frames():
    positions = _grid(60)
    coded_scene, batched_scene = Scene(positions), Scene(positions)
    coded, q = _scan(coded_scene, 60, CODED)
    batched, _ = _scan(batched_scene, 60, rd.DetectionSettings(robust=True))
    assert [led.led_id for led in coded] == list(range(60))
    assert max(_errors(coded, positions)) < 2
    assert coded_scene.frames * 5 < batched_scene.frames
    assert len(q.items) == 60
    assert all(c == DetectionControlEnum.DETECT for c, _ in q.items)


def test_ignores_stars_cars_flashlights_and_reflections():
    positions = _grid(40, spacing=30)
    scene = Scene(
        positions, base=_stars(), cars=True, flashlight=True, reflections=True
    )
    leds, _ = _scan(scene, 40, CODED)
    found = {led.led_id for led in leds}
    assert len(found) >= 36  # a car can spoil a frame for an LED; the rest are exact
    assert max(_errors(leds, positions)) < 3


def test_overlapping_leds_are_separated_by_later_rounds():
    n = 30
    positions = {i: (int(30 + 26 * (i % 10)), 120) for i in range(n)}  # 3 LEDs per spot
    leds, q = _scan(Scene(positions), n, CODED)
    assert {led.led_id for led in leds} == set(range(n))
    assert max(_errors(leds, positions)) < 2
    assert len(q.items) == n


def test_hidden_leds_cost_nothing_and_are_not_reported():
    n = 60
    positions = _grid(n)
    hidden = {i for i in range(n) if i % 20 >= 5}
    scene = Scene(positions, hidden=hidden)
    leds, q = _scan(scene, n, CODED)
    assert {led.led_id for led in leds} == set(range(n)) - hidden
    assert len(q.items) == n
    assert sum(c == DetectionControlEnum.SKIP for c, _ in q.items) == len(hidden)
    assert scene.frames < 250


def test_nothing_visible_gives_up_quickly():
    scene = Scene(_grid(20), hidden=set(range(20)))
    leds, q = _scan(scene, 20, CODED)
    assert leds == [] and len(q.items) == 20


def test_slow_camera_latency_is_measured():
    class SlowScene(Scene):  # three frames of latency instead of one
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.history = [{}, {}]

        def read(self):
            self.history.append(dict(self.lit))
            self.lit, real = self.history.pop(0), self.lit
            try:
                return super().read()
            finally:
                self.lit = real

    positions = _grid(40)
    leds, _ = _scan(SlowScene(positions), 40, CODED)
    assert {led.led_id for led in leds} == set(range(40))
    assert max(_errors(leds, positions)) < 2


def test_step_and_white_only_backend_still_work():
    positions = _grid(24)
    leds, q = _scan(
        Scene(positions), 24, rd.DetectionSettings(robust=True, coded=True, step=4)
    )
    assert [led.led_id for led in leds] == [0, 4, 8, 12, 16, 20]
    assert len(q.items) == 24
    white, _ = _scan(
        Scene(positions), 24, CODED, colour=False
    )  # falls back to --robust
    assert len(white) == 24


def test_decode_rejects_two_leds_blurred_into_one_blob():
    background = np.zeros((H, W, 3), np.uint8)
    codes = cd.codebook(2)
    frames = []
    for s in range(len(codes[0])):
        img = background.copy()
        for k, code in enumerate(codes):  # both LEDs drawn on the same spot
            r, g, b = cd.SYMBOLS[code[s]]
            img[100:106, 100 + k : 106 + k] = np.maximum(
                img[100:106, 100 + k : 106 + k], (b, g, r)
            )
        frames.append((img, img))
    assert cd.decode(frames, background, codes, [0, 1], 40, CODED) == {}


def test_auto_exposure_bloom_is_waited_out():
    class AutoExposure(
        Scene
    ):  # after the lighting changes, LEDs bloom for a few frames
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.last_count, self.adjusting = 0, 0

        def read(self):
            count = len([c for c in self.shown.values() if any(c)])
            if abs(count - self.last_count) > 5:
                self.adjusting = 6
            self.last_count = count
            img = super().read()
            if self.adjusting:
                self.adjusting -= 1
                img = cv2.add(cv2.dilate(img, np.ones((15, 15), np.uint8)), 40)
            return img

    positions = _grid(60)
    leds, _ = _scan(AutoExposure(positions), 60, CODED)
    assert {led.led_id for led in leds} == set(range(60))
    assert max(_errors(leds, positions)) < 2


def test_one_unreadable_frame_still_decodes_when_unambiguous():
    codes = cd.codebook(60)
    lookup = dict(zip(codes, range(60)))
    for led in (0, 17, 59):
        symbols = list(codes[led])
        symbols[2] = -1
        assert cd._lookup(lookup, symbols) == led
        symbols[4] = -1
        assert cd._lookup(lookup, symbols) is None  # two missing: too risky


def test_debug_dump(tmp_path):
    settings = rd.DetectionSettings(robust=True, coded=True, coded_debug=str(tmp_path))
    _scan(Scene(_grid(10)), 10, settings)
    passes = sorted(p.name for p in tmp_path.iterdir())
    assert passes[0] == "pass000"
    assert (tmp_path / "pass000" / "background.png").exists()
    assert (tmp_path / "pass000" / "codes.json").exists()
