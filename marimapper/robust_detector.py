"""Robust LED detection for outdoor / uncontrolled scenes (enabled with --robust).

The classic detector treats the brightest blob above --threshold as the LED, so stars, streetlights,
lit windows, reflections, passing cars and flashlights all get detected as LEDs. This detector:

* background subtraction: captures the scene with the LED off first and only looks at what changed;
* colour flashes: lights the LED red/green/blue and only accepts a change in that colour, so white
  light sources (headlights, flashlights, stars) cancel out;
* confirmation: flashes every LED twice in different colours and requires both detections to agree;
* peak picking: chooses the blob containing the brightest change (reflections are large but dimmer)
  and centres on the hottest pixels (a saturated core) rather than a lopsided coloured glow;
* batching: lights several far-apart LEDs at once, one colour each (~3x faster). Per view it falls
  back to one LED at a time when LEDs overlap from that angle, and stops re-trying LEDs that are
  simply hidden from that angle.

Colour/batching need a backend with set_leds(); otherwise it falls back to white flashes via set_led().
"""

import time
from dataclasses import dataclass
from multiprocessing import get_logger
from typing import Optional

import cv2
import numpy as np

from marimapper.camera import Camera
from marimapper.detector import draw_led_detections, show_image
from marimapper.led import LED2D, LEDInfo, Point2D, get_color
from marimapper.queues import DetectionControlEnum
from marimapper.timeout_controller import TimeoutController

logger = get_logger()

STALE_FRAMES = 2  # frames buffered from before the LEDs changed state
DARK_TIMEOUT = 3.0
CONFIRM_RETRIES = 3
BATCH_COLOURS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB; BGR channel = 2, 1, 0
_CHANNEL = {(255, 0, 0): 2, (0, 255, 0): 1, (0, 0, 255): 0}
UNSEEN_COLOUR = [255, 0, 255]


@dataclass
class DetectionSettings:
    robust: bool = False
    batch: int = 3  # LEDs lit per flash (1-3)
    confirm: int = 2  # flashes per LED that must agree
    step: int = 1  # scan every step-th LED (anchor scans for placing props)
    preview: bool = True  # light the scan range while aiming
    preview_colour: tuple = (64, 64, 64)
    status_brightness: float = 0.35  # coverage colours shown between views
    bg_frames: int = 4
    bg_dilate: int = 5
    core_fraction: float = 0.8
    confirm_tolerance: float = 0.015  # fraction of image width
    min_separation: float = 0.03  # fraction of image width
    adapt_window: int = 8
    overlap_limit: float = 0.6
    hidden_limit: float = 0.15
    resample: int = 5


# ---------------------------------------------------------------------------------------- images


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def capture_background(cam: Camera, settings: DetectionSettings) -> np.ndarray:
    """Per-pixel max over a few LED-off frames (covers twinkle), grown slightly (covers wobble). BGR."""
    for _ in range(STALE_FRAMES):
        cam.read()
    bg = cam.read()
    for _ in range(settings.bg_frames - 1):
        bg = cv2.max(bg, cam.read())
    if settings.bg_dilate > 1:
        bg = cv2.dilate(bg, np.ones((settings.bg_dilate, settings.bg_dilate), np.uint8))
    return bg


def find_peak_led(diff: np.ndarray, threshold: int) -> Optional[Point2D]:
    """Blob containing the brightest change (not the largest summed brightness), brightness-weighted centre."""
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    _, _, _, peak = cv2.minMaxLoc(cv2.GaussianBlur(diff, (3, 3), 0))
    led = next(
        (
            c
            for c in contours
            if cv2.pointPolygonTest(c, (float(peak[0]), float(peak[1])), False) >= 0
        ),
        max(contours, key=cv2.contourArea),
    )
    blob = np.zeros_like(diff)
    cv2.drawContours(blob, [led], -1, 255, -1)
    m = cv2.moments(cv2.bitwise_and(diff, diff, mask=blob))
    if m["m00"] == 0:
        return None
    h, w = diff.shape
    u = (m["m10"] / m["m00"]) / w
    v = (
        (m["m01"] / m["m00"]) + (w - h) / 2.0
    ) / w  # same normalisation as detector.find_led_in_image
    return Point2D(u, v, [led])


def colour_change(
    image: np.ndarray, background: np.ndarray, channel: int
) -> np.ndarray:
    """How much `channel` rose beyond the other two. White light raises all three and scores ~0."""
    d = cv2.subtract(image, background)
    others = [c for c in range(3) if c != channel]
    return cv2.subtract(
        d[:, :, channel], cv2.max(d[:, :, others[0]], d[:, :, others[1]])
    )


def locate(
    image, background, channel: Optional[int], threshold: int, settings
) -> Optional[Point2D]:
    """Find the LED in one frame. Colour picks WHICH blob; plain brightness picks WHERE its centre is."""
    bright = cv2.subtract(_gray(image), _gray(background))
    if channel is None:
        return find_peak_led(bright, threshold)
    picked = find_peak_led(colour_change(image, background, channel), threshold)
    if picked is None:
        return None
    region = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(region, picked.contours, -1, 255, -1)
    region = cv2.dilate(
        region, np.ones((5, 5), np.uint8)
    )  # a saturated core is a hole in the colour image
    bright = cv2.bitwise_and(bright, bright, mask=region)
    core_level = max(threshold, int(settings.core_fraction * int(bright.max())))
    return find_peak_led(bright, core_level) or picked


# -------------------------------------------------------------------------------------- backend


class _Leds:
    """Drives the backend: one set_leds() call per state change when available."""

    def __init__(self, backend):
        self.backend = backend
        self.coloured = hasattr(backend, "set_leds")
        self.count = backend.get_led_count() if self.coloured else 0

    def show(self, colours: dict):
        """colours: {led_id: (r, g, b)}; every other LED off."""
        if self.coloured:
            buffer = [[0, 0, 0] for _ in range(self.count)]
            for i, c in colours.items():
                if 0 <= i < self.count:
                    buffer[i] = list(c)
            self.backend.set_leds(buffer)
        else:
            for i, c in colours.items():
                self.backend.set_led(i, any(c))

    def off(self, ids):
        if self.coloured:
            self.show({})
        else:
            for i in ids:
                self.backend.set_led(i, False)


def _show_found(image, found, display):
    if display:
        for p in found:
            if p is not None:
                image = draw_led_detections(image, p)
        show_image(image)


# ------------------------------------------------------------------------------------- flashing


def flash(
    cam,
    leds: _Leds,
    colours: dict,
    wait_for: list,
    tc: TimeoutController,
    threshold,
    display,
    settings,
):
    """Light `colours` ({led_id: rgb}) at once; return {led_id: Point2D|None} for `wait_for`."""
    leds.off(colours)
    background = capture_background(cam, settings)
    channel = {
        i: (_CHANNEL.get(tuple(colours[i])) if leds.coloured else None)
        for i in wait_for
    }

    start = time.time()
    leds.show(colours)
    found = {i: None for i in wait_for}
    while any(p is None for p in found.values()) and time.time() < start + tc.timeout:
        image = cam.read()
        for i in wait_for:
            if found[i] is None:
                found[i] = locate(image, background, channel[i], threshold, settings)
                if found[i] is not None:
                    tc.add_response_time(
                        time.time() - start
                    )  # per LED: hidden ones must not stretch it
        _show_found(image, found.values(), display)
    leds.off(colours)

    # Wait until nothing we flashed is still visible, so the next background is clean.
    off_start = time.time()
    while time.time() - off_start < DARK_TIMEOUT:
        image = cam.read()
        if all(
            locate(image, background, channel[i], threshold, settings) is None
            for i in wait_for
        ):
            return found
    logger.warning(
        f"LEDs {wait_for}: still visible after switching off, discarding flash"
    )
    return {i: None for i in wait_for}


def _agree(points, settings) -> bool:
    us = [p.u() for p in points]
    vs = [p.v() for p in points]
    return (
        max(us) - min(us) <= settings.confirm_tolerance
        and max(vs) - min(vs) <= settings.confirm_tolerance
    )


def find_single(
    cam, leds, led_id, view_id, tc, threshold, display, settings
) -> Optional[LED2D]:
    """One LED, flashed `confirm` times (rotating colours); retried only when flashes disagree."""
    for _ in range(CONFIRM_RETRIES):
        points = []
        for k in range(max(1, settings.confirm)):
            colour = (
                BATCH_COLOURS[k % len(BATCH_COLOURS)]
                if leds.coloured
                else (255, 255, 255)
            )
            p = flash(
                cam, leds, {led_id: colour}, [led_id], tc, threshold, display, settings
            )[led_id]
            if p is None:
                break
            points.append(p)
        if not points:
            return None  # not visible from this angle: retrying only burns timeouts
        if len(points) == max(1, settings.confirm) and _agree(points, settings):
            return LED2D(led_id, view_id, points[-1])
    return None


def find_group(cam, leds, group, tc, threshold, display, settings):
    """Several far-apart LEDs per flash, one colour each, rotating colours between flashes.

    Returns (accepted {led_id: Point2D}, unseen {led_ids not seen on the first flash}).
    """
    flashes, active, unseen = [], list(group), set()
    for k in range(max(2, settings.confirm)):
        colours = {
            i: BATCH_COLOURS[(j + k) % len(BATCH_COLOURS)]
            for j, i in enumerate(group)
            if i in active
        }
        f = flash(cam, leds, colours, active, tc, threshold, display, settings)
        flashes.append(f)
        if k == 0:
            unseen = {i for i in group if f.get(i) is None}
        active = [
            i for i in active if f.get(i) is not None
        ]  # confirming flashes skip unseen LEDs
        if not active:
            return {}, unseen
    accepted = {
        i: flashes[-1][i] for i in active if _agree([f[i] for f in flashes], settings)
    }
    for a in list(
        accepted
    ):  # LEDs landing on top of each other may be confused: re-do singly
        for b in list(accepted):
            if a < b and a in accepted and b in accepted:
                pa, pb = accepted[a], accepted[b]
                if (
                    (pa.u() - pb.u()) ** 2 + (pa.v() - pb.v()) ** 2
                ) ** 0.5 < settings.min_separation:
                    del accepted[a], accepted[b]
    return accepted, unseen


# ----------------------------------------------------------------------------- process hooks


def find_led(
    cam, threshold: int, display: bool, settings: DetectionSettings
) -> Optional[Point2D]:
    """'Is anything already lit?' check: only reports changes, so a static bright scene passes."""
    return locate(
        cam.read(), capture_background(cam, settings), None, threshold, settings
    )


def detect_leds(
    led_id_from,
    led_id_to,
    cam,
    led_backend,
    view_id,
    tc,
    threshold,
    display,
    output_queues,
    settings: DetectionSettings,
):
    leds = _Leds(led_backend)
    ids = list(range(led_id_from, led_id_to))
    if settings.step > 1:
        skipped = [i for i in ids if (i - led_id_from) % settings.step]
        ids = [i for i in ids if not (i - led_id_from) % settings.step]
        for i in skipped:  # keep the progress bar honest
            for q in output_queues:
                q.put(DetectionControlEnum.SKIP, i)

    batch = max(1, min(len(BATCH_COLOURS), settings.batch)) if leds.coloured else 1
    stride = -(-len(ids) // batch)
    groups = [
        [ids[k + j * stride] for j in range(batch) if k + j * stride < len(ids)]
        for k in range(stride)
    ]
    batched = batch > 1

    found_leds, singles, skipped_hidden, unseen_count = [], 0, 0, 0
    retry_hits, retry_unseen = (
        [],
        True,
    )  # did one-at-a-time re-tries of batch-missed LEDs find them?
    for group in groups:
        window = retry_hits[-settings.adapt_window :]
        if batched and len(window) >= settings.adapt_window:
            rate = sum(window) / len(window)
            if rate > settings.overlap_limit:
                batched = False
                logger.warning(
                    f"view {view_id}: LEDs overlap from this angle - one LED at a time for this view"
                )
            else:
                retry_unseen = rate >= settings.hidden_limit

        accepted, unseen = (
            find_group(cam, leds, group, tc, threshold, display, settings)
            if batched
            else ({}, set())
        )
        for i in sorted(group):
            led = LED2D(i, view_id, accepted[i]) if i in accepted else None
            if led is None:
                is_unseen = batched and i in unseen
                unseen_count += is_unseen
                if is_unseen and not retry_unseen and unseen_count % settings.resample:
                    skipped_hidden += 1  # hidden from this angle; the batch already gave it its chance
                else:
                    singles += 1
                    led = find_single(
                        cam, leds, i, view_id, tc, threshold, display, settings
                    )
                    if is_unseen:
                        retry_hits.append(led is not None)
            for q in output_queues:
                (
                    q.put(DetectionControlEnum.DETECT, led)
                    if led is not None
                    else q.put(DetectionControlEnum.SKIP, i)
                )
            if led is not None:
                found_leds.append(led)

    logger.warning(
        f"view {view_id}: {len(found_leds)}/{len(ids)} LEDs found; {singles} tried one at a time, "
        f"{skipped_hidden} treated as hidden"
    )
    found_leds.sort(key=lambda led: led.led_id)
    return found_leds


def enable_and_find_led(
    cam, led_backend, led_id, view_id, tc, threshold, display, settings
):
    """Single LED (movement check)."""
    return find_single(
        cam, _Leds(led_backend), led_id, view_id, tc, threshold, display, settings
    )


def preview_buffer(led_count: int, led_range: range, settings: DetectionSettings):
    """Scan range lit dim (every step-th LED) so the camera can be aimed."""
    buffer = [[0, 0, 0] for _ in range(led_count)]
    for i in led_range[:: max(1, settings.step)]:
        if 0 <= i < led_count:
            buffer[i] = list(settings.preview_colour)
    return buffer


def coverage_buffer(
    led_info: dict,
    led_count: int,
    led_range: Optional[range],
    settings: DetectionSettings,
):
    """Reconstruction status per LED, brighter than the classic display, and magenta for LEDs in the scan
    range that no view has seen yet (classic leaves them off, indistinguishable from 'not scanned').

    green = in 3D, aqua = interpolated/merged, orange = seen but not in 3D yet, red = unreconstructable.
    """
    size = max(led_count, max(led_info, default=-1) + 1)
    level = settings.status_brightness
    buffer = [[0, 0, 0] for _ in range(size)]
    if led_range is not None:
        for i in led_range[:: max(1, settings.step)]:
            if i < size:
                buffer[i] = [int(v * level) for v in UNSEEN_COLOUR]
    for i, info in led_info.items():
        if info != LEDInfo.NONE:
            buffer[i] = [int(v * level) for v in get_color(info)]
    return buffer
