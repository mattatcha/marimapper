"""Coded scanning (--coded): every LED in the scan range blinks its own colour code at the same time.

Instead of a few LEDs per flash, all of them are lit for a short sequence of camera frames, each LED
showing red, green or blue per frame according to its code. Every blob in the image is read back as a
colour sequence and matched to a code, so a whole view takes a couple of seconds however many LEDs
there are. Codes differ from each other in at least CODE_DISTANCE frames, so a misread frame can't
turn one LED into another.

It keeps --robust's protections:

* background subtraction: only what changed since the LEDs-off frame counts;
* colour: each frame a blob must be one clear colour; white light (headlights, flashlights) and two
  LEDs blurred into one blob are rejected;
* confirmation: a blob must show the full code, each frame read twice and agreeing;
* reflections: when several blobs decode to the same LED the brightest one wins, if it clearly does.

LEDs that overlap from the camera's angle blur into one unreadable blob. They're retried in random
smaller groups (halving each round) until they separate. LEDs hidden from the angle cost nothing:
the whole range is scanned at once either way.
"""

import itertools
import random
import time
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from marimapper.led import LED2D, Point2D
from marimapper.queues import DetectionControlEnum
from marimapper.robust_detector import (
    BATCH_COLOURS,
    DARK_TIMEOUT,
    _CHANNEL,
    _gray,
    _show_found,
    capture_background,
    logger,
)

SYMBOLS = BATCH_COLOURS  # symbol k = this RGB colour
CODE_DISTANCE = 3  # any two codes differ in at least this many frames
MIN_CODE_LENGTH = 3
PURITY = (
    0.5  # second colour's share of a blob, at most this fraction of the main colour's
)
DOMINANCE = 2.0  # a duplicate decode wins only if this much brighter than the next


# ------------------------------------------------------------------------------------------ codes


@lru_cache(maxsize=None)
def codebook(n: int) -> tuple:
    """n codes (tuples of symbol indices), as short as possible, pairwise >= CODE_DISTANCE apart.
    No code is a single colour throughout, so a light that simply turns on can't match one.
    """
    length = MIN_CODE_LENGTH
    while True:
        codes = _lexicode(length, n)
        if len(codes) >= n:
            return tuple(codes[:n])
        length += 1


def _lexicode(length: int, n: int) -> list:
    chosen = np.zeros((0, length), np.int8)
    for candidate in itertools.product(range(len(SYMBOLS)), repeat=length):
        if len(set(candidate)) < 2:
            continue
        c = np.array(candidate, np.int8)
        if len(chosen) and (chosen != c).sum(1).min() < CODE_DISTANCE:
            continue
        chosen = np.vstack([chosen, c])
        if len(chosen) >= n:
            break
    return [tuple(int(x) for x in row) for row in chosen]


# ------------------------------------------------------------------------------------- decoding


def blob_symbols(image, background, labels_flat, count, threshold) -> np.ndarray:
    """Per blob label: the symbol it shows in this frame, or -1 when dark or not one clear colour."""
    d = cv2.subtract(image, background)
    sums = []
    for colour in SYMBOLS:
        ch = _CHANNEL[tuple(colour)]
        a, b = [c for c in range(3) if c != ch]
        excess = cv2.subtract(d[:, :, ch], cv2.max(d[:, :, a], d[:, :, b]))
        sums.append(np.bincount(labels_flat, weights=excess.ravel(), minlength=count))
    sums = np.stack(sums, 1)
    ordered = np.sort(sums, 1)
    lit = (
        ordered[:, -1] >= 2 * threshold
    )  # at least a couple of pixels clearly that colour
    pure = ordered[:, -2] <= PURITY * ordered[:, -1]
    return np.where(lit & pure, sums.argmax(1), -1)


def _blob_point(
    mean, labels, label, stats, threshold, core_fraction
) -> Optional[Point2D]:
    """Brightness-weighted centre of the blob's hottest pixels, normalised like the other detectors."""
    x, y, w, h = (int(v) for v in stats[label, :4])
    inside = labels[y : y + h, x : x + w] == label
    region = np.where(inside, mean[y : y + h, x : x + w], 0.0)
    top = float(region.max())
    if top <= 0:
        return None
    core = region * (region >= max(threshold, core_fraction * top))
    total = core.sum()
    if total <= 0:
        return None
    ys, xs = np.mgrid[0:h, 0:w]
    cx = x + (core * xs).sum() / total
    cy = y + (core * ys).sum() / total
    contours, _ = cv2.findContours(
        inside.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
        offset=(x, y),
    )
    height, width = labels.shape
    return Point2D(cx / width, (cy + (width - height) / 2.0) / width, list(contours))


def decode(frames, background, codes, ids, threshold, settings) -> dict:
    """frames: per code position, two frames of the same symbol. Returns {led_id: Point2D}."""
    bg_gray = _gray(background)
    brights = [cv2.subtract(_gray(a), bg_gray) for a, _ in frames]
    peak = np.maximum.reduce(brights)
    mask = cv2.dilate(
        (peak >= threshold).astype(np.uint8), np.ones((3, 3), np.uint8)
    )  # a saturated core can leave a gap between core and glow
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return {}
    flat = labels.ravel()

    read = np.full((count, len(frames)), -1, int)
    for s, (a, b) in enumerate(frames):
        sa = blob_symbols(a, background, flat, count, threshold)
        sb = blob_symbols(b, background, flat, count, threshold)
        read[:, s] = np.where(sa == sb, sa, -1)

    lookup = dict(zip(codes, ids))
    mean = np.mean(brights, axis=0)
    strength = np.zeros(count)
    np.maximum.at(strength, flat, mean.ravel())

    decoded = {}  # led -> [(strength, label)]
    for label in range(1, count):
        led = lookup.get(tuple(int(v) for v in read[label]))
        if led is not None:
            decoded.setdefault(led, []).append((strength[label], label))

    found = {}
    for led, blobs in decoded.items():
        blobs.sort(reverse=True)
        if len(blobs) > 1 and blobs[0][0] < DOMINANCE * blobs[1][0]:
            continue  # two similar blobs claim the same LED: can't tell which is real
        point = _blob_point(
            mean, labels, blobs[0][1], stats, threshold, settings.core_fraction
        )
        if point is not None:
            found[led] = point
    return found


# ------------------------------------------------------------------------------------- scanning


class Timing:
    """Camera latency in frames, measured on the first pass of a view."""

    def __init__(self):
        self.latency: Optional[int] = None


def _any_colour(image, background, threshold) -> bool:
    d = cv2.subtract(image, background)
    for colour in SYMBOLS:
        ch = _CHANNEL[tuple(colour)]
        a, b = [c for c in range(3) if c != ch]
        excess = cv2.subtract(d[:, :, ch], cv2.max(d[:, :, a], d[:, :, b]))
        if np.count_nonzero(excess >= threshold) >= 3:
            return True
    return False


def measure_latency(cam, background, threshold) -> Optional[int]:
    """Frames until coloured light shows up (white flashes don't count). None if nothing appears."""
    start, frames = time.time(), 0
    while time.time() < start + DARK_TIMEOUT:
        frames += 1
        if _any_colour(cam.read(), background, threshold):
            return frames
    return None


def coded_pass(cam, leds, ids, timing, threshold, display, settings) -> dict:
    """Light all `ids` with their codes; return {led_id: Point2D} for the ones read back."""
    codes = codebook(len(ids))
    background = capture_background(cam, settings)
    frames = []
    for s in range(len(codes[0])):
        leds.show({i: SYMBOLS[code[s]] for i, code in zip(ids, codes)})
        if timing.latency is None:
            timing.latency = measure_latency(cam, background, threshold)
            if timing.latency is None:
                leds.off(ids)
                return {}
        else:
            for _ in range(timing.latency):
                cam.read()
        frames.append(
            (cam.read(), cam.read())
        )  # two frames per symbol: catches latency jitter
    leds.off(ids)
    for _ in range(timing.latency + 1):  # let it go dark before the next background
        cam.read()
    found = decode(frames, background, codes, ids, threshold, settings)
    if display:
        _show_found(frames[-1][1].copy(), found.values(), display)
    return found


def detect_coded(ids, cam, leds, view_id, threshold, display, output_queues, settings):
    timing = Timing()
    rng = random.Random(view_id)
    remaining, found, passes = list(ids), {}, 0

    for round_ in range(max(1, settings.coded_rounds)):
        rng.shuffle(remaining)
        group_count = min(len(remaining), 2**round_)
        new = 0
        for g in range(group_count):
            group = sorted(remaining[g::group_count])
            passes += 1
            for led, point in coded_pass(
                cam, leds, group, timing, threshold, display, settings
            ).items():
                found[led] = point
                new += 1
                for q in output_queues:
                    q.put(DetectionControlEnum.DETECT, LED2D(led, view_id, point))
            if timing.latency is None:
                break  # nothing lit up at all from this angle
        remaining = [i for i in remaining if i not in found]
        if not remaining or timing.latency is None or (round_ > 0 and new == 0):
            break

    for i in sorted(remaining):
        for q in output_queues:
            q.put(DetectionControlEnum.SKIP, i)
    logger.warning(
        f"view {view_id}: {len(found)}/{len(ids)} LEDs found by coded scan in {passes} passes"
        + (
            f", camera latency {timing.latency} frames"
            if timing.latency
            else ", nothing lit up"
        )
    )
    return sorted(
        (LED2D(led, view_id, point) for led, point in found.items()),
        key=lambda led: led.led_id,
    )
