import math
import re

import numpy as np

from marimapper.placement import fit_similarity, xmodel_local_coords
from marimapper.xmodel import build_2d, build_3d, fill_missing


def _helix(n=100):
    return {
        i
        + 1: (
            (5 - 0.04 * i) * math.cos(i / 15 * 2 * math.pi),
            12 - 0.12 * i,  # y up: node 1 at the top
            (5 - 0.04 * i) * math.sin(i / 15 * 2 * math.pi),
        )
        for i in range(n)
    }


def test_3d_model_is_upright_and_not_mirrored():
    points = _helix()
    text, info = build_3d(points, "Spiral", cells_per_unit=4)
    local = xmodel_local_coords(text)  # exactly how xLights positions the nodes
    assert len(local) == len(points)

    src = np.array([local[n] for n in sorted(points)])
    dst = np.array([points[n] for n in sorted(points)])
    s, R, t, rms, mirrored = fit_similarity(src, dst)
    assert not mirrored  # the old converter wrote layers back-to-front: a mirror image
    assert np.allclose(R, np.eye(3), atol=0.05)  # same orientation: y up, z front
    assert rms / np.ptp(dst, 0).max() < 0.03


def test_2d_model_keeps_image_orientation():
    view = {
        1: (0.10, 0.10),
        2: (0.12, 0.10),
        3: (0.14, 0.10),
        4: (0.14, 0.30),
    }  # node 4 lower in the image
    text, info = build_2d(view, "Flat", cells_per_step=4)
    local = xmodel_local_coords(text)
    assert info["depth"] == 1
    assert local[4][1] < local[1][1]  # lower in the photo -> lower in xLights
    assert local[3][0] > local[1][0]


def test_fill_missing_interpolates_between_neighbours():
    filled, which = fill_missing({1: (0, 0, 0), 4: (3, 0, 0)}, 5)
    assert which == [2, 3, 5]
    assert np.allclose(filled[2], (1, 0, 0)) and np.allclose(filled[3], (2, 0, 0))
    assert np.allclose(filled[5], (3, 0, 0))


def test_strings_and_submodels_are_written():
    text, _ = build_3d(
        _helix(30),
        "Prop",
        strings=[(1, "!Ctrl:1"), (21, "!Ctrl:901")],
        submodels=[("Top", "1-10")],
    )
    assert (
        'CustomStrings="2"' in text
        and 'NodeStart2="21"' in text
        and 'String2="!Ctrl:901"' in text
    )
    assert '<subModel name="Top"' in text and 'line0="1-10"' in text
    nodes = sorted(
        int(v)
        for v in re.findall(r"\d+", re.search(r'CustomModel="([^"]*)"', text).group(1))
    )
    assert nodes == list(range(1, 31))
