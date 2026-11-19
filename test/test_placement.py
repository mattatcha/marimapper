import math
import re

import numpy as np
import pytest

from marimapper.placement import (
    euler_for_xlights,
    place,
    xlights_rotation,
    xmodel_local_coords,
)


def _xmodel(points: dict) -> str:
    """Custom model text from {node: (col, row, layer)}."""
    W = max(c for c, _, _ in points.values()) + 1
    H = max(r for _, r, _ in points.values()) + 1
    D = max(layer for _, _, layer in points.values()) + 1
    grid = {v: n for n, v in points.items()}
    layers = "|".join(
        ";".join(
            ",".join(str(grid.get((c, r, layer), "")) for c in range(W))
            for r in range(H)
        )
        for layer in range(D)
    )
    return f'<custommodel name="x" CustomWidth="{W}" CustomHeight="{H}" Depth="{D}" CustomModel="{layers}" />'


def _helix(n=120):  # a spiral "tree": 3D
    return {
        i
        + 1: (
            int(round(20 + (18 - i * 0.12) * math.cos(i / 20 * 2 * math.pi))),
            int(round(60 - i * 0.45)),
            int(round(20 + (18 - i * 0.12) * math.sin(i / 20 * 2 * math.pi))),
        )
        for i in range(n)
    }


def _arch(n=80):  # flat: 1 layer
    return {
        i
        + 1: (
            int(round(40 + 35 * math.cos(math.pi * i / (n - 1)))),
            int(round(40 - 35 * math.sin(math.pi * i / (n - 1)))),
            0,
        )
        for i in range(n)
    }


def test_local_coords_are_centred_like_xlights():
    local = xmodel_local_coords(_xmodel({1: (0, 0, 0), 2: (4, 2, 1)}))
    assert np.allclose(local[1], [-2, 1, 0.5]) and np.allclose(local[2], [2, -1, -0.5])


@pytest.mark.parametrize(
    "angles", [(10, 20, 30), (-45, 80, 170), (0, -30, 0), (179, 5, -120)]
)
def test_euler_round_trip(angles):
    R = xlights_rotation(*angles)
    assert np.allclose(xlights_rotation(*euler_for_xlights(R)), R)


def test_recovers_relative_layout_from_anchor_scan():
    rng = np.random.default_rng(3)
    models = {"Tree": (_helix(), 0), "Arch": (_arch(), 500)}
    truth = {
        "Tree": dict(pos=(0, 90, 0), rot=(0, 0, 0), scale=4.0),
        "Arch": dict(pos=(300, 60, -150), rot=(0, 35, 0), scale=3.0),
    }
    frame, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # arbitrary scan frame
    frame *= np.sign(np.linalg.det(frame))

    anchor, props, world = {}, [], {}
    for name, (points, first) in models.items():
        local = xmodel_local_coords(_xmodel(points))
        t = truth[name]
        for n, v in local.items():
            w = xlights_rotation(*t["rot"]) @ (t["scale"] * v) + np.array(t["pos"])
            world[(name, n)] = w
            if (n - 1) % 6 == 0:  # anchor scan: every 6th LED
                anchor[first + n - 1] = (
                    0.05 * frame @ w + [2, -1, 7] + rng.normal(scale=0.03, size=3)
                )
        props.append({"model": name, "local": local, "first": first})

    real = np.linalg.norm(world[("Tree", 1)] - world[("Tree", 100)])
    placements, _, skipped = place(anchor, props, measure=("Tree", 1, 100, real))
    p = {pl.model: pl for pl in placements}

    assert not skipped and not any(pl.mirrored for pl in placements)
    assert all(pl.fit_error < 0.02 for pl in placements)
    assert p["Tree"].scale == pytest.approx(4.0, rel=0.02)
    assert p["Arch"].scale == pytest.approx(3.0, rel=0.02)
    true_offset = np.subtract(truth["Arch"]["pos"], truth["Tree"]["pos"])
    got_offset = np.subtract(p["Arch"].position, p["Tree"].position)
    assert np.linalg.norm(got_offset) == pytest.approx(
        np.linalg.norm(true_offset), rel=0.02
    )
    assert got_offset[1] == pytest.approx(
        true_offset[1], abs=5
    )  # height difference survives
    relative = xlights_rotation(*p["Tree"].rotation).T @ xlights_rotation(
        *p["Arch"].rotation
    )
    assert np.allclose(relative, xlights_rotation(0, 35, 0), atol=0.02)
    lowest = min(pl.position[1] for pl in placements)
    assert lowest >= 0  # layout sits on the ground


def test_props_without_enough_anchors_are_skipped():
    local = xmodel_local_coords(_xmodel(_arch(20)))
    placements, scale, skipped = place(
        {0: np.zeros(3)}, [{"model": "Arch", "local": local, "first": 0}]
    )
    assert placements == [] and scale is None and skipped == ["Arch"]


def test_euler_prefers_upright_form():
    # X -180 / Z -169 describes the same orientation as a plain ~108 degree turn with small tilts
    R = xlights_rotation(-179.6, -71.9, -168.8)
    rx, ry, rz = euler_for_xlights(R)
    assert abs(rx) < 1 and abs(rz) < 12
    assert np.allclose(xlights_rotation(rx, ry, rz), R)


def test_set_layout_placements_only_touches_placement_attributes():
    from marimapper.placement import Placement, set_layout_placements

    tree_tag = (
        '<model name="Tree (big)" DisplayAs="Custom" WorldPosX="1.0" WorldPosY="2.0" '
        'WorldPosZ="3.0" ScaleX="1" ScaleY="1" ScaleZ="1" StartChannel="!Ctrl:1" />'
    )
    arch_tag = '<model name="Arch" DisplayAs="Arch" WorldPosX="9.0" />'
    layout = f"<xrgb>\n<models>\n{tree_tag}\n{arch_tag}\n</models>\n</xrgb>\n"
    p = Placement(
        "Tree (big)", 10, 0.01, False, (10.5, 20.25, -3.0), (1.0, 90.0, -2.0), 2.5
    )
    q = Placement("Missing", 10, 0.01, False, (0, 0, 0), (0, 0, 0), 1.0)

    new, written, missing = set_layout_placements(layout, [p, q])

    assert written == ["Tree (big)"] and missing == ["Missing"]
    tree = re.search(r'<model name="Tree \(big\)"[^>]*>', new).group(0)
    expected = {
        "WorldPosX": "10.500000",
        "WorldPosY": "20.250000",
        "WorldPosZ": "-3.000000",
        "RotateX": "1.000000",  # missing before: added
        "RotateY": "90.000000",
        "RotateZ": "-2.000000",
        "ScaleX": "2.500000",
        "ScaleY": "2.500000",
        "ScaleZ": "2.500000",
    }
    for key, value in expected.items():
        assert f'{key}="{value}"' in tree
    assert 'StartChannel="!Ctrl:1"' in tree and 'DisplayAs="Custom"' in tree
    assert new.replace(tree, "") == layout.replace(tree_tag, "")  # nothing else changed
