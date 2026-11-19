"""Place xLights custom models in 3D from an "anchor" scan.

Each prop is scanned in detail on its own (its own coordinate frame) and turned into an xLights custom
model. One wide anchor scan then lights every n-th LED across all props (--robust --step n), putting
them in a single shared frame. For each prop, its model's node positions (exactly as xLights lays out a
custom model) are fitted onto its anchor LEDs with a similarity transform, which converts directly into
xLights' ModelX/Y/Z, RotateX/Y/Z and ScaleX/Y/Z.
"""

import math
import re
from dataclasses import dataclass

import numpy as np


def xmodel_local_coords(text: str) -> dict:
    """{node: xyz} in a custom model's local space, as xLights computes it (CustomModel.cpp):
    x = col - centreCol, y = centreRow - row, z = centreLayer - layer, centred on the occupied cells.
    """
    data = re.search(r'CustomModel="([^"]*)"', text)
    if not data:
        raise ValueError(
            "no CustomModel grid found (compressed custom models aren't supported)"
        )
    cells = {}
    for layer, layer_text in enumerate(data.group(1).split("|")):
        for row, row_text in enumerate(layer_text.split(";")):
            for col, value in enumerate(row_text.split(",")):
                if value.strip():
                    cells[int(value)] = (col, row, layer)
    cols, rows, layers = zip(*cells.values())
    cc = (min(cols) + max(cols)) / 2
    cr = (min(rows) + max(rows)) / 2
    cl = (min(layers) + max(layers)) / 2
    return {
        n: np.array([c - cc, cr - r, cl - layer], float)
        for n, (c, r, layer) in cells.items()
    }


def fit_similarity(src: np.ndarray, dst: np.ndarray):
    """dst ~ s * R @ src + t (Umeyama). Returns s, R, t, rms, mirrored."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a**2).sum(1).mean()
    t = md - s * R @ ms
    rms = math.sqrt((((s * (R @ src.T)).T + t - dst) ** 2).sum(1).mean())
    return s, R, t, rms, d < 0


def rotation(axis: str, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def xlights_rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """xLights applies Rz * Ry * Rx (BoxedScreenLocation.cpp)."""
    return rotation("z", rz) @ rotation("y", ry) @ rotation("x", rx)


def euler_for_xlights(R: np.ndarray):
    """(RotateX, RotateY, RotateZ) in degrees such that R = Rz @ Ry @ Rx."""
    b = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(b)) > 1e-6:
        a = math.atan2(R[2, 1], R[2, 2])
        c = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        a = math.atan2(-R[1, 2], R[1, 1])
        c = 0.0
    a, b, c = (math.degrees(v) for v in (a, b, c))
    # Every orientation has a second, equivalent angle set (a+180, 180-b, c+180). Prefer the one with
    # less X/Z tilt, so an upright prop reads as a plain turn (RotateY) instead of e.g. X -180, Z -169.
    alt = (_wrap(a + 180), _wrap(180 - b), _wrap(c + 180))
    if abs(alt[0]) + abs(alt[2]) < abs(a) + abs(c):
        return alt
    return a, b, c


def _wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


@dataclass
class Placement:
    model: str
    anchors: int
    fit_error: float  # rms residual as a fraction of the model's size
    mirrored: bool
    position: tuple  # ModelX, ModelY, ModelZ (cm)
    rotation: tuple  # RotateX, RotateY, RotateZ (degrees)
    scale: float  # ScaleX = ScaleY = ScaleZ (cm per model cell)


def place(
    anchor: dict,
    props: list,
    led_spacing_cm: float = 10.0,
    measure=None,
    min_anchors: int = 5,
):
    """Fit every prop to the anchor scan and express the result in xLights terms.

    anchor: {led_index: xyz} from the anchor scan's led_map_3d.csv
    props:  [{"model": name, "local": {node: xyz}, "first": led_index_of_node_1}, ...]
    measure: optional (model, node_a, node_b, cm) giving the real distance between two nodes.
             Otherwise LED wire spacing sets the scale: neighbouring LEDs can't be further apart than
             their wire, and on straight runs they come close, so the 90th percentile gap ~ the spacing.

    World frame: Y = the props' average up (+Y as built), Z = their average front, origin on the ground
    (lowest LED) at the layout's centre. Returns (placements, cm_per_anchor_unit, skipped_models).
    """
    fits, skipped = [], []
    for prop in props:
        local = prop["local"]
        nodes = [n for n in sorted(local) if prop["first"] + n - 1 in anchor]
        if len(nodes) < min_anchors:
            skipped.append(prop["model"])
            continue
        src = np.array([local[n] for n in nodes])
        dst = np.array([anchor[prop["first"] + n - 1] for n in nodes])
        s, R, t, rms, mirrored = fit_similarity(src, dst)
        steps = [
            np.linalg.norm(local[n + 1] - local[n]) for n in local if n + 1 in local
        ]
        fits.append(
            dict(
                prop,
                s=s,
                R=R,
                t=t,
                n=len(nodes),
                mirrored=mirrored,
                fit_error=rms / max(np.ptp(src * s, 0)),
                cells_per_spacing=float(np.percentile(steps, 90)),
            )
        )
    if not fits:
        return [], None, skipped

    weights = np.array([f["n"] for f in fits], float)
    up = sum(w * f["R"][:, 1] for w, f in zip(weights, fits))
    up /= np.linalg.norm(up)
    front = sum(w * f["R"][:, 2] for w, f in zip(weights, fits))
    front -= up * (front @ up)
    front /= np.linalg.norm(front)
    W = np.array(
        [np.cross(up, front), up, front]
    )  # rows: world X, Y, Z in anchor coordinates

    if measure:
        name, node_a, node_b, cm = measure
        f = next(f for f in fits if f["model"] == name)
        dist = f["s"] * np.linalg.norm(
            f["local"][int(node_a)] - f["local"][int(node_b)]
        )
        cm_per_unit = float(cm) / dist
    else:
        cm_per_unit = float(
            np.median(
                [led_spacing_cm / (f["s"] * f["cells_per_spacing"]) for f in fits]
            )
        )

    def world_points(f):
        pts = np.array(list(f["local"].values())).T
        return (W @ ((f["s"] * (f["R"] @ pts)).T + f["t"]).T).T * cm_per_unit

    everything = np.vstack([world_points(f) for f in fits])
    offset = np.array(
        [
            -(everything[:, 0].min() + everything[:, 0].max()) / 2,
            -everything[:, 1].min(),
            -(everything[:, 2].min() + everything[:, 2].max()) / 2,
        ]
    )

    placements = [
        Placement(
            model=f["model"],
            anchors=f["n"],
            fit_error=f["fit_error"],
            mirrored=f["mirrored"],
            position=tuple(W @ f["t"] * cm_per_unit + offset),
            rotation=euler_for_xlights(W @ f["R"]),
            scale=f["s"] * cm_per_unit,
        )
        for f in fits
    ]
    return placements, cm_per_unit, skipped
