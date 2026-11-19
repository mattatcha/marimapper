"""Build xLights custom models (.xmodel) from marimapper maps.

xLights lays a custom model out on a grid of columns, rows and depth layers and computes each node's
local position as x = col - centreCol, y = centreRow - row, z = centreLayer - layer (CustomModel.cpp).
So row 0 is the top and layer 0 is the front: to keep the model the right way up AND not a front-to-back
mirror image of the real prop, the scan's highest y goes in row 0 and its highest z in layer 0.
"""

import itertools
import math
from typing import Optional

import numpy as np


def fill_missing(points: dict, count: int) -> tuple[dict, list]:
    """Nodes 1..count; nodes missing from `points` are interpolated between their nearest neighbours."""
    known = sorted(points)
    if not known:
        raise ValueError("no points to build a model from")
    out, filled = {}, []
    for n in range(1, count + 1):
        if n in points:
            out[n] = np.asarray(points[n], float)
            continue
        lo = max((k for k in known if k < n), default=None)
        hi = min((k for k in known if k > n), default=None)
        if lo is not None and hi is not None:
            t = (n - lo) / (hi - lo)
            out[n] = (
                np.asarray(points[lo], float)
                + (np.asarray(points[hi], float) - np.asarray(points[lo], float)) * t
            )
        else:
            out[n] = np.asarray(points[lo if lo is not None else hi], float)
        filled.append(n)
    return out, filled


def _place_on_grid(cells: dict, size: tuple) -> tuple[dict, int]:
    """{node: cell} -> {cell: node}, moving nodes that land on an occupied cell to the nearest free one."""
    grid, moved = {}, 0
    dims = len(size)
    for node in sorted(cells):
        c = cells[node]
        if c in grid:
            moved += 1
            r = 1
            while True:
                candidates = [
                    tuple(c[i] + d[i] for i in range(dims))
                    for d in itertools.product(range(-r, r + 1), repeat=dims)
                ]
                candidates = [
                    k
                    for k in candidates
                    if k not in grid and all(0 <= k[i] < size[i] for i in range(dims))
                ]
                if candidates:
                    c = min(
                        candidates,
                        key=lambda k: sum((k[i] - c[i]) ** 2 for i in range(dims)),
                    )
                    break
                r += 1
        grid[c] = node
    return grid, moved


def _xml(
    name, width, height, depth, data, start_channel, string_type, submodels, strings
):
    attrs = [
        f'name="{name}"',
        f'CustomWidth="{width}"',
        f'CustomHeight="{height}"',
        f'Depth="{depth}"',
        f'StringType="{string_type}"',
        'Transparency="0"',
        'PixelSize="2"',
        'ModelBrightness="0"',
        'Antialias="1"',
        'StrandNames=""',
        'NodeNames=""',
    ]
    if strings and len(strings) > 1:
        attrs.append(f'CustomStrings="{len(strings)}"')
        attrs += [f'NodeStart{i + 1}="{node}"' for i, (node, _) in enumerate(strings)]
        attrs += ['Advanced="1"', f'StartChannel="{strings[0][1]}"']
        attrs += [
            f'String{i + 1}="{channel}"' for i, (_, channel) in enumerate(strings)
        ]
    elif start_channel:
        attrs.append(f'StartChannel="{start_channel}"')
    attrs.append(f'CustomModel="{data}"')
    subs = "".join(
        f'  <subModel name="{n}" layout="horizontal" type="ranges" bufferstyle="Default" line0="{r}"/>\n'
        for n, r in (submodels or [])
    )
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<custommodel \n{" ".join(attrs)} >\n{subs}</custommodel>\n'


def build_3d(
    points: dict,
    name: str,
    cells_per_unit: float = 4.0,
    start_channel: Optional[str] = None,
    string_type: str = "RGB Nodes",
    submodels: Optional[list] = None,
    strings: Optional[list] = None,
) -> tuple[str, dict]:
    """points: {node (1-based): (x, y, z)} with y up. strings: [(first_node, start_channel), ...]."""
    pts = {n: np.asarray(v, float) for n, v in points.items()}
    mins = np.min(list(pts.values()), axis=0)
    cells = {
        n: tuple(int(round(c)) for c in (v - mins) * cells_per_unit)
        for n, v in pts.items()
    }
    size = tuple(max(c[i] for c in cells.values()) + 1 for i in range(3))
    grid, moved = _place_on_grid(cells, size)
    width, height, depth = size
    # row 0 = highest y (top), layer 0 = highest z (front): see module docstring
    data = "|".join(
        ";".join(
            ",".join(str(grid.get((x, y, z), "")) for x in range(width))
            for y in reversed(range(height))
        )
        for z in reversed(range(depth))
    )
    text = _xml(
        name, width, height, depth, data, start_channel, string_type, submodels, strings
    )
    return text, {"width": width, "height": height, "depth": depth, "moved": moved}


def build_2d(
    points: dict,
    name: str,
    cells_per_step: float = 8.0,
    start_channel: Optional[str] = None,
    string_type: str = "RGB Nodes",
    submodels: Optional[list] = None,
) -> tuple[str, dict]:
    """points: {node: (u, v)} from one marimapper 2D view (v grows downwards). Flat props only."""
    pts = {n: np.asarray(v, float) for n, v in points.items()}
    steps = [math.dist(pts[n], pts[n + 1]) for n in pts if n + 1 in pts]
    scale = cells_per_step / float(np.median(steps)) if steps else 1.0
    mins = np.min(list(pts.values()), axis=0)
    cells = {
        n: tuple(int(round(c)) for c in (v - mins) * scale) for n, v in pts.items()
    }
    size = tuple(max(c[i] for c in cells.values()) + 1 for i in range(2))
    grid, moved = _place_on_grid(cells, size)
    width, height = size
    data = ";".join(
        ",".join(str(grid.get((x, y), "")) for x in range(width)) for y in range(height)
    )
    text = _xml(
        name, width, height, 1, data, start_channel, string_type, submodels, None
    )
    return text, {"width": width, "height": height, "depth": 1, "moved": moved}
