import argparse
import csv
import json
import os
import time
import urllib.request

import numpy as np

from marimapper.placement import place, xmodel_local_coords


def xlights_command(url: str, cmd: str, params: dict):
    """Call an xLights automation command through xLights' built-in MCP server."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "xlights_command",
                "arguments": {"cmd": cmd, "params": params},
            },
        }
    ).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    result = json.load(urllib.request.urlopen(request, timeout=30)).get("result", {})
    return not result.get("isError"), "".join(
        c.get("text", "") for c in result.get("content", [])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Place xLights custom models in 3D from an anchor scan "
        "(marimapper ... --robust --step N across all props)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("anchor_csv", help="led_map_3d.csv from the anchor scan")
    parser.add_argument(
        "props_json",
        help='[{"model": "Tree", "xmodel": "tree/Tree.xmodel", "first": 0}, ...]; '
        '"first" is the LED index of node 1; xmodel paths are relative to this file',
    )
    parser.add_argument(
        "--led_spacing_cm", type=float, default=10.0, help="wire length between LEDs"
    )
    parser.add_argument(
        "--measure",
        help='exact scale from one real distance: "Model:node_a:node_b:cm"',
    )
    parser.add_argument("--min_anchors", type=int, default=5)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="set the placements in xLights and save the layout",
    )
    parser.add_argument(
        "--xlights_url",
        default="http://127.0.0.1:49913/mcp",
        help="xLights MCP endpoint (Preferences > Output > xFade/xSchedule = Port A)",
    )
    args = parser.parse_args()

    anchor = {
        int(r["index"]): np.array([float(r[k]) for k in "xyz"])
        for r in csv.DictReader(open(args.anchor_csv))
    }
    base = os.path.dirname(os.path.abspath(args.props_json))
    props = [
        {
            "model": p["model"],
            "first": int(p["first"]),
            "local": xmodel_local_coords(open(os.path.join(base, p["xmodel"])).read()),
        }
        for p in json.load(open(args.props_json))
    ]
    measure = tuple(args.measure.split(":")) if args.measure else None

    placements, cm_per_unit, skipped = place(
        anchor, props, args.led_spacing_cm, measure, args.min_anchors
    )
    for name in skipped:
        print(f"{name}: fewer than {args.min_anchors} anchor LEDs found, skipped")
    if not placements:
        raise SystemExit("no props could be placed")

    how = (
        f"measured {args.measure}"
        if measure
        else f"{args.led_spacing_cm} cm LED wire spacing (estimate)"
    )
    print(f"scale: {cm_per_unit:.2f} cm per scan unit ({how}); positions in cm\n")
    print(
        f"{'model':16s} {'anchors':>7s} {'fit err':>7s} {'X':>8s} {'Y':>8s} {'Z':>8s} "
        f"{'RotX':>7s} {'RotY':>7s} {'RotZ':>7s} {'Scale':>7s}"
    )
    for p in placements:
        flags = "  MIRRORED: rebuild the model flipped" if p.mirrored else ""
        flags += "  POOR FIT" if p.fit_error > 0.08 else ""
        print(
            f"{p.model:16s} {p.anchors:7d} {100 * p.fit_error:6.1f}% "
            f"{p.position[0]:8.1f} {p.position[1]:8.1f} {p.position[2]:8.1f} "
            f"{p.rotation[0]:7.1f} {p.rotation[1]:7.1f} {p.rotation[2]:7.1f} {p.scale:7.3f}{flags}"
        )

    if args.apply:
        failed = [p for p in placements if not apply_placement(args.xlights_url, p)]
        ok, message = xlights_command(args.xlights_url, "saveLayout", {})
        print("\nlayout saved" if ok else f"\nsaveLayout failed: {message}")
        if failed:
            # setModelProperty edits whichever model xLights' property grid is showing; for position,
            # rotation and scale that isn't always the model asked for. Never leave it silently wrong.
            print(
                "\nxLights did not apply these; enter them in Layout by hand "
                "(X, Y, Z, RotateX, RotateY, RotateZ, ScaleX=ScaleY=ScaleZ):"
            )
            for p in failed:
                print(
                    f"  {p.model}: {p.position[0]:.1f}, {p.position[1]:.1f}, {p.position[2]:.1f}, "
                    f"{p.rotation[0]:.1f}, {p.rotation[1]:.1f}, {p.rotation[2]:.1f}, {p.scale:.3f}"
                )


READ_KEYS = {
    "ModelX": "WorldPosX",
    "ModelY": "WorldPosY",
    "ModelZ": "WorldPosZ",
    "RotateX": "RotateX",
    "RotateY": "RotateY",
    "RotateZ": "RotateZ",
    "ScaleX": "ScaleX",
    "ScaleY": "ScaleY",
    "ScaleZ": "ScaleZ",
}


def apply_placement(url: str, p, attempts: int = 3, pause: float = 0.4) -> bool:
    """Set one model's placement and read it back. xLights refreshes its property grid asynchronously,
    so changes are paced and verified rather than trusted."""
    want = {
        "ModelX": p.position[0],
        "ModelY": p.position[1],
        "ModelZ": p.position[2],
        "RotateX": p.rotation[0],
        "RotateY": p.rotation[1],
        "RotateZ": p.rotation[2],
        "ScaleX": p.scale,
        "ScaleY": p.scale,
        "ScaleZ": p.scale,
    }
    for _ in range(attempts):
        for key, value in want.items():
            xlights_command(
                url,
                "setModelProperty",
                {"model": p.model, "key": key, "data": f"{value:.4f}"},
            )
            time.sleep(pause)
        time.sleep(1.0)
        ok, message = xlights_command(url, "getModel", {"model": p.model})
        got = json.loads(message).get("model", {}) if ok else {}
        wrong = [
            k
            for k, v in want.items()
            if got.get(READ_KEYS[k]) is None
            or abs(float(got[READ_KEYS[k]]) - v) > max(0.01, 1e-3 * abs(v))
        ]
        if not wrong:
            print(f"{p.model}: set and verified")
            return True
    print(f"{p.model}: not applied by xLights")
    return False


if __name__ == "__main__":
    main()
