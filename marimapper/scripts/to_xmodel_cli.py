import argparse
import csv

from marimapper.xmodel import build_2d, build_3d, fill_missing


def main():
    parser = argparse.ArgumentParser(
        description="Turn a marimapper map into an xLights custom model (.xmodel). Use led_map_3d.csv for a "
        "3D model, or a single led_map_2d_*.csv view with --flat for a flat prop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "map_csv", help="led_map_3d.csv, or a led_map_2d_*.csv view with --flat"
    )
    parser.add_argument("output", help="the .xmodel file to write")
    parser.add_argument("--name", required=True, help="model name in xLights")
    parser.add_argument(
        "--first", type=int, default=0, help="LED index of the model's node 1"
    )
    parser.add_argument(
        "--count",
        type=int,
        help="number of nodes (default: up to the highest LED found)",
    )
    parser.add_argument(
        "--flat", action="store_true", help="2D model from a single 2D view"
    )
    parser.add_argument(
        "--scale", type=float, default=4.0, help="3D: grid cells per LED spacing"
    )
    parser.add_argument(
        "--cells_per_step",
        type=float,
        default=8.0,
        help="--flat: grid cells per LED spacing",
    )
    parser.add_argument("--start_channel", help='e.g. "!Controller:1"')
    parser.add_argument("--string_type", default="RGB Nodes")
    parser.add_argument(
        "--submodel",
        action="append",
        default=[],
        help='"Name=ranges" (1-based nodes), e.g. "Rails=1-25"',
    )
    parser.add_argument(
        "--string",
        action="append",
        default=[],
        help='several strings on different start channels: "first_node:start_channel", '
        'e.g. --string "1:!Ctrl:1" --string "301:!Ctrl:1717"',
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.map_csv)))
    keys = ("u", "v") if args.flat else ("x", "y", "z")
    by_index = {int(r["index"]): tuple(float(r[k]) for k in keys) for r in rows}
    count = args.count or (max(by_index) - args.first + 1)
    points = {
        i - args.first + 1: p
        for i, p in by_index.items()
        if args.first <= i < args.first + count
    }
    points, filled = fill_missing(points, count)

    submodels = [tuple(s.split("=", 1)) for s in args.submodel]
    strings = [(int(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in args.string]
    if args.flat:
        text, info = build_2d(
            points,
            args.name,
            args.cells_per_step,
            args.start_channel,
            args.string_type,
            submodels,
        )
    else:
        text, info = build_3d(
            points,
            args.name,
            args.scale,
            args.start_channel,
            args.string_type,
            submodels,
            strings,
        )
    with open(args.output, "w") as f:
        f.write(text)
    print(
        f"{args.name}: {count} nodes, grid {info['width']}x{info['height']}x{info['depth']}, "
        f"{len(filled)} interpolated {filled if filled else ''}, {info['moved']} moved off shared cells"
    )


if __name__ == "__main__":
    main()
