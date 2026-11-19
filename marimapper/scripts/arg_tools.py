import logging
import importlib.metadata
from marimapper.database_populator import camera_models, camera_model_radial
from marimapper.backends.backend_utils import backend_arg_setters
from marimapper.camera import parse_device

from pathlib import Path
import os
import argparse


def add_camera_args(parser):

    camera_options = parser.add_argument_group("camera options")

    camera_options.add_argument(
        "--device",
        type=parse_device,
        help="Camera device index, set to 1 if using a laptop with a USB webcam. "
        "A network video stream URL also works, e.g. http://192.168.1.20:8080/video or rtsp://...",
        default=0,
    )

    camera_options.add_argument(
        "--exposure",
        type=int,
        help="Camera exposure, the lower the value, the darker the image",
        default=-10,
    )
    camera_options.add_argument(
        "--threshold",
        type=int,
        choices=range(0, 255),
        metavar="[0-255]",
        help="LED detection threshold, increasing this number will reduce false positive detections",
        default=128,
    )


def add_common_args(parser) -> None:

    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Print version and exit",
    )

    parser.add_argument("-v", "--verbose", action="store_true")


def add_scanner_args(parser) -> None:

    scanner_options = parser.add_argument_group("scanner options")

    scanner_options.add_argument(
        "--dir",
        type=Path,
        default=os.getcwd(),
        help="the location for your maps, defaults to the current working directory",
    )

    scanner_options.add_argument(
        "--start", type=int, help="Index of the first led you want to scan", default=0
    )

    scanner_options.add_argument(
        "--end",
        type=int,
        help="Index of the last led you want to scan up to the backend's limit",
        default=10000,
    )

    scanner_options.add_argument(
        "--interpolation_max_fill",
        type=int,
        default=5,
        help="The maximum number of consecutive LEDs that can be interpolated based on adjacent LEDs. Set to -1 for no limit",
    )

    scanner_options.add_argument(
        "--interpolation_max_error",
        type=float,
        default=0.2,
        help="The maximum variance in led spacing whilst interpolating LEDs, set to -1 for no limit",
    )

    scanner_options.add_argument(
        "--disable_movement_check",
        action="store_false",
        help="Disables checking of movement when a scan completes",
    )

    scanner_options.add_argument(
        "--camera_model",
        type=str,
        choices=[model.__name__ for model in camera_models],
        default=camera_model_radial.__name__,
        help="Sets the camera model used for reconstruction, choose camera_model_opencv_full for higher accuracy",
    )


def add_robust_args(parser) -> None:

    robust_options = parser.add_argument_group(
        "robust detection options (outdoor / uncontrolled scenes)"
    )

    robust_options.add_argument(
        "--robust",
        action="store_true",
        help="Ignore stars, streetlights, reflections, cars and flashlights: background subtraction, "
        "coloured confirmation flashes and batched scanning. --threshold then means how much the "
        "flashed colour must stand out (try 25-60). Colour and batching need a backend with set_leds",
    )

    robust_options.add_argument(
        "--batch",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="With --robust: LEDs lit per flash, each in its own colour. Falls back to one at a time "
        "automatically when LEDs overlap from the current angle",
    )

    robust_options.add_argument(
        "--confirm",
        type=int,
        default=2,
        help="With --robust: flashes per LED that must land in the same place",
    )

    robust_options.add_argument(
        "--step",
        type=int,
        default=1,
        help="With --robust: only scan every n-th LED, e.g. an anchor scan across several props "
        "for placing them relative to each other",
    )

    robust_options.add_argument(
        "--no_preview",
        action="store_true",
        help="With --robust: don't light the scan range while aiming the camera",
    )

    robust_options.add_argument(
        "--status_brightness",
        type=float,
        default=0.35,
        help="With --robust: brightness (0-1) of the per-LED coverage colours shown between views",
    )


def get_detection_settings(args: argparse.Namespace):
    from marimapper.robust_detector import DetectionSettings

    return DetectionSettings(
        robust=args.robust,
        batch=args.batch,
        confirm=max(1, args.confirm),
        step=max(1, args.step),
        preview=not args.no_preview,
        status_brightness=min(1.0, max(0.0, args.status_brightness)),
    )


def parse_common_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.version:
        version = importlib.metadata.version("marimapper")
        print(f"Marimapper version: {version}")
        quit()

    if not hasattr(args, "backend"):
        args.backend = "dummy"


def add_all_backend_parsers(parser, required=False) -> list[argparse.ArgumentParser]:
    backend_subparsers = []

    backend_subparser = parser.add_subparsers(
        help="Backend to use to control the LEDs", required=required
    )

    for backend_name, backend_arg_setter in backend_arg_setters.items():
        backend_parser = backend_subparser.add_parser(
            backend_name,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            usage=argparse.SUPPRESS,
        )
        backend_parser.set_defaults(backend=backend_name)
        backend_parser_group = backend_parser.add_argument_group(
            f"{backend_name} options"
        )

        backend_arg_setter(backend_parser_group)

        backend_subparsers.append(backend_parser)

    return backend_subparsers
