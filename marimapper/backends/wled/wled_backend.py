import requests
from ipaddress import ip_address
from functools import partial
import argparse


def wled_backend_factory(argparse_args: argparse.Namespace):
    return partial(Backend, argparse_args.server)


def wled_backend_set_args(parser):
    parser.add_argument("--server", default="4.3.2.1")


class Backend:

    def __init__(self, wled_base_url):

        try:
            ip_address(wled_base_url)
        except ValueError:
            raise RuntimeError(
                f"WLED backend failed to start due as {wled_base_url} is not a valid IP address"
            )

        self.state_endpoint = f"http://{wled_base_url}/json/state"
        self.info_endpoint = f"http://{wled_base_url}/json/info"
        # one kept-alive connection: a scan sends thousands of small requests
        self.session = requests.Session()
        self.led_count = None
        self.reset_wled()

    def _post(self, payload, what: str):
        response = self.session.post(self.state_endpoint, json=payload, timeout=10)
        if response.status_code != 200:
            raise ConnectionError(
                f"WLED Backend failed to {what}. Status code: {response.status_code}"
            )

    def get_led_count(self):

        if self.led_count is None:
            # Send the HTTP GET request to WLED info API
            response = self.session.get(self.info_endpoint, timeout=10)

            # Get the LED Count straight from the WLED Device :D
            if response.status_code != 200:
                raise ConnectionError(
                    f"Failed to retrieve LED count. Status code: {response.status_code}"
                )

            self.led_count = response.json()["leds"]["count"]
        return self.led_count

    def reset_wled(self):

        # One segment over the whole strip, every LED black, in a single request
        # (one request per LED takes minutes on strips with thousands of LEDs)
        count = self.get_led_count()
        payload = {
            "on": True,
            "seg": [
                {
                    "start": 0,
                    "stop": count,
                    "sel": True,
                    "fx": 0,
                    "i": [0, count, "000000"],
                },
                {"stop": 0},
            ],
        }
        self._post(payload, "reset the strip")

    def set_led(self, led_index: int, on: bool):
        self._post(
            {"seg": {"i": [led_index, "FFFFFF" if on else "000000"]}}, "set LED state"
        )

    def set_leds(self, buffer: list):
        """Set every LED at once from a list of [r, g, b], sent as runs of equal colour."""
        hexes = [
            "%02X%02X%02X" % tuple(int(max(0, min(255, v))) for v in colour[:3])
            for colour in buffer
        ]
        segment = []
        start = 0
        for i in range(1, len(hexes) + 1):
            if i == len(hexes) or hexes[i] != hexes[start]:
                if i - start > 1:
                    segment += [start, i, hexes[start]]
                else:
                    segment += [start, hexes[start]]
                start = i
        if segment:
            self._post({"seg": {"i": segment}}, "set LED colours")
