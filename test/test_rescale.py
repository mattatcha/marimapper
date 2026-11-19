from marimapper.led import rescale, LED3D


def test_rescale_basic():

    led_0 = LED3D(0)
    led_1 = LED3D(1)
    led_2 = LED3D(2)

    led_0.point.position[0] = 0
    led_1.point.position[0] = 2
    led_2.point.position[0] = 4

    scale = rescale([led_0, led_1, led_2])

    assert scale == 0.5

    assert led_0.point.position[0] == 0
    assert led_1.point.position[0] == 1
    assert led_2.point.position[0] == 2


def test_rescale_sparse_scan():
    # every 8th LED (an anchor scan): no neighbours with gap 1, used to produce NaN everywhere
    leds = [LED3D(i) for i in (0, 8, 16, 24)]
    for k, led in enumerate(leds):
        led.point.position[0] = k * 16  # 2 units per LED index

    scale = rescale(leds)

    assert scale == 0.5
    assert [led.point.position[0] for led in leds] == [0, 8, 16, 24]


def test_rescale_single_led_is_left_alone():
    led = LED3D(5)
    led.point.position[0] = 3

    assert rescale([led]) == 1.0
    assert led.point.position[0] == 3
