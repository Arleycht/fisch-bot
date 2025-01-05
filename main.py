import cv2 as cv2
import datetime
import mss
import keyboard
import numpy as np
import pydirectinput
import time
import pywinctl

import fisch

# Hard coded areas to screenshot

monitor_index = 0
reel_prompt_rect = (870, 790, 176, 16)
reel_rect = (572, 876, 776, 31)
debug_mode = False

# Prepare template images

reel_prompt = cv2.imread("reel_prompt.png", cv2.IMREAD_UNCHANGED)
reel_prompt = cv2.cvtColor(reel_prompt, cv2.COLOR_BGR2GRAY)

button_background = cv2.imread("base_button.png", cv2.IMREAD_UNCHANGED)
button_text = cv2.imread("base_text.png", cv2.IMREAD_UNCHANGED)
button_template = fisch.paste(button_background, button_text, background_alpha=0.6)
button_template = cv2.cvtColor(np.array(button_template), cv2.COLOR_BGR2GRAY)
button_scales = [
    121, # Maximized window sizes
    203,
]

fish_color = np.array((220, 26, 35))
fish_color = np.multiply(fish_color, (0.5, 2.55, 2.55))
fish_color_epsilon = (1, 3, 3)
lower_fish_color = fish_color - fish_color_epsilon
upper_fish_color = fish_color + fish_color_epsilon

# Runtime variables

auto_cast = False
auto_shake = True
auto_reel = True

offset_x = 0
offset_y = 0


def process_sobel(image):
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.Sobel(
        image,
        cv2.CV_16S,
        1,
        1,
        ksize=3,
        scale=1,
        delta=0,
        borderType=cv2.BORDER_DEFAULT,
    )
    return cv2.convertScaleAbs(image)


def get_shake_button_pos(image, threshold=0.45):
    a = process_sobel(image)

    for scale in button_scales:
        b = cv2.resize(button_template, (scale, scale))
        b = process_sobel(b)

        matched = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED, None, None)
        _, max_value, _, max_location = cv2.minMaxLoc(matched)

        if max_value > threshold:
            top_left = np.array(max_location)
            bottom_right = top_left + b.shape

            return (top_left + bottom_right) // 2

    return np.array((-1, 1))


def is_reeling():
    global offset_x, offset_y

    with mss.mss() as sct:
        image = np.array(
            sct.grab(
                (
                    offset_x + reel_prompt_rect[0],
                    offset_y + reel_prompt_rect[1],
                    offset_x + reel_prompt_rect[0] + reel_prompt_rect[2],
                    offset_y + reel_prompt_rect[1] + reel_prompt_rect[3],
                )
            )
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    prompt_match = cv2.matchTemplate(
        image, reel_prompt, cv2.TM_CCOEFF_NORMED, None, None
    )
    _, max_value, _, _ = cv2.minMaxLoc(prompt_match)

    return max_value > 0.5


def get_reel_state():
    global offset_x, offset_y

    with mss.mss() as sct:
        image = np.array(
            sct.grab(
                (
                    offset_x + reel_rect[0],
                    offset_y + reel_rect[1],
                    offset_x + reel_rect[0] + reel_rect[2],
                    offset_y + reel_rect[1] + reel_rect[3],
                )
            )
        )

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    kernel = np.ones((reel_rect[3] // 2, 3))
    fish = cv2.inRange(hsv, lower_fish_color, upper_fish_color)
    fish = cv2.morphologyEx(fish, cv2.MORPH_OPEN, kernel)
    fish = cv2.morphologyEx(fish, cv2.MORPH_CLOSE, kernel)
    fish_rect = cv2.boundingRect(fish)

    _, s, v = cv2.split(hsv)

    v = cv2.bitwise_and(v, cv2.bitwise_not(fish))
    v_max = v.max().item()

    current = cv2.inRange(v, v_max / 2, 255)

    if v_max < 255:
        composite = s.astype(np.uint16) * v
        c_max = composite.max().item()
        current &= cv2.inRange(composite, c_max / 2, c_max)

    kernel = np.ones((reel_rect[3] // 2, fish_rect[2]))
    current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
    current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
    current_rect = cv2.boundingRect(current)

    fish_position = fish_rect[0] + (fish_rect[2] / 2)
    current_position = current_rect[0] + (current_rect[2] / 2)

    fish_position /= reel_rect[2]
    current_position /= reel_rect[2]

    return current_position, current_rect[2] / reel_rect[2], fish_position


def get_is_window_focused():
    return pywinctl.getActiveWindowTitle() == "Roblox"


def get_active_window_rect():
    return pywinctl.getActiveWindow().rect


def toggle_auto_cast():
    global auto_cast
    auto_cast = not auto_cast


def toggle_auto_shake():
    global auto_shake
    auto_shake = not auto_shake


def toggle_auto_reel():
    global auto_reel
    auto_reel = not auto_reel


def main():
    if debug_mode:
        with mss.mss() as sct:
            monitor = sct.monitors[monitor_index]
            monitor_image = np.array(sct.grab(monitor))

            print("Selected monitor:", monitor)
            print("All monitors:", sct.monitors)

        from PIL import Image

        monitor_image = cv2.cvtColor(monitor_image, cv2.COLOR_BGR2RGB)
        Image.fromarray(monitor_image).save("full_preview.png")

        return

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]

        global offset_x, offset_y

        offset_x = monitor["left"]
        offset_y = monitor["top"]

    keyboard.add_hotkey("ctrl+shift+c", toggle_auto_cast)
    keyboard.add_hotkey("ctrl+shift+f", toggle_auto_shake)
    keyboard.add_hotkey("ctrl+shift+r", toggle_auto_reel)

    pydirectinput.PAUSE = 0

    failsafe_active = False
    last_active_time = time.time()

    print("Fisch bot is active")

    while True:
        # AFK fail safe

        last_active_elapsed = time.time() - last_active_time

        if last_active_elapsed > 60 and not failsafe_active:
            print("No shaking nor reeling detected in 1 minute")
            print(f"AFK fail safe activated at { datetime.datetime.now() }")
            failsafe_active = True
        elif last_active_elapsed > 60 * 10:
            print(
                "Last successful shake or reel was over 10 minutes ago, breaking loop"
            )
            print(f"Current time { datetime.datetime.now() }")
            break

        if not get_is_window_focused():
            time.sleep(1.5)
            continue

        # Cast

        if auto_cast and not failsafe_active:
            pydirectinput.moveTo(
                offset_x + reel_rect[0] + (reel_rect[2] // 2),
                offset_y + reel_rect[1] + reel_rect[3]
            )
            time.sleep(0.02)
            pydirectinput.mouseDown(button="left")
            time.sleep(np.random.uniform(0.25, 0.35))
            pydirectinput.mouseUp(button="left")
            time.sleep(2)

        # Shake

        was_shaking = False

        while auto_shake and get_is_window_focused():
            (x0, y0, x1, y1) = get_active_window_rect()

            with mss.mss() as capture:
                image = np.array(capture.grab((x0, y0, x1, y1)))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pos = get_shake_button_pos(image)

            if pos[0] >= 0 and pos[1] >= 0:
                was_shaking = True

                pos += (int(x0 + button_template.shape[1] / 6), y0)

                pydirectinput.moveTo(pos[0], pos[1] + 10)
                time.sleep(0.02)
                pydirectinput.moveTo(pos[0], pos[1])
                time.sleep(0.08)
                pydirectinput.click()
                time.sleep(0.5)
            else:
                break

        # Reel

        if auto_cast or was_shaking:
            # Wait for reeling minigame to start
            for _ in range(4):
                if is_reeling():
                    break
                else:
                    time.sleep(0.5)

        was_reeling = False
        last_reel_check_time = 0
        dt = 1 / 60

        estimator = fisch.ReelStateEstimator()
        controller_gains = {
            "default": (1, 0.5, 0),
            "edge": (1, 0, 0),
        }
        controller = fisch.Controller()

        start_time = time.time()

        is_holding = False

        while auto_reel:
            now = time.time()

            if now - last_reel_check_time > 0.1 or not get_is_window_focused():
                if not is_reeling():
                    break

                was_reeling = True
                last_reel_check_time = now

            position, width, target = get_reel_state()

            # Clip

            position = np.clip(position, width / 2, 1 - width / 2)
            target = np.clip(target, (width * 0.9 / 2), 1 - (width * 0.9 / 2))

            # Update kinematic metrics

            estimator.update(position, target, is_holding, dt)

            # Initial compensation

            alpha = (now - start_time) / 3

            if alpha < 1:
                target += (1 - alpha) * 0.025

            error = target - position

            # Acceleration compensation

            if estimator.forces[0] > 0 and estimator.forces[1] > 0:
                input_ratio = estimator.forces[1] / estimator.forces[0]

                if error > 0:
                    error /= input_ratio
                elif error < 0:
                    error *= input_ratio

            pydirectinput.moveTo(
                offset_x + reel_rect[0] + int(reel_rect[2] * np.clip(target, 0, 1)),
                offset_y + reel_rect[1] + (reel_rect[3] // 2),
            )

            if target < width / 2 or target > 1 - width / 2:
                controller.p, controller.d, controller.i = controller_gains["edge"]
            else:
                controller.p, controller.d, controller.i = controller_gains["default"]

            control_value = controller.update(error, dt)

            if control_value > 0:
                if not is_holding:
                    pydirectinput.mouseDown(button="left")

                is_holding = True
            else:
                pydirectinput.mouseUp(button="left")
                is_holding = False

            time.sleep(dt)

        if was_reeling:
            pydirectinput.mouseUp(button="left")

        if not (was_shaking or was_reeling):
            time.sleep(1)
        else:
            if failsafe_active:
                if was_shaking and was_reeling:
                    s = "shake and reel"
                elif was_shaking:
                    s = "shake"
                else:
                    s = "reel"

                print(f"AFK fail safe deactivated after successful { s }")
                print(f"Current time { datetime.datetime.now() }")

            failsafe_active = False
            last_active_time = time.time()

    print("Fisch bot is inactive")


if __name__ == "__main__":
    main()
