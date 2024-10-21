import cv2 as cv2
import mss
import keyboard
import numpy as np
import pydirectinput
import time
import win32gui

from PIL import Image


# Hard coded areas to screenshot
reel_rect = (572, 876, 776, 31)
reel_prompt_rect = (870, 790, 176, 16)

MONITOR_INDEX = 0
auto_cast = False
auto_shake = True
auto_reel = True


def paste(
    background: Image.Image,
    foreground: Image.Image,
    position=(0, 0),
    background_alpha=1,
    foreground_alpha=1,
):
    background = background.copy().convert("RGBA")
    foreground = foreground.convert("RGBA")

    alpha = background.split()[3]
    alpha = alpha.point(lambda x: x * background_alpha)
    background.putalpha(alpha)

    alpha = foreground.split()[3]
    alpha = alpha.point(lambda x: x * foreground_alpha)
    foreground.putalpha(alpha)

    background.paste(foreground, position, foreground)

    return background


button_background = Image.open("base_button.png")
button_text = Image.open("base_text.png")
button_template = paste(button_background, button_text, background_alpha=0.6)
button_template = cv2.cvtColor(np.array(button_template), cv2.COLOR_BGR2GRAY)
button_scales = [121, 205]

reel_prompt = np.array(Image.open("reel_prompt.png"))
reel_prompt = cv2.cvtColor(reel_prompt, cv2.COLOR_RGBA2GRAY)


def get_shake_button_pos(threshold=0.25):
    for scale in button_scales:
        button = cv2.resize(button_template, (scale, scale))

        with mss.mss() as capture:
            image = np.array(capture.grab(capture.monitors[MONITOR_INDEX]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        matched = cv2.matchTemplate(image, button, cv2.TM_CCOEFF_NORMED, None, None)
        _, max_value, _, max_location = cv2.minMaxLoc(matched)
        a = max_location
        b = (a[0] + button.shape[0], a[1] + button.shape[1])

        x = (a[0] + b[0]) // 2
        y = (a[1] + b[1]) // 2

        if max_value > threshold:
            return (x, y)

    return (-1, 1)


def is_reeling():
    with mss.mss() as sct:
        image = np.array(
            sct.grab(
                {
                    "left": reel_prompt_rect[0],
                    "top": reel_prompt_rect[1],
                    "width": reel_prompt_rect[2],
                    "height": reel_prompt_rect[3],
                }
            )
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (image > int(255 * 0.7)).astype(np.uint8) * 255

    prompt_match = cv2.matchTemplate(
        image, reel_prompt, cv2.TM_CCOEFF_NORMED, None, None
    )
    _, max_value, _, _ = cv2.minMaxLoc(prompt_match)

    return max_value > 0.75


def get_reel_state():
    with mss.mss() as sct:
        image = np.array(
            sct.grab(
                {
                    "left": reel_rect[0],
                    "top": reel_rect[1],
                    "width": reel_rect[2],
                    "height": reel_rect[3],
                }
            )
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fish = cv2.inRange(image, 70, 80)
    fish = cv2.morphologyEx(fish, cv2.MORPH_OPEN, np.ones((5, 7)))
    fish_rect = cv2.boundingRect(fish)

    current = cv2.equalizeHist(image)
    current = cv2.inRange(current, 150, 255)
    current = cv2.morphologyEx(current, cv2.MORPH_OPEN, np.ones((127, 127)))
    current_rect = cv2.boundingRect(cv2.inRange(current, 200, 255))

    fish_position = fish_rect[0] + (fish_rect[2] // 2)
    fish_position /= reel_rect[2]

    current_position = current_rect[0] + (current_rect[2] // 2)
    current_position /= reel_rect[2]

    return current_position, current_rect[2] / reel_rect[2], fish_position


def get_window_title():
    return win32gui.GetWindowText(win32gui.GetForegroundWindow())


def toggle_auto_cast():
    global auto_cast
    auto_cast = not auto_cast


def toggle_auto_shake():
    global auto_shake
    auto_shake = not auto_shake


def toggle_auto_reel():
    global auto_reel
    auto_reel = not auto_reel


class KinematicEstimator:
    """
    Second order estimator
    """

    def __init__(self, x=0, v=0, a=0):
        self.state = (x, v, a)
        self.prev_state = (x, v, a)

    def update(self, x: float, delta_time: float):
        (prev_x, prev_v, _) = self.prev_state

        v = (x - prev_x) / delta_time
        a = (v - prev_v) / delta_time

        self.prev_state = self.state
        self.state = (x, v, a)

    def estimate(self, projected_time: float, ignore_acceleration=False):
        (x, v, a) = self.state

        v_p = v + (a * projected_time)
        x_p = x + (v * projected_time)

        if not ignore_acceleration:
            x_p += 0.5 * a * projected_time**2

        return (x_p, v_p)


def main():
    pydirectinput.PAUSE = 0

    keyboard.add_hotkey("ctrl+shift+c", toggle_auto_cast)
    keyboard.add_hotkey("ctrl+shift+f", toggle_auto_shake)
    keyboard.add_hotkey("ctrl+shift+r", toggle_auto_reel)

    while True:
        if get_window_title() != "Roblox":
            time.sleep(1.5)
            continue

        # Cast
        if auto_cast:
            pydirectinput.moveTo(
                reel_rect[0] + reel_rect[2] // 2, reel_rect[1] + reel_rect[3]
            )
            time.sleep(0.02)
            pydirectinput.mouseDown(button="left")
            time.sleep(np.random.uniform(0.25, 0.35))
            pydirectinput.mouseUp(button="left")
            time.sleep(2)

        # Shake button loop
        was_shaking = False

        while auto_shake:
            (x, y) = get_shake_button_pos(threshold=0.25)

            if x >= 0 and y >= 0:
                was_shaking = True

                pydirectinput.moveTo(x + button_template.shape[1] // 6, y + 10)
                time.sleep(0.02)
                pydirectinput.moveTo(x + button_template.shape[1] // 6, y)
                time.sleep(0.08)
                pydirectinput.click()
                time.sleep(np.random.uniform(0.4, 0.6))
            else:
                break

        if was_shaking:
            # Wait for reeling minigame to start
            time.sleep(1.5)

        was_reeling = False
        last_reel_check_time = 0

        last_time = time.time() - 0.1
        reel_estimator = KinematicEstimator()
        target_estimator = KinematicEstimator()

        is_holding = False

        while auto_reel:
            # Check reeling conditions
            if (
                time.time() - last_reel_check_time > 0.5
                or get_window_title() != "Roblox"
            ):
                if not is_reeling():
                    break

                was_reeling = True
                last_reel_check_time = time.time()

            delta_time = time.time() - last_time
            last_time = time.time()

            # Get current state

            x, width, target = get_reel_state()

            left_edge = width / 2
            right_edge = 1 - (width / 2)

            # Update kinematic metrics

            reel_estimator.update(x, delta_time)
            target_estimator.update(target, delta_time)

            projected_time = 0.2

            projected_position, projected_velocity = reel_estimator.estimate(
                projected_time
            )
            projected_target, _ = target_estimator.estimate(projected_time)

            # Improve deceleration stability

            _, target_velocity, target_accleration = target_estimator.state

            # if np.sign(target_velocity) != np.sign(target_accleration):
            # projected_target = (projected_target + target) / 2

            # projected_position = np.clip(projected_position, left_edge, right_edge)
            # projected_target = np.clip(projected_target, left_edge, right_edge)

            projected_error = projected_target - projected_position

            # _, projected_target_velocity, _ = target_estimator.state
            # projected_error += 0.2 * (projected_target_velocity - projected_velocity)

            pydirectinput.moveTo(
                int(
                    np.clip(
                        reel_rect[0] + reel_rect[2] * projected_position,
                        reel_rect[0],
                        reel_rect[0] + reel_rect[2],
                    )
                ),
                int(reel_rect[1] + (reel_rect[3] / 2)),
            )

            max_target_speed = 3

            hold = False
            edge_tolerance = 0

            if target >= left_edge + edge_tolerance:
                if (
                    (projected_error > 0 or projected_velocity < -max_target_speed)
                    and not projected_velocity > max_target_speed
                ) or target > right_edge - edge_tolerance:
                    hold = True

            if hold:
                if not is_holding:
                    pydirectinput.mouseDown(button="left")
                    is_holding = True
            else:
                pydirectinput.mouseUp(button="left")
                is_holding = False

            time.sleep(1 / 15)

        if was_reeling:
            pydirectinput.mouseUp(button="left")
            time.sleep(3.5)

        if not (was_shaking or was_reeling):
            time.sleep(0.5)


if __name__ == "__main__":
    main()
