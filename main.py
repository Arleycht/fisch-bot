import cv2 as cv2
import mss
import keyboard
import numpy as np
import pydirectinput
import time
import pywinctl

from matplotlib import pyplot as plt

import fisch


use_right_docked_window = False

# Hard coded areas to screenshot
if not use_right_docked_window:
    reel_prompt_rect = (870, 790, 176, 16)
    reel_rect = (572, 876, 776, 31)
    reel_prompt = cv2.imread("reel_prompt.png", cv2.IMREAD_UNCHANGED)
else:
    reel_prompt_rect = (1354, 815, 168, 46)
    reel_rect = (1246, 892, 387, 15)
    reel_prompt = cv2.imread("reel_prompt_half.png", cv2.IMREAD_UNCHANGED)

reel_prompt = cv2.cvtColor(reel_prompt, cv2.COLOR_BGR2GRAY)

MONITOR_INDEX = 0
auto_cast = False
auto_shake = True
auto_reel = True

plot_controller_data = False

button_background = cv2.imread("base_button.png", cv2.IMREAD_UNCHANGED)
button_text = cv2.imread("base_text.png", cv2.IMREAD_UNCHANGED)
button_template = fisch.paste(button_background, button_text, background_alpha=0.6)
button_template = cv2.cvtColor(np.array(button_template), cv2.COLOR_BGR2GRAY)
button_scales = [
    115,  # Half window sizes
    190,
    121,  # Maximized window sizes
    203,
]


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
            top_left = max_location
            bottom_right = (top_left[0] + b.shape[0], top_left[1] + b.shape[1])

            center_x = (top_left[0] + bottom_right[0]) // 2
            center_y = (top_left[1] + bottom_right[1]) // 2

            return (center_x, center_y)

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

    prompt_match = cv2.matchTemplate(
        image, reel_prompt, cv2.TM_CCOEFF_NORMED, None, None
    )
    _, max_value, _, _ = cv2.minMaxLoc(prompt_match)

    return max_value > 0.5


def get_reel_state():
    with mss.mss() as sct:
        image = np.array(
            sct.grab(
                (
                    reel_rect[0],
                    reel_rect[1],
                    reel_rect[0] + reel_rect[2],
                    reel_rect[1] + reel_rect[3],
                )
            )
        )
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    fish = cv2.inRange(grayscale, 70, 80)
    fish = cv2.morphologyEx(fish, cv2.MORPH_OPEN, np.ones((8, 4)))
    fish = cv2.morphologyEx(fish, cv2.MORPH_CLOSE, np.ones((8, 4)))
    fish_rect = cv2.boundingRect(fish)

    current = cv2.morphologyEx(grayscale, cv2.MORPH_CLOSE, np.ones((16, 16)))
    current = cv2.morphologyEx(current, cv2.MORPH_OPEN, np.ones((16, 16)))
    current = cv2.inRange(current, current.max().item() - 10, 255)
    current_rect = cv2.boundingRect(current)

    fish_position = fish_rect[0] + (fish_rect[2] // 2)
    fish_position /= reel_rect[2]

    current_position = current_rect[0] + (current_rect[2] // 2)
    current_position /= reel_rect[2]

    return current_position, current_rect[2] / reel_rect[2], fish_position


def get_is_window_focused():
    return pywinctl.getActiveWindowTitle() == "Roblox"


def get_active_window_rect():
    (x0, y0, x1, y1) = pywinctl.getActiveWindow().rect
    return (x0, y0, x1 - x0, y1 - y0)


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
    keyboard.add_hotkey("ctrl+shift+c", toggle_auto_cast)
    keyboard.add_hotkey("ctrl+shift+f", toggle_auto_shake)
    keyboard.add_hotkey("ctrl+shift+r", toggle_auto_reel)

    pydirectinput.PAUSE = 0

    while True:
        if not get_is_window_focused():
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

        # Shake

        was_shaking = False

        while auto_shake and get_is_window_focused():
            (ox, oy, w, h) = get_active_window_rect()

            with mss.mss() as capture:
                image = np.array(capture.grab((ox, oy, ox + w, oy + h)))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            (x, y) = get_shake_button_pos(image)

            if x >= 0 and y >= 0:
                was_shaking = True

                x += ox
                y += oy

                pydirectinput.moveTo(x + button_template.shape[1] // 6, y + 10)
                time.sleep(0.02)
                pydirectinput.moveTo(x + button_template.shape[1] // 6, y)
                time.sleep(0.08)
                pydirectinput.click()
                time.sleep(np.random.uniform(0.4, 0.5))
            else:
                break

        # Reel

        if was_shaking:
            # Wait for reeling minigame to start
            time.sleep(1.75)

        was_reeling = False
        last_reel_check_time = 0
        dt = 1 / 60

        estimator = fisch.ReelStateEstimator()
        controller = fisch.Controller(
            1.5, 0.45, 0.05, clip_error=True, error_bounds=(-0.5, 0.5)
        )

        start_time = time.time()

        is_holding = False

        positions = []
        target_positions = []
        velocities = []
        accelerations = []

        while auto_reel:
            if time.time() - last_reel_check_time > 0.2 or not get_is_window_focused():
                if not is_reeling():
                    break

                was_reeling = True
                last_reel_check_time = time.time()

            position, width, target = get_reel_state()

            # Clip

            position = np.clip(position, width, 1 - width / 2)

            # Update kinematic metrics

            estimator.update(position, target, is_holding, dt)

            positions.append(estimator.reel.position)
            velocities.append(estimator.reel.velocity)
            accelerations.append(estimator.reel.acceleration)
            target_positions.append(estimator.fish.position)

            error = target - position

            if estimator.forces[0] > 0 and estimator.forces[1] > 0:
                input_ratio = estimator.forces[1] / estimator.forces[0]

                if error > 0:
                    error /= input_ratio
                elif error < 0:
                    error *= input_ratio

            pydirectinput.moveTo(
                int(reel_rect[0] + reel_rect[2] * np.clip(target, 0, 1)),
                int(reel_rect[1] + reel_rect[3] / 2),
            )

            elapsed_percentage = (time.time() - start_time) / 2

            if elapsed_percentage < 1:
                a = (target - position) + (width * 0.1)

                error = a + (error - a) * elapsed_percentage

            controller.error_bounds = (-width * 0.1, width * 0.1)
            control_value = controller.update(error, dt)

            if control_value > 0:
                if not is_holding:
                    pydirectinput.mouseDown(button="left")

                is_holding = True
            else:
                pydirectinput.mouseUp(button="left")
                is_holding = False

            time.sleep(dt)

        if len(positions) > 4 and plot_controller_data:
            valid_slice = slice(6, -6)

            positions = positions[valid_slice]
            velocities = velocities[valid_slice]
            accelerations = accelerations[valid_slice]
            target_positions = target_positions[valid_slice]

            time_interval = np.linspace(0, 1, len(positions))

            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            axs[0].plot(time_interval, positions, label="Position", color="blue")
            axs[0].plot(time_interval, target_positions, label="Target", color="red")
            axs[0].set_ylabel("Position")
            axs[0].legend()
            axs[0].grid()

            axs[1].plot(time_interval, velocities, label="Velocity", color="orange")
            axs[1].set_ylabel("Velocity")
            axs[1].legend()
            axs[1].grid()

            axs[2].plot(
                time_interval, accelerations, label="Acceleration", color="green"
            )
            axs[2].set_xlabel("Time (s)")
            axs[2].set_ylabel("Acceleration")
            axs[2].legend()
            axs[2].grid()

            plt.tight_layout()
            fig.savefig("fig.png")

        if was_reeling:
            pydirectinput.mouseUp(button="left")
            time.sleep(2.5)

        if not (was_shaking or was_reeling):
            time.sleep(0.5)


if __name__ == "__main__":
    main()
