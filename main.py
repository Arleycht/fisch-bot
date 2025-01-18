import cv2
import datetime
import mss
import keyboard
import numpy as np
import pydirectinput
import time
import pywinctl

import kinematics

# Hard coded areas to screenshot

debug_mode = False

monitor_index = 1
prompt_rect = (720, 670, 480, 70)
reel_rect = (435, 820, 1050, 84)
edge_rect = (435, 814, 1050, 1)
sample_coord = (432, 817)

# Prepare template images

prompt_template = cv2.imread("reel_prompt.png", cv2.IMREAD_UNCHANGED)
prompt_template = cv2.cvtColor(prompt_template, cv2.COLOR_BGR2GRAY)

reel_color = np.array((152, 152, 152))

# Runtime variables

auto_strike = False
auto_reel = True


def grab_image(x: int, y: int, w: int, h: int, ignore_offset=False):
    with mss.mss() as sct:
        coords = np.array((x, y, x + w, y + h))

        if not ignore_offset:
            monitor = sct.monitors[monitor_index]
            offset = (monitor["left"], monitor["top"])

            coords += (offset[0], offset[1], offset[0], offset[1])

            image = np.array(sct.grab(tuple(coords.tolist())))

    return image


def is_control_minigame_active():
    image = grab_image(*prompt_rect)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    prompt_match = cv2.matchTemplate(
        image, prompt_template, cv2.TM_CCOEFF_NORMED, None, None
    )
    _, max_value, _, _ = cv2.minMaxLoc(prompt_match)

    return max_value > 0.5


def get_current_pos(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv[hsv.shape[0] // 2, :, :]
    sat, val = hsv[:, 1], hsv[:, 2]

    mask = (sat == 0) & (val > 127)
    mask = mask[np.newaxis, :].astype(np.uint8)

    (x, _, w, _) = cv2.boundingRect(mask)
    pos = (x + w / 2) / reel_rect[2]

    return pos


def get_rects(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for contour in contours:
        rects.append(cv2.boundingRect(contour))

    return np.array(rects)


def get_target_pos(image, target_color_hsv):
    kernel_size = 32
    eps = (5, 10, 255)

    # Create a normalized HSV image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hsv = cv2.GaussianBlur(hsv, (13, 13), 0)
    color_mask = cv2.inRange(hsv, target_color_hsv - eps, target_color_hsv + eps)
    hsv = cv2.bitwise_and(hsv, hsv, mask=color_mask)
    hsv = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size)))
    hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size)))

    rects = get_rects(
        cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    )

    if len(rects) == 0:
        print("Failed to find target, defaulting to 0.5")
        return 0.5, 0.5
    elif len(rects) > 1:
        rect = rects[rects[:, 2].argmax()]
    else:
        rect = rects[0]

    # Normalize position and width
    (x, _, w, _) = rect
    pos = (x + w / 2) / reel_rect[2]

    return pos, w / reel_rect[2]


def get_state():
    image = grab_image(*reel_rect)
    edge_image = grab_image(*edge_rect)
    target_color = grab_image(
        sample_coord[0], sample_coord[1], sample_coord[0] + 1, sample_coord[1] + 1
    )
    target_color = cv2.cvtColor(target_color, cv2.COLOR_BGR2HSV)[0, 0]

    current_pos = get_current_pos(edge_image)
    target_pos, target_width = get_target_pos(image, target_color)

    return current_pos, target_pos, target_width


def is_window_focused():
    return pywinctl.getActiveWindowTitle() == "Roblox"


def get_active_window_rect():
    return pywinctl.getActiveWindow().rect


def toggle_auto_strike():
    global auto_strike
    auto_strike = not auto_strike


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

    keyboard.add_hotkey("ctrl+shift+c", toggle_auto_strike)
    keyboard.add_hotkey("ctrl+shift+r", toggle_auto_reel)

    pydirectinput.PAUSE = 0

    failsafe_active = False
    last_active_time = time.time()

    print("Dig bot is active")

    while True:
        # AFK fail safe

        last_active_elapsed = time.time() - last_active_time

        if last_active_elapsed > 60 and not failsafe_active:
            print("No action detected in 1 minute")
            print(f"AFK fail safe activated at { datetime.datetime.now() }")
            failsafe_active = True
        elif last_active_elapsed > 60 * 10:
            print("Last successful action was over 10 minutes ago, breaking loop")
            print(f"Current time { datetime.datetime.now() }")
            break

        if not is_window_focused():
            time.sleep(1.5)
            continue

        with mss.mss() as sct:
            monitor = sct.monitors[monitor_index]
            offset = (monitor["left"], monitor["top"])
        
        # Strike

        if auto_strike and not failsafe_active:
            pydirectinput.moveTo(
                offset[0] + reel_rect[0] + (reel_rect[2] // 2),
                offset[1] + reel_rect[1] + reel_rect[3]
            )
            time.sleep(0.02)
            pydirectinput.click()
            time.sleep(0.5)

        # Dig

        was_digging = False
        last_dig_check_time = 0
        dt = 1 / 60

        estimator = kinematics.KinematicEstimator()
        controller_gains = {
            "default": (1, 0.05, 0),
            "close": (1, 0.025, 0),
        }
        controller = kinematics.Controller()

        is_holding = False

        while auto_reel:
            now = time.time()

            if now - last_dig_check_time > 0.1 or not is_window_focused():
                if not is_control_minigame_active():
                    break

                was_digging = True
                last_dig_check_time = now

            position, target, width = get_state()

            # Clip

            position = np.clip(position, width / 2, 1 - width / 2)
            target = np.clip(target, (width * 0.9 / 2), 1 - (width * 0.9 / 2))

            error = target - position

            # Update kinematic metrics

            estimator.update(position, dt)
            error -= (estimator.velocity * dt) + (estimator.acceleration * 0.5 * dt)

            pydirectinput.moveTo(
                offset[0] + reel_rect[0] + int(reel_rect[2] * np.clip(target, 0, 1)),
                offset[1] + reel_rect[1] + (reel_rect[3] // 2),
            )

            if error < width / 2:
                controller.p, controller.d, controller.i = controller_gains["close"]
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

        if was_digging:
            pydirectinput.mouseUp(button="left")

            if failsafe_active:
                if was_digging:
                    s = "dig"

                print(f"AFK fail safe deactivated after successful { s }")
                print(f"Current time { datetime.datetime.now() }")

            failsafe_active = False
            last_active_time = time.time()

            time.sleep(0.5)

    print("Dig bot is inactive")


if __name__ == "__main__":
    main()
