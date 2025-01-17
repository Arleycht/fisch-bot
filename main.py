import cv2
import datetime
import mss
import keyboard
import numpy as np
import pydirectinput
import time
import pywinctl

import fisch

# Hard coded areas to screenshot

debug_mode = False

monitor_index = 0
reel_prompt_rect = (720, 670, 480, 70)
reel_rect = (435, 820, 1050, 84)
edge_rect = (435, 814, 1050, 1)

# Prepare template images

reel_prompt = cv2.imread("reel_prompt.png", cv2.IMREAD_UNCHANGED)
reel_prompt = cv2.cvtColor(reel_prompt, cv2.COLOR_BGR2GRAY)

reel_color = np.array((152, 152, 152))

# Runtime variables

auto_reel = True

offset_x = 0
offset_y = 0


def is_digging():
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


def get_current_pos(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv[hsv.shape[0] // 2,:,:]
    sat, val = hsv[:,1], hsv[:,2]
    
    mask = (sat == 0) & (val > 127)
    mask = mask[np.newaxis,:].astype(np.uint8)
    
    (x, _, w, _) = cv2.boundingRect(mask)
    pos = (x + w / 2) / reel_rect[2]
    
    return pos


def get_rects(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    
    for i, v in enumerate(contours):
        contours[i] = cv2.boundingRect(v)

    return np.array(contours)


def get_target_pos(image):
    # Create a normalized HSV image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    composite = (hsv[:,:,0] / 180) * (hsv[:,:,1] / 255)

    composite = cv2.GaussianBlur(composite, (7, 7), 0)
    
    composite -= composite.min()
    composite /= composite.max()
    
    kernel_size = 32
    threshold_value = 0.5
    
    thresh = cv2.compare(composite, threshold_value, cv2.CMP_GT)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size)))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size)))
    rects = get_rects(thresh)

    # If there are too many rects, attempt to gather the target from the inverted composite
    if len(rects) == 0 or rects[:,2].max() > thresh.shape[1] / 2 or len(rects) > 4:
        composite_inv = 1 - composite
        thresh_inv = cv2.compare(composite_inv, threshold_value, cv2.CMP_GT)
        thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size)))
        thresh_inv = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size)))
        rects_inv = get_rects(thresh_inv)

        # If it is an improvement, swap the images with their inversions
        if len(rects_inv) < len(rects):
            thresh = thresh_inv
            rects = rects_inv

    # If no rects are found, default target position to the center
    if len(rects) == 0:
        print("Failed to find target, defaulting to 0.5")
        return 0.5, 0.5
    
    # Take rect with maximum width as it is likely the target
    rect = rects[rects[:,2].argmax()]
    
    # Normalize position and width
    (x, _, w, _) = rect
    pos = (x + w / 2) / reel_rect[2]
    
    return pos, w / reel_rect[2]


def get_state():
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

        edge_image = np.array(
            sct.grab(
                (
                    offset_x + edge_rect[0],
                    offset_y + edge_rect[1],
                    offset_x + edge_rect[0] + edge_rect[2],
                    offset_y + edge_rect[1] + edge_rect[3],
                )
            )
        )
    
    current_pos = get_current_pos(edge_image)
    target_pos, target_width = get_target_pos(image)

    return current_pos, target_pos, target_width


def get_is_window_focused():
    return pywinctl.getActiveWindowTitle() == "Roblox"


def get_active_window_rect():
    return pywinctl.getActiveWindow().rect


def toggle_auto_reel():
    global auto_reel
    auto_reel = not auto_reel


def main():
    if debug_mode:
        while not (get_is_window_focused() and is_digging()):
            time.sleep(1)
        
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
            print(
                "Last successful action was over 10 minutes ago, breaking loop"
            )
            print(f"Current time { datetime.datetime.now() }")
            break

        if not get_is_window_focused():
            time.sleep(1.5)
            continue

        # Dig

        was_digging = False
        last_dig_check_time = 0
        dt = 1 / 60

        estimator = fisch.ReelStateEstimator()
        controller_gains = {
            "default": (1, 0.1, 0.02),
            "edge": (1, 0.1, 0),
        }
        controller = fisch.Controller()

        start_time = time.time()

        is_holding = False

        while auto_reel:
            now = time.time()

            if now - last_dig_check_time > 0.1 or not get_is_window_focused():
                if not is_digging():
                    break

                was_digging = True
                last_dig_check_time = now

            position, target, width = get_state()

            # Clip

            position = np.clip(position, width / 2, 1 - width / 2)
            target = np.clip(target, (width * 0.9 / 2), 1 - (width * 0.9 / 2))

            # Update kinematic metrics

            estimator.update(position, target, is_holding, dt)

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

        if was_digging:
            pydirectinput.mouseUp(button="left")

        if not was_digging:
            time.sleep(1)
        else:
            if failsafe_active:
                if was_digging:
                    s = "dig"

                print(f"AFK fail safe deactivated after successful { s }")
                print(f"Current time { datetime.datetime.now() }")

            failsafe_active = False
            last_active_time = time.time()

    print("Dig bot is inactive")


if __name__ == "__main__":
    main()
