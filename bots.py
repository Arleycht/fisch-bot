from typing import Dict
import cv2
import datetime
import mss
import numpy as np
import time
import pydirectinput
import yaml

import util, kinematics


class Bot:
    def __init__(self):
        self.failsafe_active = False
        self.last_active_time = time.time()

        self.auto_start = False
        self.auto_control = True

        self.is_running = True

    def stop(self):
        self.is_running = False


class FischConfig:
    def __init__(self):
        self.prompt_rect = np.array((0, 0, 0, 0))
        self.reel_rect = np.array((0, 0, 0, 0))

        self.monitor_index = 1
        self.prompt_template = None

    def load(self, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        use_preset = config["use_preset"]
        preset = config["presets"][use_preset]

        try:
            reel_prompt_image_path = preset["prompt_image"]
            self.prompt_template = cv2.imread(
                reel_prompt_image_path, cv2.IMREAD_UNCHANGED
            )
            self.prompt_template = cv2.cvtColor(
                self.prompt_template, cv2.COLOR_BGR2GRAY
            )
        except cv2.error:
            print(f'Failed to find image "{ reel_prompt_image_path }"')
            exit()

        def load_as_array(k):
            return np.array(preset[k]).astype(np.int64)

        self.monitor_index = preset["monitor_index"]
        self.button_base_path = preset["button_base_image"]
        self.button_text_path = preset["button_text_image"]
        self.prompt_rect = load_as_array("prompt_rect")
        self.reel_rect = load_as_array("reel_rect")
        self.button_scales = load_as_array("button_scales")
        self.fish_color_hsv = load_as_array("fish_color_hsv")


class Fisch(Bot):
    def __init__(self, config: FischConfig):
        super().__init__()

        self.config = config

        button_background = cv2.imread(config.button_base_path, cv2.IMREAD_UNCHANGED)
        button_text = cv2.imread(config.button_text_path, cv2.IMREAD_UNCHANGED)
        template = util.paste(button_background, button_text, background_alpha=0.6)

        self.button_template = cv2.cvtColor(np.array(template), cv2.COLOR_BGR2GRAY)
        self.offset = np.array((0, 0))

    def process_sobel(self, image):
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

    def get_shake_button_pos(self, image, threshold=0.45):
        a = self.process_sobel(image)

        for scale in self.config.button_scales:
            b = cv2.resize(self.button_template, (scale, scale))
            b = self.process_sobel(b)

            matched = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED, None, None)
            _, max_value, _, max_location = cv2.minMaxLoc(matched)

            if max_value > threshold:
                top_left = np.array(max_location)
                bottom_right = top_left + b.shape

                return (top_left + bottom_right) // 2

        return np.array((-1, 1))

    def is_control_minigame_active(self):
        image = util.grab_image(
            *self.config.prompt_rect[0:2],
            *(self.config.prompt_rect[0:2] + self.config.prompt_rect[2:4]),
            self.config.monitor_index,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prompt_match = cv2.matchTemplate(
            image, self.config.prompt_template, cv2.TM_CCOEFF_NORMED, None, None
        )
        _, max_value, _, _ = cv2.minMaxLoc(prompt_match)

        return max_value > 0.5

    def get_state(self):
        image = util.grab_image(
            *self.config.reel_rect[0:2],
            *(self.config.reel_rect[0:2] + self.config.reel_rect[2:4]),
            self.config.monitor_index,
        )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        fish_color_epsilon = (1, 3, 3)
        fish_color = np.multiply(self.config.fish_color_hsv, (0.5, 2.55, 2.55))
        lower_fish_color = fish_color - fish_color_epsilon
        upper_fish_color = fish_color + fish_color_epsilon

        kernel = np.ones((self.config.reel_rect[3] // 2, 3))
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

        kernel = np.ones((self.config.reel_rect[3] // 2, fish_rect[2]))
        current = cv2.morphologyEx(current, cv2.MORPH_CLOSE, kernel)
        current = cv2.morphologyEx(current, cv2.MORPH_OPEN, kernel)
        current_rect = cv2.boundingRect(current)

        fish_position = fish_rect[0] + (fish_rect[2] / 2)
        current_position = current_rect[0] + (current_rect[2] / 2)

        fish_position /= self.config.reel_rect[2]
        current_position /= self.config.reel_rect[2]

        return (
            current_position,
            current_rect[2] / self.config.reel_rect[2],
            fish_position,
        )

    def run(self):
        pydirectinput.PAUSE = 0

        while self.is_running:
            # AFK fail safe

            last_active_elapsed = time.time() - self.last_active_time

            if last_active_elapsed > 60 and not self.failsafe_active:
                print("No shaking nor reeling detected in 1 minute")
                print(f"AFK fail safe activated at { datetime.datetime.now() }")
                self.failsafe_active = True
            elif last_active_elapsed > 60 * 10:
                print(
                    "Last successful shake or reel was over 10 minutes ago, breaking loop"
                )
                print(f"Current time { datetime.datetime.now() }")
                break

            if not util.is_window_focused():
                time.sleep(1.5)
                continue

            # Cast

            if self.auto_start and not self.failsafe_active:
                pydirectinput.moveTo(
                    self.offset[0]
                    + self.config.reel_rect[0]
                    + (self.config.reel_rect[2] // 2),
                    self.offset[1]
                    + self.config.reel_rect[1]
                    + self.config.reel_rect[3],
                )
                time.sleep(0.02)
                pydirectinput.mouseDown(button="left")
                time.sleep(np.random.uniform(0.25, 0.35))
                pydirectinput.mouseUp(button="left")
                time.sleep(2)

            # Shake

            was_shaking = False

            while self.auto_control and util.is_window_focused():
                (x0, y0, x1, y1) = util.get_active_window_rect()

                with mss.mss() as capture:
                    image = np.array(capture.grab((x0, y0, x1, y1)))

                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                pos = self.get_shake_button_pos(image)

                if pos[0] >= 0 and pos[1] >= 0:
                    was_shaking = True

                    pos += (int(x0 + self.button_template.shape[1] / 6), y0)

                    pydirectinput.moveTo(pos[0], pos[1] + 10)
                    time.sleep(0.02)
                    pydirectinput.moveTo(pos[0], pos[1])
                    time.sleep(0.08)
                    pydirectinput.click()
                    time.sleep(0.5)
                else:
                    break

            # Reel

            if self.auto_start or was_shaking:
                # Wait for reeling minigame to start
                for _ in range(4):
                    if self.is_control_minigame_active():
                        break
                    else:
                        time.sleep(0.5)

            was_reeling = False
            last_reel_check_time = 0
            dt = 1 / 60

            estimator = kinematics.ReelStateEstimator()
            controller_gains = {
                "default": (1, 0.5, 0),
                "edge": (1, 0, 0),
            }
            controller = kinematics.Controller()

            start_time = time.time()

            is_holding = False

            while self.auto_control:
                now = time.time()

                if now - last_reel_check_time > 0.1 or not util.is_window_focused():
                    if not self.is_control_minigame_active():
                        break

                    was_reeling = True
                    last_reel_check_time = now

                position, width, target = self.get_state()

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
                    self.offset[0]
                    + self.config.reel_rect[0]
                    + int(self.config.reel_rect[2] * np.clip(target, 0, 1)),
                    self.offset[1]
                    + self.config.reel_rect[1]
                    + (self.config.reel_rect[3] // 2),
                )

                if target < width / 2 or target > 1 - width / 2:
                    controller.p, controller.d, controller.i = controller_gains["edge"]
                else:
                    controller.p, controller.d, controller.i = controller_gains[
                        "default"
                    ]

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
                if self.failsafe_active:
                    if was_shaking and was_reeling:
                        s = "shake and reel"
                    elif was_shaking:
                        s = "shake"
                    else:
                        s = "reel"

                    print(f"AFK fail safe deactivated after successful { s }")
                    print(f"Current time { datetime.datetime.now() }")

                self.failsafe_active = False
                self.last_active_time = time.time()

        print("Fisch bot is inactive")


class DigItConfig:
    def __init__(self):
        self.prompt_rect = np.array((0, 0, 0, 0))
        self.reel_rect = np.array((0, 0, 0, 0))
        self.edge_rect = np.array((0, 0, 0, 0))
        self.sample_coord = np.array((0, 0, 0, 0))

        self.monitor_index = 1
        self.prompt_template = None

    def load(self, path):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        use_preset = config["use_preset"]
        preset = config["presets"][use_preset]

        try:
            reel_prompt_image_path = preset["prompt_image"]
            self.prompt_template = cv2.imread(
                reel_prompt_image_path, cv2.IMREAD_UNCHANGED
            )
            self.prompt_template = cv2.cvtColor(
                self.prompt_template, cv2.COLOR_BGR2GRAY
            )
        except cv2.error:
            print(f'Failed to find image "{ reel_prompt_image_path }"')
            exit()

        def load_as_array(k):
            return np.array(preset[k]).astype(np.int64)

        self.monitor_index = preset["monitor_index"]
        self.prompt_rect = load_as_array("prompt_rect")
        self.reel_rect = load_as_array("reel_rect")
        self.edge_rect = load_as_array("edge_rect")
        self.sample_coord = load_as_array("sample_coord")


class DigIt(Bot):
    def __init__(self, config: DigItConfig):
        super().__init__()
        self.config = config

    def is_control_minigame_active(self):
        image = util.grab_image(
            *self.config.prompt_rect[0:2],
            *(self.config.prompt_rect[0:2] + self.config.prompt_rect[2:4]),
            self.config.monitor_index,
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prompt_match = cv2.matchTemplate(
            image, self.config.prompt_template, cv2.TM_CCOEFF_NORMED, None, None
        )
        _, max_value, _, _ = cv2.minMaxLoc(prompt_match)

        return max_value > 0.45

    def get_current_pos(self, image: np.ndarray):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv[hsv.shape[0] // 2, :, :]
        sat, val = hsv[:, 1], hsv[:, 2]

        mask = (sat == 0) & (val > 127)
        mask = mask[np.newaxis, :].astype(np.uint8)

        (x, _, w, _) = cv2.boundingRect(mask)

        return (x + w / 2) / int(image.shape[1])

    def get_target_pos(self, image: np.ndarray, target_color_hsv: np.ndarray):
        kernel_size = 40
        eps = np.array((5, 5, 200))

        # Create a normalized HSV image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Blur
        hsv = cv2.GaussianBlur(hsv, (13, 13), 0)

        # Mask by color
        color_mask = cv2.inRange(hsv, target_color_hsv - eps, target_color_hsv + eps)
        hsv = cv2.bitwise_and(hsv, hsv, mask=color_mask)

        # Remove artifacts
        hsv = cv2.morphologyEx(
            hsv, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size))
        )
        hsv = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size)))

        image = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        rects = util.get_rects(image)

        if len(rects) == 0:
            return None, None
        elif len(rects) > 1:
            rect = rects[rects[:, 2].argmax()]
        else:
            rect = rects[0]

        (x, _, w, _) = rect

        # Center of mass method

        _, x = np.where(image > 0)

        if len(x) > 0:
            pos = np.average(x).item()
        else:
            pos = w / 2

        pos /= int(image.shape[1])
        w /= int(image.shape[1])

        return pos, w

    def get_state(self):
        reel_a = self.config.reel_rect[0:2]
        reel_b = self.config.reel_rect[0:2] + self.config.reel_rect[2:4]

        edge_a = self.config.edge_rect[0:2]
        edge_b = self.config.edge_rect[0:2] + self.config.edge_rect[2:4]

        sample = self.config.sample_coord

        min_coord = np.min((reel_a, reel_b, sample), axis=0)
        max_coord = np.max((reel_a, reel_b, sample), axis=0)

        reel_a = reel_a - min_coord
        reel_b = reel_b - min_coord
        edge_a = edge_a - min_coord
        edge_b = edge_b - min_coord
        sample = sample - min_coord

        image = util.grab_image(*min_coord, *max_coord)

        reel_image = image[reel_a[1] : reel_b[1], reel_a[0] : reel_b[0]]
        edge_image = image[edge_a[1] : edge_b[1], edge_a[0] : edge_b[0]]

        target_color_hsv = image[sample[1], sample[0], 0:3]
        target_color_hsv = np.array(((target_color_hsv,),))
        target_color_hsv = cv2.cvtColor(target_color_hsv, cv2.COLOR_BGR2HSV)[0, 0]
        # Use slightly higher valued color to better represent the midtone of the target bar
        if target_color_hsv[2] < 255 - 3:
            target_color_hsv[2] += 3

        current_pos = self.get_current_pos(edge_image)
        target_pos, target_width = self.get_target_pos(reel_image, target_color_hsv)

        return current_pos, target_pos, target_width

    def run(self):
        print("Dig bot is active")

        pydirectinput.PAUSE = 0

        while self.is_running:
            # AFK fail safe

            last_active_elapsed = time.time() - self.last_active_time

            if last_active_elapsed > 60 and not self.failsafe_active:
                print("No action detected in 1 minute")
                print(f"AFK fail safe activated at { datetime.datetime.now() }")
                self.failsafe_active = True
            elif last_active_elapsed > 60 * 10:
                print("Last successful action was over 10 minutes ago, breaking loop")
                print(f"Current time { datetime.datetime.now() }")
                break

            if not util.is_window_focused():
                time.sleep(1.5)
                continue

            with mss.mss() as sct:
                monitor = sct.monitors[self.config.monitor_index]
                offset = (monitor["left"], monitor["top"])

            # Strike

            if self.auto_start and not self.failsafe_active:
                pos = (
                    offset
                    + self.config.reel_rect[0:2]
                    + (self.config.reel_rect[2:4] // 2)
                )
                pydirectinput.moveTo(*pos)
                time.sleep(0.2)
                pydirectinput.click(*pos)
                time.sleep(0.75)

                for _ in range(30):
                    if self.is_control_minigame_active():
                        break
                    else:
                        time.sleep(1 / 10)

            # Dig

            was_digging = False
            last_dig_check_time = 0

            max_speed = 0.35
            max_frequency = 60

            estimator = kinematics.KinematicEstimator()
            controller = kinematics.Controller(
                1, 0.04, 0.015, error_bounds=max_speed / 4
            )

            is_holding = False
            last_click_time = 0
            last_target_pos = 0.5
            last_target_width = 0.5

            last_time = time.perf_counter()

            while self.auto_control:
                now = time.perf_counter()

                if now - last_dig_check_time > 0.25 or not util.is_window_focused():
                    if not self.is_control_minigame_active():
                        break

                    was_digging = True
                    last_dig_check_time = now

                true_dt = now - last_time
                dt = max(true_dt, 1 / 60)
                last_time = now

                current_pos, target_pos, target_width = self.get_state()

                if target_pos is None:
                    target_pos = last_target_pos
                else:
                    last_target_pos = target_pos

                if target_width is None:
                    target_width = last_target_width
                else:
                    last_target_width = target_width

                # Clip

                current_pos = np.clip(current_pos, 0, 1)
                target_pos = np.clip(target_pos, 0, 1)

                # Move mouse
                pydirectinput.moveTo(
                    offset[0]
                    + self.config.reel_rect[0]
                    + int(self.config.reel_rect[2] * target_pos),
                    offset[1]
                    + self.config.reel_rect[1]
                    + (self.config.reel_rect[3] // 2),
                )

                # Update kinematic metrics

                estimator.update(current_pos, dt)
                velocity = estimator.velocity

                position_error = target_pos - current_pos
                velocity_error = (
                    np.clip(position_error * 2, -max_speed, max_speed) - velocity
                )

                error = velocity_error
                error += np.clip(position_error, -target_width, target_width)

                control_value = controller.update(error, dt)

                if control_value > 0:
                    if not is_holding or (
                        position_error > target_width / 2
                        and now - last_click_time > 0.25
                    ):
                        pydirectinput.mouseDown(button="left")
                        last_click_time = now

                    is_holding = True
                else:
                    pydirectinput.mouseUp(button="left")
                    is_holding = False
                    last_click_time = now

                time.sleep(max((1 / max_frequency) - true_dt, 0))

            if was_digging:
                pydirectinput.mouseUp(button="left")

                if self.failsafe_active:
                    if was_digging:
                        s = "dig"

                    print(f"AFK fail safe deactivated after successful { s }")
                    print(f"Current time { datetime.datetime.now() }")

                self.failsafe_active = False
                self.last_active_time = time.time()

                time.sleep(2.5)

        print("Dig bot is inactive")
