import cv2
from collections import deque
import numpy as np

from PIL import Image


def paste(
    background: np.ndarray,
    foreground: np.ndarray,
    position=(0, 0),
    background_alpha=1,
    foreground_alpha=1,
):
    background = Image.fromarray(background)
    foreground = Image.fromarray(foreground)

    alpha = background.split()[3]
    alpha = alpha.point(lambda x: x * background_alpha)
    background.putalpha(alpha)

    alpha = foreground.split()[3]
    alpha = alpha.point(lambda x: x * foreground_alpha)
    foreground.putalpha(alpha)

    background.paste(foreground, position, foreground)

    return background


class KinematicEstimator:
    def __init__(self, window_len: int = 4):
        self.position = 0.5
        self.velocity = 0
        self.acceleration = 0

        self.positions = deque(maxlen=max(4, window_len))
        self.window_len = window_len

        for _ in range(self.positions.maxlen):
            self.positions.append(self.position)

    def update(self, position: float, dt: float):
        self.positions.append(position)

        duration = dt * len(self.positions)

        position_diffs = np.diff(self.positions)
        average_velocity = np.mean(position_diffs) / duration

        velocity_diffs = np.diff(position_diffs)
        average_acceleration = np.mean(velocity_diffs) / duration

        self.position = np.mean(self.positions)
        self.velocity = average_velocity
        self.acceleration = average_acceleration


class ReelStateEstimator:
    def __init__(self):
        self.reel = KinematicEstimator(window_len=6)
        self.fish = KinematicEstimator(window_len=4)

        self.forces = [0, 0]
        self.current_time = 0
        self.last_measure_time = 0
        self.measure_state = False

    def update(
        self, reel_position: float, fish_position: float, is_holding: bool, dt: float
    ):
        self.reel.update(reel_position, dt)
        self.fish.update(fish_position, dt)

        elapsed = self.current_time - self.last_measure_time

        if self.measure_state == is_holding:
            if elapsed >= 8 * dt:
                if self.reel.acceleration != 0:
                    if is_holding and self.reel.acceleration >= 0:
                        self.forces[1] = abs(self.reel.acceleration)
                    elif self.reel.acceleration <= 0:
                        self.forces[0] = abs(self.reel.acceleration)

                self.last_measure_time = self.current_time
        else:
            self.measure_state = is_holding
            self.last_measure_time = self.current_time

        self.current_time += dt


class Controller:
    def __init__(self, p=1, d=0, i=0, clip_error=False, error_bounds=(0, 0)):
        self.p = p
        self.d = d
        self.i = i

        self.clip_error = clip_error
        self.error_bounds = error_bounds

        self.prev_error = 0
        self.sum_error = 0

    def update(self, error: float, dt: float):
        self.sum_error += self.i * error * dt

        if self.clip_error:
            self.sum_error = np.clip(
                self.sum_error, self.error_bounds[0], self.error_bounds[1]
            )

        result = (
            (self.p * error)
            + (self.d * (error - self.prev_error) / dt)
            + self.sum_error
        )

        self.prev_error = error

        return result
