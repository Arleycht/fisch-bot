from collections import deque
import numpy as np


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
