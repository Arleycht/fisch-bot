import numpy as np


class KinematicEstimator():
    def __init__(self):
        self.position = 0
        self.velocity = 0
        self.acceleration = 0

    def update(self, position: float, dt: float):
        prev_position = self.position
        prev_velocity = self.velocity

        self.position = position
        self.velocity = (self.position - prev_position) / dt
        self.acceleration = (self.velocity - prev_velocity) / dt


class ReelStateEstimator():
    def __init__(self):
        self.reel = KinematicEstimator()
        self.fish = KinematicEstimator()

        self.forces = [0, 0]
        self.current_time = 0
        self.last_measure_time = 0
        self.measure_state = False

    def update(self, reel_position: float, fish_position: float, is_holding: bool, dt: float):
        self.reel.update(reel_position, dt)
        self.fish.update(fish_position, dt)

        elapsed = self.current_time - self.last_measure_time

        if self.measure_state == is_holding:
            if elapsed >= dt * 4:
                if is_holding:
                    self.forces[1] = abs(self.reel.acceleration)
                else:
                    self.forces[0] = abs(self.reel.acceleration)

                self.last_measure_time = self.current_time
        else:
            self.measure_state = is_holding
            self.last_measure_time = self.current_time

        self.current_time += dt


class Controller():
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
                self.sum_error, self.error_bounds[0], self.error_bounds[1])

        result = (self.p * error) + (self.d *
                                     (error - self.prev_error) / dt) + self.sum_error

        self.prev_error = error

        return result
