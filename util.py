import cv2
import mss
import numpy as np
import pywinctl


def grab_image(x0: int, y0: int, x1: int, y1: int, monitor_index=1):
    with mss.mss() as sct:
        coords = np.array((x0, y0, x1, y1))

        monitor = sct.monitors[monitor_index]
        offset = (monitor["left"], monitor["top"])

        coords += (offset[0], offset[1], offset[0], offset[1])

        image = np.array(sct.grab(tuple(coords.tolist())))

    return image


def is_window_focused():
    return pywinctl.getActiveWindowTitle() == "Roblox"


def get_active_window_rect():
    return pywinctl.getActiveWindow().rect


def get_rects(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []

    for contour in contours:
        rects.append(cv2.boundingRect(contour))

    return np.array(rects)
