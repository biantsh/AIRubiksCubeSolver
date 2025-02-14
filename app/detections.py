import cv2 as cv
import numpy as np


class Detection:
    def __init__(self, position: tuple, score: float) -> None:
        self.position = position
        self.score = score

    def draw(self, frame: np.ndarray) -> None:
        top, left, bot, right = self.position
        height, width, _ = frame.shape

        top, bot = int(top * height), int(bot * height)
        left, right = int(left * width), int(right * width)

        cv.rectangle(frame, (left, top), (right, bot), (0, 255, 0), 2)
