from enum import Enum


class Color(Enum):
    WHITE = 0, (255, 255, 255)
    ORANGE = 1, (0, 165, 255)
    GREEN = 2, (0, 128, 0)
    RED = 3, (0, 0, 255)
    BLUE = 4, (255, 0, 0)
    YELLOW = 5, (0, 255, 255)

    def __new__(cls, value: int, bgr: tuple[int, ...]) -> 'Color':
        color = object.__new__(cls)
        color._value_ = value
        color.bgr = bgr

        return color
