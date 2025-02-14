import argparse

import cv2 as cv

from app.detector import TFLiteDetector
from app.webcam import WebcamInteractor


def main(detector_path: str) -> None:
    detector = TFLiteDetector(detector_path)
    webcam = WebcamInteractor()

    while webcam.isOpened():
        frame = webcam.get_frame()
        detection = detector.detect(frame)

        if detection.score > 0.5:
            detection.draw(frame)

        webcam.show_frame(frame)
        webcam.await_input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_path', required=True)

    args = parser.parse_args()
    main(args.detector_path)
