from dataclasses import dataclass

import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite


@dataclass
class Detection:
    score: float
    position: tuple


class TFLiteDetector(tflite.Interpreter):
    def __init__(self, model_path: str) -> None:
        super().__init__(model_path)
        self.allocate_tensors()

        self.input_details = self.get_input_details()
        self.output_details = self.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        self.input_address = self.input_details[0]['index']

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        _, height, width, _ = self.input_shape

        image = cv.resize(image, (width, height))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        tensor = image.astype(self.input_dtype)
        tensor = (tensor - 127.5) / 127.5

        return np.expand_dims(tensor, 0)

    def _predict(self, tensor: np.ndarray) -> tuple[float, tuple]:
        self.set_tensor(self.input_address, tensor)
        self.invoke()

        score = self.get_tensor(self.output_details[0]['index'])
        position = self.get_tensor(self.output_details[1]['index'])

        return float(score), tuple(np.squeeze(position))

    def detect(self, image: np.ndarray) -> Detection:
        tensor = self._preprocess(image)
        score, position = self._predict(tensor)

        return Detection(score, position)
