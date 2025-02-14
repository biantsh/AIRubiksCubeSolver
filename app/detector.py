import cv2 as cv
import numpy as np
import tflite_runtime.interpreter as tflite

from app.detections import Detection


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

    def _predict(self, tensor: np.ndarray) -> tuple[tuple, float]:
        self.set_tensor(self.input_address, tensor)
        self.invoke()

        position = self.get_tensor(self.output_details[1]['index'])
        score = self.get_tensor(self.output_details[0]['index'])

        return tuple(np.squeeze(position)), float(score)

    def detect(self, image: np.ndarray) -> Detection:
        tensor = self._preprocess(image)
        position, score = self._predict(tensor)

        return Detection(position, score)
