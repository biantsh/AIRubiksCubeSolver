import tensorflow as tf
from object_detection.utils import config_util, visualization_utils
from object_detection.builders import model_builder

from display_utils import detect_colors, colors_image, show_colors, crop_detected, check_consistency
from kociemba_utils import dict2Solution

import cv2 as cv
import numpy as np
import textwrap

# This function takes the image from the webcam and returns the possible location of a Rubik's cube face
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    return detection_model.postprocess(prediction_dict, shapes)

# Loading the Tensorflow model from file and setting it up
configs = config_util.get_configs_from_pipeline_file('TF_model\\pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('TF_model\\ckpt-6').expect_partial()

category_index = {1: {'id': 1, 'name': 'Cube face'}}
info_string = '_'
faces_dict = {}
counter = 0

# Setting up the webcam capture
cap = cv.VideoCapture(0)

# Mainloop
while True: 
    ret, img = cap.read()
    img = cv.resize(img, (750,750))

    input_tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    if detections['detection_scores'][0][0] >= 0.5:
        ## If cube is detected with >50% confidence, crop the image and detect the individual colors
        cropped_img = crop_detected(img, detections['detection_boxes'])
        colors = detect_colors(cropped_img)
        img = show_colors(img, colors_image(colors), detections['detection_scores'], detections['detection_boxes'])
        face_string = ''.join([element[::-1] for element in textwrap.wrap(colors, 3)]) ## face_string is a list that contains the colors as strings

        ## This function checks if the list of strings arriving is the same as the one from the previous frame (for continuity reasons)
        info_string, faces_dict = check_consistency(face_string, info_string, faces_dict)

        ## If all 6 faces have been detected, find the solution, print it, and terminate the program.
        if len(faces_dict) == 6:  
            solution = dict2Solution(faces_dict)
            print(solution)
            break

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    ## Drawing a green rectangle around the cube for visualization 
    visualization_utils.visualize_boxes_and_labels_on_image_array(img,
        detections['detection_boxes'], detections['detection_classes']+1, detections['detection_scores'],
        category_index, use_normalized_coordinates=True, max_boxes_to_draw=1)

    cv.imshow('Cube solver', img)
    
    counter += 1
    cv.waitKey(1)
