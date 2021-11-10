import numpy as np
import cv2 as cv
import pickle

with open('KNN_model', 'rb') as f:
	model = pickle.load(f)

def img2color(img):
	# Classifies the color from an image of roughly uniform color
	lab = cv.cvtColor(img.astype(np.uint8), cv.COLOR_BGR2LAB)
	lab = np.mean(lab, axis=(0, 1)).reshape(1, -1)
	return {0: 'U', 1: 'L', 2: 'F', 3: 'R', 4: 'B', 5: 'D'}[model.predict(lab)[0]]

def crop_detected(image, detection_boxes):
	# Crops and returns only the part of the image that's detected as a cube face
	y_min, x_min, y_max, x_max = (np.array(detection_boxes[0][0]) * 750).astype(np.int16)
	image = image[y_min:y_max, x_min:x_max]
	return cv.resize(image, (100,100))

def detect_colors(img):
	# Takes an image of a rubik's cube face and returns a list containing the colors as strings ('White', 'Yellow', etc.)
	coordinates = [(12,12), (46,12), (80,12), (12,46), (46,46), (80,46), (12,80), (46,80), (80,80)]
	return ''.join([img2color(img[y:y+8,x:x+8]) for (x, y) in coordinates])

def colors_image(colors):
	# Takes the list of color strings and displays them as a virtual Rubik's cube face 300x300 background (for visualization reasons)
	coordinates = [(6,6), (104,6), (202,6), (6,104), (104,104), (202,104), (6, 202), (104, 202), (202,202)]
	color_dict = {'U': (240,240,240), 'L': (0,143,230), 'F': (54,194,15), 'R': (28,31,222), 
	'B': (207,87,23), 'D': (41,227,227)}

	colors_image = np.zeros((300,300,3))
	for i in range(len(coordinates)):
		(x, y) = coordinates[i]
		cv.rectangle(colors_image, (x, y), (x+92, y+92), color_dict[colors[i]], -1)
	return colors_image

def show_colors(image, colors_image, detection_scores, detection_boxes):
	# Displays the colors_image on the main image, based on the position and size of detected cube face so that it shows right next to it
	if detection_scores[0][0] >= 0.5:
		y_min, x_min, y_max, x_max = (np.array(detection_boxes[0][0]) * 750).astype(np.int16)
		width = height = y_max - y_min

		if x_max + width < 750:
			image[y_min:y_max, x_max:x_max+width] = cv.resize(colors_image, (width, height))
		elif x_min - width > 0:
			image[y_min:y_max, x_min-width:x_min] = cv.resize(colors_image, (width, height))

	return image

def check_consistency(new_string, info_string, faces_dict):
	# Continuously checks if the string arriving is the same as the previous one for continuity reasons
	previous_string, number = info_string.split('_')
	
	if new_string == previous_string:
		number = int(number) + 1
		info_string = new_string + '_' + str(number)
	else:
		info_string = new_string + '_1'

	if number == 15 and new_string[4] in faces_dict:
		print("Already registered face")
	elif number == 15:
		faces_dict[new_string[4]] = new_string
		print(faces_dict)

	return info_string, faces_dict
