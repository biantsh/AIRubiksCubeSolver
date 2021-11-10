import numpy as np
import itertools
import kociemba

number_to_face = {0: 'U', 1: 'R', 2: 'F', 3: 'D', 4: 'L', 5: 'B'}

def isSolvable(scramble):
	# Checks if a the state of the cube that is passed into this fuction is solvable
	try:
		solve = kociemba.solve(scramble)
	except:
		return False
	return True

def rotateFace(face, n):
	# Rotates a face of colors 90, 180, or 270 degrees clockwise depending on if n == 1, 2, or 3. This is useful so that all possible rotation
	# combinations are checked, until the solvable one is found
	matrix = np.array([letter for letter in face]).reshape(3,3)
	rotated = np.rot90(matrix, k=4-int(n))
	
	return ''.join([letter for letter in rotated.reshape(9)])

def dict2Solution(dict_original):
	# Takes a dictionary of the 6 individual Rubik's cube faces and figures out how to combine them so that it matches the true scramble
	# that was shown to the camera
	combinations = [i for i in itertools.product('0123', repeat = 6)]
	solvable, scrambles = 0, []

	for combination in combinations:
		dict_ = dict_original.copy()
		for k in range(len(combination)):
			dict_[number_to_face[k]] = rotateFace(dict_[number_to_face[k]], combination[k])

		scrambled_state = dict_['U'] + dict_['R'] + dict_['F'] + dict_['D'] + dict_['L'] + dict_['B']

		if isSolvable(scrambled_state):
			return kociemba.solve(scrambled_state)
