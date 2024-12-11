import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Ensures no GUI is required for Qt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

''' Detects a face (or multiple faces)
	Draws a rectangle around them '''
def detect_face_rect(self, filename, out_filename):
	# filename = input("Enter the filename: ").strip()
	image = cv2.imread(filename=filename)
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	plt.figure(figsize=(15, 7))
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	
	# out_filename = input("Enter output filename: ").strip()
	
	plt.savefig(out_filename)
	print(f"Output saved to '{out_filename}'")

''' Detects face and returns the face and its coordinates '''
def detect_face_coord(self, filename):
	# filename = input("Enter the filename: ").strip()
	image = cv2.imread(filename=filename)
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
	if len(faces) == 0:
		return None, None
	(x, y, w, h) = faces[0]
	return gray_img[y:y+w, x:+h], faces[0]

''' We need to recognize which face belongs to whom.
	We provide multiple folders, each containing pictures of a different person.
	We first prepare the training data by storing the faces and their coordinates.
	Then we begin training the model using face coordinates and labels.
	For GUI simplicity, it is implemented to support only one face and one directory
	of faces, but it can be extended.
'''
def prepare_train_data(self, data_folder_path, filename, file_path):
	faces = []
	labels = []
	label = 0
	subject_images_names = os.listdir(data_folder_path)
	for image_name in subject_images_names:
		image_path = data_folder_path + "/" + image_name
		image = cv2.imread(image_path)
		face, rect = detect_face_coord(image, image_path)
		if face is not None and face.size > 0:
			resized_face = cv2.resize(face, (500, 800))
			faces.append(resized_face)
			labels.append(label)
	# Training
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.train(faces, np.array(labels))
	face, rect = detect_face_coord(self, file_path)

	''' Predictions are not accurate using a small set of data '''
	# if face is None or face.size == 0 or rect is None:
	# 	return None
	# label, confidence = face_recognizer.predict(face)
	# if confidence >= 100:
	# 	return None

	img = self.image
	(x, y, w, h) = rect

	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	cv2.putText(img, "", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	return img

def attach_face_to_image(image_class):
	image_class.detect_face = prepare_train_data
