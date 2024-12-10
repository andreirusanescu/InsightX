import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Ensures no GUI is required for Qt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def detect_face(self):
	filename = input("Enter the filename: ").strip()
	image = cv2.imread(filename=filename)
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
	for (x, y, w, h) in faces:
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	
	plt.figure(figsize=(15, 7))
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	
	out_filename = input("Enter output filename: ").strip()
	
	plt.savefig(out_filename)
	print(f"Output saved to '{out_filename}'")

# TODO: recognize face and prepare_training_data
def recognize_face(self):
	filename = input("Enter the filename: ").strip()
	image = cv2.imread(filename=filename)
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
	(x, y, w, h) = faces[0]
	return gray_img[y:y+w, x:+h], faces[0]

def prepare_training_data(data_folder_path):
	dirs = os.listdir(data_folder_path)
	faces = []
	labels = []
	for dir_name in dirs:
		if not dir_name.startswith("s"):
			continue
	label = int(dir_name.replace("s", ""))
	subject_dir_path = data_folder_path + "/" + dir_name
	subject_images_names = os.listdir(subject_dir_path)
	for image_name in subject_images_names:
		image_path = subject_dir_path + "/" + image_name
		image = cv2.imread(image_path)
		face, rect = detect_face(image)
		if face is not None:
			faces.append(face)
			labels.append(label)

	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.train(faces, np.array(labels))
	subjects = ["Kanye"]
	img = "kanye_exemplu.jpg"
	face, rect = detect_face(img)
	label = face_recognizer.predict(face)[0]
	label_text = subjects[label]
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	cv2.putText(img, label_text, (rect[0], rect[1]-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def attach_face_to_image(image_class):
	image_class.detect_face = detect_face
