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

def attach_face_to_image(image_class):
	image_class.detect_face = detect_face
