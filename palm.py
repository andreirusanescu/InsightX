import cv2
import os
import matplotlib.pyplot as plt

# Ensures no GUI is required for Qt
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def find_palm_lines(self, filename, out_filename):
	# filename = input("Enter palm filename: ").strip()
	image = cv2.imread(filename=filename)
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 40, 55, apertureSize=3)
	edges = cv2.bitwise_not(edges)
	
	# Convert image color so it can be merged with the original image
	edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
	
	# Merge original image with detected palm lines
	img = cv2.addWeighted(edges, 0.3, image, 0.7, 0)

	plt.figure(figsize=(15, 7))
	# Convert BGR to RGB
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.axis('off')

	# out_filename = input("Enter output filename: ").strip()
	
	plt.savefig(out_filename)
	# print(f"Output saved to '{out_filename}'")

def attach_palm_to_image(image_class):
	image_class.find_palm_lines = find_palm_lines
