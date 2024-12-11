import cv2
import numpy as np
from image_utils import attach_utils_to_image
from filters import attach_filters_to_image
from adjust import attach_adjust_to_image
from sift import attach_sift_to_image
from palm import attach_palm_to_image
from ransac import attach_ransac_to_image
from detect_faces import attach_face_to_image

class MyImage:
	def __init__(self, filename):
		"""
		Initializes the Image by loading it into memory using OpenCV.
		:param filename: The path to the image file.
		:param image: The image itself
		"""
		self.filename = filename
		self.image = None

		try:
			# Load the image using OpenCV, BGR format
			self.image = cv2.imread(filename)
			if self.image is None:
				raise FileNotFoundError(f"Image '{filename}' not found or file is not a valid image.")
			print(f"Image '{filename}' loaded successfully. Size: {self.image.shape[1]}x{self.image.shape[0]}")
		except FileNotFoundError as e:
			print(e)
			exit(1)
		except Exception as e:
			print(f"Error loading image: {e}")
			exit(1)

	def __str__(self):
		""" Return the string representation of the image """
		if self.image is not None:
			height, width, channels = self.image.shape
			return f"Image '{self.filename}' with size {width}x{height} and {channels} channels."
		else:
			return "No image loaded."
	
	def blend(self, filename2, alpha, beta):
		try:
			image2 = cv2.imread(filename2)
			if self.image is None:
				raise FileNotFoundError(f"Image '{filename2}' not found or file is not a valid image.")
			print(f"Image to be blended '{filename2}' loaded successfully. Size: {image2.shape[1]}x{image2.shape[0]}")
		except FileNotFoundError as e:
			print(e)
			exit(1)
		except Exception as e:
			print(f"Error loading image: {e}")
			exit(1)

		# alpha, beta = map(float, input("Enter alpha (0.0, 1.0), beta (0.0, 1.0), (alpha + beta = 1.0): ").split())
		if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0) or round(alpha + beta, 5) != 1.0:
			raise ValueError("Alpha and beta should be between 0.0 and 1.0, alpha + beta = 1.0")

		# Resize image to the main image sizes
		height, width = self.image.shape[:2]
		image2 = cv2.resize(image2, (width, height))

		# Making sure they have the same channels
		if len(self.image.shape) != len(image2.shape):
			image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

		result = cv2.addWeighted(self.image, alpha, image2, beta, 0.0)
		print(f"Blended the two images")
		return result

attach_utils_to_image(MyImage)
attach_filters_to_image(MyImage)
attach_adjust_to_image(MyImage)
attach_sift_to_image(MyImage)
attach_ransac_to_image(MyImage)
attach_palm_to_image(MyImage)
attach_face_to_image(MyImage)
