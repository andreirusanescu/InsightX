import cv2
import numpy as np
from image_utils import attach_utils_to_image
from filters import attach_filters_to_image
from adjust import attach_adjust_to_image

class Image:
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

	def pad_kernel(self):
		""" Called in wiener_deconvolution() to return
			a kernel the size of the image """
		kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
		padded_kernel = np.zeros(self.image.shape)
		kh, kw = kernel.shape
		center_h = (self.image.shape[0] - kh) // 2
		center_w = (self.image.shape[1] - kw) // 2
		padded_kernel[center_h:center_h + kh, center_w:center_w + kw] = kernel
		return padded_kernel

	def wiener_deconvolution(self, K=0.01):
		""" Applies Wiener Deconvolution to remove blur for grayscale images
		:param K: Noise factor (controls the removing of the noise)
		F(u,v)= H*(u,v) / (|H(u,v)|^2 + K) * G(u,v)
		"""
		if len(self.image.shape) != 2:
			print("Image is not grayscale!")
			return

		img_FFT = np.fft.fft2(self.image)

		kernel = self.pad_kernel()
		kernel_FFT = np.fft.fft2(kernel, s=self.image.shape)

		# Hermitian matrix
		kernel_FFT_conj = np.conjugate(kernel_FFT)
		deconvolvedFFT = (kernel_FFT_conj / (np.abs(kernel_FFT) ** 2 + K)) * img_FFT
		deconvolved = np.fft.ifft2(deconvolvedFFT)
		deconvolved = np.abs(deconvolved)
		
		# Normalize the image
		deconvolved = (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved)) * 255
		self.image = deconvolved.astype(np.uint8)
		print("Unblurred the image")
	
	def blend(self):
		filename2 = input("Enter filename to be merged: ").strip()
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

		alpha, beta = map(float, input("Enter alpha (0.0, 1.0), beta (0.0, 1.0), (alpha + beta = 1.0): ").split())
		if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0) or round(alpha + beta, 5) != 1.0:
			raise ValueError("Alpha and beta should be between 0.0 and 1.0, alpha + beta = 1.0")

		# Resize image to the main image sizes
		height, width = self.image.shape[:2]
		image2 = cv2.resize(image2, (width, height))

		# Making sure they have the same channels
		if len(self.image.shape) != len(image2.shape):
			image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

		# Making sure they are the same type
		self.image = self.image.astype('float32')
		image2 = image2.astype('float32')

		self.image = cv2.addWeighted(self.image, alpha, image2, beta, 0.0)
		print(f"Blended the two images")

attach_utils_to_image(Image)
attach_filters_to_image(Image)
attach_adjust_to_image(Image)
