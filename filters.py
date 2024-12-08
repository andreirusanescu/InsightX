import cv2
import numpy as np

def apply_filter(self, filterType):
	""" Apply a filter to the image (blur, median_blur, edge, sharpen, etc.) """
	if self.image is not None:
		if filterType == "BLUR":
			""" the bigger the kernel size, the blurrier the image
				the bigger sigmaX, the blurrier the image """
			self.image = cv2.GaussianBlur(self.image, (5, 5), 1.0)
		elif filterType == "MEDIAN_BLUR":
			self.image = cv2.medianBlur(self.image, (5, 5))
		elif filterType == "SHARPEN":
			kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
			# ddepth -1, auto, keeps the same pixel intensity as the input
			self.image = cv2.filter2D(self.image, -1, kernel)
		elif filterType == "EDGE":
			""" First threshold filters the noise,
				second threshold selects the clear edges.
				The gradient is computed for each pixel, using Hysteresis
				to find pixels between thresholds. The pixels above
				second threshold become strong edges """
			self.image = cv2.Canny(self.image, 100, 200)
		else:
			print(f"Invalid filter type: {filterType}")
			return
		print(f"Applied {filterType} filter.")
	else:
		print("No image to apply filter to")

def gray_scale(self):
	""" Turns the image from RGB to Grayscale """
	if self.image is not None and len(self.image.shape) == 3:
		self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		print("Image turned to gray scale")

def equalize(self):
	""" Equalize the image using OpenCV's histogram equalization. """
	if self.image is not None:
		# Grayscale image
		if len(self.image.shape) == 2:
			self.image = cv2.equalizeHist(self.image)
		# RGB image
		else:
			# Equalize each channel separately
			channels = cv2.split(self.image)

			# Equalize with Histograms for each channel (B, G, R)
			equalized_channels = [cv2.equalizeHist(channel) for channel in channels]

			# Merge the RGB channels into the image
			self.image = cv2.merge(equalized_channels)
		print("Image equalized.")
	else:
		print("No image to equalize.")

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

def attach_filters_to_image(image_class):
	"""
	Attaches the methods below to the given image class.
	:param image_class: The class to which the methods will be attached.
	"""
	image_class.apply_filter = apply_filter
	image_class.gray_scale = gray_scale
	image_class.equalize = equalize
	image_class.unblur = wiener_deconvolution
