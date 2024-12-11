import numpy as np
import cv2

def apply_filter(self, filter_type, kernel_size=1, sigmaX=0.0, intensity=1.0, denoise_strength=10):
	"""
	Apply a filter to the image (blur, median_blur, edge, sharpen, etc.)
	:param filter_type: the applied filter
	:param kernel_size: the kernel size (int)
	:param sigmaX: standard deviation on the X axis (float)
	:param intensity: the intensity applied for sharpen (0.0 - 3.0)
	:param denoise_strength: (0.0 - 30.0)
	"""
	result = None
	if self.image is not None:
		if filter_type == "BLUR":
			""" the bigger the kernel size, the blurrier the image
				the bigger sigmaX, the blurrier the image """
			result = cv2.GaussianBlur(self.image, ksize=(kernel_size, kernel_size), sigmaX=sigmaX)
		elif filter_type == "MEDIAN BLUR":
			result = cv2.medianBlur(self.image, ksize=kernel_size)
		elif filter_type == "SHARPEN":
			intensity = max(0, intensity)
			center_value = 9 + intensity
			sharpen_kernel = np.array([[-1, -1, -1], [-1, center_value, -1], [-1, -1, -1]])
			sharpen = cv2.filter2D(self.image, -1, sharpen_kernel)
			deblurred = cv2.fastNlMeansDenoisingColored(sharpen, None, denoise_strength, denoise_strength, 7, 21)
			result = deblurred
		elif filter_type == "EDGE":
			""" First threshold filters the noise,
				second threshold selects the clear edges.
				The gradient is computed for each pixel, using Hysteresis
				to find pixels between thresholds. The pixels above
				second threshold become strong edges """
			result = cv2.Canny(self.image, 100, 200)
	return result

def gray_scale(self):
	""" Turns the image from RGB to Grayscale """
	result = None
	if self.image is not None and len(self.image.shape) == 3:
		result = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
	return result

def equalize(self):
	""" Equalize the image using OpenCV's histogram equalization """
	result = None
	if self.image is not None:
		# Grayscale image
		if len(self.image.shape) == 2:
			result = cv2.equalizeHist(self.image)
		# RGB image
		else:
			# Equalize each channel separately
			channels = cv2.split(self.image)

			# Equalize with Histograms for each channel (B, G, R)
			equalized_channels = [cv2.equalizeHist(channel) for channel in channels]

			# Merge the RGB channels into the image
			result = cv2.merge(equalized_channels)
	return result


def pad_kernel(kernel, target_shape):
	"""Pad the kernel to match the target shape """
	padded = np.zeros(target_shape, dtype=kernel.dtype)
	kh, kw = kernel.shape
	th, tw = target_shape

	# Center-align the kernel within the padded array
	start_y = (th - kh) // 2
	start_x = (tw - kw) // 2
	padded[start_y:start_y + kh, start_x:start_x + kw] = kernel
	# Apply shift to center the kernel in the frequency domain
	padded = np.fft.ifftshift(padded) 
	return padded

def pad_kernel(kernel, target_shape):
	"""Pad the kernel to match the target shape """
	padded = np.zeros(target_shape, dtype=kernel.dtype)
	kh, kw = kernel.shape
	th, tw = target_shape

	# Center-align the kernel within the padded array
	start_y = (th - kh) // 2
	start_x = (tw - kw) // 2
	padded[start_y:start_y + kh, start_x:start_x + kw] = kernel

	# Shift kernel center for frequency domain alignment
	padded = np.fft.ifftshift(padded)
	return padded

def wiener_deconvolution(self, K=0.01):
	""" Wiener deconvolution - unblurring algorithm """
	if len(self.image.shape) == 3:
		target_shape = self.image.shape[:2]
	else:
		target_shape = self.image.shape

	kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
	kernel_padded = pad_kernel(kernel, target_shape)

	if len(self.image.shape) == 3:
		deconvolved = []
		for c in range(self.image.shape[2]):
			img_FFT = np.fft.fft2(self.image[:, :, c])
			kernel_FFT = np.fft.fft2(kernel_padded, s=self.image.shape[:2])
			kernel_FFT_conj = np.conjugate(kernel_FFT)
			deconvolvedFFT = (kernel_FFT_conj / (np.abs(kernel_FFT) ** 2 + K)) * img_FFT
			deconvolved_channel = np.fft.ifft2(deconvolvedFFT)
			deconvolved.append(np.abs(deconvolved_channel))
		deconvolved = np.stack(deconvolved, axis=-1)
	else:
		img_FFT = np.fft.fft2(self.image)
		kernel_FFT = np.fft.fft2(kernel_padded, s=self.image.shape)
		kernel_FFT_conj = np.conjugate(kernel_FFT)
		deconvolvedFFT = (kernel_FFT_conj / (np.abs(kernel_FFT) ** 2 + K)) * img_FFT
		deconvolved = np.abs(np.fft.ifft2(deconvolvedFFT))

	min_val = np.min(deconvolved)
	max_val = np.max(deconvolved)
	if max_val - min_val > 0:
		deconvolved = (deconvolved - min_val) / (max_val - min_val) * 255
	else:
		deconvolved = np.zeros_like(deconvolved)

	return np.clip(np.round(deconvolved), 0, 255).astype(np.uint8)

def attach_filters_to_image(image_class):
	""" Monkey patching helper function """
	image_class.apply_filter = apply_filter
	image_class.gray_scale = gray_scale
	image_class.equalize = equalize
	image_class.unblur = wiener_deconvolution
