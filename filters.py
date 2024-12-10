import numpy as np
import cv2

def apply_filter(self, filterType, kernel_size, sigmaX):
	""" Apply a filter to the image (blur, median_blur, edge, sharpen, etc.) """
	result = None
	if self.image is not None:
		if filterType == "BLUR":
			""" the bigger the kernel size, the blurrier the image
				the bigger sigmaX, the blurrier the image """
			result = cv2.GaussianBlur(self.image, ksize=(kernel_size, kernel_size), sigmaX=sigmaX)
		elif filterType == "MEDIAN BLUR":
			result = cv2.medianBlur(self.image, ksize=kernel_size)
		elif filterType == "SHARPEN":
			kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
			# ddepth -1, auto, keeps the same pixel intensity as the input
			result = cv2.filter2D(self.image, -1, kernel)
		elif filterType == "EDGE":
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
	""" Equalize the image using OpenCV's histogram equalization. """
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
    """Pad the kernel to match the target shape."""
    padded = np.zeros(target_shape, dtype=kernel.dtype)
    kh, kw = kernel.shape
    th, tw = target_shape

    # Center-align the kernel within the padded array
    start_y = (th - kh) // 2
    start_x = (tw - kw) // 2
    padded[start_y:start_y + kh, start_x:start_x + kw] = kernel
    return padded

def wiener_deconvolution(self, K=0.01):
	# Ensure kernel is padded to the same size as the image
	if len(self.image.shape) == 3:  # Multichannel image
		target_shape = self.image.shape[:2]
	else:
		target_shape = self.image.shape
	
	kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

	kernel_padded = pad_kernel(kernel, target_shape)

	if len(self.image.shape) == 3:  # Multichannel image
		deconvolved = []
		for c in range(self.image.shape[2]):
			# FFT for the image channel and kernel
			img_FFT = np.fft.fft2(self.image[:, :, c])
			kernel_FFT = np.fft.fft2(kernel_padded, s=self.image.shape[:2])
			kernel_FFT_conj = np.conjugate(kernel_FFT)

			# Wiener deconvolution
			deconvolvedFFT = (kernel_FFT_conj / (np.abs(kernel_FFT) ** 2 + K)) * img_FFT
			deconvolved_channel = np.fft.ifft2(deconvolvedFFT)
			deconvolved.append(np.abs(deconvolved_channel))

		# Combine channels back into a single image
		deconvolved = np.stack(deconvolved, axis=-1)

	else:  # Single-channel image
		# FFT for the image and kernel
		img_FFT = np.fft.fft2(self.image)
		kernel_FFT = np.fft.fft2(kernel_padded, s=self.image.shape)
		kernel_FFT_conj = np.conjugate(kernel_FFT)

		# Wiener deconvolution
		deconvolvedFFT = (kernel_FFT_conj / (np.abs(kernel_FFT) ** 2 + K)) * img_FFT
		deconvolved = np.abs(np.fft.ifft2(deconvolvedFFT))

	# Normalize the image
	deconvolved = (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved)) * 255

	return deconvolved.astype(np.uint8)

def attach_filters_to_image(image_class):
	image_class.apply_filter = apply_filter
	image_class.gray_scale = gray_scale
	image_class.equalize = equalize
	image_class.unblur = wiener_deconvolution
	image_class.pad_kernel = pad_kernel
