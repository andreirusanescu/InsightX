import sys
import cv2
import numpy as np

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
	
	def show(self, windowName="Image"):
		""" Displays the image """
		if self.image is not None:
			cv2.imshow(windowName, self.image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		else:
			print("No image to show.")
	
	def crop(self, x1, y1, x2, y2):
		""" Crop the image using the specified coordinates. """
		if self.image is not None:
			self.image = self.image[y1 : y2, x1 : x2]
			print(f"Image cropped to coordinates ({x1}, {y1}), ({x2}, {y2}).")
		else:
			print("No image to crop.")
	
	def rotate(self, angle):
		""" Rotate the image to the specified angle (in degrees). """
		if self.image is not None:
			height, width = self.image.shape[:2]

			# Center point of the image
			center = (width // 2, height // 2)

			# Rotate matrix around its center with angle degrees
			rotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
			
			# Apply the rotation matrix to the image by an affine transformation
			self.image = cv2.warpAffine(self.image, rotationMatrix, (width, height))
			print(f"Image rotated by {angle} degrees.")
		else:
			print("No image to rotate.")
	
	def flip(self, direction):
		""" Flip the image (0 = vertical, 1 = horizontal, -1 = both) """
		if self.image is not None:
			self.image = cv2.flip(self.image, direction)
			print(f"Image flipped (direction: {direction}).")
		else:
			print("No image to flip.")
	
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
				equalizedChannels = [cv2.equalizeHist(channel) for channel in channels]

				# Merge the RGB channels into the image
				self.image = cv2.merge(equalizedChannels)
			print("Image equalized.")
		else:
			print("No image to equalize.")
	
	def toGrayScale(self):
		""" Turns the image from RGB to Grayscale """
		if self.image is not None and len(self.image.shape) == 3:
			self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
			print("Image turned to gray scale")

	def applyFilter(self, filterType):
		""" Apply a filter to the image (blur, edge, sharpen, etc.) """
		if self.image is not None:
			if filterType == "BLUR":
				""" the bigger the kernerl size, the blurrier the image
					the bigger sigmaX, the blurrier the image """
				self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
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
			elif filterType == "MEDIAN_BLUR":
				self.image = cv2.medianBlur(self.image, (5, 5))
			else:
				print(f"Invalid filter type: {filterType}")
				return
			print(f"Applied {filterType} filter.")
		else:
			print("No image to apply filter to")
	
	def padKernel(self, kernel):
		""" Called in wienerDeconvolution() to return
			a kernel the size of the image """
		padded_kernel = np.zeros(self.image.shape)
		kh, kw = kernel.shape
		center_h = (self.image.shape[0] - kh) // 2
		center_w = (self.image.shape[1] - kw) // 2
		padded_kernel[center_h:center_h + kh, center_w:center_w + kw] = kernel
		return padded_kernel

	def wienerDeconvolution(self, K=0.01):
		""" Applies Wiener Deconvolution to remove blur for grayscale images
		:param K: Noise factor (controls the removing of the noise)
		F(u,v)= H*(u,v) / (|H(u,v)|^2 + K) * G(u,v)
		"""
		if len(self.image.shape) != 2:
			print("Image is not grayscale!")
			return

		img_fft = np.fft.fft2(self.image)

		kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
		kernel = self.padKernel(kernel)
		kernelFFT = np.fft.fft2(kernel, s=self.image.shape)

		# Hermitian matrix
		kernelFFTConj = np.conjugate(kernelFFT)
		deconvolvedFFT = (kernelFFTConj / (np.abs(kernelFFT) ** 2 + K)) * img_fft
		deconvolved = np.fft.ifft2(deconvolvedFFT)
		deconvolved = np.abs(deconvolved)
		
		# Normalize the image
		deconvolved = (deconvolved - np.min(deconvolved)) / (np.max(deconvolved) - np.min(deconvolved)) * 255
		self.image = deconvolved.astype(np.uint8)
		print("Unblurred the image")
	
	def adjustBrightnessContrast(self, brightness=0, contrast=1.0):
		""" Adjust brightness and contrast. Brightness (0-100), contrast (1.0-3.0) """
		if self.image is not None:
			self.image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=brightness)
			print(f"Brightness set to {brightness}, contrast set to {contrast}.")
		else:
			print("No image to adjust.")
	
	def save(self, outputFilename):
		""" Save the image to the specified filename. """
		if self.image is not None:
			cv2.imwrite(outputFilename, self.image)
			print(f"Image saved as '{outputFilename}'.")
		else:
			print("No image to save.")
	
	def blend(self):
		filename2 = input("Enter filename to be merged: ").strip()
		try:
			image2 = cv2.imread(filename2)
			if self.image is None:
				raise FileNotFoundError(f"Image '{filename2}' not found or file is not a valid image.")
			print(f"Image to be blended '{filename2}' loaded successfully. Size: {self.image.shape[1]}x{self.image.shape[0]}")
		except FileNotFoundError as e:
			print(e)
			exit(1)
		except Exception as e:
			print(f"Error loading image: {e}")
			exit(1)

		alpha, beta = map(float, input("Enter alpha (0.0, 1.0), beta(0.0, 1.0), (alpha + beta = 1.0): ").split())
		self.image = cv2.addWeighted(self.image, alpha, image2, beta, 0.0)
		print(f"Blended the two images")

def main():
	if len(sys.argv) < 2:
		print("Usage: python3 main.py <filename>")
		exit(1)

	filename = sys.argv[1]
	image = Image(filename)

	while True:
		print("\nAvailable commands:\nCROP, EQUALIZE, ROTATE, APPLY, UNBLUR, GRAYSCALE, FLIP, BRIGHTNESS, CONTRAST, SHOW, SAVE, EXIT, BLEND")
		command = input("Enter command: ").strip().upper()
		
		if command == "CROP":
			x1, y1, x2, y2 = map(int, input("Enter coordinates (x1, y1, x2, y2): ").split())
			image.crop(x1, y1, x2, y2)
		
		elif command == "EQUALIZE":
			image.equalize()
		
		elif command == "ROTATE":
			angle = int(input("Enter angle to rotate: "))
			image.rotate(angle)
		
		elif command == "APPLY":
			filter_type = input("Enter filter (BLUR, SHARPEN, EDGE): ").strip().upper()
			image.applyFilter(filter_type)
		
		elif command == "UNBLUR":
			image.wienerDeconvolution()

		elif command == "GRAYSCALE":
			image.toGrayScale()

		elif command == "BRIGHTNESS":
			brightness = int(input("Enter brightness value (-100 to 100): "))
			image.adjustBrightnessContrast(brightness=brightness)

		elif command == "CONTRAST":
			contrast = float(input("Enter contrast value (0.5 to 3.0): "))
			image.adjustBrightnessContrast(contrast=contrast)
		
		elif command == "FLIP":
			direction = int(input("Enter flip direction (0=vertical, 1=horizontal, -1=both): "))
			image.flip(direction)
		
		elif command == "SHOW":
			image.show()
		
		elif command == "SAVE":
			outputFilename = input("Enter filename to save: ")
			image.save(outputFilename)
		
		elif command == "EXIT":
			print("Exiting the application.")
			break

		elif command == "BLEND":
			image.blend()

		else:
			print("Invalid command. Please try again.")


if __name__ == "__main__":
	main()
