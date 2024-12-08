import cv2

def show(self, window_name="Image"):
	"""
	Displays the image using OpenCV in a new window.
	:param windowName: The name of the window in which the image will be displayed.
	"""
	try:
		if self.image is not None:
			cv2.imshow(window_name, self.image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		else:
			print("No image to show.")
	except Exception as e:
		print(f"Error displaying image: {e}")

def save(self, output_filename):
	""" Save the image to the specified filename. """
	if self.image is not None:

		cv2.imwrite(output_filename, self.image)
		print(f"Image saved as '{output_filename}'.")
	else:
		print("No image to save.")

def attach_utils_to_image(image_class):
	"""
	Attaches the methods to the given image class.
	:param image_class: The class to which the methods will be attached.
	"""
	image_class.show = show
	image_class.save = save
