import cv2

def crop(self, x1, y1, x2, y2):
	""" Crop the image using the specified coordinates """
	if self.image is not None:
		self.image = self.image[y1 : y2, x1 : x2]
		print(f"Image cropped to coordinates ({x1}, {y1}), ({x2}, {y2})")
	else:
		print("No image to crop")
	
def rotate(self, angle):
	""" Rotate the image to the specified angle (in degrees) """
	if self.image is not None:
		height, width = self.image.shape[:2]

		# Center point of the image
		center = (width // 2, height // 2)

		# Rotate matrix around its center with angle degrees
		rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

		# Apply the rotation matrix to the image by an affine transformation
		rotated_image = cv2.warpAffine(self.image, rotation_matrix, (width, height))
		return rotated_image

def flip(self, direction):
	""" Flip the image (0 = vertical, 1 = horizontal) """
	if self.image is not None:
		self.image = cv2.flip(self.image, direction)
		print(f"Image flipped (direction: {direction})")
	else:
		print("No image to flip")
	
def resize(self):
	""" Resizes the image to given input sizes """
	height, width = map(int, input("Enter new sizes for the image (height, width): "))
	if height < 0 or width < 0:
		print("Height and width should be positive")
	image2 = cv2.resize(image2, (width, height))
	print(f"Image resized to {self.image.shape[1]}x{self.image.shape[0]}")

def adjust_brightnes_contrast(self, brightness=0, contrast=1.0):
	"""
	Adjust :param brightness: and :param contrast:
	Brightness (0-100), contrast (1.0-3.0)
	"""
	if self.image is not None:
		self.image = cv2.convertScaleAbs(self.image, alpha=contrast, beta=brightness)
		print(f"Brightness set to {brightness}, contrast set to {contrast}")
	else:
		print("No image to adjust")

def attach_adjust_to_image(image_class):
	"""
	Attaches the methods to the given image class
	:param image_class: The class to which the methods will be attached
	"""
	image_class.resize = resize
	image_class.flip = flip
	image_class.rotate = rotate
	image_class.crop = crop
	image_class.adjust_brightnes_contrast = adjust_brightnes_contrast
