import os
# Ensures no GUI required for Qt, may cause errors
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import cv2
import matplotlib.pyplot as plt

def extract_sift_features(img):
	""" Extracts SIFT features from the given image """
	# SIFT algorithm is stored in the sift_initialize variable
	sift_initialize = cv2.SIFT_create()
	key_points, descriptors = sift_initialize.detectAndCompute(img, None)
	return key_points, descriptors

def showing_sift_features(img1, img2, key_points):
	""" Draws the keypoints present in the image """
	plt.imshow(cv2.drawKeypoints(img1, key_points, img2.copy()))

def sift(self):
	x = input("Enter First Image Name: ").strip()
	image1 = cv2.imread(x)
	if image1 is None:
		raise FileNotFoundError(f"Image '{x}' not found.")

	y = input("Enter Second Image Name: ").strip()
	image2 = cv2.imread(y)
	if image2 is None:
		raise FileNotFoundError(f"Image '{y}' not found.")

	h1, w1 = image1.shape[:2]
	h2, w2 = image2.shape[:2]

	if h1 * w1 > h2 * w2:
		image1 = cv2.resize(image1, (w2, h2))
	else:
		image2 = cv2.resize(image2, (w1, h1)) 

	# Convert images to grayscale for SIFT
	image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
	image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

	# Extract key points and descriptors
	image1_key_points, image1_descriptors = extract_sift_features(image1_gray)
	image2_key_points, image2_descriptors = extract_sift_features(image2_gray)

	showing_sift_features(image1_gray, image1, image1_key_points)

	""" Match descriptors using the norm.
		This could be problematic because it finds the most common pixels,
		even if the images are completely different (could be the corners/borders)
		that are similar.
		Find distance between key points using the Manhattan distance (i.e. norm).
	"""
	bruteForce = cv2.BFMatcher(cv2.NORM_L2)

	# Matches descriptors
	matches = bruteForce.match(image1_descriptors, image2_descriptors)

	# Sort matches by Manhattan distance
	matches = sorted(matches, key=lambda match: match.distance)

	# Save only the first 100 matches
	matched_img = cv2.drawMatches(image1, image1_key_points, 
								image2, image2_key_points, 
								matches[:100], image2.copy())

	plt.figure(figsize=(30, 15))
	plt.imshow(matched_img)
	out_filename = input("Enter output filename: ").strip()
	plt.savefig(out_filename)

def attach_sift_to_image(image_class):
	"""
	Attaches the method to the given image class.
	:param image_class: The class to which the method will be attached.
	"""
	image_class.sift = sift
