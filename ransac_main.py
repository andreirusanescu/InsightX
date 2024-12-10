import numpy as np
import cv2

# RANSAC PARAMETERS

K = 3 	# Neccessary points to estimate a transformation
th = 1	# Maximum threshold for error to consider a point as an inlier

# Can be modified, the bigger the more precise
ITER_NUM = 2000

def residual_lengths(X, Y, source, target):
	"""
	Determine the errors present in the model make sure
	the affine matrices that we generate, or the descriptors
	that are matched, have as few errors as possible
	:param X: Rotation matrix
	:param Y: Translation vector
	:param source: Source Points (coords)
	:param target: Target Points (coords)
	"""

	# Apply the transformation to the source points
	e = np.dot(X, source) + Y
	diff_square = np.power(e - target, 2)
	residual = np.sqrt(np.sum(diff_square, axis=0))
	return residual

def ransac_fit(pts_source, pts_target):
	"""
	Applies RANSAC to find the best transformation
	matrix between the source points and the target points
	"""
	# maximum number of inliers found
	inliers_num = 0

	# transformation matrix
	A = None

	# translation vector
	t = None

	# Positions of inliers
	inliers = None
	for i in range(ITER_NUM):
		# indexes are generated randomly
		idx = np.random.randint(0, pts_source.shape[1], (K, 1))

		# estimate transformation matrix and translation vector
		A_tmp, t_tmp = estimate_affine(pts_source[:, idx], pts_target[:, idx])

		# estimate errors
		residual = residual_lengths(A_tmp, t_tmp, pts_source, pts_target)
		if not(residual is None):
			inliers_tmp = np.where(residual < th)
			inliers_num_tmp = len(inliers_tmp[0])
			if inliers_num_tmp > inliers_num:
				inliers_num = inliers_num_tmp
				inliers = inliers_tmp
				A = A_tmp
				t = t_tmp

	return A, t, inliers


def extract_SIFT(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT_create()
	kp, desc = sift.detectAndCompute(img_gray, None)
	kp = np.array([p.pt for p in kp]).T
	return kp, desc

def match_SIFT(descriptor_source, descriptor_target):
	# obtains the best two matches among all the matched descriptors
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(descriptor_source, descriptor_target, k=2)
	pos = np.array([], dtype=np.int32).reshape((0, 2))
	matches_num = len(matches)
	for i in range(matches_num):
		# consider only the points that have a ratio <= 0.8 (D. Lowe)
		if matches[i][0].distance <= 0.8 * matches[i][1].distance:
			# trainIdx returns the index of the descriptor in source, 
			# queryIdx returns the index of the descriptor in target.
			temp = np.array([matches[i][0].queryIdx, matches[i][0].trainIdx])
			# These are the actual positions that are stacked vertically
			pos = np.vstack((pos, temp))
	return pos

def affine_matrix(s, t, pos):
	# store all the key points in the s and 
	# t variables, based on the best descriptor 
	# positions stored in the pos variable.
	s = s[:, pos[:, 0]]
	t = t[:, pos[:, 1]]

	# Inliers are the points in the two images 
	# that show maximum similarity, used to draw RANSAC models

	_, _, inliers = ransac_fit(s, t)
	s = s[:, inliers[0]]
	t = t[:, inliers[0]]

	for n in inliers:
		print(f"{n}",end=" ")

	A, t = estimate_affine(s=s, t=t)
	# Homography  matrix
	H = np.hstack((A, t))
	return H

""" Used for applying to rotation, translation"""

def estimate_affine(s, t):
	"""
	Computes the affine transformation matrix that
	maps the source points to the target points
	"""
	num = s.shape[1]
	M = np.zeros((2 * num, 6))
	for i in range(num):
		temp = [[s[0, i], s[1, i], 0, 0, 1, 0], [0, 0, s[0, i], s[1, i], 0, 1]]
		M[2 * i : 2 * i + 2, :] = np.array(temp)
	b = t.T.reshape((2 * num, 1))

	# M * theta = b
	theta = np.linalg.lstsq(M, b)[0]

	# Rotation matrix
	X = theta[:4].reshape((2, 2))

	# Translation vector
	Y = theta[4:]
	return X, Y

def ransac_main(self):
	filename1 = input("Enter first filename: ").strip()
	filename2 = input("Enter second filename: ").strip()

	""" Source image is the image we want
	to register over the target image """
	src_image = cv2.imread(filename=filename1)
	target_image = cv2.imread(filename=filename2)

	# Extract key points and their related descriptors
	keypoint_source, descriptor_source = extract_SIFT(src_image)
	keypoint_target, descriptor_target = extract_SIFT(target_image)

	# Obtain the position of all the points found in the previous step 
	pos = match_SIFT(descriptor_source, descriptor_target)

	# Get the Homography matrix (used in image registration)
	H = affine_matrix(keypoint_source, keypoint_target, pos)

	rows, cols, _ = src_image.shape
	warp = cv2.warpAffine(src_image, H, (cols, rows))

	# blending the images
	merge = np.uint8(target_image * 0.5 + warp * 0.5)

	cv2.imshow('img', merge)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def attach_ransac_to_image(image_class):
	image_class.ransac = ransac_main

