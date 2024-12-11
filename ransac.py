import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# Ensures no GUI required for Qt, may cause errors
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def extract_SIFT(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.SIFT_create()
	kp, desc = sift.detectAndCompute(img_gray, None)
	kp = np.array([p.pt for p in kp]).T
	return kp, desc

def match_SIFT(descriptor_source, descriptor_target):
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(descriptor_source, descriptor_target, k=2)
	pos = np.array([], dtype=np.int32).reshape((0, 2))
	for m, n in matches:
		if m.distance < 0.8 * n.distance:
			pos = np.vstack((pos, [m.queryIdx, m.trainIdx]))
	return pos

def estimate_affine(s, t):
	num = s.shape[1]
	M = np.zeros((2 * num, 6))
	for i in range(num):
		M[2 * i] = [s[0, i], s[1, i], 0, 0, 1, 0]
		M[2 * i + 1] = [0, 0, s[0, i], s[1, i], 0, 1]
	b = t.T.reshape((2 * num, 1))
	theta, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
	X = theta[:4].reshape((2, 2))
	Y = theta[4:].reshape((2, 1))
	return X, Y

def residual_lengths(X, Y, s, t):
	e = np.dot(X, s) + Y
	residual = np.sqrt(np.sum((e - t) ** 2, axis=0))
	return residual

def ransac_fit(pts_s, pts_t, k=3, threshold=1, iter_num=2000):
	max_inliers_num = 0
	best_A, best_t = None, None
	best_inliers = None

	for i in range(iter_num):
		idx = np.random.choice(pts_s.shape[1], k, replace=False)
		A_tmp, t_tmp = estimate_affine(pts_s[:, idx], pts_t[:, idx])
		residual = residual_lengths(A_tmp, t_tmp, pts_s, pts_t)
		inliers = np.where(residual < threshold)[0]
		inliers_num = len(inliers)
		if inliers_num > max_inliers_num:
			max_inliers_num = inliers_num
			best_A, best_t = A_tmp, t_tmp
			best_inliers = inliers

	return best_A, best_t, best_inliers

def affine_matrix(s, t, pos):
	s = s[:, pos[:, 0]]
	t = t[:, pos[:, 1]]
	_, _, inliers = ransac_fit(s, t)
	s = s[:, inliers]
	t = t[:, inliers]
	A, t = estimate_affine(s, t)
	M = np.hstack((A, t))
	return M

def ransac(self, img1, img2, output_file):
	keypoint_source, descriptor_source = extract_SIFT(img1)
	keypoint_target, descriptor_target = extract_SIFT(img2)
	
	pos = match_SIFT(descriptor_source, descriptor_target)
	
	H = affine_matrix(keypoint_source, keypoint_target, pos)
	rows, cols, _ = img2.shape
	
	warp = cv2.warpAffine(img1, H, (cols, rows))
	merge = np.uint8(img2 * 0.5 + warp * 0.5)
	
	plt.figure(figsize=(15, 7))
	plt.imshow(merge)
	plt.axis('off')
	plt.savefig(output_file)


def attach_ransac_to_image(image_class):
	image_class.ransac = ransac
