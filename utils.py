import numpy as np
import torch
import cv2

RAW_IMG_SIZE=228
MODEL_INPUT_SIZE=128
MODEL_OUTPUT_SIZE=32
HEATMAP_SIGMA=0.8
N_JOINTS=21

def projectPoints(xyz, K):
    """
    Projects 3D coordinates into image space.
    Function taken from https://github.com/lmb-freiburg/freihand
    """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def gen_heatmap(img, pt, sigma):
	"""
	from Minimal-Hand model

    generate heatmap based on pt coord

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """
	#pt = pt.astype(np.int32)

    # Check that any part of the gaussian is in-bounds

	vector = np.vectorize(np.int32)
	ul = [(pt[0] - 3 * sigma), (pt[1] - 3 * sigma)]
	ul = vector(ul)
	br = [(pt[0] + 3 * sigma + 1), (pt[1] + 3 * sigma + 1)]
	br = vector(br)
	"""
	if (
	        ul[0] >= img.shape[1]
	        or ul[1] >= img.shape[0]
	        or br[0] < 0
	        or br[1] < 0 ):
	    # If not, just return the image as is
	    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	    return img, 0
	"""

	# Generate gaussian
	size = 6 * sigma + 1
	x = np.arange(0, size, 1, float)
	y = x[:, np.newaxis]
	x0 = y0 = size // 2
	g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
	# Usable gaussian range
	g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
	g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
	# Image range
	img_x = max(0, ul[0]), min(br[0], img.shape[1])
	img_y = max(0, ul[1]), min(br[1], img.shape[0])

	img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
	return img, 1

def vector_to_heatmaps(keypoints):
    """
    Creates 2D heatmaps from keypoint locations for a single image
    Input: array of size N_JOINTS x 2
    Output: array of size N_JOINTS x MODEL_IMG_SIZE x MODEL_IMG_SIZE
    """
    heatmaps = np.zeros([N_JOINTS, MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * MODEL_OUTPUT_SIZE), int(y * MODEL_OUTPUT_SIZE)
        if (0 <= x < MODEL_OUTPUT_SIZE) and (0 <= y < MODEL_OUTPUT_SIZE):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


def blur_heatmaps(heatmaps):
    """Blurs heatmaps using GaussinaBlur of defined size"""
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / heatmaps_blurred[k].max()
    return heatmaps_blurred