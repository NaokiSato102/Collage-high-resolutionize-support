import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

# 画像1
img2 = cv2.imread("img1.png")
# 画像2
img1 = cv2.imread("img2.png")

# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()

# 特徴量の検出と特徴量ベクトルの計算
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)


# Brute-Force Matcher生成
bf = cv2.BFMatcher()

# 特徴量ベクトル同士をBrute-Force&KNNでマッチング
matches = bf.knnMatch(des1, des2, k=2)


# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
	if m.distance < 0.7*n.distance:
		good.append(m)

if len(good)>MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()
	
	h,w = img1.shape[:2]

	img4 = cv2.warpPerspective(img1, M, (w, h) )
	cv2.imwrite('img4.png', img4)
else:
	print ('hoge')
	matchesMask = None



