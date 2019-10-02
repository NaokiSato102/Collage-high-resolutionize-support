
'''
本プログラムはコラ画像を高画質な元画像と合成するためにコラ画像を回転、縮小するプログラムである。
コラ画像が元画像の大きさから漏れることは現段階では想定されていないので注意。書いてて気が付いたが対応すべきだよね。

【opencv 基礎知識 #4】動画の手ぶれ補正をpython実装 (AKAZE, KNN, RANSAC) - MotoJapan's Tech-Memo
http://motojapan.hateblo.jp/entry/2017/08/17/081917
を発見することにより、ほかの記事をキーワード検索しやすくなった。

画像から特徴量を抽出し、透視変換行列を導出して画像を変形する - Qiita
https://qiita.com/ka10ryu1/items/bd05aed321a7a154d8a1
を基に記述が抜けている部分を保管したがうまくいかなかった。

試行錯誤のうちに判明したが、画像1と画像2、どちらを基準に処理しているかをしっかりと理解できずにいたために処理がおかしくなっていた。
とくに、画像の行列による変形がおかしくなっていた。

逆行列にして処理するとうまくいった。
その後、理解しやすくするため変数名を変更した。さらに理解が深まった。
'''

import numpy as np
import cv2

MIN_MATCH_COUNT = 10
RATIO = 0.5

# 画像1
orig = cv2.imread("orig.png")
# 画像2
collage = cv2.imread("collage.png")

# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()

# 特徴量の検出と特徴量ベクトルの計算
kp_orig, des_orig = akaze.detectAndCompute(orig, None)
kp_collage, des_collage = akaze.detectAndCompute(collage, None)

# Brute-Force Matcher生成
bf = cv2.BFMatcher()

# 特徴量ベクトル同士をBrute-Force&KNNでマッチング
matches = bf.knnMatch(des_orig, des_collage, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
	if m.distance < RATIO*n.distance:
		good.append(m)

if len(good)>=MIN_MATCH_COUNT:
	orig_pts = np.float32([ kp_orig[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	collage_pts = np.float32([ kp_collage[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(orig_pts, collage_pts, cv2.RANSAC,5.0)

	h,w = orig.shape[:2]

	converted_collage = cv2.warpPerspective(collage, np.linalg.inv(M), (w, h) )

	cv2.imwrite('converted_collage.png', converted_collage)

else:
	print("not enough matching key point")
