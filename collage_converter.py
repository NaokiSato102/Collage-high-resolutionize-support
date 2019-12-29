import glob
import sys
import numpy as np
import cv2

MIN_MATCH_COUNT = 10
RATIO = 0.5
SPRT_EXT_LIST = ["png","jp*","bmp"]

def ex_img(img,multiple = 2):
	#===========================================================
	# 表示用の余白あり画像の作成
	# multiple : 何倍の大きさの画像にするか
	#===========================================================

	h, w, ch = img.shape

	margin_h = int( (multiple - 1)*0.5*h )
	margin_w = int( (multiple - 1)*0.5*w )

	left   = margin_w
	right  = margin_w + w
	top    = margin_h
	bottom = margin_h + h

	ex_h, ex_w = int(h*multiple), int(w*multiple)

	ex_img = np.ones( (ex_h, ex_w, ch), dtype=np.uint8 )
	ex_img[top:bottom, left:right, :] = img

	return ex_img

def trim_img(img):# トリミング実行
	# print("[debug]縦の開始点走査開始")
	for i in range(img.shape[0]):
		# print("[debug]第{}層".format(i) )
		# print( img[i,:,:] ) 

		j = np.max(img[i,:,:])
		# print("[debug]最大値は{}\n".format(j) )
		if(j!=0):
			# print("[debug]第{}層から開始\n".format(i) )
			top = i
			break

	# print("[debug]縦の終了点走査開始")
	for i in reversed( range(img.shape[0]) ):
		# print("[debug]第{}層".format(i) )
		# print( img[i,:,:] ) 

		j = np.max(img[i,:,:])
		# print("[debug]最大値は{}\n".format(j) )
		if(j!=0):
			# print("[debug]第{}層で終了\n".format(i) )
			bottom = i
			break
	
	# print("[debug]左から横の開始点走査開始")
	for i in range(img.shape[1]):
		# print("[debug]第{}層".format(i) )
		# print( img[:,i,:] ) 

		j = np.max(img[:,i,:])
		# print("[debug]最大値は{}\n".format(j) )
		if(j!=0):
			# print("[debug]左から第{}層から開始\n".format(i) )
			left = i
			break

	# print("[debug]左から横の終了点走査開始")
	for i in reversed( range(img.shape[1]) ):
		# print("[debug]第{}層".format(i) )
		# print( img[:,i,:] ) 

		j = np.max(img[:,i,:])
		# print("[debug]最大値は{}\n".format(j) )
		if(j!=0):
			# print("[debug]縦の第{}層で終了\n".format(i) )
			right = i
			break
	return img[top:bottom, left:right, :]

def collages_transform(file_name,akaze,bf,orig):
	collages = cv2.imread(file_name)
	if (not collages is None):
		print("[{}]の読み込み成功".format(file_name) )
		# 特徴量の検出と特徴量ベクトルの計算
		kp_orig, des_orig = akaze.detectAndCompute(orig, None)
		kp_collages, des_collages = akaze.detectAndCompute(collages, None)

		# 特徴量ベクトル同士をBrute-Force&KNNでマッチング
		matches = bf.knnMatch(des_orig, des_collages, k=2)

		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
			if m.distance < RATIO * n.distance:
				good.append(m)

		if ( len(good) >= MIN_MATCH_COUNT ):
			orig_pts     = np.float32( [ kp_orig    [m.queryIdx].pt for m in good ] ).reshape(-1,1,2)
			collages_pts = np.float32( [ kp_collages[m.trainIdx].pt for m in good ] ).reshape(-1,1,2)

			M, mask = cv2.findHomography(orig_pts, collages_pts, cv2.RANSAC,5.0)

			h,w = orig.shape[:2]

			converted_collages = cv2.warpPerspective(collages, np.linalg.inv(M), (w, h) )

			cv2.imwrite('converted_'+file_name, converted_collages)
			print("[{}]の変形完了".format(file_name) )

		else:
			print("Warning：[{}]は対応点不足".format(file_name) )
		
	else:
		print("Warning：[{}]の読み込み失敗".format(file_name) )


def main():

	file_name_list = []
	for i in SPRT_EXT_LIST:
		file_name_list += glob.glob("*."+i)


	if  ( 1 < len( [i for i in file_name_list if 'orig' in i] )  ):
		print("Error：origが複数あります")
		return 1
	else:
		orig_file      = [i for i in file_name_list if     'orig' in i]
		file_name_list = [i for i in file_name_list if not 'orig' in i]


	# 合成元画像
	orig = ex_img( cv2.imread("".join(orig_file) ),3 )
	if (orig is None):# 原画の読み込みに失敗したならば中止
		print("Error：原画の読み込みに失敗")
		return 1 
	else:
		print("原画の読み込みに成功")
		cv2.imwrite("converted_orig.png",orig)# 原画の読み込みに成功したならば余白付き画像生成

	# A-KAZE検出器の生成
	akaze = cv2.AKAZE_create()
	# Brute-Force Matcher生成
	bf = cv2.BFMatcher()

	# 成功件数カウント用
	success_count = 0

	# 処理開始
	for index, file_name in enumerate(file_name_list,1):# 1から始まるインデックス付きでfile_name_listを走査
		# 部品画像
		file_name = "".join(file_name)

		#print("{:>5}枚目[{}]の処理".format(index, file_name))

		collages_transform(file_name,akaze,bf,orig)



if __name__ == "__main__":
	sys.exit(main())
