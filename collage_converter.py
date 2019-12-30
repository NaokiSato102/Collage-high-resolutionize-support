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

	ex_img = np.zeros( (ex_h, ex_w, ch), dtype=np.uint8 )
	ex_img[top:bottom, left:right, :] = img

	return ex_img


def trim_img(img, calc):# トリミング実行
	return img[calc[0]:calc[1], calc[2]:calc[3], :]


def trim_calc(img):# トリミング計算
	for i in range(img.shape[0]):
		if(
			(  0 != np.max(img[i,:,:]) ) and
			(255 != np.min(img[i,:,:]) )
		):
			top = i
			break

	for i in reversed( range(img.shape[0]) ):
		if( 
			(  0 != np.max(img[i,:,:]) ) and
			(255 != np.min(img[i,:,:]) ) 
		):
			bottom = i
			break

	for i in range(img.shape[1]):
		if( 
			(  0 != np.max(img[:,i,:]) ) and
			(255 != np.min(img[:,i,:]) )
		):
			left = i
			break

	for i in reversed( range(img.shape[1]) ):
		if(
			(  0 != np.max(img[:,i,:]) ) and
			(255 != np.min(img[:,i,:]) )
		):
			right = i
			break
	return top, bottom, left, right


def ct_formater():
	# A-KAZE検出器の生成
	akaze = cv2.AKAZE_create()
	# Brute-Force Matcher生成
	bf = cv2.BFMatcher()

	return akaze, bf

 
def collage_transformer(file_name, ctf, orig, trim_option="n"):

	collage = cv2.imread(file_name)
	if (collage is None):
		print("Warning：[{}]の読み込み失敗".format(file_name) )
		return 1


	print("[{}]の読み込み成功".format(file_name) )
	
	#if   (trim_option == 'y'):
	collage = trim_img(collage,trim_calc(collage) )
	#elif (trim_option == 'n'):
	#	pass
	
	# 特徴量の検出と特徴量ベクトルの計算
	kp_orig,    des_orig    = ctf[0].detectAndCompute(orig,    None)
	kp_collage, des_collage = ctf[0].detectAndCompute(collage, None)

	# 特徴量ベクトル同士をBrute-Force&KNNでマッチング
	matches = ctf[1].knnMatch(des_orig, des_collage, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
		if m.distance < RATIO * n.distance:
			good.append(m)

	if ( len(good) < MIN_MATCH_COUNT ):
		print("Warning：[{}]は対応点不足".format(file_name) )
		return 1

	orig_pts    = np.float32( [ kp_orig   [m.queryIdx].pt for m in good ] ).reshape(-1,1,2)
	collage_pts = np.float32( [ kp_collage[m.trainIdx].pt for m in good ] ).reshape(-1,1,2)

	M, mask = cv2.findHomography(orig_pts, collage_pts, cv2.RANSAC,5.0)

	h,w = orig.shape[:2]

	converted_collage = cv2.warpPerspective(collage, np.linalg.inv(M), (w, h) )

	
	print("[{}]の変形完了".format(file_name) )
	return file_name, converted_collage


def main():

	file_name_list = []
	for i in SPRT_EXT_LIST:
		file_name_list += glob.glob("*."+i) # 拡張子リストに載っている拡張子を条件として検索しリスト化


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

	# 成功件数カウント用
	success_count = 0

	ctf = ct_formater()
	collages = []
	len_fn_list = len(file_name_list)

	# 処理開始
	for index, file_name in enumerate(file_name_list,1):# 1から始まるインデックス付きでfile_name_listを走査
		# 部品画像
		file_name = "".join(file_name)

		print("[{}]の変形開始{:^5}/{:^5}".format(file_name, index, len_fn_list ) )
		result = collage_transformer(file_name, ctf, orig)
		if( 1!= result ):
			success_count += 1
			collages.append(result)
			print("[{}]の変形成功 全体{:^5}/{:^5} 成功{:^5}/{:^5}".format(
					file_name, 
					index, len_fn_list, 
					success_count, len_fn_list 
				)
			)
			
		else:
			print("[{}]の変形失敗 全体{:^5}/{:^5}".format(file_name, index, len_fn_list) )

	print("全行程終了  成功{:^5}/{:^5}".format(success_count, len_fn_list ) )

	# トリミング計算用全合成画像作成
	tmp_image = orig
	for i in collages:
		tmp_image = cv2.addWeighted(tmp_image, 0.5, i[1], 0.5, 0)
	
	# トリミングの実行
	trim = trim_calc(tmp_image)


	print("書き込み開始")

	cv2.imwrite( "converted_orig.png",trim_img(orig,trim) )

	for index, img in enumerate(collages,1):# 1から始まるインデックス付きでfile_name_listを走査
		tmp_file_name = ("converted_{}.png".format( img[0].rsplit(".",1 )[0] ) )
		cv2.imwrite(
			tmp_file_name,
			trim_img(img[1],trim)
		)
		print("[{}]の書き込み完了 {:^5}/{:^5}".format(tmp_file_name, index, success_count ) )
	
	print("全書き込み終了")


if __name__ == "__main__":
	sys.exit(main())
