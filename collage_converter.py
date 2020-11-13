# coding:utf-8

"""
コラージュ
"""

import re
import sys
import os
import numpy as np
import cv2
import mbiocv2 as mb

"""
# pylint: disable=C0111
# ↑プログラムの説明ドキュメントがないよ！というエラーの防止
"""
# pylint: disable=E1121
# ↑Too many positional arguments for method call を防止

MIN_MATCH_COUNT = 10
RATIO = 0.5

#対応拡張子リスト
SPRT_EXT_LIST = ["png", "jp*", "bmp"]


class PATH_FETCH_FAILED(Exception):
	"""
	パス取得失敗
	"""
	def __str__(self):
		return "パス取得失敗"

class FILENAME_DUPLICATE(Exception):
	"""
	ファイル名重複
	"""
	def __str__(self):
		return "ファイル名重複"

class ORIG_DUPLICATE(Exception):
	"""
	orig重複
	"""
	def __str__(self):
		return "orig重複"

class ORIG_NOTHING(Exception):
	"""
	origなし
	"""
	def __str__(self):
		return "origなし"

def get_filename_list(dirname):
	"""
		ファイル名リストを取得する。

		Returns
		-------
		filename_list : list
			ファイル名リスト

		Raises
		------
		PATH_FETCH_FAILED
			パス取得失敗
	"""

	# 対応拡張子リストをスキャンの為に変形
	condition = [".*"+ f for f in SPRT_EXT_LIST]
	#print(f"[debug]{'|'.join(condition)}")

	# ディレクトリから種類を問わずにファイルを検索しリスト化
	law_filename_list = os.listdir(dirname)
	filename_list = [f for f in law_filename_list if os.path.isfile(os.path.join(dirname, f))]
	#print(f"[debug]{filename_list}")

	# 対応拡張子のみのリストへ加工
	filename_list = [f for f in filename_list if re.match("|".join(condition), f)]
	#print(f"[debug]{filename_list}")

	return filename_list


def check_filename_list(filename_list):
	"""
		ファイル名リストをチェックし
		様々なエラーをあぶりだす。

		Parameters
		----------
		filename_list : list
			ファイル名リスト

		Raises
		------
		ORIG_NOTHING
			origなし
		ORIG_DUPLICATE
			orig重複
		FILENAME_DUPLICATE
			ファイル名重複
			ファイル名重複は処理ミスを引き起こす。
	"""

	filename_list_no_ext = [os.path.splitext(f)[0] for f in filename_list]
	print(filename_list_no_ext)



	if not "orig" in filename_list_no_ext:
		raise ORIG_NOTHING

	elif filename_list_no_ext.count("orig") >1:
		raise ORIG_DUPLICATE

	elif len(filename_list_no_ext) != len(set(filename_list_no_ext)):
		raise FILENAME_DUPLICATE


def ex_img(img, multiple = 2):
	"""
		表示用の余白あり画像の作成

		Parameters
		----------
		img : OpemCV image array
			OpenCVの画像配列

		multiple : int[倍], オプション
			何倍の余白とするかの設定
			デフォルトでは2

		Returns
		-------
		result : OpemCV image array
			OpenCVの画像配列
	"""

	h, w, ch = img.shape

	margin_h = int( (multiple - 1) * 0.5 * h )
	margin_w = int( (multiple - 1) * 0.5 * w )

	left   = margin_w
	right  = margin_w + w
	top    = margin_h
	bottom = margin_h + h

	ex_h, ex_w = int(h*multiple), int(w*multiple)

	result = np.zeros( (ex_h, ex_w, ch), dtype=np.uint8 )
	result[top:bottom, left:right, :] = img

	return result


def trim_img(img, calc):
	"""
		trim_calcで計算されたパロメータによってトリミング実行

		Parameters
		----------
		img : OpemCV image array
			OpenCVの画像配列

		calc : [type]
			[description]

		Returns
		-------
		result : OpemCV image array
			トリミングされた OpenCVの画像配列
	"""
	return img[calc[0]:calc[1], calc[2]:calc[3], :]


def trim_calc(img):
	"""
		トリミング計算

		Parameters
		----------
		img : OpemCV image array
			OpenCVの画像配列

		Returns
		-------
		list
			上下左右のトリミング位置
	"""
	for i in range(img.shape[0]):
		if(
			( np.max(img[i,:,:]) != 0 ) and
			( np.min(img[i,:,:]) != 255 )
		):
			top = i
			break

	for i in reversed( range(img.shape[0]) ):
		if(
			( np.max(img[i,:,:]) != 0 ) and
			( np.min(img[i,:,:]) != 255 )
		):
			bottom = i
			break

	for i in range(img.shape[1]):
		if(
			( np.max(img[:,i,:]) != 0 ) and
			( np.min(img[:,i,:]) != 255 )
		):
			left = i
			break

	for i in reversed( range(img.shape[1]) ):
		if(
			( np.max(img[:,i,:]) != 0 ) and
			( np.min(img[:,i,:]) != 255 )
		):
			right = i
			break
	return top, bottom, left, right


def ct_formater():
	"""
		検出器、Matcherの生成を一度に行う関数。
		ココの中身を変えれば、別の物へと一括で変更可能

		Returns
		-------
		ctf : list
			検出器、Matcherをリストにして返す
	"""
	# A-KAZE検出器の生成
	akaze = cv2.AKAZE_create()
	# Brute-Force Matcher生成
	bf = cv2.BFMatcher()

	return akaze, bf


def collage_transformer(file_name, ctf, orig):
	"""
		変形を行う

		Parameters
		----------
		file_name : char
			変形対象のファイル名
		ctf : list
			検出器とMatcher

		orig : OpemCV image array
			オリジナル画像のOpenCVの画像配列
		Returns
		-------
		[type]
			[description]
	"""
	collage = mb.imread(file_name)
	if collage is None:
		print("Warning：[{}]の読み込み失敗".format(file_name) )
		return 1


	print("[{}]の読み込み成功".format(file_name) )


	collage = trim_img(collage,trim_calc(collage) )


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

	if len(good) < MIN_MATCH_COUNT:
		print("Warning：[{}]は対応点不足".format(file_name) )
		return 1

	orig_pts    = np.float32( [ kp_orig   [m.queryIdx].pt for m in good ] ).reshape(-1,1,2)
	collage_pts = np.float32( [ kp_collage[m.trainIdx].pt for m in good ] ).reshape(-1,1,2)

	M = cv2.findHomography(orig_pts, collage_pts, cv2.RANSAC,5.0)[0]

	h,w = orig.shape[:2]

	converted_collage = cv2.warpPerspective(collage, np.linalg.inv(M), (w, h) )


	print("[{}]の変形完了".format(file_name) )
	file_name = os.path.basename(file_name)
	print(f"{file_name}:{np.linalg.det(M):1.5}倍？") #現状仮設。後で追加機能含め本実装しよう。
	return file_name, converted_collage


def main():
	"""
	main
	"""
	dirname = os.path.dirname(sys.argv[1] )
	print(dirname)


	try:
		filename_list = get_filename_list(dirname)
		check_filename_list(filename_list)
	except PATH_FETCH_FAILED as e:
		input(e)
		return 1
	except FILENAME_DUPLICATE as e:
		input(e)
		return 1
	except ORIG_DUPLICATE as e:
		input(e)
		return 1
	except ORIG_NOTHING as e:
		input(e)
		return 1


	orig_file     = [i for i in filename_list if     'orig' in i]
	filename_list = [i for i in filename_list if not 'orig' in i]


	# 合成元画像
	orig = ex_img( mb.imread(os.path.join(dirname, "".join(orig_file) ) ),3 )
	if orig is None:# 原画の読み込みに失敗したならば中止
		print("Error：原画の読み込みに失敗")
		return 1
	else:
		print("原画の読み込みに成功")

	# 成功件数カウント用
	success_count = 0

	ctf = ct_formater()
	collages = []
	len_fn_list = len(filename_list)

	# 処理開始
	# 1から始まるインデックス付きでfilename_listを走査
	for index, file_name in enumerate(filename_list,1):
		# 部品画像
		print(f"file_nameは[{file_name}]")
		file_name = "".join(file_name)

		print("[{}]の変形開始{:^5}/{:^5}".format(file_name, index, len_fn_list ) )
		result = collage_transformer(
			os.path.join(dirname, file_name ),
			ctf, orig
		)
		if result != 1:
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
	dirname += "\\transformed\\"
	os.makedirs(dirname, exist_ok=True)
	mb.imwrite( os.path.join(dirname,"orig.png"), trim_img(orig,trim) )

	# 1から始まるインデックス付きでfilename_listを走査
	for index, img in enumerate(collages,1):
		tmp_file_name = dirname+ os.path.splitext(img[0])[0] +".png"
		mb.imwrite(
			tmp_file_name,
			trim_img(img[1],trim)
		)
		print("[{}]の書き込み完了 {:^5}/{:^5}".format(tmp_file_name, index, success_count ) )

	input("全書き込み終了")


if __name__ == "__main__":
	sys.exit(main())
