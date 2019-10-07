'''
https://tat-pytone.hatenablog.com/entry/2019/03/20/195949
を参考に
'''

import cv2 #OpenCVのインポート
import numpy as np #numpyをnpという名前でインポート

def conv_image1(img):
	img = cv2.erode( img, np.ones( (3,3),np.uint8 ), iterations = 3 )
	img = cv2.GaussianBlur(img,(13,13),10,10)
	return img

def conv_image2(img):
	img = cv2.erode( img, np.ones( (3,3),np.uint8 ), iterations = 3 )
	img = cv2.GaussianBlur(img,(13,13),10,10)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	return img

#img1,img2の各要素を比較し、等しければ0、異なれば1の値を持つ配列comparisonを生成。どこまでを等しいとするかの閾値は関数によって異なる。
def diff_compa1(img1,img2):
	return np.where(img1 == img2, 0, 1)

def diff_compa2(img1,img2,ratio = 5):
	return np.where(
		( 
			( ( img1 - ratio ) <= img2 ) 
			&
			( img2 <= ( ratio + img1 ) )
		)
		, 0, 1
	)

def diff_compa3(img1,img2,ratio = 5):
	return np.where(
		( 
			( img2 <= ( img1 - ratio ) ) 
			&
			( ( ratio + img1 ) <= img2 )
		)
		, 0, 1
	)

def save_compa_img(compa):
	if np.ndim(compa) == 3:
		for i in range(3):
			filename = "collage_mask_"+str(i+1)+".png"
			cv2.imwrite(filename,compa[:,:,i]*255)
	else:
		cv2.imwrite("collage_mask.png",compa*255)





orig = cv2.imread("orig.png")
conv_collage = cv2.imread("converted_collage.png")

edit_orig = conv_image2(orig)
cv2.imwrite('edit_orig.png',edit_orig) 
edit_conv_collage = conv_image2(conv_collage)
cv2.imwrite('edit_conv_collage.png',edit_conv_collage) 


#配列comparisonを生成
comparison=diff_compa1(edit_orig,edit_conv_collage)

save_compa_img(comparison)

if np.ndim(comparison) == 3:#次元数が多かったら削減する。
	comparison= np.amax(comparison,axis = 2)
	conv_collage = cv2.cvtColor(conv_collage,cv2.COLOR_BGR2GRAY)


#img1とcomparisonの要素の積、配列differenceを生成
#2つの画像の要素が等しい部分は0が掛けられるため0(黒)となり
#2つの画像の要素が等しい部分は1が掛けられるため値を持つ
#グレースケール画像として出力するためdtype = np.uint8を指定する
difference=np.array( conv_collage*comparison ,dtype = np.uint8)

cv2.imwrite('collage_difference.png',difference) #ファイル名difference.pngでdifferenceを保存

