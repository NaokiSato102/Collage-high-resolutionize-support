'''
https://tat-pytone.hatenablog.com/entry/2019/03/20/195949
'''

import cv2 #OpenCVのインポート
import numpy as np #numpyをnpという名前でインポート

def conv_image1(img):
	conv_img = img
	conv_img = cv2.erode( conv_img, np.ones( (3,3),np.uint8 ), iterations = 3 )
	conv_img = cv2.GaussianBlur(conv_img,(13,13),10,10)
	return conv_img

def diff_compa1(edit_orig,edit_conv_collage):
	return np.where(edit_orig == edit_conv_collage, 0, 1)

def diff_compa2(edit_orig,edit_conv_collage,ratio = 5):
	return np.where(
		( 
			( ( edit_orig - ratio ) <= edit_conv_collage ) 
			&
			( edit_conv_collage <= ( ratio + edit_orig ) )
		)
		, 0, 1
	)

orig = cv2.imread("orig.png",cv2.IMREAD_GRAYSCALE)
conv_collage = cv2.imread("converted_collage.png",cv2.IMREAD_GRAYSCALE)

edit_orig = conv_image1(orig)
cv2.imwrite('edit_orig.png',edit_orig) 

edit_conv_collage = conv_image1(conv_collage)
cv2.imwrite('edit_conv_collage.png',edit_conv_collage) 

#img1,img2の各要素を比較し、等しければ0、異なれば1の値を持つ配列comparisonを生成

comparison=diff_compa1(edit_orig,edit_conv_collage)





#img1とcomparisonの要素の積、配列differenceを生成
#2つの画像の要素が等しい部分は0が掛けられるため0(黒)となり
#2つの画像の要素が等しい部分は1が掛けられるため値を持つ
#グレースケール画像として出力するためdtype = np.uint8を指定する
difference=np.array( conv_collage*comparison ,dtype = np.uint8)

cv2.imwrite('collage_difference.png',difference) #ファイル名difference.pngでdifferenceを保存