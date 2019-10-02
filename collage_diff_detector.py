'''
https://tat-pytone.hatenablog.com/entry/2019/03/20/195949
'''

import cv2 #OpenCVのインポート
import numpy as np #numpyをnpという名前でインポート


orig = cv2.imread("orig.png",0)
collage_conv = cv2.imread("collage_converted.png",0)



#img1,img2の各要素を比較し、等しければ0、異なれば1の値を持つ配列comparisonを生成
comparison=np.where(orig == collage_conv, 0, 1)
print(type(comparison))

print(type(collage_conv))
#img1とcomparisonの要素の積、配列differenceを生成
#2つの画像の要素が等しい部分は0が掛けられるため0(黒)となり
#2つの画像の要素が等しい部分は1が掛けられるため値を持つ
#グレースケール画像として出力するためdtype = np.uint8を指定する
#comparison = collage_conv*comparison
comparison = np.dot(collage_conv,comparison)
difference=np.array( comparison ,dtype = np.uint8)

cv2.imwrite('collage_difference.png',difference) #ファイル名difference.pngでdifferenceを保存