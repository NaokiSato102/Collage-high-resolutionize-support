import cv2 #OpenCVのインポート
import numpy as np #numpyをnpという名前でインポート

img1 = cv2.imread('castle1.png',0) #画像castle1.pngをグレースケールで読み込む
img2 = cv2.imread('castle2.png',0) #画像castle2.pngをグレースケールで読み込む

#img1,img2の各要素を比較し、等しければ0、異なれば1の値を持つ配列comparisonを生成
comparison=np.where(img1 == img2, 0, 1)
#img1とcomparisonの要素の積、配列differenceを生成
#2つの画像の要素が等しい部分は0が掛けられるため0(黒)となり
#2つの画像の要素が等しい部分は1が掛けられるため値を持つ
#グレースケール画像として出力するためdtype = np.uint8を指定する
difference=np.array(img1*comparison,dtype = np.uint8)

cv2.imwrite('difference.png',difference) #ファイル名difference.pngでdifferenceを保存