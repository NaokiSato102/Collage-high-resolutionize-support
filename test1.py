# https://qiita.com/hitomatagi/items/caac014b7ab246faf6b1 より。


# -*- coding: utf-8 -*-
import cv2

# 画像1
img1 = cv2.imread("img1.jpg")
# 画像2
img2 = cv2.imread("img2.jpg")

# A-KAZE検出器の生成
akaze = cv2.AKAZE_create()

# 特徴量の検出と特徴量ベクトルの計算
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)


# Brute-Force Matcher生成
bf = cv2.BFMatcher()

# 特徴量ベクトル同士をBrute-Force&KNNでマッチング
matches = bf.knnMatch(des1, des2, k=2)

# データを間引きする
ratio = 0.3
good = []
for m, n in matches:
	if m.distance < ratio * n.distance:
		good.append([m])

'''
f = open('list.txt', 'w')
for x in kp1:
    #f.write(str(x.pt) + "\n") #特徴点の位置をタプルで表示
	f.write(
		str(x.pt[0]) + "\t" + 
		str(x.pt[1]) + "\n"
	)#特徴点の位置をx､y分けて表示
f.close()'''

f = open('kp1.csv', 'w')
fileprint = "x[pic],y[pic],size,ang[deg],response,octave,class_id\n"
for x in kp1:	
	fileprint += (
		str(x.pt[0]) + "," + 
		str(x.pt[1]) + "," +
		str(x.size) + "," +
		str(x.angle) + "," +
		str(x.response) + "," +
		str(x.octave) + "," +
		str(x.class_id) + "\n"
	)	
	#特徴点の位置をx､y分けて表示
f.write(fileprint)
f.close()

f = open('kp2.csv', 'w')
fileprint = "x[pic],y[pic],size,ang[deg],response,octave,class_id\n"
for x in kp2:	
	fileprint += (
		str(x.pt[0]) + "," + 
		str(x.pt[1]) + "," +
		str(x.size) + "," +
		str(x.angle) + "," +
		str(x.response) + "," +
		str(x.octave) + "," +
		str(x.class_id) + "\n"
	)	
	#特徴点の位置をx､y分けて表示
f.write(fileprint)
f.close()

'''
# 対応する特徴点同士を描画
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# 画像表示
#cv2.imshow ('img', img3)
cv2.imwrite('img3.png', img3)
# キー押下で終了
cv2.waitKey(0)
cv2.destroyAllWindows()
'''