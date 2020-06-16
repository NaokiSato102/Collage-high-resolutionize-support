# pylint: disable=C0111
# ↑プログラムの説明ドキュメントがないよ！というエラーの防止

# pylint: disable=W0312
# ↑Found indentation with tabs instead of spacesを防止



#import glob
import sys
import os
import numpy as np
import cv2
import mbiocv2 as mb
#import pprint as pp


class images:
	def __init__(self):
		self.filename = []
		self.h = np.empty(0, dtype=np.int)
		self.w = np.empty(0, dtype=np.int)
		self.hpw = np.empty(0, dtype=np.float)
		self.wph = np.empty(0, dtype=np.float)
		self.max_h = 0
		self.max_w = 0
	def calc_rate(self):
		self.hpw = []




def main():
	argv = sys.argv
	argc = len(sys.argv)


	for n in range(1,argc):
		#print(sys.argv[n])
		collage = mb.imread(argv[n])
		if (collage is None):
			print("Warning：[{}]の読み込み失敗".format(argv[n]) )
		else:
			h = collage.shape[0]
			w = collage.shape[1]
			hpw = h/w
			wph = w/h
			filename = os.path.basename(argv[n])
			print("ファイル名:{:>20}, 高:{:>5}, 幅:{:>5}, 高/幅:{:>5.3f}, 幅/高:{:>5.3f}".format(
				filename, h, w, hpw, wph)
			)



	input("終")





if __name__ == "__main__":
	sys.exit(main())
