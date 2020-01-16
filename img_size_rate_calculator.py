import glob
import sys
from os import path as osp
import numpy as np
import cv2
import mbiocv2 as mb


SPRT_EXT_LIST = ["png","jp*","bmp"]


def imgread(filename, dirname):
	img = mb.imread(osp.join(dirname, filename) )
	if (img is None):
		print("Warning：[{}]の読み込み失敗".format(file_name) )
		return 1
	print("[{}]の読み込み成功".format(file_name) )

	h, w, _ = img.shape
	h_rate = h/w
	w_rate = w/h

	return img, h, w, h_rate, w_rate




def main():
	dirname  = osp.dirname (sys.argv[1] )
	origname = osp.basename(sys.argv[1] )
	origname_nx = osp.splitext(origname)[0]
	file_name_list = []
	for i in SPRT_EXT_LIST:
		file_name_list += glob.glob( dirname + "/*." + i ) # 拡張子リストに載っている拡張子を条件として検索しリスト化 
	 
	if (not len(file_name_list)):
		return 1
	
	tmp_list = file_name_list
	file_name_list = []
	for i in tmp_list:
		file_name_list.append(osp.basename(i) )

	print(file_name_list)

	file_name_list_nx = []
	for i in file_name_list:
		file_name_list_nx.append(osp.splitext(i)[0])
	

	print(file_name_list_nx)

	if(1<file_name_list_nx.count(origname_nx) ):
		print("拡張子を除いたファイル名でオリジナル指定を受けたファイルと同名のファイルがあります")
		return 1
	
	print("[debug]orignameと同じものをfile_name_listから省く")
	file_name_list.remove(origname)

	print(file_name_list)

	





	


if __name__ == "__main__":
	sys.exit(main())
