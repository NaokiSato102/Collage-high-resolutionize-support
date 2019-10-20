import cv2
import numpy as np

def spectrum_edit(img):
	img = np.fft.fft2(img)
	return np.fft.fftshift(img)

def spectrum_edit_I(img):
	img = np.fft.ifftshift(img)
	return np.fft.ifft2(img)

def nomalizeSpectrum(spectrum, shifted = False):
    if shifted:
        fshift = spectrum
    else:
        fshift = np.fft.fftshift(spectrum)
    fabs = np.abs(fshift) # 複素数の絶対値を計算
    nomalized_shifted_spectrum = np.log10(fabs + 1) # logをとるパワーをデシベルであらわすには20をかけるが正規化するのでかけない
    nomalized_shifted_spectrum = np.uint8(nomalized_shifted_spectrum / np.amax(nomalized_shifted_spectrum) * 255) # [0,1]に正規化してから8bitに直す
    return nomalized_shifted_spectrum

orig = cv2.imread("f1.png")
collage = cv2.imread("f0.png")
dst = cv2.imread("a0.png")

sp_orig = spectrum_edit(orig)
sp_collage = spectrum_edit(collage)
sp_dst = spectrum_edit(dst)

diff = sp_collage / sp_orig

sp_orig_db = nomalizeSpectrum(sp_orig, shifted = True)
sp_collage_db = nomalizeSpectrum(sp_collage, shifted = True)

cv2.imwrite('sp_orig_db.png',sp_orig_db) 
cv2.imwrite('sp_collage_db.png',sp_collage_db) 

cv2.imwrite('dst1.png',spectrum_edit_I( sp_dst * diff ) )
cv2.imwrite('dst2.png',spectrum_edit_I( sp_dst / diff ) )

