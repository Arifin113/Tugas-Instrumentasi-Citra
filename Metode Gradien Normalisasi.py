import cv2
import numpy as np

#membaca citra
image = cv2.imread(r'D:\INSTRUMENTASI CITRA DIGITAL\kuliah_citra_20_5_2024\Modul Praktik 2\cameraman.tif', cv2.IMREAD_GRAYSCALE)

#definisikan kernel gradien
Gx = np.array([[-1, 1]])
Gy = np.array([[-1], [1]])

#operasi konvolusi kernel dengan citra
grad_x = cv2.filter2D(image, cv2.CV_64F, Gx)
grad_y = cv2.filter2D(image, cv2.CV_64F, Gy)

#cari nilai magnitude hasil konvolusi
magnitude = cv2.magnitude(grad_x, grad_y)

#normalisasi hasil citra
grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX)
grad_y = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX)
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)


#ubah ke tipe data 8 bit
magnitude = magnitude.astype(np.uint8)
grad_x = grad_x.astype(np.uint8)
grad_y = grad_y.astype(np.uint8)

#tampilkan hasil
cv2.imshow('citra asli', image)
cv2.imshow('Gradien X', grad_x)
cv2.imshow('Gradien Y', grad_y)
cv2.imshow('Edge detection', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()