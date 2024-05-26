import cv2
import numpy as np

#membaca citra
image = cv2.imread(r'D:\INSTRUMENTASI CITRA DIGITAL\kuliah_citra_20_5_2024\Modul Praktik 2\cameraman.tif', cv2.IMREAD_GRAYSCALE)

#definisikan kernel gradien
kernel_x = np.array([[-1,0, 1]])
kernel_y = np.array([[-1], [0], [1]])

#operasi konvolusi kernel dengan citra
grad_sp_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
grad_sp_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)

#cari nilai magnitude hasil konvolusi
magnitude = cv2.magnitude(grad_sp_x, grad_sp_y)

#normalisasi hasil citra
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

#ubah ke tipe data 8 bit
magnitude = magnitude.astype(np.uint8)

#tampilkan hasil
cv2.imshow('citra asli', image)
cv2.imshow('Gradien X', grad_sp_x)
cv2.imshow('Gradien Y', grad_sp_y)
cv2.imshow('Edge detection', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
