import cv2
import numpy as np

#membaca citra
image = cv2.imread(r'D:\INSTRUMENTASI CITRA DIGITAL\kuliah_citra_20_5_2024\Modul Praktik 2\cameraman.tif', cv2.IMREAD_GRAYSCALE)

#smoothing image
image_blurred = cv2.GaussianBlur(image, (3,3), 0)

#definisikan kernel laplacian
laplacian_kernel = np.array([[0,1,0], [1,-4,1], [0,1,0]])

#operasi konvolusi kernel dengan citra
laplacian_conv = cv2.filter2D(image_blurred, cv2.CV_64F, laplacian_kernel)


#cari nilai magnitude hasil konvolusi
magnitude = cv2.magnitude(laplacian_conv, laplacian_conv)

#normalisasi hasil citra
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

#ubah ke tipe data 8 bit
magnitude = magnitude.astype(np.uint8)

#tampilkan hasil
cv2.imshow('citra asli', image)
cv2.imshow('Edge detection', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
