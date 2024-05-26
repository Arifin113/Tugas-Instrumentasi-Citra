import cv2

#membaca citra
image = cv2.imread(r'D:\INSTRUMENTASI CITRA DIGITAL\Project Tugas Citra\Modul Praktik 2\cameraman.tif', cv2.IMREAD_GRAYSCALE)

#operasi edge detection
#analisis perbedaan nilai threshold
edges = cv2.Canny(image, 50, 150)

#tampilkan hasil
cv2.imshow('Citra Asli', image)
cv2.imshow('Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
