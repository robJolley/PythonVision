import numpy as np
import statistics
import math
import cv2
from scipy import fftpack
from numpy import zeros
import matplotlib.pyplot as plt

imgSobl = cv2.imread('sobel.jpg')
imgDir = cv2.imread('dirE.jpg')
width, height, channels = imgSobl.shape
cannyImage = zeros([width,height,3])

for h in range (1,(height-2), 1):  # Canny filter
	for w in range (1,(width-2), 1):
		dirE = imgDir[h,w,0]
		maxPoint = True
		curPoint = imgSobl[h,w,0]
		if curPoint > 0.0:
			if 22.5 < dirE <= 67.5:
#				print('122-157',dirE)
				if (imgSobl[h-1,w+1,0] >= curPoint):
					maxPoint = False				
				if (curPoint <= imgSobl[h+1,w-1,0]):
					maxPoint = False
			elif 67.5 < dirE <=122.5:
#				print('67-112 :',dirE,)
				if (imgSobl[h-1,w,0] >= curPoint):
					maxPoint = False
				if (curPoint <= imgSobl[h+1,w,0]):
					maxPoint = False
			elif 122.5 < dirE <= 157.5:
#				print('22-67',dirE)
				if (imgSobl[h-1,w-1,0] >= curPoint):
					maxPoint = False
				if (curPoint <= imgSobl[h+1,w+1,0]):
					maxPoint = False
			else:
#				print('All else',dirE)
				if (imgSobl[h,w-1,0] >= curPoint):
					maxPoint = False
				if (curPoint <= imgSobl[h,w+1,0]):
					maxPoint = False
				
			if maxPoint:
				cannyImage[h,w] = [255,255,255]
			#	print('Current point',dirE)
			
#bilateral_filtered_image8 = np.uint8(bilateral_filtered_image*255)
#edge_detected_image1 = cv2.Canny((bilateral_filtered_image8), 75, 200)

#edge_detected_image1 = edge_detected_image1/255
#imgFP = img
#imgSobl = imgSobl/(imgSobl.max()/255.0)
cv2.imshow('Edge Sobel',imgSobl/255)
#cv2.waitKey(0)
#imgSobl = imgSobl/255
plt.figure()
plt.imshow(cannyImage/255)
cv2.imwrite('cannyImage.jpg',cannyImage);
plt.title('Canny Filter')
plt.figure()
plt.imshow(imgDir/255)
plt.title('ImageDir')
plt.show()
