"""
=======================================================
Simple image blur by convolution with a Gaussian kernel
=======================================================

Blur an an image (:download:`../../../../data/elephant.png`) using a
Gaussian kernel.

Convolution is easy to perform with FFT: convolving two signals boils
down to multiplying their FFTs (and performing an inverse FFT).

"""
import numpy as np
import statistics
import math
import cv2
from scipy import fftpack
from numpy import zeros
import matplotlib.pyplot as plt

#####################################################################
# The original image
#####################################################################

# read image
img = plt.imread('resize.bmp')
#img =cv2.resize(img,(200,200))
#bilateral_filtered_image = cv2.bilateralFilter(img, 5, 175, 175)

img = img/255
thold = 15	
plt.figure()
plt.title('Raw')
plt.imshow(img)

#bilateral_filtered_image = bilateral_filtered_image/255
#plt.figure()
#plt.title('Bilateral CV2')
#plt.imshow(bilateral_filtered_image)

width, height, channels = img.shape

imga = zeros([width,height,3])
imgDir = zeros([width,height])	
imgSobl = imga

t = imga[1,1]
sobelX =[[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]
sobelY =[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]
imgSob = 0.0
tS =0.0
tSx = tS
tSy = tS

tr = t

mat = 5
mp =1+(int)(mat/2)
ta = zeros([mat,mat,3])
#plt.figure()
#plt.imshow(imga)
#####################################################################
# Prepare an Gaussian convolution kernel
#####################################################################

# First a 1-D  Gaussian
t = np.linspace(-mat, mat, mat)  # esecentialy for loop starting at -10 ending at 10 with 30 steps

bump = np.exp(-0.1*t**2)  # Used to generate line that erodes in an expodential patern -10 to + 10 building a bell curve
bump /= np.trapz(bump) # normalize the integral to 1

#print ('bump',bump)

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

#print ('kernel',kernel)
for xk in range (0,(mat-1), 1):  # nested loop to parce through raw image	
	for yk in range (0,(mat-1), 1):
		ta[xk,yk,0] = kernel[xk,yk]
		ta[xk,yk,1] = kernel[xk,yk]
		ta[xk,yk,2] = kernel[xk,yk]

for h in range (0,(height-1), 1):  # nested loop to parce through raw image	
	for w in range (0,(width-1), 1):
		img[h,w] = (img[h,w,0] + img[h,w,1] + img[h,w,2])/3
#		print(img[h,w])
		if img[h,w,0] < 0.02:
			img[h,w] = [0,0,0]

for h in range (mp,(height-mp), 1):  # nested loop to parce through raw image	
	for w in range (mp,(width-mp), 1):
		tr = img[h,w]
#		trAvg = (tr[0]+tr[1]+tr[2])/3
		if tr[0] > 0.0:	
			for xk in range (0,(mat-1), 1):  # nested loop to parce through raw image	
				for yk in range (0,(mat-1), 1):
					intPixel = ((img[(h+(xk-mp)),(w+(yk-mp)),0])+(img[(h+(xk-mp)),(w+(yk-mp)),1])+(img[(h+(xk-mp)),(w+(yk-mp)),2]))/3 
#					if (intPixel -(thold/255)) <= trAvg <= (intPixel + (thold/255)):
					t[0] += img[(h+(xk-mp)),(w+(yk-mp)),0]*ta[xk,yk,0]
					t[1] += img[(h+(xk-mp)),(w+(yk-mp)),1]*ta[xk,yk,1]
					t[2] += img[(h+(xk-mp)),(w+(yk-mp)),2]*ta[xk,yk,2]								
#					else:
#						t[0] += tr[0]*ta[xk,yk,0]
#						t[1] += tr[1]*ta[xk,yk,1]
#						t[2] += tr[2]*ta[xk,yk,2]
		else:
			t[0] = 0
			t[1] = 0
			t[2] = 0
				
#		print(intPixel)
		imga[h,w,0] = t[0]
		imga[h,w,1] = t[1]
		imga[h,w,2] = t[2]
		t[0] = 0
		t[1] = 0
		t[2] = 0
		
plt.figure()
plt.imshow(imga)
plt.title('Bilateral filter')

imga1 = np.uint8(imga*255)
cv2.imshow('Filtered Image ', imga1)
#cv2.waitKey(0)
for h in range (1,(height-2), 1):  # nested loop to parce through raw image	
	for w in range (1,(width-2), 1):
		if imga[h,w,0] > 0.0:
#			tS = imga1[h,w,0]
			for xk in range (0,3,1):  # nested loop to parce through raw image	
				for yk in range (0,3,1):
					tSx+=((sobelX[xk][yk])*imga1[((h-1)+xk),((w-1)+yk),0])
					tSy+=((sobelY[xk][yk])*imga1[((h-1)+xk),((w-1)+yk),0])
			tS =math.sqrt(tSx*tSx + tSy*tSy)
#			print('tS :',tS)
			if tS >76:
				imgSobl[h,w] =[tS,tS,tS]
				if tSx > 0.0:
					dirE = (tSy/tSx)
					imgDir[h,w] = dirE
			else: 
				imgSobl[h,w] = [0,0,0]
		tS = 0.0
		tSx = 0.0
		tSy = 0.0
	

#bilateral_filtered_image8 = np.uint8(bilateral_filtered_image*255)
#edge_detected_image1 = cv2.Canny((bilateral_filtered_image8), 75, 200)

#edge_detected_image1 = edge_detected_image1/255
#imgFP = img
#imgSobl = imgSobl/(imgSobl.max()/255.0)
cv2.imshow('Edge Sobel',imgSobl/255)
#cv2.waitKey(0)
#imgSobl = imgSobl/255
plt.figure()
plt.imshow(imgSobl/255)
plt.title('Sobel Filter')

plt.figure()
plt.imshow(imgDir/255)
plt.title('direction')

plt.show()