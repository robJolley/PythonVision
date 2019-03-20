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
import cv2
from scipy import fftpack
from numpy import zeros
import matplotlib.pyplot as plt

#####################################################################
# The original image
#####################################################################

# read image
img = plt.imread('resize.bmp')
img =cv2.resize(img,(200,200))
img = img/255
plt.figure()
plt.imshow(img)

width, height, channels = img.shape

imga = zeros([width,height,3])

t = imga[1,1]
tr = t
ta = zeros([5,5,3])
#plt.figure()
#plt.imshow(imga)
#####################################################################
# Prepare an Gaussian convolution kernel
#####################################################################

# First a 1-D  Gaussian
t = np.linspace(-5, 5, 5)  # esecentialy for loop starting at -10 ending at 10 with 30 steps
bump = np.exp(-0.1*t**2)  # Used to generate line that erodes in an expodential patern -10 to + 10 building a bell curve
bump /= np.trapz(bump) # normalize the integral to 1

#print ('bump',bump)

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

#print ('kernel',kernel)
for xk in range (0,(4), 1):  # nested loop to parce through raw image	
	for yk in range (0,(4), 1):
		ta[xk,yk,0] = kernel[xk,yk]
		ta[xk,yk,1] = kernel[xk,yk]
		ta[xk,yk,2] = kernel[xk,yk]




for h in range (4,(height-4), 1):  # nested loop to parce through raw image	
	for w in range (4,(width-4), 1):
		tr = img[h,w]
		
		for xk in range (0,(4), 1):  # nested loop to parce through raw image	
			for yk in range (0,(4), 1):
				t[0] += img[(h+(xk-4)),(w+(yk-4)),0]*ta[xk,yk,0]
				t[1] += img[(h+(xk-4)),(w+(yk-4)),1]*ta[xk,yk,1]
				t[2] += img[(h+(xk-4)),(w+(yk-4)),2]*ta[xk,yk,2]
		imga[h,w,0] = t[0]
		imga[h,w,1] = t[1]
		imga[h,w,2] = t[2]
		t[0] = 0
		t[1] = 0
		t[2] = 0
		
plt.figure()
plt.imshow(imga)
plt.title('Gaussian filter')
plt.show()