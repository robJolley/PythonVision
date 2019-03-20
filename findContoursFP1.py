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
cannyImageA = cv2.imread('cannyImage.jpg')
print('D-type',cannyImageA.dtype)
print('Shape',cannyImageA.shape)
cannyImage = cv2.cvtColor(cannyImageA, cv2.COLOR_RGB2GRAY)
print('D-type',cannyImage.dtype)
print('Shape',cannyImage.shape)
_, contours, _= cv2.findContours(cannyImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list = []
ellipse_list = []
size = 0.0
standardDev = 0.0
mat = 0
meanSize = 0.0
cumeccent = 0.0
meanCumeccent = 0.0
sizeArray = []
cnt = contours[0]
i = 0
for contour in contours:
	i+=1
	approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
	area = cv2.contourArea(contour)
	if ((area > 10) and (len(approx) > 8) and (len(approx) < 23)):
		contour_list.append(contour)
		ellipse = cv2.fitEllipse(contour)
		(x,y),(ma, MA), angle = ellipse
		if (MA/ma < 2.5):
			print('Area: ', area)
			print ('X', x)
			print ('Y',y)
			print ('Number:',i) 
			cv2.ellipse(cannyImageA,ellipse,(0,255,0))
			cv2.putText(cannyImageA,str(i),(int(x),int(y)),
				cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
			size += ((ma+MA)/2)
			cumeccent += (MA/ma)
			sizeArray.append((ma+MA)/2)
#			print(MA/ma)
#cv2.drawContours(rawImage, contour_list,  -1, (255,0,0), 2)
cv2.imshow('Objects Detected',cannyImageA)
meanSize =5.4*(size/i)
meanCumeccent = cumeccent/i
mat = np.array(sizeArray)
standardDev = statistics.stdev(mat)

#	minr, minc, maxr, maxc = props.bbox
#	bx = (minc, maxc, maxc, minc, minc)
#	by = (minr, minr, maxr, maxr, minr)
#	ax.plot(bx, by, '-b', linewidth=2.5)


#stop = timeit.default_timer()
#print('Time: ', stop - start)
print('Mean Size =', meanSize)
print('Mean Eccentricity =', meanCumeccent)
print('Standard Deviation  =', standardDev*5.4)
cv2.imwrite('cannyImageEF.jpg',cannyImageA);
print(mat)
cv2.waitKey(0)

