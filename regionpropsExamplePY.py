import timeit
import statistics
start = timeit.default_timer()
import math
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate
kernel = np.ones((7,7),np.uint8)

image = cv2.imread('D:/resize.bmp',0)
#image = cv2.imread('brace200.jpg',0)
#image = cv2.imread('wurst1.jpg',0)
#hist_full = cv2.calcHist([image],[0],None,[256],[0,256])
#plt.plot(hist_full)
#plt.show()
#cv2.waitKey(0)
imageC = image;
width, height = image.shape[:2]
_,image = cv2.threshold(image,80,255,cv2.THRESH_BINARY)
#image = cv2.equalizeHist(image)
#hist_full = cv2.calcHist([image],[0],None,[256],[0,256])
#plt.plot(hist_full)
#plt.show()
#cv2.imshow('Objects Detected',image)
#cv2.waitKey(0)
#image =image.astype(float)
#image = image*image
#cv2.imshow('Objects Detected',image)
#cv2.waitKey(0)
#image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#cv2.imshow('Objects Detected',image)
#cv2.waitKey(0)
image= cv2.GaussianBlur(image,(5,5),0)
_,image = cv2.threshold(image,70,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
image = cv2.erode(image,kernel,iterations = 3)
#imageC = cv2.imread('wurst1.jpg')
#imageC = cv2.resize(imageC,None,fx =4, fy = 4,interpolation = cv2.INTER_CUBIC)
label_img = label(image)
regions = regionprops(label_img)

fig, ax = plt.subplots()
ax.imshow(imageC, cmap=plt.cm.gray)
size = 0.0
standardDev = 0.0
meanSize = 0.0
cumeccent = 0.0
sizeArray = []
meanCumeccent = 0.0
i = 1
for props in regions:
	y0, x0  = props.centroid
	eccent = props.eccentricity
	orientation = -props.orientation
	degOr = (math.degrees(orientation))
	major = (props.major_axis_length)+25			
	minor = (props.minor_axis_length)+25
	print('Number:'+str(i)+' Angle:' +str(degOr)+' Major:' +str(major)+' Minor:' + str(minor))
	x1 = x0 + math.cos(orientation) * 0.5 * major
	y1 = y0 + math.sin(orientation) * 0.5 * major
	x2 = x0 + math.sin(orientation) * 0.5 * minor
	y2 = y0 - math.cos(orientation) * 0.5 * minor
	ell = Ellipse([x0,y0], (major), (minor), degOr,
	edgecolor='b', lw=1, facecolor='none')
	ax.add_artist(ell)
	
	ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
	ax.plot((x0, x2), (y0, y2), '-g', linewidth=2.5)
	ax.annotate (i, (x0, y0))
	size += ((major+minor)/2)
	cumeccent += (minor/major)
	sizeArray.append((major+minor)/2)
i +=1
meanSize =5.4*(size/i)
meanCumeccent = cumeccent/i
mat = np.array(sizeArray)
standardDev = statistics.stdev(mat)

#	minr, minc, maxr, maxc = props.bbox
#	bx = (minc, maxc, maxc, minc, minc)
#	by = (minr, minr, maxr, maxr, minr)
#	ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, height, width, 0))
stop = timeit.default_timer()
print('Time: ', stop - start)
print('Mean Size =', meanSize)
print('Mean Eccentricity =', meanCumeccent)
print('Standard Deviation  =', standardDev*5.4)
print(mat)


plt.show()

plt.close()
