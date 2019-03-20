import timeit
import statistics
start = timeit.default_timer()
import cv2
import numpy as np
#rawImage = cv2.imread('resize.bmp')
rawImage = cv2.imread('brace200.jpg')
#cv2.imshow('Original Image', rawImage)
#cv2.waitKey(0)
bilateral_filtered_image = cv2.bilateralFilter(rawImage, 5, 175, 175)
cv2.imshow('Bilateral', bilateral_filtered_image)
cv2.waitKey(0)
edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)
cv2.imshow('Edge', edge_detected_image)
cv2.imwrite('cvCanny.jpg',edge_detected_image)
print('D-type',edge_detected_image.dtype)
print('Shape',edge_detected_image.shape)
edge_detected_image = cv2.imread('cannyImage.jpg')
edge_detected_image = edge_detected_image
edge_detected_image= cv2.cvtColor(edge_detected_image, cv2.COLOR_RGB2GRAY)
cv2.waitKey(0)
_, contours, _= cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_list = []
ellipse_list = []
size = 0.0
standardDev = 0.0
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
	if ((area > 30)):
		# (len(approx) > 8) & (len(approx) < 23)
		contour_list.append(contour)
		ellipse = cv2.fitEllipse(contour)
		(x,y),(ma, MA), angle = ellipse
		if (MA/ma < 1.5):
			print('Area: ', area)
			print ('X', x)
			print ('Y',y)
			print ('Number:',i) 
			cv2.ellipse(rawImage,ellipse,(0,255,0))
			cv2.putText(rawImage,str(i),(int(x),int(y)),
				cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
			size += ((ma+MA)/2)
			cumeccent += (MA/ma)
			sizeArray.append((ma+MA)/2)
#			print(MA/ma)
#cv2.drawContours(rawImage, contour_list,  -1, (255,0,0), 2)
cv2.imshow('Objects Detected',rawImage)
meanSize =5.4*(size/i)
meanCumeccent = cumeccent/i
mat = np.array(sizeArray)
standardDev = statistics.stdev(mat)

#	minr, minc, maxr, maxc = props.bbox
#	bx = (minc, maxc, maxc, minc, minc)
#	by = (minr, minr, maxr, maxr, minr)
#	ax.plot(bx, by, '-b', linewidth=2.5)


stop = timeit.default_timer()
print('Time: ', stop - start)
print('Mean Size =', meanSize)
print('Mean Eccentricity =', meanCumeccent)
print('Standard Deviation  =', standardDev*5.4)
print(mat)
cv2.waitKey(0)

