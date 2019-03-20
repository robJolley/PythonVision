import timeit
import statistics
start = timeit.default_timer()
import cv2
import numpy as np
rawImage = cv2.imread('resize.bmp')
rawImageR= cv2.resize(rawImage, (200, 200)) 
#rawImage = cv2.imread('brace200.jpg')
#cv2.imshow('Original Image', rawImage)
#cv2.waitKey(0)
width, height, channels = rawImageR.shape
biImage = rawImageR
print('Width', width)
print ('Height', height)
validpoints = 0
#bilatMatrix = [[0.5,0.5,0.5,0.5,0.5],[0.5,1,1,1,0], [0.5,1,1,1,0],[0.5,1,1,1,0.5],[0.5,0.5,0.5,0.5,.5]]# Matrix for basic Gausian filter
#left to right
for h in range (3,(height-3), 1):  # nested loop to parce through raw image	
	for w in range (3,(width-3), 1):
		rawPixValue = rawImageR[w,h]
		
		raw0P25 = rawPixValue[0]+ 2
		raw0M25 = rawPixValue[0]- 2
		raw1P25 = rawPixValue[1]+ 2
		raw1M25 = rawPixValue[1]- 2
		raw2P25 = rawPixValue[2]+ 2
		raw2M25 = rawPixValue[2]- 2
#		print ('Raw Pixel value :', rawPixValue)
		bimatrix = rawImageR[(w-2):(w+2),(h-2):(h+2)]
#		print ('bimatrix: ', bimatrix)
		bimult = [0,0,0]
		for x in range(0,4,1): #nested loop to walk through matrix sized portion of image 
			for y in range(0,4,1):
				bipix = bimatrix[x,y]  # Working through the RGB to see if they are close
				#print ('Pixel value :', bipix)
				if ((raw0M25 >= bipix[0] <= raw0P25) and (raw1M25 >= bipix[1] <= raw1P25) and (raw2M25 >= bipix[2] <= raw2P25)):
					validpoints += 1
					bimult[0] = bimult[0] + bipix[0] # accumulates values for RGB in order to build mean if values are close
					bimult[1] = bimult[1] + bipix[1]
					bimult[2] = bimult[2] + bipix[2]
#					print('Bimult', bimult)
			if validpoints > 0:
				bimult[0] = bimult[0]/validpoints
				bimult[1] = bimult[1]/validpoints
				bimult[2] = bimult[2]/validpoints
#				print('Bimult', bimult)

				biImage[w,h] = bimult
			else:
				biImage[w,h] = [0,0,0]
			validpoints = 0
#right to left			
for h in range ((height-3),3, -1):  # nested loop to parce through raw image	
	for w in range ((width-3),3 -1):
		rawPixValue = rawImageR[w,h]
		
		raw0P25 = rawPixValue[0]+ 2
		raw0M25 = rawPixValue[0]- 2
		raw1P25 = rawPixValue[1]+ 2
		raw1M25 = rawPixValue[1]- 2
		raw2P25 = rawPixValue[2]+ 2
		raw2M25 = rawPixValue[2]- 2
#		print ('Raw Pixel value :', rawPixValue)
		bimatrix = rawImageR[(w-2):(w+2),(h-2):(h+2)]
#		print ('bimatrix: ', bimatrix)
		bimult = [0,0,0]
		for x in range(0,4,1): #nested loop to walk through matrix sized portion of image 
			for y in range(0,4,1):
				bipix = bimatrix[x,y]  # Working through the RGB to see if they are close
				#print ('Pixel value :', bipix)
				if ((raw0M25 >= bipix[0] <= raw0P25) and (raw1M25 >= bipix[1] <= raw1P25) and (raw2M25 >= bipix[2] <= raw2P25)):
					validpoints += 1
					bimult[0] = bimult[0] + bipix[0] # accumulates values for RGB in order to build mean if values are close
					bimult[1] = bimult[1] + bipix[1]
					bimult[2] = bimult[2] + bipix[2]
#					print('Bimult', bimult)
			if validpoints > 0:
				bimult[0] = bimult[0]/validpoints
				bimult[1] = bimult[1]/validpoints
				bimult[2] = bimult[2]/validpoints
#				print('Bimult', bimult)

				biImage[w,h] = bimult
			else:
				biImage[w,h] = [0,0,0]
			validpoints = 0
			
cv2.imshow('Bilateral filter', biImage)
cv2.waitKey(0)
cv2.imwrite('filteredFile.bmp',biImage)
print('filted content', biImage)

