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
img = plt.imread('circle.jpg')
img =cv2.resize(img,(200,200))
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
cannyImage = zeros([width,height])
imgDir = zeros([width,height])
imgSobl = imga

t = imga[1,1]
sobelX =[[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]]
sobelY =[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]
cannyArray = zeros([3,3])
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
for h in range (1,(height-2), 1):  # Sobel filter
    for w in range (1,(width-2), 1):
        if imga[h,w,0] > 0.0:
            #			tS = imga1[h,w,0]
            for xk in range (0,3,1):  # nested loop to parce through raw image
                for yk in range (0,3,1):
                    tSx+=((sobelX[xk][yk])*imga1[((h-1)+xk),((w-1)+yk),0])
                    tSy+=((sobelY[xk][yk])*imga1[((h-1)+xk),((w-1)+yk),0])
            tS =math.sqrt(tSx*tSx + tSy*tSy)
            #			print('tS :',tS)
            #			tSx = abs(tSx)
            #			tSy = abs(tSy)
            if tS >76:
                imgSobl[h,w] =[tS,tS,tS]
                #				if tSx > 0.0:
                dirE = ((math.atan(tSy/tSx))*180/(math.pi))
                #					print('Dir:',dirE)
                if dirE < 0:
                    dirE = dirE + 180
                #				print('Dir:',dirE)
                dirE = 180-dirE
                imgDir[h,w] = dirE
            else:
                imgSobl[h,w] = [0,0,0]
        tS = 0.0
        tSx = 0.0
        tSy = 0.0

cv2.imwrite('sobel.jpg',imgSobl)
cv2.imwrite('dirE.jpg',imgDir)

for h in range (1,(height-2), 1):  # Canny filter,  modifed so that direction is coded into solution only requires 3 bits per pixel,  will speed up contour identification
    for w in range (1,(width-2), 1):
        dirE = imgDir[h,w]
        maxPoint = True
        maxDir = 0
        curPoint = imgSobl[h,w,0]
        if curPoint > 0.0:
            if 22.5 < dirE <= 67.5:#NW
                maxDir = 2
                #				print('122-157',dirE)
                if (imgSobl[h-1,w+1,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h+1,w-1,0]):
                    maxPoint = False
            elif 67.5 < dirE <=122.5:#N
                #				print('67-112 :',dirE,)
                maxDir = 3
                if (imgSobl[h-1,w,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h+1,w,0]):
                    maxPoint = False
            elif 122.5 < dirE <= 157.5:#NE
                maxDir = 4
                #				print('22-67',dirE)
                if (imgSobl[h-1,w-1,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h+1,w+1,0]):
                    maxPoint = False
            else:#E
                #				print('All else',dirE)
                maxDir = 1
                if (imgSobl[h,w-1,0] >= curPoint):
                    maxPoint = False
                if (curPoint <= imgSobl[h,w+1,0]):
                    maxPoint = False

            if maxPoint:
                cannyImage[h,w] = maxDir
        #	print('Current point',dirE)

contourList = [[0 for i in range(3)] for j in range(5000)]
contNum = 0
contPlace = 0
contDirection = 0#  0 for CW,  1 for CWW 2 for end of contour
for h in range (1,(height-2), 1): #parces through modified canny
    for w in range (1,(width-2), 1):
        cannyImgPix = cannyImage[h,w]
        if ((cannyImgPix !=0) and (cannyImgPix !=5)):
            #			print('Value First:',cannyImgPix)
            #			print('Well I got this far')
            conh = h  #location of current interigated pixel in corrent contour
            conhOld = h
            conw = w
            conwOld = w
            constartH = h #start location of contour,  will first move clockwise then antilockwise
            constartW = w
            #N/S
            #			contourList[contPlace] = {contNum,constartH,constartW}
            #			print(contourList[0])
            while(contDirection !=2):
                cannyImgPix = cannyImage[conh,conw]
                #				print('Value:',cannyImgPix)
                #				print('Height',conh)
                #				print('Width',conw)
                if (cannyImgPix == 3 and contDirection == 0):
                    print('N/S 0')
                    if(cannyImage[conh,conw+1] !=0):
                        contPlace+=1
                        conw+=1
                    elif(cannyImage[conh+1,conw+1] !=0):
                        print('*********************************')
                        contPlace+=1
                        conh+=1
                        conw+=1
                    elif(cannyImage[conh-1,conw+1] !=0):
                        contPlace+=1
                        conh-=1
                        conw+=1
                    else:
                        contDirection = 1#dirrection of movment around contour
                elif (cannyImgPix == 3 and contDirection == 1):
                    print('N/S 1')
                    if(cannyImage[conh,conw-1] !=0):
                        contPlace+=1
                        conw-=1
                    elif(cannyImage[conh-1,conw-1] !=0):
                        contPlace+=1
                        conh-=1
                        conh-=1
                    elif(cannyImage[conh+1,conw-1] !=0):
                        contPlace+=1
                        conh+=1
                        conw-=1
                    else:
                        print('*******************************NS')
                        contDirection = 2#End of contour
                        contNum +=1

                # NE/SW

                elif (cannyImgPix == 2 and contDirection == 0):
                    print('NE 0')
                    if(cannyImage[conh+1,conw+1] !=0):
                        contPlace+=1
                        conh+=1
                        conw+=1
                    elif(cannyImage[conh,conw+1] !=0):
                        contPlace+=1
                        conw+=1
                    elif(cannyImage[conh-1,conw] !=0):
                        contPlace+=1
                        conh-=1
                    else:
                        contDirection = 1#dirrection of movment around contour

                elif (cannyImgPix == 2 and contDirection == 1):
                    print('NE 1')
                    if(cannyImage[conh-1,conw-1] !=0):
                        contPlace+=1
                        conw-=1
                        conh-=1
                    elif(cannyImage[conh,conw-1] !=0):
                        contPlace+=1
                        conh-=1
                    elif(cannyImage[conh-1,conw] !=0):
                        contPlace+=1
                        conh-=1

                    else:
                        print('*********************************NE')
                        contDirection = 2#End of contour
                        contNum +=1

                # W/E
                elif (cannyImgPix == 1 and contDirection == 0):
                    print('W/E 0')
                    if(cannyImage[conh+1,conw] !=0):
                        contPlace+=1
                        conh+=1
                    elif(cannyImage[conh+1,conw+1] !=0):
                        contPlace+=1
                        conh+=1
                        conw+=1
                    elif(cannyImage[conh+1,conw-1] !=0):
                        contPlace+=1
                        conh+=1
                        conw-=1
                    else:
                        contDirection = 1#dirrection of movment around contour
                elif (cannyImgPix == 1 and contDirection == 1):
                    print('W/E 1')
                    if(cannyImage[conh-1,conw] !=0):
                        contPlace+=1
                        conh-=1
                    elif(cannyImage[conh-1,conw-1] !=0):
                        contPlace+=1
                        conh-=1
                        conw-=1
                    elif(cannyImage[conh-1,conw+1] !=0):
                        contPlace+=1
                        conh-=1
                        conw+=1

                    else:
                        print('*********************************WE')
                        contDirection = 2#End of contour
                        contNum +=1

                #NW/SE
                elif (cannyImgPix == 4 and contDirection == 0):
                    print('NW 0')
                    if(cannyImage[conh+1,conw+1] !=0):
                        contPlace+=1
                        conh+=1
                        conw+=1
                    elif(cannyImage[conh+1,conw] !=0):
                        contPlace+=1
                        conh+=1
                    elif(cannyImage[conh,conw+1] !=0):
                        contPlace+=1
                        conw-=1
                    else:
                        contDirection = 1#dirrection of movment around contour

                elif (cannyImgPix == 4 and contDirection == 1):
                    print('NW 1')
                    if(cannyImage[conh-1,conw-1] !=0):
                        contPlace+=1
                        conh-=1
                        conw-=1
                    elif(cannyImage[conh-1,conw] !=0):
                        contPlace+=1
                        conh-=1
                    elif(cannyImage[conh,conw-1] !=0):
                        contPlace+=1
                        conw+=1
                    else:
                        contDirection = 2#End of contour
                        contNum +=1
                        print('*********************************NW')

                else:
                    print('*********************************Outer')
                    contDirection = 2#End of contour
                    contNum +=1
                if(conh >=height-1) or (conw >= width-1) or (conh <=1) or (conw <=1):
                    break
                if (contDirection !=2):
                    cannyImage[conhOld,conwOld] = 5
                    conhOld=conh
                    conwOld=conw
                    print('ContNum ',contNum)
                    contourList[contPlace] = {contNum,conh,conw}
                    print('contourList ',contourList[contPlace])
#print(contourList)

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
print('D-type: ',cannyImage.dtype)
print('Shape: ',cannyImage.shape)
cannyImage = cannyImage.astype(np.uint8)
#plt.figure()
#plt.imshow(imgDir/255)
#plt.title('direction')
#imgConv= cv2.cvtColor(cannyImage, cv2.COLOR_RGB2GRAY)

plt.show()