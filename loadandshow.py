import numpy as np
import cv2
from matplotlib import pyplot as plt

kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])

# Load an color image in grayscale
img = cv2.imread('D:\\resize.bmp',0)
cv2.imshow('image',img)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
plt.hist(img.ravel(),256,[0,256]); plt.show()
_, th2 = cv2.threshold(img,200,255,cv2.THRESH_TRUNC)
#th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
#th2 = cv2.morphologyEx(th2, cv2.MORPH_ERODE, kernel)
#th2 = cv2.morphologyEx(th2, cv2.MORPH_ERODE, kernel)
#th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
#th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
cv2.imshow('simpleThresholdGS',th2)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
th3 = cv2.adaptiveThreshold(th2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imshow('dynamicGasianThreshold',th3)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
cv2.imwrite('D:\\dynamicthreshold.bmp',th3)
dst	=	cv2.bitwise_and(th2, th3)
cv2.imshow('anded',dst)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
plt.hist(dst.ravel(),256,[0,256]); plt.show()
_, dst = cv2.threshold(dst,60,255,cv2.THRESH_BINARY)
dst	=	cv2.bitwise_and(th2, th3)
cv2.imshow('post threshold',dst)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
dst = cv2.morphologyEx(dst, cv2.MORPH_DILATE, kernel)
cv2.imshow('post threshold and dilation',dst)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
_, contours, _ = cv2.findContours(th2.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
print(len(contours))
print(contours)
centres = []
for i in range(len(contours)):
  moments = cv2.moments(contours[i])
  centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
  cv2.circle(img, centres[-1], 3, (0, 0, 0), -1)

print(centres)

cv2.imshow('image', img)
#cv2.imwrite('output.png',img)
cv2.waitKey(0)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()