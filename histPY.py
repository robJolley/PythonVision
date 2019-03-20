import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('pears.png',0)
cv2.imshow("Input", img)
plt.hist(img.ravel(),256,[0,256]); plt.show()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)
plt.hist(img.ravel(),256,[0,256]); plt.show()
cv2.imshow("Output", img)
cv2.waitKey(0)
