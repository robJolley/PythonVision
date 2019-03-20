import cv2
import numpy as np
from skimage import morphology
image = cv2.imread('pears.png', 0)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
cv2.imshow("Sobel X", sobelx)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
cv2.imshow("Sobel Y", sobely)
# Compute gradient magnitude
grad_magn = np.sqrt(sobelx**2 + sobely**2)
# Put it in [0, 255] value range
grad_magn = 255*(grad_magn - np.min(grad_magn)) / (np.max(grad_magn) - np.min(grad_magn))
cv2.imshow("grad_magn", grad_magn)
selem = morphology.disk(20)
opened = morphology.opening(image, selem)
cv2.imshow("opened disks", opened)
th3 = cv2.adaptiveThreshold(opened,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,1)
#ret3,th3 = cv2.threshold(opened,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#cv2.imshow("Thresholded", th3)
def imregionalmax(image, ksize=3):
  """Similar to matlab's imregionalmax"""
  filterkernel = np.ones((ksize, ksize)) # 8-connectivity
  reg_max_loc = peak_local_max(image,
                               footprint=filterkernel, indices=False,
                               exclude_border=0)
  return reg_max_loc.astype(np.uint8)

foreground_1 = imregionalmax(recon_erosion_recon_dilation, ksize=65)
cv2.imshow("Foreground", foreground_1)
cv2.waitKey(0)