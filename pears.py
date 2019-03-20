import cv2
from skimage import morphology
import numpy as np
image = cv2.imread('pears.png', cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
# Compute gradient magnitude
grad_magn = np.sqrt(sobelx**2 + sobely**2)
# Put it in [0, 255] value range
grad_magn = 255*(grad_magn - np.min(grad_magn)) / (np.max(grad_magn) - np.min(grad_magn))
selem = morphology.disk(20)
opened = morphology.opening(image, selem)
eroded = morphology.erosion(image, selem)
opening_recon = morphology.reconstruction(seed=eroded, mask=image, method='dilation')
closed_opening = morphology.closing(opened, selem)
dilated_recon_dilation = morphology.dilation(opening_recon, selem)
recon_erosion_recon_dilation = morphology.reconstruction(dilated_recon_dilation,
                                                    opening_recon,
                                                    method='erosion').astype(np.uint8)
def imcomplement(image):
  """Equivalent to matlabs imcomplement function"""
  min_type_val = np.iinfo(image.dtype).min
  max_type_val = np.iinfo(image.dtype).max
  return min_type_val + max_type_val - image

recon_dilation_recon_dilation = morphology.reconstruction(imcomplement(dilated_recon_dilation),
                                                     imcomplement(opening_recon),
                                                     method='dilation').astype(np.uint8)
recon_dilation_recon_dilation_c = cv2.imcomplement(recon_dilation_recon_dilation)

print(np.linalg.norm(recon_erosion_recon_dilation - recon_dilation_recon_dilation_c))
# Outputs 0.0
print(np.allclose(recon_erosion_recon_dilation, recon_dilation_recon_dilation_c))
# Outputs True
def imregionalmax(image, ksize=3):
  """Similar to matlab's imregionalmax"""
  filterkernel = np.ones((ksize, ksize)) # 8-connectivity
  reg_max_loc = peak_local_max(image,
                               footprint=filterkernel, indices=False,
                               exclude_border=0)
  return reg_max_loc.astype(np.uint8)

foreground_1 = imregionalmax(recon_erosion_recon_dilation, ksize=65)
fg_superimposed_1 = image.copy()
fg_superimposed_1[foreground_1 == 1] = 255
foreground_2 = morph.closing(foregorund_1, np.ones((5, 5)))
foreground_3 = morph.erosion(foregorund_2, np.ones((5, 5)))
from skimage.color import label2rgb

foreground_4 = morph.remove_small_objects(foreground_3.astype(bool), min_size=20)

_, labeled_fg = cv2.connectedComponents(foreground_4.astype(np.uint8))
col_labeled_fg = label2rgb(labeled_fg)
_, thresholded = cv2.threshold(recon_erosion_recon_dilation, 0, 255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Uncomment if you did find the skiz (assumed boolean here)
# grad_magn = grad_magn + (skiz*255).astype(np.uint8)

labels = morph.watershed(grad_magn, labeled_fg)
superimposed = image.copy()
watershed_boundaries = segmentation.find_boundaries(labels)
superimposed[watershed_boundaries] = 255
superimposed[foreground_4] = 255
#superimposed[skiz_im] = 255 # Uncomment if you computed skiz
col_labels = label2rgb(labels)
col_labels_merged = label2rgb(labels, image)

