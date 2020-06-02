#MORPHOLOGY

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data,io
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import threshold_local
#for structuring element
from skimage.morphology import square,rectangle,disk
#*********
from skimage.morphology import binary_dilation, binary_opening
from skimage.morphology import skeletonize, thin


#normal image
img1 = io.imread("try3.jpg")
print("NORMAL IMAGE")
print(img1.dtype)
print("****")
#print(img1.shape)
#io.imshow(img1)
#io.show()

#img in grayscale (float64)
img_g = rgb2gray(img1)
print("GRAYSCALE FLOAT")
print(img_g.dtype)
print("****")
#print(img_g.shape)

#img as ubyte in uint8 for thresholding
gray = img_as_ubyte(img_g)
print("GRAYSCALE UINT8")
print(gray.dtype)
print("****")
#print(gray.shape)
#io.imshow(gray)
#io.show()

# adaptive thresholding
block_size=35
adaptive_thresholding = threshold_local(gray, block_size, method = "gaussian", mode = "wrap", param = 40, offset = 10)

binary_adaptive_img = gray < adaptive_thresholding
print("binary_adaptive_img")
print(binary_adaptive_img.dtype)
print("****")
#print(binary_adaptive_img.shape)
#io.imshow(binary_adaptive_img)
#io.show()


## opening to remove white spots

sel = square(8)
#opening
open_bin = binary_opening(binary_adaptive_img, sel)
print("open_bin")
print(open_bin.dtype)
print("****")
#print(open_bin.shape)


fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(binary_adaptive_img, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('thres')
ax[0].axis('off')

ax[1].imshow(open_bin, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('opening')
ax[1].axis('off')
plt.show()



##DIlation
sel_1 = disk(10)
dil_dil = binary_dilation(open_bin, sel_1)
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(open_bin, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('opening')
ax[0].axis('off')

ax[1].imshow(dil_dil, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('dilation')
ax[1].axis('off')
plt.show()


## Thinning

thinning_1 = thin(dil_dil, max_iter=8)
print(thinning_1.dtype)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(dil_dil, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Dilated')
ax[0].axis('off')

ax[1].imshow(thinning_1, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Thinning')
ax[1].axis('off')

plt.show()


## SKELETONIZE 

skel = skeletonize(dil_dil)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(dil_dil, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('boolean dilation')
ax[0].axis('off')

ax[1].imshow(skel, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeletonized image')
ax[1].axis('off')
plt.show()






'''
#binary_close (CHOOSE THIS ONE CZ DILATION-FIRST
# (filling of empty space) and then erosion-
#shrinking of large spaces)
closing_bin = binary_closing(binary_adaptive_img,selem)
print(closing_bin)
print("BINARY CLOSING")
print(closing_bin.dtype)
print(closing_bin.shape)
io.imshow(closing_bin)
io.show()

## morph-opening with cv2
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(binary_adaptive_img, cv2.MORPH_OPEN, kernel)
cv2.imshow(opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Medial axis or skeletonize

#medial_axis
thin = medial_axis(closing_bin)
print(thin)
print("MEDIAL AXIS THINNING")
print(thin.dtype)
print(thin.shape)
io.imshow(thin)
io.show()

#Skeletonize
skel = skeletonize(closing_bin)
print(skel)
print("SKELETONIZATION")
print(skel.dtype)
print(skel.shape)
io.imshow(skel)
io.show()

#EROSION in cv2
#setup kernel
kernel = np.ones((5,5),np.uint8)
erosion_cv = cv2.erode(binary_adaptive_img, kernel, iterations = 1)
print(erosion_cv)
io.imshow(erosion_cv)
io.show()


#DILATION in cv2
#setup kernel
kernel = np.ones((5,5),np.uint8)
dilation_cv = cv2.dilate(binary_adaptive_img,kernel,iterations = 1)
print(dilation_cv.shape)
io.imshow(dilation_cv)
io.show()
'''

