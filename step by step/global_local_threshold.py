from skimage.filters import threshold_otsu, threshold_local
from skimage import data
from skimage.util import img_as_uint
import matplotlib.pyplot as plt

image = data.page()
print(image)
print(image.dtype)
print(image.shape)

img1= img_as_uint(image)
print(img1.dtype)
print(img1.shape)

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh
print(binary_global.dtype)
print(binary_global.shape)

block_size = 35
adaptive_thresh = threshold_local(image, block_size, offset=10)
binary_adaptive = image > adaptive_thresh
print(binary_adaptive.dtype)
print(binary_adaptive.shape)


fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()

ax[0].imshow(image)
ax[0].set_title('Original')

ax[1].imshow(binary_global)
ax[1].set_title('Global thresholding')

ax[2].imshow(binary_adaptive)
ax[2].set_title('Adaptive thresholding')

for a in ax:
    a.axis('off')

plt.show()
