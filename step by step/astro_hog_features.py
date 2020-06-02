#astro_HOG feature
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.filters import threshold_isodata
from skimage import data, exposure,io


image = io.imread("c.jpg")

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(2,2),
                    cells_per_block=(1, 1), block_norm = "L2-Hys", visualize=True, multichannel=True)


print(hog_image.dtype)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

print(hog_image_rescaled.dtype)

io.imshow(hog_image_rescaled, cmap=plt.cm.gray)
io.show()
