# Trying to mix cv2 and skimage together to read and perform preprocessing
# working but the file is read in the program
# not by the user

import numpy as np
import cv2
from skimage import data,io
from skimage.util import img_as_ubyte
from skimage.filters import threshold_local
from skimage.color import rgb2gray
from skimage.feature import hog


#reading in skimage
latest = io.imread("d.jpg")
print(latest.dtype) 	#uint8
print(latest.shape) 	#(r,c,3)
io.imshow(latest)
io.show()

#converting to gray in skimage
grayscale = rgb2gray(latest)
print(grayscale.dtype)  #float64
print(grayscale.shape)	#(r,c)

#converting the image datatype to the previous one as the shape
#of grayscale (float64) is only = (rows, columns)
#and original loaded image (uint8) is (rows, columns, channels)

#hence, we convert the float64 to uint8, but 
#now both have only (rows, columns)
#which is helpful for adaptive_local_thresholding 

#we get one runtime error (possible precision loss), but we can ignore it

gray= img_as_ubyte(grayscale)
print(gray.dtype)	#uint8
print(gray.shape)	#(r,c)
io.imshow(gray)
io.show()

#converting to binary using skimage
#here using adaptive, that is local thresholding
block_size = 35
adaptive_t = threshold_local(gray, block_size, method ="gaussian", mode= "wrap",
param = 40, offset=10)
binary_adaptive = gray < adaptive_t

print(binary_adaptive.dtype)
print(binary_adaptive.shape)

io.imshow(binary_adaptive)
io.show()

