import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
img = cv.imread('image.png')
print(img.shape)
f = np.fft.fft2(img)
print(f.shape)
fshift = np.fft.fftshift(f)
print(fshift)
img1 = np.abs(fshift)
img1 = img1.astype("float32")
fimg = np.log(np.abs(fshift))
print(img1.dtype)
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
plt.show()
