import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('image.png')
print(img.shape)
f = np.fft.fft2(img)
print(f.shape)
fshift = np.fft.fftshift(f)
print(fshift)
img1 = np.abs(fshift)
fimg = np.log(np.abs(fshift))
print(img1.shape)
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
plt.show()
