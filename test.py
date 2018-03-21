"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
img = cv.imread('image.png')
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
img1 = np.abs(fshift)
img1 = img1.astype("float32")
fimg = np.log(np.abs(fshift))
print(fimg)
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Fourier')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('Fourier Fourier')
plt.show()
"""
import tensorflow as tf
import numpy as np
a =tf.constant([1])

with tf.Session() as sess:
    #b = a.eval()
    print(np.array(a).shape)
