import tensorflow as tf
import config
import os
import scipy
import scipy.misc
import tensorlayer as tl
from tensorlayer.prepro import *

def crop_imgs(x, is_random=True):
    x = crop(x, wrg=128, hrg=128, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[32, 32], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def Load_Train_Data(datadir_HR,path_HR):
    train_set_HR = []
    imgs_files_list_HR = os.listdir(datadir_HR)
    for i in range(len(imgs_files_list_HR)):
        imgs_HRs = scipy.misc.imread(path_HR+imgs_files_list_HR[i])
        train_set_HR.append(imgs_HRs)
        print(i)
    return train_set_HR

def Load_Test_Data(datadir_HR,datadir_LR,path_HR,path_LR):
    test_set_HR = []
    test_set_LR = []
    imgs_files_list_HR = os.listdir(datadir_HR)
    imgs_files_list_LR = os.listdir(datadir_LR)
    for i in range(len(imgs_files_list_HR)):
        imgs = scipy.misc.imread(path_HR+imgs_files_list_HR[i])
        test_set_HR.append(imgs)
    for i in range(len(imgs_files_list_LR)):
        imgs = scipy.misc.imread(path_LR+imgs_files_list_LR[i])
        test_set_LR.append(imgs)
    return test_set_HR,test_set_LR
