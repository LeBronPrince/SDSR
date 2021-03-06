from easydict import EasyDict as edict
Train = edict()
Train.DataSet_HR = 'data/train/DIV2K_train_HR/'
Train.DataSet_LR = 'data/train/DIV2K_train_LR_bicubic/X4/'
Train.BatchSize = 16
Train.init_learning_rate = 0.0001
Train.momentum = 0.9
Train.epochs = 10000
Train.lr_delay = 2000
Train.beta = 0.9

Test = edict()
Test.DataSet_HR = '/home/wangyang/Desktop/code/SRGAN/data2017/DIV2K_valid_HR/1068*2040/'
Test.DataSet_LR = '/home/wangyang/Desktop/code/SRGAN/data2017/DIV2K_valid_LR_bicubic/X4/267*510/'
