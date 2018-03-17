from easydict import EasyDict as edict
Train = edict()
Train.DataSet_HR = 'data/train/DIV2K_train_HR/'
Train.DataSet_LR = 'data/train/DIV2K_train_LR_bicubic/X4/'
Train.BatchSize = 16
Train.init_learning_rate = 0.001
Train.momentum = 0.9
Train.epochs = 1000

Test = edict()
Test.DataSet_HR = 'data/test/set5'
Test.DataSet_LR = ''
