import tensorflow as tf
from model import *
from config import *
from data import *
import os
import scipy
import scipy.misc
import tensorlayer as tl
from tensorlayer.prepro import *
import time
from time import localtime, strftime
from skimage import measure

def train():
    ###make some dirs
    checkpoint_dir = os.getcwd()+'/checkpoint'
    tl.files.exists_or_mkdir(checkpoint_dir)
    tensorboard_dir = os.getcwd()+'/tensorboard'
    tl.files.exists_or_mkdir(tensorboard_dir)

    data_HR,data_LR = Load_Train_Data(Train.DataSet_HR,Train.DataSet_HR)
### time domain

    input_LR_placeholder = tf.placeholder(tf.float32,[Train.BatchSize,96,96,3])
    input_HR_placeholder = tf.placeholder(tf.float32,[Train.BatchSize,384,384,3])
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    out = SD_Net(input_LR_placeholder,'SD_Net')

    ###Loss
    Loss_L1 = tf.reduce_mean(input_HR_placeholder - out.model)
    tf.summary.scalar('Loss',Loss_L1)
    tf.summary.scalar('Learning_rate',learning_rate)
    #Loss_F1
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=Train.momentum, use_nesterov=True)
    train_run = optimizer.minimize(Loss_L1)


### frequence domain
    input_LR_placeholder_F = tf.placeholder(tf.float32,[Train.BatchSize,96,96,3])
    input_HR_placeholder_F = tf.placeholder(tf.float32,[Train.BatchSize,384,384,3])
    out_F = SD_Net(input_LR_placeholder_F,'SD_Net_F')

    ###Loss

    Loss_L1_F = tf.reduce_mean(input_HR_placeholder_F - out_F.model)
    tf.summary.scalar('Loss_F',Loss_L1_F)
    saver = tf.train.Saver(tf.global_variables())
    #Loss_F1
    optimizer_F = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=Train.momentum, use_nesterov=True)
    train_run_F = optimizer.minimize(Loss_L1_F)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.getcwd()+'/tensorboard',sess.graph)
        epoch_learning_rate = Train.init_learning_rate
        batch_HR = [None] * Train.BatchSize
        batch_LR = [None] * Train.BatchSize
        batch_HR_F = [None] * Train.BatchSize
        batch_LR_F = [None] * Train.BatchSize
        for i in range(Train.epochs):
            step_time = time.time()
            if i % 200 == 0:
                epoch_learning_rate = epoch_learning_rate/10
            pre_index = 0

            for step in range(0,len(data_HR),Train.BatchSize):
                for n in range(0,Train.BatchSize):
                    batch_HR[n] = data_HR[pre_index+n]
                    batch_LR[n] = data_LR[pre_index+n]

                Loss_run , _ = sess.run([Loss_L1,train_run],{input_LR_placeholder: batch_LR, input_HR_placeholder: batch_HR, learning_rate:epoch_learning_rate})
                print("Epoch is %d , step is %d ,time is %4.4f,loss is %.8f "%(i ,step ,time.time() - step_time ,Loss_run))
                step_time_F = time.time()

                for index in range(0,Train.BatchSize):
                    batch_HR_F[index] = np.fft.fft2(data_HR[pre_index+index])
                    batch_HR_F[index] = np.abs(batch_HR_F[index])
                    batch_LR_F[index] = np.fft.fft2(data_LR[pre_index+index])
                    batch_LR_F[index] = np.abs(batch_LR_F[index])
                print("batch_LR_F shape")
                print(np.array(batch_LR_F).shape())

                Loss_run_F , _, summary = sess.run([Loss_L1_F,train_run_F,merged],{input_LR_placeholder_F:batch_LR_F,input_HR_placeholder_F:batch_HR_F,learning_rate:epoch_learning_rate})
                pre_index = pre_index+Train.BatchSize
                print("Epoch is %d , step is %d ,time is %4.4f,Loss_F is %.8f "%(i ,step ,time.time() - step_time_F ,Loss_run_F))
                summary_writer.add_summary(summary,i*Train.BatchSize+step)

            if i % 100 == 0:
                saver.save(sess=sess,save_path=checkpoint_dir+'SD_Net.ckpt')



def evalution():

    checkpoint_dir = os.getcwd()+'/checkpoint'
    image_out_dir = os.getcwd()+'/image'

    Test_Data_LR,Test_Data_HR = Load_Test_Data(Test.DataSet_HR,Test.DataSet_LR,Test.DataSet_HR,Test.DataSet_LR)
    size = Test_Data_LR[0].shape
    input_LR_placeholder = tf.placeholder(tf.float32,[None,size[0],size[1],3])
    out_HR = SD_Net(input_LR_placeholder,'Test')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_dir+'SD_Net.ckpt')
        for i in range(len(Test_Data_LR)):
            start_time = time.time()
            ssim_sum = 0
            psnr_sum = 0
            HR_image = Test_Data_HR[i]
            LR_image = Test_Data_LR[i]/127.5 -1
            out_image =sess.run([out_HR],{input_LR_placeholder:LR_image})
            print("costing time %4.4f"%(time.time()-start_time))
            print("save image")
            tl.vis.save_image(out_image.model[0], image_out_dir+'/HR_SD%d.png' % i)
            tl.vis.save_image(HR_image, image_out_dir+'/Origin%d.png' % i)
            tl.vis.save_image(LR_image, image_out_dir+'/LR%d.png' % i)
            out_bicu = scipy.misc.imresize(LR_image, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
            tl.vis.save_image(out_bicu, image_out_dir+'/HR_Bicubic%d.png' % i)
            HR_FLOAT = HR_image.astype('float32')
            ssim = measure.compare_ssim(HR_FLOAT,out_image.model[0],win_size=11,gradient=False,multichannel=True,gaussian_weights=True,full=False)
            ssim_sum += ssim
            psnr = measure.compare_psnr(out_image.model[0],HR_FLOAT)
            psnr_sum += psnr
            print("image%d . ssim is %4.4f. psnr is %4.4f"%(i,ssim,psnr))
        ssim_ave = ssim_sum/len(Test_Data_LR)
        psnr_ave = psnr_sum/len(Test_Data_LR)
        print("average ssim is %4.4f.average psnr is %4.4f"%(ssim_ave,ssim_ave))

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train')
    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] =='train':
        train()
    elif tl.global_flag['mode'] =='evalution':
        evalution()
    else:
        raise Exception("Unknow --mode")
