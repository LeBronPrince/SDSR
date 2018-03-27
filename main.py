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

    data_HR = Load_Train_Data(Train.DataSet_HR,Train.DataSet_HR)
### time domain

    input_LR_placeholder = tf.placeholder('float32',[Train.BatchSize,32,32,3],name='input_LR_placeholder')
    input_HR_placeholder = tf.placeholder('float32',[Train.BatchSize,128,128,3],name='input_HR_placeholder')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    out = SD_Net(input_LR_placeholder,'SD_Net')

    ###Loss
    #Loss_L1 = tf.abs((tf.reduce_mean(input_HR_placeholder - out.model)))
    Mse_Loss = tl.cost.mean_squared_error(out.model , input_HR_placeholder, is_mean=True)
    tf.summary.scalar('Loss',Mse_Loss)
    tf.summary.scalar('Learning_rate',learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate,beta1=Train.beta)
    train_run = optimizer.minimize(Mse_Loss)



### frequence domain
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        merged = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.getcwd()+'/tensorboard',sess.graph)
        epoch_learning_rate = Train.init_learning_rate
        batch_HR = [None] * Train.BatchSize
        batch_LR = [None] * Train.BatchSize

        for i in range(Train.epochs):
            step_time = time.time()
            pre_index = 0
            for step in range(0,len(data_HR),Train.BatchSize):
                for n in range(0,Train.BatchSize):
                    batch_HR[n] = crop_imgs(data_HR[pre_index+n], is_random=True)
                    batch_LR[n] = downsample(batch_HR[n])
                    batch_HR[n] = batch_HR[n].astype('float32')
                    batch_LR[n] = batch_LR[n].astype('float32')
                Loss_run , _ ,summary= sess.run([Mse_Loss,train_run,merged],{input_LR_placeholder: batch_LR, input_HR_placeholder: batch_HR, learning_rate:epoch_learning_rate})
                print("Epoch is %d , step is %d ,time is %4.4f,loss is %.8f "%(i ,step ,time.time() - step_time ,Loss_run))
                pre_index = pre_index+Train.BatchSize
            epoch_iter = (i + 1) // Train.lr_delay
            epoch_learning_rate = epoch_learning_rate/2*epoch_iter
            summary_writer.add_summary(summary,i)

            if i % 1000 == 0:
                saver.save(sess=sess,save_path=checkpoint_dir+'/SD_Net.ckpt')



def evalution():

    checkpoint_dir = os.getcwd()+'/checkpoint'
    image_out_dir = os.getcwd()+'/image'
    tl.files.exists_or_mkdir(image_out_dir)

    Test_Data_HR,Test_Data_LR = Load_Test_Data(Test.DataSet_HR,Test.DataSet_LR,Test.DataSet_HR,Test.DataSet_LR)
    size = Test_Data_LR[0].shape
    input_LR_placeholder = tf.placeholder(tf.float32,[1,size[0],size[1],3],name='LR')
    input_HR_placeholder = tf.placeholder(tf.float32,[1,size[0]*4,size[1]*4,3],name='HR')
    out_HR = tf.identity(SD_Net(input_LR_placeholder,'Test').model)
    #Loss_L1 = tf.abs((tf.reduce_mean(input_HR_placeholder - out_HR.model)))
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess,checkpoint_dir+'/SD_Net.ckpt')
        for i in range(len(Test_Data_LR)):
            start_time = time.time()
            ssim_sum = 0
            psnr_sum = 0
            HR_image = Test_Data_HR[i]
            LR_image = Test_Data_LR[i]/127.5 -1
            out_image =sess.run(out_HR,{input_LR_placeholder:[LR_image]})
            print("costing time %4.4f"%(time.time()-start_time))
            print("save image")
            print(np.array(out_image[0]).shape)
            scipy.misc.imsave(image_out_dir+'/HR_SD%d.png' % i,out_image[0])
            scipy.misc.imsave(image_out_dir+'/Origin%d.png' % i,HR_image)
            scipy.misc.imsave(image_out_dir+'/LR%d.png' % i,LR_image)
            out_bicu = scipy.misc.imresize(LR_image, [size[0]*4, size[1]*4], interp='bicubic', mode=None)
            scipy.misc.imsave(image_out_dir+'/HR_Bicubic%d.png' % i,out_bicu)
            HR_FLOAT = HR_image.astype('float32')
            """
            ssim = measure.compare_ssim(HR_FLOAT,out_HR.model[0],win_size=11,gradient=False,multichannel=True,gaussian_weights=True,full=False)
            ssim_sum += ssim
            psnr = measure.compare_psnr(out_HR.model[0],HR_FLOAT)
            psnr_sum += psnr
            print("image%d . ssim is %4.4f. psnr is %4.4f"%(i,ssim,psnr))
        ssim_ave = ssim_sum/len(Test_Data_LR)
        psnr_ave = psnr_sum/len(Test_Data_LR)
        print("average ssim is %4.4f.average psnr is %4.4f"%(ssim_ave,ssim_ave))
            """
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
