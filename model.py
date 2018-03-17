import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np

reduction_ratio = 4

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        if activation :
            network = tf.nn.relu(network)
        return network

def Fully_connected(x, units, layer_name='fully_connected'):
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def get_incoming_shape(incoming):
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, np.ndarray, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")

def global_avg_pool(incoming, name="GlobalAvgPool"):
    input_shape = get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    with tf.name_scope(name):
        inference = tf.reduce_mean(incoming, [1, 2])
    # Track output tensor.
    #tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)
    return inference


def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = global_avg_pool(input_x,layer_name+'global_avg_pool')
        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = tf.nn.relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = tf.nn.sigmoid(excitation)
        excitation = tf.reshape(excitation, [-1,1,1,out_dim])
        scale = input_x * excitation
    return scale

class SD_Unit():
    def __init__(self,x,name_scope):
        self.model = self.Build_Unit(x,name_scope)

    def Build_Unit(self,x,name_scope):
        with tf.name_scope(name_scope):
            conv1 = conv_layer(x,filter=32,kernel=[3,3],stride=1,padding='SAME',layer_name="conv3")
            conv2 = conv_layer(x,filter=32,kernel=[3,3],stride=1,padding='SAME',layer_name="conv5_1")
            conv3 = conv_layer(conv2,filter=48,kernel=[3,3],stride=1,padding='SAME',layer_name="conv5_2")
            conv4 = conv_layer(x,filter=32,kernel=[3,3],stride=1,padding='SAME',layer_name="conv7_1")
            conv5 = conv_layer(conv4,filter=48,kernel=[3,3],stride=1,padding='SAME',layer_name="conv7_2")
            conv6 = conv_layer(conv5,filter=64,kernel=[3,3],stride=1,padding='SAME',layer_name="conv7_3")
            inception = tf.concat([conv1,conv3,conv6],axis=3)
            print("this is Squeeze_excitation_layeroutyput shape")
            print(int(inception.get_shape().as_list()[-1]))
            out = Squeeze_excitation_layer(inception,int(inception.get_shape().as_list()[-1]),ratio=reduction_ratio,layer_name=name_scope)
        return out


class SD_Block():
    def __init__(self,input_x,name_scope,layers_num):
        self.model = self.Build_Block(input_x,name_scope,layers_num)

    def Build_Block(self,input_x,name_scope,layers_num):
        with tf.name_scope(name_scope):
            layers_cat = list()
            layers_cat.append(input_x)
            x = SD_Unit(input_x,name_scope=name_scope+str(0)).model
            layers_cat.append(x)
            for i in range(layers_num-1):
                x = tf.concat(layers_cat,axis=3)
                x = SD_Unit(x,name_scope=name_scope+str(i+1)).model
                layers_cat.append(x)
            x = tf.concat(layers_cat,axis=3)
            x = conv_layer(x,filter=int(input_x.get_shape().as_list()[-1]),kernel=[1,1],stride=1,padding='VALID',layer_name=name_scope+"conv11")
            out = x +input_x
        return out


class SD_Net():
    def __init__(self,x,name_scope):
        self.model = self.Build_Net(x,name_scope)

    def Build_Net(self,x,name_scope):
        with tf.name_scope(name_scope):
            conv1 = conv_layer(x,filter=64,kernel=[3,3],padding='SAME',stride=2,layer_name=name_scope+'conv1')
            conv2 = conv_layer(x,filter=64,kernel=[3,3],stride=2,padding='SAME',layer_name=name_scope+"conv2")

            ###SD_Block
            SD_B1 = SD_Block(conv2,name_scope=name_scope+'Block1',layers_num=4).model
            SD_B2 = SD_Block(SD_B1,name_scope=name_scope+'Block2',layers_num=4).model
            SD_B3 = SD_Block(SD_B2,name_scope=name_scope+'Block3',layers_num=4).model
            SD_B4 = SD_Block(SD_B3,name_scope=name_scope+'Block4',layers_num=4).model
            SD_B5 = SD_Block(SD_B4,name_scope=name_scope+'Block5',layers_num=4).model
            SD_B6 = SD_Block(SD_B5,name_scope=name_scope+'Block6',layers_num=4).model

            Block_End = tf.concat([SD_B1,SD_B2,SD_B3,SD_B4,SD_B5,SD_B6],axis=3)
            GLL_11 = conv_layer(Block_End,filter=64,kernel=[1,1],stride=1,padding='VALID',layer_name=name_scope+"GLL_11")
            GLL_33 = conv_layer(GLL_11,filter=64,kernel=[3,3],stride=1,padding='SAME',layer_name=name_scope+"GLL_33")
            UpSample = conv2+GLL_33
            UpSample = conv_layer(UpSample,filter=64,kernel=[3,3],stride=1,padding='SAME',layer_name=name_scope+"UpSample")
            input_layer = InputLayer(UpSample, name=name_scope+'in_tensorlayer')
            Subpixel = SubpixelConv2d(input_layer, scale=8, n_out_channel=None, act=tf.nn.relu, name=name_scope+'Subpixel')
            out = conv_layer(Subpixel.outputs,filter=3,kernel=[3,3],stride=1,padding='SAME',layer_name=name_scope+"out")
        return out
