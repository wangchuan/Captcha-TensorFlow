import numpy as np
import math
import tensorflow as tf
import os

def conv2d(input, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        return conv

def relu(x):
    return tf.nn.relu(x, name='relu')

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def fc(input, output_dim, name='fc', stddev=0.02, bias_start=0.0):
    shape = input.get_shape().as_list()
    with tf.variable_scope(name):
        matrix = tf.get_variable("matrix", [shape[1], output_dim], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input, matrix) + bias

def compute_size(s, stride):
    return int(math.ceil(float(s) / float(stride)))

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)
        #deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, w, biases
        else:
            return deconv

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            scale=True, scope=self.name)

def get_subdirs(dir):
    return sorted([name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))])

def get_files(dir, ext='.ann'):
    return sorted([name for name in os.listdir(dir)
				   if os.path.isfile(os.path.join(dir, name)) and os.path.splitext(os.path.join(dir, name))[1]==ext])
