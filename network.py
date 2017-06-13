import tensorflow as tf
from utils import *

class Net:
    ph_image = None
    ph_label = None

    image_h, image_w = (60, 150)
    num_chars = 6

    bn = batch_norm(name='bn')

    batch_size = None

    def __init__(self, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.ph_image = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_h, self.image_w, 1), name='image')
        self.ph_label = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_chars), name='label')

        self.logits = self.inference(self.ph_image)
        self.loss = self.compute_loss()
        self.total_acc_in_char, self.total_acc_in_word = self.compute_acc()
        self.optim = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss)

    def placeholders(self):
        return self.ph_image, self.ph_label

    def _conv_layer(self, input, out_channels, ksize, psize, name):
        with tf.variable_scope(name) as scope:
            l_conv = conv2d(input, out_channels, name='conv')
            l_pool = tf.nn.max_pool(l_conv, ksize=[1,ksize,ksize,1], strides=[1,psize,psize,1], padding='SAME', name='pool')
            l_bn = self.bn(l_pool)
            l_relu = relu(l_bn)
        return l_relu

    def inference(self, image):
        l0 = image
        l1 = self._conv_layer(l0, 32, 5, 2, 'l1')
        l2 = self._conv_layer(l1, 32, 5, 2, 'l2')
        l3 = self._conv_layer(l2, 64, 5, 2, 'l3')
        reshape = tf.reshape(l3, [self.batch_size, -1])
        fcs = []
        for branch in xrange(self.num_chars):
            with tf.variable_scope('fc-char-%s' % branch) as scope:
                fc1 = fc(reshape, 64, name='fc1')
                fc2 = fc(fc1, 64, name='fc2')
            fcs.append(fc2)
        return fcs

    def compute_loss(self):
        label = tf.cast(self.ph_label, tf.int32)
        total_loss = tf.constant(0.0)
        for branch in xrange(self.num_chars):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits[branch], labels=label[:,branch], name='xentropy')
            cross_entropy = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            total_loss += cross_entropy
        total_loss /= self.num_chars
        return total_loss

    def compute_acc(self):
        label = tf.cast(self.ph_label, tf.int64)
        total_acc_in_char = tf.constant(0.0)
        acc_word = tf.constant(True, dtype=tf.bool, shape=(self.batch_size, 1))
        for branch in xrange(self.num_chars):
            acc = tf.equal(tf.argmax(self.logits[branch], 1), label[:,branch])
            acc_word = tf.logical_and(acc_word, acc)
            acc = tf.reduce_mean(tf.cast(acc, tf.float32))
            total_acc_in_char += acc
        total_acc_in_char /= self.num_chars
        total_acc_in_word = tf.reduce_mean(tf.cast(acc_word, tf.float32))
        return total_acc_in_char, total_acc_in_word

