from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import pdb
import cv2

import do_validate

def run(FLAGS, sess, net, saver, data_train, data_test):
    loss = net.loss
    acc0, acc1 = net.total_acc_in_char, net.total_acc_in_word

    loss_summary = tf.summary.scalar('loss', loss)
    acc0_summary = tf.summary.scalar('total_acc_in_char', acc0)
    acc1_summary = tf.summary.scalar('total_acc_in_word', acc1)
    all_summary = tf.summary.merge([loss_summary, acc0_summary, acc1_summary])
    summary_writer = tf.summary.FileWriter(FLAGS.log_path, sess.graph)

    optim = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    ph_image, ph_label = net.placeholders()
    prev_epoch = data_train.epoch
    while data_train.epoch < FLAGS.epoches:
        image_batch = None
        while (True):
            image_batch, label_batch = data_train.next_batch()
            if image_batch.shape[0] == FLAGS.batch_size:
                break
        if False:
            im1 = image_batch[0]
            cv2.imshow('im1', im1)
            print(label_batch[0])
            cv2.waitKey(0)

        image_batch = image_batch.astype(np.float32) / 127.5 - 1.0
        feed_dict = {
            ph_image: image_batch,
            ph_label: label_batch
        }

        _, loss_val, summary_str = sess.run([optim, loss, all_summary], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, data_train.iteration)

        if data_train.iteration % FLAGS.disp == 0:
            print('Iter[%04d]: loss: %3.6f' % (data_train.iteration, loss_val))
        if prev_epoch != data_train.epoch:
            print('Epoch[%03d] finished' % data_train.epoch, end=' ')
            do_validate.run(sess, net, data_test)
            saver.save(sess, os.path.join(FLAGS.log_path, 'model.ckpt'), data_train.iteration)
        prev_epoch = data_train.epoch

