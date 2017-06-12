import tensorflow as tf
import numpy as np

def run(sess, net, data_test):
    ph_image, ph_label = net.placeholders()
    acc0, acc1 = net.total_acc_in_char, net.total_acc_in_word
    data_test.reset()
    total_acc0, total_acc1 = 0.0, 0.0
    count = 0
    while data_test.epoch < 1:
        image_batch, label_batch = data_test.next_batch()
        image_batch = image_batch.astype(np.float32) / 127.5 - 1.0
        feed_dict = {
            ph_image: image_batch,
            ph_label: label_batch
        }
        acc0_val, acc1_val = sess.run([acc0, acc1], feed_dict=feed_dict)
        total_acc0 += acc0_val
        total_acc1 += acc1_val
        count += 1
    total_acc0 /= count
    total_acc1 /= count
    print('Validation: acc_in_char = %3.6f, acc_in_word = %3.6f' % (total_acc0, total_acc1))