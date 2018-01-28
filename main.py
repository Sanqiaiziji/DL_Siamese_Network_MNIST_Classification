import tensorflow as tf
import numpy as np
import cv2
from utils import DataProvider


# params
margin = 0.2


# placeholders
images_1_pl = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
images_2_pl = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
the_same_pl = tf.placeholder(dtype=tf.float32, shape=None)

# model
def get_model(inputs):
    with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv2d(inputs,
                                 filters=4,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.leaky_relu)
        batch_norm1 = tf.layers.batch_normalization(conv1)
        dropout1 = tf.layers.dropout(batch_norm1, 0.2)

        conv2 = tf.layers.conv2d(dropout1,
                                 filters=8,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.leaky_relu)
        batch_norm2 = tf.layers.batch_normalization(conv2)
        dropout2 = tf.layers.dropout(batch_norm2, 0.2)

        conv3 = tf.layers.conv2d(dropout2,
                                 filters=8,
                                 kernel_size=[3, 3],
                                 padding='same',
                                 activation=tf.nn.leaky_relu)
        batch_norm3 = tf.layers.batch_normalization(conv3)
        dropout3 = tf.layers.dropout(batch_norm3, 0.2)

        flatten = tf.layers.flatten(dropout3)

        dense1 = tf.layers.dense(flatten, 500, activation=tf.nn.leaky_relu)
        dense2 = tf.layers.dense(dense1, 500, activation=tf.nn.leaky_relu)

        output = tf.layers.dense(dense2, 10)
        return output

out_1 = get_model(images_1_pl)
out_2 = get_model(images_2_pl)

# loss
Dw = tf.reduce_sum(tf.sqrt(tf.square(out_1 - out_2)), axis=1, keep_dims=True)
loss = tf.multiply(1-the_same_pl,tf.square(Dw))# + the_same_pl * 0.5 * tf.square(tf.maximum(0.0, margin - Dw))






batch_size = 40
provider = DataProvider('train_images')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    images_1, images_2, the_same = provider.get_batch(batch_size)
    out = sess.run(loss, feed_dict={images_1_pl: images_1, images_2_pl: images_2, the_same_pl: the_same})
    print(out.shape)

