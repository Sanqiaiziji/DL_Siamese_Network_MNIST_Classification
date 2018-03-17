import os

import cv2
import numpy as np
import tensorflow as tf

from utils import DataProvider

# params
batch_size = 40
eta = 0.0001
margin = 0.2
num_epochs = 100

# placeholders
images_1_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
images_2_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
label_placeholder = tf.placeholder(dtype=tf.float32, shape=None)


# reusable model for each of two branches
def get_model(inputs):
    with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
        net = tf.layers.conv2d(inputs,
                               filters=64,
                               kernel_size=[3, 3],
                               activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.2)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding='same')

        net = tf.layers.conv2d(net,
                               filters=128,
                               kernel_size=[3, 3],
                               activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.2)
        net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding='same')

        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=[3, 3],
                               activation=tf.nn.relu)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dropout(net, 0.2)

        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 4096, activation=tf.nn.sigmoid)
        return net


branch_1 = get_model(images_1_placeholder)
branch_2 = get_model(images_2_placeholder)

logits = tf.layers.dense(branch_1 - branch_2, 1, activation=None)
sigmoid_logits = tf.squeeze(tf.nn.sigmoid(logits), -1)
thresholded_logits = tf.cast(tf.cast(sigmoid_logits + 0.5, tf.uint8), tf.float32)

loss = tf.losses.sigmoid_cross_entropy(label_placeholder, logits)
accuracy = tf.reduce_sum(tf.cast(tf.equal(thresholded_logits, label_placeholder), tf.float32)) / batch_size

tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)

# optimizer
train_op = tf.train.AdamOptimizer(eta).minimize(loss)

data_provider = DataProvider(batch_size, [1, 5])
full_num_batches = data_provider.num_batches()

if not os.path.isdir('summaries'):
    os.mkdir('summaries')

merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summaries/model', sess.graph)
    sess.run(tf.global_variables_initializer())

    # full training
    for epoch in range(num_epochs):
        for batch in range(full_num_batches):
            data, labels = data_provider.get_full_data()
            labels = [np.argmax(label) for label in labels]

            a_data = data[:batch_size // 2]
            b_data = data[batch_size // 2:]
            labels = np.equal(labels[:batch_size // 2], labels[batch_size // 2:]).astype(np.float32)

            _, cost, acc, summ = sess.run([train_op, loss, accuracy, merged],
                                          feed_dict={images_1_placeholder: a_data, images_2_placeholder: b_data, label_placeholder: labels})
            print(
                'Training on full data: epoch {} of {}, batch {} of {}, cost: {:.4f}, full acc: {:.4f}'.format(epoch + 1, num_epochs, batch + 1,
                                                                                                               full_num_batches, cost,
                                                                                                               acc))
            train_writer.add_summary(summ, epoch * full_num_batches + batch)

        # one shot validation
        val_batches = data_provider.get_one_shot_data(20)
        accuracies = []
        for batch in val_batches:
            org_imgs = batch[0]
            comp_imgs = batch[1]
            the_same = batch[2]

            probs = sess.run(sigmoid_logits, feed_dict={images_1_placeholder: org_imgs, images_2_placeholder: comp_imgs})

            # accuracy
            output = np.zeros_like(probs)
            output[np.argmax(probs)] = 1
            equal = int(np.array_equal(the_same, output))
            accuracies.append(equal)
        print('--------Accuracy after {} epochs: {}'.format(epoch + 1, sum(accuracies) / len(accuracies)))

    # one shot testing
    if not os.path.isdir('sample_imgs'):
        os.mkdir('sample_imgs')
    val_batches = data_provider.get_one_shot_data(20)
    for i, batch in enumerate(val_batches):
        org_imgs = batch[0]
        comp_imgs = batch[1]
        digits = batch[3]
        probs = sess.run(sigmoid_logits, feed_dict={images_1_placeholder: org_imgs, images_2_placeholder: comp_imgs})

        id = np.argmax(probs)
        digit = digits[id]
        cv2.imwrite('sample_imgs/sample_' + str(i) + '_digit_' + str(digit) + '.jpg', (org_imgs[0] * 255).astype(np.uint8))
