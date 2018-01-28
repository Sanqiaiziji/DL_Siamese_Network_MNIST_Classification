import tensorflow as tf
import numpy as np
import cv2
import os
from shutil import rmtree

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


def generate_data(path, dataset, images_per_class):
    """
    Generate folder with images
    :param: path to folder
    :param dataset: string, either train or validation
    :return:
    """
    if os.path.isdir(path):
        rmtree(path)
    os.mkdir(path)
    if dataset=='train':
        images = train_data
        labels = train_labels
    elif dataset=='validation':
        images = eval_data
        labels = eval_labels
    else:
        raise Exception("No such dataset!")

    counters = dict(zip([i for i in range(10)], [0]*10))
    class_full = [0] * 10

    i = 0
    while sum(class_full)<10:
        image = images[i].reshape((28,28))
        image*= 255.0
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        label = labels[i]

        if class_full[label]==0:
            cv2.imwrite(path + '/' + str(label) + '_' + str(i) + '.png', image)
            counters[label]+=1
            if counters[label] == images_per_class:
                class_full[label] = 1

            print("\rGenerated: %.2f" % (sum(counters.values())/images_per_class*10), flush=True, end='')
        i += 1
    print()


generate_data('train_images', 'train', 30)
generate_data('val_images', 'validation', 10)