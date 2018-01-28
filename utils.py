import random

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


class DataProvider:
    def __init__(self, path):
        filenames = os.listdir(path)
        self.images = [np.expand_dims(cv2.imread(os.path.join(path, filename), 0), 3) for filename in filenames]
        self.labels = [filename.split('_')[0] for filename in filenames]
        self.data_len = len(self.images)
        self.i = 0
        self.j = 0

    def get_batch(self, batch_size):
        indices = random.sample(range(0, self.data_len), batch_size*2)
        indices_1 = indices[:int(len(indices)/2)]
        indices_2 = indices[int(len(indices) / 2):]
        images_1 = [self.images[i]/255 for i in indices_1]
        images_2 = [self.images[i]/255 for i in indices_2]
        labels_1 = [int(self.labels[i]) for i in indices_1]
        labels_2 = [int(self.labels[i]) for i in indices_2]
        the_same = np.expand_dims(1-np.equal(labels_1, labels_2).astype(np.float32), 1)
        return images_1, images_2, the_same

    def get_single_data(self):
        image1 = self.images[self.i] / 255
        image2 = self.images[self.j] / 255
        label1 = int(self.labels[self.i])
        label2 = int(self.labels[self.j])
        distance = 0 if label1==label2 else 1
        self.j+=1
        if self.j == self.data_len:
            self.j=0
            self.i+=1
        return [image1], [image2], [distance]




# generate_data('train_images', 'train', 30)
# generate_data('val_images', 'validation', 5)