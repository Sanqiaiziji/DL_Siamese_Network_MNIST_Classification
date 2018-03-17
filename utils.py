from collections import Counter

import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data


class DataProvider:
    """
    Provides data for network
    """

    def __init__(self, batch_size=50, exluded_digits=(1, 5)):
        """
        Downloads MNIST, segregates data into full training, one shot training and validation
        """
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape=False, validation_size=0)
        self.train_data = mnist.train.images  # Returns np.array
        self.train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        self.data_len = self.train_data.shape[0]

        self.excluded_digits = exluded_digits
        self.split_full_oneshot(self.excluded_digits)
        self.full_batch_id = 0
        self.one_shot_batch_id = 0

        if (batch_size % 2) != 0:
            raise Exception("Batch size must be divisible by two!")
        self.batch_size = batch_size

    def num_batches(self):
        """
        Returns batch count for training set
        """
        return len(self.full_data) // self.batch_size

    def split_full_oneshot(self, one_shot_digits):
        """
        Splits training data into full training and one shot training sets.
        :param one_shot_digits: List of digits which will be trained in one shot manner
        """
        self.full_data = []
        self.full_labels = []
        self.one_shot_data = []
        self.one_shot_labels = []

        for i, (img, label) in enumerate(zip(self.train_data, self.train_labels)):
            print('\rSplitting data: {:.2f}%'.format(i / self.data_len * 100), end='', flush=True)
            if np.argmax(label) in one_shot_digits:
                self.one_shot_data.append(img)
                self.one_shot_labels.append(label)
            else:
                self.full_data.append(img)
                self.full_labels.append(label)
        print()

    def get_full_data(self):
        """
        Provides full data for network
        """
        if self.full_batch_id % (len(self.full_data) // self.batch_size) == 0:
            self.full_data, self.full_labels = shuffle(self.full_data, self.full_labels)
            self.full_batch_id = 0

        data = self.full_data[self.full_batch_id * self.batch_size: (self.full_batch_id + 1) * self.batch_size]
        labels = self.full_labels[self.full_batch_id * self.batch_size: (self.full_batch_id + 1) * self.batch_size]
        self.full_batch_id += 1
        return data, labels

    def get_one_shot_data(self, num_data_per_digit):
        """
        Provides one_shot data for network
        """
        self.one_shot_data, self.one_shot_labels = shuffle(self.one_shot_data, self.one_shot_labels)
        cnt = Counter()

        examples = []
        for img, lbl in zip(self.one_shot_data, self.one_shot_labels):
            digit = np.argmax(lbl)
            if cnt[digit] < len(self.excluded_digits) * (1 + num_data_per_digit) * 2:
                examples.append([img, digit])
                cnt[digit] += 1

        examples = sorted(examples, key=lambda item: item[1])

        batches = []

        for i in range(num_data_per_digit):
            for out_digit in self.excluded_digits:
                digits = [example[1] for example in examples]
                id = digits.index(out_digit)
                org_images = [examples[id][0]] * len(self.excluded_digits)
                examples.pop(id)

                comp_images = []
                the_same = []
                label_digits = []
                for in_digit in self.excluded_digits:
                    digits = [example[1] for example in examples]
                    id = digits.index(in_digit)
                    comp_images.append(examples[id][0])
                    the_same.append(1 if in_digit == out_digit else 0)
                    label_digits.append(in_digit)
                    examples.pop(id)

                comp_images, label_digits, the_same = shuffle(comp_images, label_digits, the_same)
                batches.append([org_images, comp_images, the_same, label_digits])

        return batches

