#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   14.06.2017
#-------------------------------------------------------------------------------

import random
import cv2
import os

import numpy as np

from glob import glob

#-------------------------------------------------------------------------------


def build_file_list(images_root, labels_root, sample_name):
    image_sample_root = images_root + '/' + sample_name
    image_root_len = len(image_sample_root)
    label_sample_root = labels_root + '/' + sample_name
    image_files = glob(image_sample_root + '/**/*png')
    file_list = []
    for f in image_files:
        f_relative = f[image_root_len:]
        f_dir = os.path.dirname(f_relative)
        f_base = os.path.basename(f_relative)
        f_base_gt = f_base.replace('leftImg8bit', 'gtFine_color')
        f_label = label_sample_root + f_dir + '/' + f_base_gt
        if os.path.exists(f_label):
            file_list.append((f, f_label))
    return file_list


class LipSource:
    #---------------------------------------------------------------------------
    def __init__(self):
        self.image_size = (416, 416)
        self.num_classes = 2
        self.label_colors = {0: np.array([0, 0, 0]),
                             1: np.array([255, 0, 100])}

        self.num_training = None
        self.num_validation = None
        self.train_generator = None
        self.valid_generator = None

    #---------------------------------------------------------------------------
    def load_data(self, data_dir, valid_fraction):
        """
        Load the data and make the generators
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        images_root = data_dir + '/train/images/'
        labels_root = data_dir + '/train/masks/'

        train_images = build_file_list(images_root, labels_root, 'train')
        valid_images = build_file_list(images_root, labels_root, 'val')

        if len(train_images) == 0:
            raise RuntimeError('No training images found in ' + data_dir)
        if len(valid_images) == 0:
            raise RuntimeError('No validatoin images found in ' + data_dir)

        self.num_training = len(train_images)
        self.num_validation = len(valid_images)
        self.train_generator = self.batch_generator(train_images)
        self.valid_generator = self.batch_generator(valid_images)

    #---------------------------------------------------------------------------
    def batch_generator(self, image_paths):
        def gen_batch(batch_size, names=False):
            road_color = np.array([255, 0, 255])

            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:offset+batch_size]

                images = []
                labels = []
                names_images = []
                names_labels = []
                for image_file in files:
                    label_file = self.label_paths[os.path.basename(image_file)]

                    image = cv2.resize(cv2.imread(image_file), self.image_size)
                    label = cv2.resize(cv2.imread(label_file), self.image_size)

                    label_road = np.all(label == road_color, axis=2)
                    label_bg = np.any(label != road_color, axis=2)
                    label_all = np.dstack([label_bg, label_road])
                    label_all = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)

                    if names:
                        names_images.append(image_file)
                        names_labels.append(label_file)

                if names:
                    yield np.array(images), np.array(labels), \
                        names_images, names_labels
                else:
                    yield np.array(images), np.array(labels)

        return gen_batch

#-------------------------------------------------------------------------------


def get_source():
    return LipSource()
