#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

import gzip
import os
import glob
import shutil
import json

import numpy as np
from six.moves import urllib
import tensorflow as tf



def check_dataset_dir(dataset_dir):
    """Validate that dataset directory contains at least two classes."""

    classes = os.listdir(dataset_dir)
    k = len(classes)
    print("Number of train classes is as shown below,")
    print(k)
    if k<2:
        raise ValueError('Invalid data directory %s: Expected at least two classes, found %d' %(dataset_dir, k))


def check_class_dir(class_dir, params):
    """Validate that class directory contains at least 1 image."""

    image_list = glob.glob(class_dir+"/*."+params.image_type)
    m = len(image_list)
    print("Number of images is as shown below,")
    print(m)
    if m<1:
        raise ValueError('Invalid class directory %s: Expected at least 1 ', params.image_type, ' image, found %d' %(class_dir, m))


def _parse_function(filename, label, image_size, channels):
    # Read an image from a file
    # Decode it into a dense vector
    # Resize it to fixed shape
    # Reshape it to 1 dimensonal tensor
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
    image_resized = tf.image.resize_images(image_decoded, [image_size, image_size])
    features = tf.reshape(image_resized, [image_size*image_size*channels])
    features_normalized = features / 255.0
    return features_normalized, label


def save_class_dict(d, json_path):
    """Saves dict of class indexes and their labels into json file

    Args:
        d: (dict) of string values
        json_path: (string) path to json file
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump(d, f, indent=4)


def dataset(dataset_dir, params, class_dict_dir, metadata=False):
    """Load and parse dataset.

    Args:
        dataset_dir: directory containing the train, validation or test folder
        params: contains hyperparameters of the model (ex: `params.learning_rate`)
        class_dict_dir: directory where to save the class dictionary
        metadata: boolean indicating whether to return filenames or features
    """

    check_dataset_dir(dataset_dir)
    print(check_dataset_dir)
    filenames = []
    labels = []
    class_idx = 0
    class_dict = {}

    for d in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, d)
        check_class_dir(class_dir, params)
        if os.path.isdir(class_dir):
            # get all images from each class folder
            image_list = glob.glob(class_dir+"/*."+params.image_type)
            # add class images to filenames
            filenames = filenames + image_list
            # add class labels to labels
            labels = labels + [class_idx] * len(image_list)
            # add index-to-label mapping into class dictionary
            class_dict[class_idx] = d
            # prepare label for next class
            class_idx += 1

    # Save the class dictionary
    json_path = os.path.join(class_dict_dir, 'class_dict_' + os.path.basename(dataset_dir) + '.json')
    save_class_dict(class_dict, json_path)

    # Create the dataset using a python generator to save memory and reduce size of events dump file
    def generator():
        for sample in zip(*(filenames, labels)):
            yield sample            
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_types=(tf.string, tf.int32),
                                             output_shapes=(tf.TensorShape([]), tf.TensorShape([])))

    image_size = params.image_size
    if not metadata:
        channels = 3 if params.rgb else 1
        dataset = dataset.map(lambda filename, label: _parse_function(filename, label, image_size, channels))

    return dataset


def train(data_dir, params, class_dict_dir):
    """tf.data.Dataset object for training data."""
    return dataset(os.path.join(data_dir, "train"), params, class_dict_dir)


def val(data_dir, params, class_dict_dir):
    """tf.data.Dataset object for validation data."""
    return dataset(os.path.join(data_dir, "validation"), params, class_dict_dir)


def test(data_dir, params, class_dict_dir):
    """tf.data.Dataset object for test data."""
    return dataset(os.path.join(data_dir, "test"), params, class_dict_dir)
