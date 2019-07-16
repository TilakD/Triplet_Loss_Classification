"""Create the sprite image for any dataset"""

import argparse
import glob
import os

import numpy as np
import scipy
from scipy import ndimage

from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model_v2',
                    help="Experiment directory containing params.json")
parser.add_argument('--dataset_dir', default='data_for_model_resized_448*448/test/',
                    help="Directory containing the images")


def _images_to_sprite(dataset_dir, params):
    """Creates the sprite image along with any necessary padding
    Args:
        dataset_dir: directory containing the dataset
        params: contains hyperparameters of the model (ex: `params.image_size`)

    Returns:
        data: properly shaped sprite image with any necessary padding
    """
    data = []

    for d in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, d)
        if os.path.isdir(class_dir):
            # get all jpg images from each class folder
            image_list = glob.glob(class_dir+"/*."+params.image_type)
            print("Loading images of class '%s'..." % os.path.basename(class_dir))
            for addr in image_list:
                img = scipy.misc.imread(addr)
                img =  scipy.misc.imresize(img, (64, 64))
                data.append(img)

    data = np.array(data)
    print("Initial data is of shape: {}".format(data.shape))

    # find out the number of images per row and column in the sprite image (square matrix)
    n = int(np.ceil(np.sqrt(data.shape[0])))

    # pad with empty images (0 values) to achieve an n x n square matrix
    padding = ((0, n**2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim-3)
    data = np.pad(data, padding, mode='constant', constant_values=0)

    # tile the individual thumbnauls into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0,2,1,3) + tuple(range(4, data.ndim+1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    print("Sprite image is of shape: {}".format(data.shape))
    print("Number of images per row and column: {}".format(n))

    return data


if __name__ == '__main__':

    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    sprite = _images_to_sprite(args.dataset_dir, params)
    scipy.misc.imsave(os.path.join(args.model_dir, "sprite_test"+".png"), sprite)


