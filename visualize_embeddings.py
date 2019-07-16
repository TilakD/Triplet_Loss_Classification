"""Visualize the embeddings on training and validation sets"""

import argparse
import os
import glob
import pathlib
import shutil
import json

import numpy as np
import scipy
from scipy import ndimage

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model.utils import Params
from model.model_fn import model_fn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model_v2',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data_for_model_resized_448*448/',
                    help="Directory containing the dataset")


def check_dataset_dir(dataset_dir):
    """Validate that dataset directory contains at least two classes."""
    
    classes = os.listdir(dataset_dir)
    k = len(classes)
    if k<2:
        raise ValueError('Invalid data directory %s: Expected at least two classes, found %d' %(dataset_dir, k))


def check_class_dir(class_dir, params):
    """Validate that class directory contains at least 3 images."""

    image_list = glob.glob(class_dir+"/*."+params.image_type)
    m = len(image_list)
    if m<1:
        raise ValueError('Invalid class directory %s: Expected at least 1 ', params.image_type, ' image, found %d' %(class_dir, m))


def save_class_dict(d, json_path):
    """Saves dict of class indexes and their labels into json file

    Args:
        d: (dict) of string values
        json_path: (string) path to json file
    """
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump(d, f, indent=4)


def _get_metadata(dataset_dir, params, eval_dir):
    """Load and parse dataset.

    Args:
        dataset_dir: directory containing the train and test folders
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        filenames: List of filenames on which to build the dataset
        labels: List of labels corresponding to filenames
    """
    check_dataset_dir(dataset_dir)

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
    json_path = os.path.join(eval_dir, 'class_dict_' + os.path.basename(dataset_dir) + '.json')
    save_class_dict(class_dict, json_path)

    # Random selection of images for visualization if the size is above 1000
    if len(filenames) > 1000:
        tf.logging.info("SPECIAL EVENT: Subsampling dataset {} to 1000 observations.".format(os.path.basename(dataset_dir)))
        idx = np.random.choice(range(len(filenames)), 1000, replace=False)
        filenames = [filenames[i] for i in idx]
        labels = [labels[i] for i in idx]

    return filenames, labels


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


def _get_dataset(filenames, labels, params):

    # A tensor of filenames
    filename_tensor = tf.constant(filenames)

    # The corresponding tensor of labels
    label_tensor = tf.constant(labels)

    # Create the dataset
    image_size = params.image_size
    channels = 3 if params.rgb else 1
    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, label_tensor))
    dataset = dataset.map(lambda filename, label: _parse_function(filename, label, image_size, channels))
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)

    return dataset


def _get_embeddings(dataset_name, filenames, labels, estimator, params):
        
    # Compute embeddings
    tf.logging.info("Predicting on dataset '" + dataset_name + "'...")

    predictions = estimator.predict(lambda: _get_dataset(filenames, labels, params))

    embeddings = np.zeros((len(filenames), params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape for dataset '" + dataset_name + "': {}.".format(embeddings.shape))

    return embeddings


def _images_to_sprite(filenames, params):
    """Creates the sprite image along with any necessary padding
    Args:
        dataset_dir: directory containing the dataset
        params: contains hyperparameters of the model (ex: `params.image_size`)

    Returns:
        data: properly shaped sprite image with any necessary padding
    """
    data = []

    for addr in filenames:
        img = scipy.misc.imread(addr)
        img =  scipy.misc.imresize(img, (params.image_size, params.image_size))
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

    tf.logging.info("Sprite image is of shape: {}".format(data.shape))
    tf.logging.info("Number of images per row and column: {}".format(n))

    return data


def _add_embeddings(dataset_name, filenames, labels, embedding_var, args, params, config, eval_dir):

    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name

    # Load the class labels from json file
    json_path = os.path.join(eval_dir, "class_dict_" + dataset_name + ".json")
    assert os.path.isfile(json_path), "No json class file found at {}".format(json_path)
    with open(json_path) as f:
        class_dict = json.load(f)
    '''
    # Specify where to find the sprite
    sprite_filename = os.path.join(args.model_dir, "sprite_" + dataset_name + ".png")
    # Create sprite image
    sprite = _images_to_sprite(filenames, params)
    scipy.misc.imsave(os.path.join(args.model_dir, "sprite_" + dataset_name + ".png"), sprite)
    # Add sprite image to embedding attributes
    embedding.sprite.image_path = pathlib.Path(sprite_filename).name
    embedding.sprite.single_image_dim.extend([params.image_size, params.image_size])
    # Copy the sprite image to the eval directory
    shutil.copy2(sprite_filename, eval_dir)
    '''
    # Specify where to find the metadata
    metadata_filename = os.path.basename(args.data_dir)+"_metadata_" +dataset_name + ".tsv"
    # Save the metadata file needed for Tensorboard projector
    with open(os.path.join(eval_dir, metadata_filename), 'w') as f:
        f.write('label\tfilename\n')
        for i in range(len(filenames)):
            filename = os.path.basename(filenames[i])
            label = class_dict[str(labels[i])]
            f.write('{}\t{}\n'.format(label, filename))
    embedding.metadata_path = metadata_filename
 
    return config



if __name__ == '__main__':

    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the model
    tf.logging.info("Preparing the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Indicate the evaluation directory
    eval_dir = os.path.join(args.model_dir, "eval")

    # Get metadata
    filenames_train, labels_train = _get_metadata(os.path.join(args.data_dir, "train"), params, eval_dir)
    filenames_val, labels_val = _get_metadata(os.path.join(args.data_dir, "validation"), params, eval_dir)

    # Get training set embeddings and define tensorflow variable
    train_embeddings = _get_embeddings("train", filenames_train, labels_train, estimator, params)
    train_embedding_var = tf.Variable(train_embeddings, "data_for_model_resized_448*448"+"_embeddings_train")
    
    # Get validation set embeddings and define tensorflow variable
    val_embeddings = _get_embeddings("validation", filenames_val, labels_val, estimator, params)
    val_embedding_var = tf.Variable(val_embeddings, "data_for_model_resized_448*448" +"_embeddings_val")
      
    # Save the embedding variables
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(val_embedding_var.initializer)
        sess.run(train_embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))

    # Add embeddings to projector
    config = projector.ProjectorConfig()
    config = _add_embeddings("train", filenames_train, labels_train, train_embedding_var, args, params, config, eval_dir)
    config = _add_embeddings("validation", filenames_val, labels_val, val_embedding_var, args, params, config, eval_dir)

    # Save a config file that TensorBoard will read during startup to visualise the embeddings
    summary_writer = tf.summary.FileWriter(eval_dir)
    projector.visualize_embeddings(summary_writer, config)
