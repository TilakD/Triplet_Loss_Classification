"""Find the nearest neighbor Save the embeddings and labels of all landmarks"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import model.create_dataset as create_dataset
from model.utils import Params
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model_v2',
                    help="Experiment directory containing params.json")
parser.add_argument('--filename', default='data_for_model_resized_448*448/test/image_193.jpg',
                    help="Path to inference image")


def _parse_function(filename, image_size, channels):
    # Read an image from a file
    # Decode it into a dense vector
    # Resize it to fixed shape
    # Reshape it to 1 dimensional tensor
    # Normalize it from [0, 255] to [0.0, 1.0]
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=channels)
    image_resized = tf.image.resize_images(image_decoded, [image_size, image_size])
    features = tf.reshape(image_resized, [image_size*image_size*channels])
    features_normalized = features / 255.0
    return features_normalized


def _get_dataset(filename, params):

    # A tensor of only one filename
    filename_tensor = tf.constant([filename])

    # Load necessary params from config file
    image_size = params.image_size
    channels = 3 if params.rgb else 1

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
    dataset = dataset.map(lambda filename: _parse_function(filename, image_size, channels))
    dataset = dataset.batch(1)

    return dataset


def _get_embeddings(filename, estimator, params):
        
    # Compute embeddings for an image file
    tf.logging.info("Predicting on "+filename)

    predictions = estimator.predict(lambda: _get_dataset(filename, params))

    embeddings = np.zeros((1, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Image embedding shape: {}".format(embeddings.shape))

    return embeddings



if __name__ == '__main__':

    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Get embeddings
    embeddings = _get_embeddings(args.filename, estimator, params)
    embeddings = np.round(embeddings, 6)

    # Load landmark embeddings from disk
    data = pd.read_csv(os.path.join(args.model_dir, "landmarks/embeddings.txt"), sep="\t")
    data['embeddings'] = data['embeddings'].apply(lambda x : list(map(float, x.split(','))))

    # Compute image distance to all landmarks
    data['dist'] = data['embeddings'].apply(lambda x : np.linalg.norm(embeddings - np.array(x)))
    data['dist'] = np.round(data['dist'], 6)

    # Get index of nearest neighbor
    nnidx = data['dist'].idxmin()
    
    # Output predicted class and distance
    label = data['label'][nnidx]
    dist = data['dist'][nnidx]
    print("Predicted class: {}".format(label))
    print("Distance: {}".format(dist))
