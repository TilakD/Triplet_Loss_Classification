"""Save the embeddings and labels of all landmarks"""

import argparse
import os
import json

import numpy as np
import tensorflow as tf

import model.create_dataset as create_dataset
from model.utils import Params
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model_v2',
                    help="Experiment directory containing params.json")
parser.add_argument('--landmark_dir', default='data_for_model_resized_448*448/train',
                    help="Directory containing the landmark dataset")



def _get_dataset(landmark_dir, params, class_dict_dir):

    dataset = create_dataset.dataset(landmark_dir, params, class_dict_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)

    return dataset


def _get_metadata(landmark_dir, params, class_dict_dir, landmark_size):

    dataset = create_dataset.dataset(landmark_dir, params, class_dict_dir, metadata=True)
    dataset = dataset.batch(landmark_size)
    metadata = dataset.make_one_shot_iterator().get_next()

    return metadata


def _get_dataset_size(landmark_dir, image_type):

	size = 0
	for root, dirs, files in os.walk(landmark_dir):
	    files = [f for f in files if "."+image_type in f ]
	    size += len(files)

	tf.logging.info("Found {} {} landmarks in {}".format(size, image_type, landmark_dir))

	return size


def _get_embeddings(landmark_dir, estimator, params, class_dict_dir, landmark_size):
        
    # Compute embeddings
    tf.logging.info("Predicting on "+landmark_dir)    

    predictions = estimator.predict(lambda: _get_dataset(landmark_dir, params, class_dict_dir))

    embeddings = np.zeros((landmark_size, params.embedding_size))
    for i, p in enumerate(predictions):
        embeddings[i] = p['embeddings']

    tf.logging.info("Embeddings shape in "+os.path.basename(landmark_dir)+": {}".format(embeddings.shape))

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

    # Create a new folder to save embeddings
    embeddings_dir = os.path.join(args.model_dir, "landmarks")
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Get the number of landmarks
    landmark_size = _get_dataset_size(os.path.normpath(args.landmark_dir), params.image_type)

    # Get embeddings and define tensorflow variables
    embeddings = _get_embeddings(args.landmark_dir, estimator, params, embeddings_dir, landmark_size)
    embeddings = np.round(embeddings, 6)

    # Get the metadata (filenames and labels)
    metadata = _get_metadata(args.landmark_dir, params, embeddings_dir, landmark_size)

    # Load the class labels from json file
    json_path = os.path.join(embeddings_dir, "class_dict_"+os.path.basename(args.landmark_dir)+".json")
    assert os.path.isfile(json_path), "No json class file found at {}".format(json_path)
    with open(json_path) as f:
        class_dict = json.load(f)

    # Save the landmark embeddings to disk in a text file
    with tf.Session() as sess:
      	filenames, labels = sess.run(metadata)
      	with open(os.path.join(embeddings_dir, 'embeddings.txt'), 'w') as f:
      		f.write('label\tfilename\tembeddings\n')
      		for i in range(landmark_size):
      			emb = embeddings[i,:]
      			filename = filenames[i].decode("utf-8")
      			label = class_dict[str(labels[i])]
      			f.write("%s\t%s\t%s\n" %(label, filename, ','.join(map(str,emb))))
