"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

import model.create_dataset as create_dataset


def train_input_fn(data_dir, params, class_dict_dir):
    """Train input function.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = create_dataset.train(data_dir, params, class_dict_dir)
    dataset = dataset.shuffle(params.train_size, reshuffle_each_iteration=True)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def val_input_fn(data_dir, params, class_dict_dir):
    """Validation input function.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = create_dataset.val(data_dir, params, class_dict_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


def test_input_fn(data_dir, params, class_dict_dir):
    """Test input function.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = create_dataset.test(data_dir, params, class_dict_dir)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset
