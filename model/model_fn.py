"""Define the model."""

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
from model.triplet_loss import batch_hard_pos_triplet_loss
from model import models


def model_fn(features, labels, mode, params):
    """Model function for tf.estimator

    Args:
        features: input batch of images
        labels: labels of the images
        mode: can be one of tf.estimator.ModeKeys.{TRAIN, EVAL, PREDICT}
        params: contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        model_spec: tf.estimator.EstimatorSpec object
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    channels = 3 if params.rgb else 1
    image_size = params.image_size

    images = tf.reshape(features, [-1, image_size, image_size, channels])
    assert images.shape[1:] == [image_size, image_size, channels], "{}".format(images.shape)

    # MODEL: Compute the embeddings using the specified model
    # -------------------------------------------------------
    tf.logging.info("Current model: {}".format(params.model))
    with tf.variable_scope('model'):
        embeddings = getattr(models, params.model)(is_training, images, params, mode)
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    # Send tensorflow INFO if minimizing the maximum loss instead of mean
    if params.minimize_max:
        tf.logging.info("Minimizing the maximum individual loss at each batch.")

    # Define triplet loss
    if params.triplet_strategy == "batch_all":
        loss_dict = batch_all_triplet_loss(labels, embeddings, params)

    elif params.triplet_strategy == "batch_hard":
        if params.minimize_max:
            tf.logging.info("Batch hard strategy applied while minimizing maximum loss.\nSetting the margin higher than zero may be useless for triplet selection and optimization.")
        loss_dict = batch_hard_triplet_loss(labels, embeddings, params)
        # specific metrics to batch hard
        hardest_positive_dist = loss_dict['hardest_positive_dist']
        hardest_negative_dist = loss_dict['hardest_negative_dist']

    elif params.triplet_strategy == "batch_hard_pos":
        loss_dict = batch_hard_pos_triplet_loss(labels, embeddings, params)
        # specific metrics to batch hard
        hardest_positive_dist = loss_dict['hardest_positive_dist']

    else:
        raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # common general metrics
    loss = loss_dict['triplet_loss']
    fraction_positive_triplets = loss_dict['fraction_positive_triplets']
    max_batch_triplet_loss = loss_dict['max_batch_triplet_loss']
    min_batch_triplet_loss = loss_dict['min_batch_triplet_loss']
    rank1_acc = loss_dict['rank1_accuracy']

    # METRICS AND SUMMARIES
    # -----------------------------------------------------------
    with tf.variable_scope("metrics"):
        eval_metric_ops = {'embedding_mean_norm': tf.metrics.mean(embedding_mean_norm),
                           'fraction_positive_triplets' : tf.metrics.mean(fraction_positive_triplets),
                           'max_batch_triplet_loss': tf.metrics.mean(max_batch_triplet_loss),
                           'min_batch_triplet_loss': tf.metrics.mean(min_batch_triplet_loss),
                           'rank1_accuracy': tf.metrics.mean(rank1_acc)}

        if params.triplet_strategy == "batch_hard":
            eval_metric_ops['hardest_positive_dist'] = tf.metrics.mean(hardest_positive_dist)
            eval_metric_ops['hardest_negative_dist'] = tf.metrics.mean(hardest_negative_dist)

        if params.triplet_strategy == "batch_hard_pos":
            eval_metric_ops['hardest_positive_dist'] = tf.metrics.mean(hardest_positive_dist)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)
    tf.summary.scalar("fraction_positive_triplets", fraction_positive_triplets)
    tf.summary.scalar("max_batch_triplet_loss", max_batch_triplet_loss)
    tf.summary.scalar("min_batch_triplet_loss", min_batch_triplet_loss)
    tf.summary.scalar("rank1_accuracy", rank1_acc)

    if params.triplet_strategy == "batch_hard":
        tf.summary.scalar("hardest_positive_dist", hardest_positive_dist)
        tf.summary.scalar("hardest_negative_dist", hardest_negative_dist)

    if params.triplet_strategy == "batch_hard_pos":
        tf.summary.scalar("hardest_positive_dist", hardest_positive_dist)

    tf.summary.image('train_image', images, max_outputs=8)

    # Define the optimizer based on choice in the configuration file
    optimizers = {'adam': tf.train.AdamOptimizer, 
                  'adagrad': tf.train.AdagradOptimizer,
                  'adadelta': tf.train.AdadeltaOptimizer,
                  'rmsprop': tf.train.RMSPropOptimizer,
                  'gradient_descent': tf.train.GradientDescentOptimizer}

    if params.optimizer in list(optimizers.keys()):
        optimizer = optimizers[params.optimizer](params.learning_rate)
    else:
        raise ValueError("Optimizer not recognized: {}\nShould be in the list {}".format(params.optimizer, list(optimizers.keys())))

    tf.logging.info("Current optimizer: {}".format(params.optimizer))


    # Define training step that minimizes the loss with the chosen optimizer
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        # Add a dependency to update the moving mean and variance for batch normalization
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_op = optimizer.minimize(loss, global_step=global_step)
    else:
        train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
