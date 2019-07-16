
"""Model architectures"""

import tensorflow as tf


def custom_v1(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    # Define the number of filters for each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    image_size_in = params.image_size
    num_filters = params.num_filters
    num_blocks = params.num_blocks
    bn_momentum = params.bn_momentum
    filters = [num_filters*(2**i) for i in range(num_blocks) ] # each element in this list indicates the number of filters to use in a new conv bloc

    if image_size_in % (2**num_blocks) != 0:
        raise ValueError("Image size ({}) must be a multiple of 2^num_blocks (2^{}).".format(image_size_in, num_blocks))

    out = input_dropout
    for i, f in enumerate(filters):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, f, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    image_size_out = int(image_size_in / (2 ** num_blocks)) # reduction by 2*2 maxpool
    assert out.shape[1:] == [image_size_out, image_size_out, filters[-1]], "filters: {}\nout shape: {}\nimage_size_out: {}".format(filters[-1], out.shape, image_size_out)

    out = tf.reshape(out, [-1, image_size_out * image_size_out * filters[-1]])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


def custom_v2(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    # Define the number of filters for each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    image_size_in = params.image_size
    num_filters = params.num_filters
    num_blocks = params.num_blocks
    bn_momentum = params.bn_momentum
    filters = [num_filters*(2**i) for i in range(num_blocks) ] # each element in this list indicates the number of filters to use in a new conv block

    if image_size_in % (2**num_blocks) != 0:
        raise ValueError("Image size ({}) must be a multiple of 2^num_blocks (2^{}).".format(image_size_in, num_blocks))

    out = input_dropout
    for i, f in enumerate(filters):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, f, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, f, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    image_size_out = int(image_size_in / (2 ** num_blocks)) # reduction by 2*2 maxpool
    assert out.shape[1:] == [image_size_out, image_size_out, filters[-1]], "filters: {}\nout shape: {}\nimage_size_out: {}".format(filters[-1], out.shape, image_size_out)

    out = tf.reshape(out, [-1, image_size_out * image_size_out * filters[-1]])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, 10*params.embedding_size)
    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


def custom_v3(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).

    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    # Define the number of filters for each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    image_size_in = params.image_size

    num_filters = params.num_filters
    num_blocks = params.num_blocks
    bn_momentum = params.bn_momentum
    filters = [32, 64, 128] # each element in this list indicates the number of filters to use in a new conv block

    if params.image_size != 96:
        raise ValueError("Image size should be equal to 96 if you want to use custom_v3.")

    out = input_dropout
    for i, f in enumerate(filters):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, f, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.conv2d(out, f, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    image_size_out = int(image_size_in / (2 ** 3)) # 3 reductions by 2*2 maxpool
    assert out.shape[1:] == [image_size_out, image_size_out, filters[-1]], "filters: {}\nout shape: {}\nimage_size_out: {}".format(filters[-1], out.shape, image_size_out)
    # 12 x 12 x 128

    out = tf.layers.conv2d(out, 64, 1, padding='same')
    # 12 x 12 x 64

    out = tf.layers.average_pooling2d(out, 12, strides=1)
    # 1 x 1 x 64

    out = tf.reshape(out, [-1, 1 * 1 * 64])

    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)
        out = tf.divide(out, tf.expand_dims(tf.norm(out, ord='euclidean', axis=1) + 1e-16, 1))
        out = params.alpha * out
    # 1 x 1 x 64

    return out


def miniception_v1(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    # Define the number of filters for each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    image_size_in = params.image_size
    num_blocs = params.num_blocs

    if params.image_size != 131:
        raise ValueError("image size should be equal to 112 if you want to use miniception_v1.")

    out = input_dropout
    # 131 x 131 x 3

    out = tf.layers.conv2d(out, 32, 7, strides=2)
    assert out.shape[1:] == [63, 63, 32], "output has shape {}".format(out.shape)
    # 63 x 63 x 32

    out = tf.layers.max_pooling2d(out, 3, strides=2)
    assert out.shape[1:] == [31, 31, 32], "output has shape {}".format(out.shape)
    # 31 x 31 x 32

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.conv2d(out, 96, 3, padding='same')
    assert out.shape[1:] == [31, 31, 96], "output has shape {}".format(out.shape)
    # 31 x 31 x 96

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.max_pooling2d(out, 3, strides=2)
    assert out.shape[1:] == [15, 15, 96], "output has shape {}".format(out.shape)
    # 15 x 15 x 96


    # Miniception module 1
    # -------------------
    with tf.variable_scope('miniception_block1'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 32, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 8, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 16, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 48, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 64, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 16, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 15 x 15 x 128

    # Transitional max pooling layer
    # ------------------------------
    out = tf.layers.max_pooling2d(out, 3, strides=2)
    assert out.shape[1:] == [7, 7, 128], "output has shape {}".format(out.shape)
    # 7 x 7 x 128

    # Miniception module 2
    # -------------------------------
    with tf.variable_scope('miniception_block2'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 64, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 16, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 48, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 64, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 96, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 32, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 7 x 7 x 240

    assert out.shape[1:] == [7, 7, 240], "out shape: {}".format(out.shape)

    # Average pooling reduction
    # -------------------------
    out = tf.layers.average_pooling2d(out, 7, strides=1)
    # 1 x 1 x 240

    # Flatten layer with dropout
    # -------------
    out = tf.reshape(out, [-1, 1 * 1 * 240])
    out = tf.layers.dropout(out, rate=output_dropout, training=is_training, name='output_dropout')

    # Final dense layer (embeddings)
    # ------------------------------
    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


def miniception_v2(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')


    if params.image_size != 112:
        raise ValueError("Image size should be equal to 112 if you want to use miniception_v2.")

    out = input_dropout
    # 112 x 112 x 3

    out = tf.layers.conv2d(out, 32, 7, strides=2, padding='same')
    assert out.shape[1:] == [56, 56, 32], "output has shape {}".format(out.shape)
    # 56 x 56 x 32

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [28, 28, 32], "output has shape {}".format(out.shape)
    # 28 x 28 x 32

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.conv2d(out, 96, 3, padding='same')
    assert out.shape[1:] == [28, 28, 96], "output has shape {}".format(out.shape)
    # 28 x 28 x 96

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [14, 14, 96], "output has shape {}".format(out.shape)
    # 14 x 14 x 96

    # Miniception module 1
    # ------------------
    with tf.variable_scope('miniception_block1'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 32, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 8, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 16, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 48, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 64, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 16, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 14 x 14 x 128

    # Transitional max pooling layer
    # ------------------------------
    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [7, 7, 128], "output has shape {}".format(out.shape)
    # 7 x 7 x 128

    # Miniception module 2
    # ------------------
    with tf.variable_scope('miniception_block2'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 64, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 16, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 48, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 64, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 96, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 32, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 7 x 7 x 240

    assert out.shape[1:] == [7, 7, 240], "out shape: {}".format(out.shape)

    # Average pooling reduction
    # -------------------------
    out = tf.layers.average_pooling2d(out, 7, strides=1)
    # 1 x 1 x 240

    # Flatten layer with dropout
    # --------------------------
    out = tf.reshape(out, [-1, 1 * 1 * 240])
    out = tf.layers.dropout(out, rate=params.output_dropout, training=is_training, name='output_dropout')

    # Final dense layer (embeddings)
    # ------------------------------
    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


def miniception_v3(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    if params.image_size != 112:
        raise ValueError("Image size should be equal to 112 if you want to use miniception_v3.")

    out = input_dropout
    # 112 x 112 x 3

    out = tf.layers.conv2d(out, 32, 7, strides=2, padding='same')
    assert out.shape[1:] == [56, 56, 32], "output has shape {}".format(out.shape)
    # 56 x 56 x 32

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [28, 28, 32], "output has shape {}".format(out.shape)
    # 28 x 28 x 32

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.conv2d(out, 32, 1, padding='valid')
    assert out.shape[1:] == [28, 28, 32], "output has shape {}".format(out.shape)
    # 28 x 28 x 32

    out = tf.layers.conv2d(out, 96, 3, padding='same')
    assert out.shape[1:] == [28, 28, 96], "output has shape {}".format(out.shape)
    # 28 x 28 x 96

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [14, 14, 96], "output has shape {}".format(out.shape)
    # 14 x 14 x 96

    # Miniception module 1
    # ------------------
    with tf.variable_scope('miniception_block1'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 32, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 8, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 16, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 48, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 64, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 16, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 14 x 14 x 128

    # Transitional max pooling layer
    # ------------------------------
    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [7, 7, 128], "output has shape {}".format(out.shape)
    # 7 x 7 x 128

    # Miniception module 2
    # ------------------
    with tf.variable_scope('miniception_block2'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 64, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 16, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 48, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 64, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 96, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 32, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 7 x 7 x 240

    assert out.shape[1:] == [7, 7, 240], "out shape: {}".format(out.shape)

    # Average pooling reduction
    # -------------------------
    out = tf.layers.average_pooling2d(out, 7, strides=1)
    # 1 x 1 x 240

    # Flatten layer with dropout
    # --------------------------
    out = tf.reshape(out, [-1, 1 * 1 * 240])
    out = tf.layers.dropout(out, rate=params.output_dropout, training=is_training, name='output_dropout')

    # Final dense layer (embeddings)
    # ------------------------------
    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)

    return out


def miniception_v4(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).
    Adding L2-norm layer to miniception_v2
    (maybe add a learnable scaling parameter alpha, see paper L2-constraint softmax)
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    out = input_dropout
    # 112 x 112 x 3

    if params.image_size != 112:
        raise ValueError("image size should be equal to 112 if you want to use miniception_v4.")

    out = tf.layers.conv2d(out, 32, 7, strides=2, padding='same')
    assert out.shape[1:] == [56, 56, 32], "output has shape {}".format(out.shape)
    # 56 x 56 x 32

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [28, 28, 32], "output has shape {}".format(out.shape)
    # 28 x 28 x 32

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.conv2d(out, 96, 3, padding='same')
    assert out.shape[1:] == [28, 28, 96], "output has shape {}".format(out.shape)
    # 28 x 28 x 96

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [14, 14, 96], "output has shape {}".format(out.shape)
    # 14 x 14 x 96

    # Miniception module 1
    # ------------------
    with tf.variable_scope('miniception_block1'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 32, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 8, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 16, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 48, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 64, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 16, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 14 x 14 x 128

    # Transitional max pooling layer
    # ------------------------------
    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [7, 7, 128], "output has shape {}".format(out.shape)
    # 7 x 7 x 128

    # Miniception module 2
    # ------------------
    with tf.variable_scope('miniception_block2'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 64, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 16, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 48, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 64, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 96, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 32, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 7 x 7 x 240

    assert out.shape[1:] == [7, 7, 240], "out shape: {}".format(out.shape)

    # Average pooling reduction
    # -------------------------
    out = tf.layers.average_pooling2d(out, 7, strides=1)
    # 1 x 1 x 240

    # Flatten layer with dropout
    # --------------------------
    out = tf.reshape(out, [-1, 1 * 1 * 240])
    out = tf.layers.dropout(out, rate=params.output_dropout, training=is_training, name='output_dropout')

    # Final dense layer (embeddings) followed by L2 normalization
    # ------------------------------
    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)
        out = tf.divide(out, tf.expand_dims(tf.norm(out, ord='euclidean', axis=1) + 1e-16, 1))
        out = params.alpha * out

    return out


def miniception_v5(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).
    Adding L2-norm layer to miniception_v2
    (maybe add a learnable scaling parameter alpha, see paper L2-constraint softmax)
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    out = input_dropout
    # 448 x 448 x num_channels

    if params.image_size != 448:
        raise ValueError("Image size should be equal to 448 if you want to use miniception_v5.")

    out = tf.layers.conv2d(out, 16, 7, strides=2, padding='same')
    assert out.shape[1:] == [224, 224, 16], "output has shape {}".format(out.shape)
    # 224 x 224 x 16

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [112, 112, 16], "output has shape {}".format(out.shape)
    # 112 x 112 x 16

    out = tf.layers.conv2d(out, 32, 3, strides=1, padding='same')
    assert out.shape[1:] == [112, 112, 32], "output has shape {}".format(out.shape)
    # 112 x 112 x 32

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [56, 56, 32], "output has shape {}".format(out.shape)
    # 56 x 56 x 16

    out = tf.layers.conv2d(out, 64, 3, strides=1, padding='same')
    assert out.shape[1:] == [56, 56, 64], "output has shape {}".format(out.shape)
    # 56 x 56 x 64

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [28, 28, 64], "output has shape {}".format(out.shape)
    # 28 x 28 x 64

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.conv2d(out, 96, 3, padding='same')
    assert out.shape[1:] == [28, 28, 96], "output has shape {}".format(out.shape)
    # 28 x 28 x 96

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [14, 14, 96], "output has shape {}".format(out.shape)
    # 14 x 14 x 96

    # Miniception module 1
    # ------------------
    with tf.variable_scope('miniception_block1'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 32, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 8, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 16, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 48, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 64, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 16, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 14 x 14 x 128

    # Transitional max pooling layer
    # ------------------------------
    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [7, 7, 128], "output has shape {}".format(out.shape)
    # 7 x 7 x 128

    # Miniception module 2
    # ------------------
    with tf.variable_scope('miniception_block2'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 64, 1, padding='same')
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 16, 1)
            branch5x5 = tf.layers.conv2d(branch5x5, 48, 5, padding='same')
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 64, 1)
            branch3x3 = tf.layers.conv2d(branch3x3, 96, 3, padding='same')
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 32, 1, padding='same')
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 7 x 7 x 240

    assert out.shape[1:] == [7, 7, 240], "out shape: {}".format(out.shape)

    # Average pooling reduction
    # -------------------------
    out = tf.layers.average_pooling2d(out, 7, strides=1)
    # 1 x 1 x 240

    # Flatten layer with dropout
    # --------------------------
    out = tf.reshape(out, [-1, 1 * 1 * 240])
    out = tf.layers.dropout(out, rate=params.output_dropout, training=is_training, name='output_dropout')

    # Final dense layer (embeddings) followed by L2 normalization
    # -----------------------------------------------------------
    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)
        out = tf.divide(out, tf.expand_dims(tf.norm(out, ord='euclidean', axis=1) + 1e-16, 1))
        out = params.alpha * out

    return out


def miniception_v6(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).
    Adding L2-norm layer to miniception_v2
    (maybe add a learnable scaling parameter alpha, see paper L2-constraint softmax)
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    out = input_dropout
    # 448 x 448 x num_channels

    if params.image_size != 448:
        raise ValueError("Image size should be equal to 448 if you want to use miniception_v5.")

    out = tf.layers.conv2d(out, 16, 7, strides=2, padding='same', activation=tf.nn.relu)
    assert out.shape[1:] == [224, 224, 16], "output has shape {}".format(out.shape)
    # 224 x 224 x 16

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [112, 112, 16], "output has shape {}".format(out.shape)
    # 112 x 112 x 16

    out = tf.layers.conv2d(out, 32, 3, strides=1, padding='same', activation=tf.nn.relu)
    assert out.shape[1:] == [112, 112, 32], "output has shape {}".format(out.shape)
    # 112 x 112 x 32

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [56, 56, 32], "output has shape {}".format(out.shape)
    # 56 x 56 x 16

    out = tf.layers.conv2d(out, 64, 3, strides=1, padding='same', activation=tf.nn.relu)
    assert out.shape[1:] == [56, 56, 64], "output has shape {}".format(out.shape)
    # 56 x 56 x 64

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [28, 28, 64], "output has shape {}".format(out.shape)
    # 28 x 28 x 64

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.conv2d(out, 96, 3, padding='same', activation=tf.nn.relu)
    assert out.shape[1:] == [28, 28, 96], "output has shape {}".format(out.shape)
    # 28 x 28 x 96

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [14, 14, 96], "output has shape {}".format(out.shape)
    # 14 x 14 x 96

    # Miniception module 1
    # ------------------
    with tf.variable_scope('miniception_block1'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 32, 1, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 8, 1, activation=tf.nn.relu)
            branch5x5 = tf.layers.conv2d(branch5x5, 16, 5, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 48, 1, activation=tf.nn.relu)
            branch3x3 = tf.layers.conv2d(branch3x3, 64, 3, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 16, 1, padding='same', activation=tf.nn.relu)
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 14 x 14 x 128

    # Transitional max pooling layer
    # ------------------------------
    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [7, 7, 128], "output has shape {}".format(out.shape)
    # 7 x 7 x 128

    # Miniception module 2
    # ------------------
    with tf.variable_scope('miniception_block2'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 64, 1, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 16, 1, activation=tf.nn.relu)
            branch5x5 = tf.layers.conv2d(branch5x5, 48, 5, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 64, 1, activation=tf.nn.relu)
            branch3x3 = tf.layers.conv2d(branch3x3, 96, 3, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 32, 1, padding='same', activation=tf.nn.relu)
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 7 x 7 x 240

    assert out.shape[1:] == [7, 7, 240], "out shape: {}".format(out.shape)

    # Average pooling reduction
    # -------------------------
    out = tf.layers.average_pooling2d(out, 7, strides=1)
    # 1 x 1 x 240

    # Flatten layer with dropout
    # --------------------------
    out = tf.reshape(out, [-1, 1 * 1 * 240])
    out = tf.layers.dropout(out, rate=params.output_dropout, training=is_training, name='output_dropout')

    # Final dense layer (embeddings) followed by L2 normalization
    # -----------------------------------------------------------
    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)
        out = tf.divide(out, tf.expand_dims(tf.norm(out, ord='euclidean', axis=1) + 1e-16, 1))
        out = params.alpha * out

    return out



def miniception_v7(is_training, images, params, mode):
    """Compute outputs of the model (embeddings for triplet loss).
    Adding L2-norm layer to miniception_v2
    (maybe add a learnable scaling parameter alpha, see paper L2-constraint softmax)
    Args:
        is_training: (bool) whether we are training or not
        images: (dict) contains the inputs of the graph (features)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    # Apply dropout to the input layer
    input_dropout = tf.layers.dropout(images, rate=params.input_dropout, training=is_training, name='input_dropout')

    out = input_dropout
    # 448 x 448 x num_channels

    if params.image_size != 448:
        raise ValueError("Image size should be equal to 448 if you want to use miniception_v5.")

    out = tf.layers.conv2d(out, 16, 7, strides=2, padding='same', activation=tf.nn.relu)
    assert out.shape[1:] == [224, 224, 16], "output has shape {}".format(out.shape)
    # 224 x 224 x 16

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [112, 112, 16], "output has shape {}".format(out.shape)
    # 112 x 112 x 16

    out = tf.layers.conv2d(out, 32, 3, strides=1, padding='same', activation=tf.nn.relu)
    assert out.shape[1:] == [112, 112, 32], "output has shape {}".format(out.shape)
    # 112 x 112 x 32

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [56, 56, 32], "output has shape {}".format(out.shape)
    # 56 x 56 x 16

    out = tf.layers.conv2d(out, 64, 3, strides=1, padding='same', activation=tf.nn.relu)
    assert out.shape[1:] == [56, 56, 64], "output has shape {}".format(out.shape)
    # 56 x 56 x 64

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [28, 28, 64], "output has shape {}".format(out.shape)
    # 28 x 28 x 64

    out = tf.nn.local_response_normalization(out)

    out = tf.layers.conv2d(out, 96, 3, padding='same')
    assert out.shape[1:] == [28, 28, 96], "output has shape {}".format(out.shape)
    # 28 x 28 x 96

    out = tf.nn.local_response_normalization(out)

    out = tf.nn.relu(out)

    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [14, 14, 96], "output has shape {}".format(out.shape)
    # 14 x 14 x 96

    # Miniception module 1
    # ------------------
    with tf.variable_scope('miniception_block1'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 32, 1, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 8, 1, activation=tf.nn.relu)
            branch5x5 = tf.layers.conv2d(branch5x5, 16, 5, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 48, 1, activation=tf.nn.relu)
            branch3x3 = tf.layers.conv2d(branch3x3, 64, 3, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 16, 1, padding='same', activation=tf.nn.relu)
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 14 x 14 x 128

    # Transitional max pooling layer
    # ------------------------------
    out = tf.layers.max_pooling2d(out, 3, strides=2, padding='same')
    assert out.shape[1:] == [7, 7, 128], "output has shape {}".format(out.shape)
    # 7 x 7 x 128

    # Miniception module 2
    # ------------------
    with tf.variable_scope('miniception_block2'):
        with tf.variable_scope('branch1x1'):
            branch1x1 = tf.layers.conv2d(out, 64, 1, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch5x5'):
            branch5x5 = tf.layers.conv2d(out, 16, 1, activation=tf.nn.relu)
            branch5x5 = tf.layers.conv2d(branch5x5, 48, 5, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch3x3'):
            branch3x3 = tf.layers.conv2d(out, 64, 1, activation=tf.nn.relu)
            branch3x3 = tf.layers.conv2d(branch3x3, 96, 3, padding='same', activation=tf.nn.relu)
        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling2d(out, 3, strides=1, padding='same')
            branch_pool = tf.layers.conv2d(branch_pool, 32, 1, padding='same', activation=tf.nn.relu)
        out = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3, branch_pool])
        # 7 x 7 x 240

    assert out.shape[1:] == [7, 7, 240], "out shape: {}".format(out.shape)

    # Average pooling reduction
    # -------------------------
    out = tf.layers.average_pooling2d(out, 7, strides=1)
    # 1 x 1 x 240

    # Flatten layer with dropout
    # --------------------------
    out = tf.reshape(out, [-1, 1 * 1 * 240])
    out = tf.layers.dropout(out, rate=params.output_dropout, training=is_training, name='output_dropout')

    # Final dense layer (embeddings) followed by L2 normalization
    # -----------------------------------------------------------
    with tf.variable_scope('fc'):
        out = tf.layers.dense(out, params.embedding_size)
        out = tf.divide(out, tf.expand_dims(tf.norm(out, ord='euclidean', axis=1) + 1e-16, 1))
        out = params.alpha * out

    return out