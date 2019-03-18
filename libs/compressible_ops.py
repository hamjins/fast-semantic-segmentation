r"""Ops compatable with filter pruner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.slim.layers import convolution


@add_arg_scope
def convolution2d_compressible(inputs,
                               num_outputs,
                               kernel_size,
                               stride=1,
                               padding='SAME',
                               compression_ratio=1.0,  # The additional arg
                               data_format=None,
                               rate=1,
                               activation_fn=None,
                               normalizer_fn=None,
                               normalizer_params=None,
                               weights_initializer=None,
                               weights_regularizer=None,
                               biases_initializer=None,
                               biases_regularizer=None,
                               reuse=None,
                               variables_collections=None,
                               outputs_collections=None,
                               trainable=True,
                               prediction_output=False,
                               scope=None):
    if weights_initializer is None:
        weights_initializer = tf.contrib.layers.xavier_initializer()
    if biases_initializer is None:
        biases_initializer = tf.zeros_initializer()
    if not prediction_output:
        activation_fn = tf.nn.relu
        normalizer_fn = None
    num_filter_with_compression = num_outputs // compression_ratio
    return convolution(inputs,
                       num_filter_with_compression,
                       kernel_size,
                       stride,
                       padding,
                       data_format,
                       rate,
                       activation_fn,
                       normalizer_fn,
                       normalizer_params,
                       weights_initializer,
                       weights_regularizer,
                       biases_initializer,
                       biases_regularizer,
                       reuse,
                       variables_collections,
                       outputs_collections,
                       trainable,
                       scope,
                       conv_dims=2)


# Export aliases.
compressible_conv2d = convolution2d_compressible  # pylint: disable=C0103
conv2d_compressible = convolution2d_compressible  # pylint: disable=C0103,E0303
conv2d = convolution2d_compressible  # pylint: disable=C0103,E0303
