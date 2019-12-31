import numpy as np
import tensorflow as tf
from .backend import keras
from math import sqrt


SCALING_NORMALIZATION = 'scaling'
SOFTMAX_NORMALIZATION = 'softmax'


def _scaling(x, mask):
    if mask is None:
        n = x.shape[1]
        k = sqrt(int(n))
    else:
        k = keras.backend.sqrt(keras.backend.sum(mask))
    return x / k


def _softmax_row(x, mask):
    return keras.activations.softmax(x, axis=-1)


def _softmax_column(x, mask):
    sequence_length = x.shape[1]
    if mask is not None:
        x *= keras.backend.reshape(mask, [-1, sequence_length, 1])
    values = keras.activations.softmax(x, axis=-2)
    return values


class EfficientAttention(keras.layers.Layer):
    def __init__(self, normalization=SCALING_NORMALIZATION, **kwargs):
        super(EfficientAttention, self).__init__(**kwargs)
        assert normalization in [SCALING_NORMALIZATION, SOFTMAX_NORMALIZATION]
        self.supports_masking = True
        self.normalization = normalization

        if normalization == SCALING_NORMALIZATION:
            self.normalization_q = _scaling
            self.normalization_k = _scaling
        elif normalization == SOFTMAX_NORMALIZATION:
            self.normalization_q = _softmax_row
            self.normalization_k = _softmax_column

    def get_config(self):
        cls_config = {
            'normalization': self.normalization,
        }
        base_config = super(EfficientAttention, self).get_config()
        config = dict(base_config, **cls_config)
        return config

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        batch_size, sequence_length, _ = q
        _, _, features_dim = v
        output_shape = (batch_size, sequence_length, features_dim)
        return output_shape

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        super(EfficientAttention, self).build(input_shape)
        self.kt_v_meta = keras.backend.constant(np.zeros([1, v[-1], v[-1]]))
        self.e_meta = keras.backend.constant(np.zeros([1, v[1], v[-1]]))

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            mask = mask[1]
        q = self.normalization_q(q, mask)
        k = self.normalization_k(k, mask)

        _, _, kv = keras.backend.map_fn(
            fn=lambda x: (
                x[0],
                x[1],
                keras.backend.dot(
                    keras.backend.transpose(x[0]),
                    x[1]
                ),
            ),
            elems=(k, v, self.kt_v_meta)
        )
        _, _, e = keras.backend.map_fn(
            fn=lambda x: (
                x[0],
                x[1],
                keras.backend.dot(x[0], x[1]),
            ),
            elems=(q, kv, self.e_meta)
        )
        return e
