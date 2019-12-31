import numpy as np
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
    r"""
    The attention layer that implements https://arxiv.org/pdf/1812.01243.pdf paper \
        (near to scaled-dot-product attention in case of softmax activations, equal in case of scaling activations).
    """
    def __init__(self, normalization=SCALING_NORMALIZATION, **kwargs):
        """
        Initialize attention layer.
        :param normalization: normalization type. Could be one of "scaling" (SCALING_NORMALIZATION), "softmax" (SOFTMAX_NORMALIZATION)
        """
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
            q, _, v = input_shape
        else:
            q = _ = v = input_shape
        batch_size, sequence_length, _ = q
        _, _, features_dim = v
        output_shape = (batch_size, sequence_length, features_dim)
        return output_shape

    def build(self, input_shape):
        # I need to build two placeholders - to pass info `tf.map_fn` function.
        if isinstance(input_shape, list):
            _, _, v = input_shape
        else:
            _ = _ = v = input_shape
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
        # Implemented attention defined next way (if we're talking about individual batch elements):
        # `E(Q, K, V) = norm_Q(Q) x (norm_K(K)^T x V)`

        # So, we're applying normalizations
        q = self.normalization_q(q, mask)
        k = self.normalization_k(k, mask)
        # Then dot product each batch element's `norm_K(K)^T x V`
        _, _, kv = keras.backend.map_fn(
            fn=lambda x: (x[0], x[1],
                          keras.backend.dot(
                              keras.backend.transpose(x[0]),
                              x[1]
                          )),
            elems=(k, v, self.kt_v_meta)
        )
        # And finally dot product each batch element's `norm_Q(Q) x (norm_K(K)^T x V)`
        _, _, e = keras.backend.map_fn(
            fn=lambda x: (x[0], x[1],
                          keras.backend.dot(x[0], x[1])),
            elems=(q, kv, self.e_meta)
        )
        return e
