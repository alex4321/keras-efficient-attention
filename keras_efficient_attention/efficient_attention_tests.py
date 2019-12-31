from unittest import TestCase
import tensorflow as tf
import numpy as np
from .efficient_attention import EfficientAttention
from keras_self_attention import ScaledDotProductAttention
from .backend import keras


RANDOM_SEED = 20191231
LOOP_COUNT = 50
DIFF_MEAN_THRESOLD = 0.01
DIFF_MAX_THRESHOLD = 0.03
MASKED_MEAN_THRESHOLD = 0.031
MASKED_MAX_THRESHOLD = 0.15


class TestEfficientAttention(TestCase):
    def test_efficient_attention(self):
        input_shape = (16, 64, 128)
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        for i in range(LOOP_COUNT):
            q = np.random.rand(*input_shape)
            k = np.random.rand(*input_shape)
            v = np.random.rand(*input_shape)

            inputs = [keras.Input(input_shape[1:]),
                      keras.Input(input_shape[1:]),
                      keras.Input(input_shape[1:])]
            ea = EfficientAttention(normalization='softmax')(inputs)
            sda = ScaledDotProductAttention()(inputs)
            model = keras.Model(inputs, [ea, sda])
            model.compile(optimizer='sgd', loss='mse')

            out1, out2 = model.predict([q, k, v])
            diff = np.abs(out1 - out2)
            self.assertLess(diff.mean(), DIFF_MEAN_THRESOLD)
            self.assertLess(diff.max(), DIFF_MAX_THRESHOLD)

    def test_masked_efficient_attention(self):
        input_shape = (16, 64, 128)
        mask_length = 32
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        for i in range(LOOP_COUNT):
            q = np.random.rand(*input_shape)
            k = np.random.rand(*input_shape)
            v = np.random.rand(*input_shape)
            mask = np.array([
                [1] * mask_length + [0] * (input_shape[1] - mask_length)
                for _ in range(input_shape[0])
            ])

            inputs = [keras.Input(input_shape[1:]),
                      keras.Input(input_shape[1:]),
                      keras.Input(input_shape[1:])]
            mask_input = keras.Input((input_shape[1],))
            ea = EfficientAttention(normalization='softmax')(inputs, mask=mask_input)
            sda = ScaledDotProductAttention()(inputs, mask=mask_input)
            model = keras.Model(inputs + [mask_input], [ea, sda])
            model.compile(optimizer='sgd', loss='mse')

            out1, out2 = model.predict([q, k, v, mask])
            diff = np.abs(out1 - out2)
            self.assertLess(diff.mean(), MASKED_MEAN_THRESHOLD)
            self.assertLess(diff.max(), MASKED_MAX_THRESHOLD)
