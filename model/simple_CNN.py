import re
import tensorflow as tf
from .module import Mish


class SimpleCNN(tf.keras.Model):
    def __init__(self, dropout=None, class_num=4, depth=5):
        super(SimpleCNN, self).__init__()
        self.depth = depth
        fs = [ 32, 64, 96, 112, 128 ]
        """ Backbone """
        self.cv = [
            tf.keras.layers.Conv2D(
                fs[di], 3, strides=2, padding="same", activation=tf.nn.silu
            ) for di in range(depth)
        ]
        self.bn = [
            tf.keras.layers.BatchNormalization() for _ in range(depth)
        ]
        """ Classifier """
        self.classifier = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(192, activation=Mish()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(1, activation=tf.nn.relu),
        ]

    @property
    def name(self):
        return "simpleCNN"

    def call(self, x):
        for di in range(self.depth):
            x = self.cv[di](x)
            x = self.bn[di](x)
        for c in self.classifier: x = c(x)
        return x