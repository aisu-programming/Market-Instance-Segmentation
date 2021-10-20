import tensorflow as tf
from module import Mish, MyConv, CSPSPP


class Unet(tf.keras.Model):
    def __init__(self, dropout=None, class_num=4, depth=6):
        super(Unet, self).__init__()
        self.depth = depth
        fs = [ 32, 64, 128, 256, 512, 1024 ]
        """ Backbone """
        self.bb_cv1 = [
            tf.keras.layers.Conv2D(
                fs[di], 3, strides=min(2, di+1), padding="same", activation=tf.nn.silu
            ) for di in range(depth)
        ]
        self.bb_cv2 = [
            tf.keras.layers.Conv2D(
                fs[di], 3, strides=1, padding="same", activation=tf.nn.silu
            ) for di in range(depth)
        ]
        """ Upsample """
        self.us_us = tf.keras.layers.UpSampling2D()
        self.us_cv = [
            tf.keras.layers.Conv2D(
                fs[di], 3, strides=1, padding="same", activation=tf.nn.silu
            ) for di in range(depth-1)
        ]
        """ Detector """
        self.detector = tf.keras.layers.Conv2D(
            class_num+1, 1, padding="same", activation=tf.nn.sigmoid
        )

    @property
    def name(self):
        return "Unet"

    def call(self, x):
        bb_p = {}
        for di in range(self.depth):
            x = self.bb_cv1[di](x)
            x = self.bb_cv2[di](x)
            bb_p[di] = x

        us_p = { self.depth-1: bb_p[self.depth-1]}
        for di in range(self.depth-2, -1, -1):
            us_p[di] = self.us_us(us_p[di+1])
            us_p[di] = tf.concat(
                [ bb_p[di], self.us_cv[di](us_p[di]) ], axis=-1
            )
        return self.detector(us_p[0])


class MyUnet(tf.keras.Model):
    def __init__(self, dropout, class_num=4, depth=6):
        super(MyUnet, self).__init__()
        self.depth = depth
        fs = [ 24, 32, 48, 56, 64, 72 ]
        """ Backbone """
        self.bb_cv = [
            MyConv(fs[di], 3, strides=min(2, di+1)) for di in range(depth)
        ]
        """ Upsample """
        self.cspspp = CSPSPP(fs[depth-1])
        self.us_us = tf.keras.layers.UpSampling2D()
        self.us_cv = [
            MyConv(fs[di], 3, strides=1) for di in range(depth-1)
        ]
        """ Detector """
        # self.detector = tf.keras.layers.Conv2D(class_num+1, 1, padding="same", activation=tf.nn.sigmoid)
        """ Classifier """
        self.classifier = [
            MyConv(32, 3, strides=2),
            MyConv(32, 3, strides=2),
            MyConv(32, 3, strides=2),
            MyConv(24, 3, strides=2),
            MyConv(24, 3, strides=2),
            MyConv(24, 3, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=Mish()),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(3, activation=tf.nn.softmax),
        ]

    @property
    def name(self):
        return "MyUnet"

    def call(self, x):
        bb_p = {}
        for di in range(self.depth):
            x = self.bb_cv[di](x)
            bb_p[di] = x

        us_p = { self.depth-1: tf.concat(
            [ bb_p[self.depth-1], self.cspspp(bb_p[self.depth-1]) ], axis=-1
        )}
        for di in range(self.depth-2, -1, -1):
            us_p[di] = self.us_cv[di](self.us_us(us_p[di+1]))
            us_p[di] = tf.concat(
                [ bb_p[di], us_p[di] ], axis=-1
            )

        # return self.detector(us_p[0])

        x = us_p[0]
        for l in self.classifier: x = l(x)
        return x