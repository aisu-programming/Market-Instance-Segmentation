import tensorflow as tf


class Mish(tf.keras.layers.Layer):
    def forward(self, x):
        return x * tf.nn.softplus(x).tanh()


class MyConv(tf.keras.layers.Layer):
    def __init__(self, filter, kernel_size, strides):
        super(MyConv, self).__init__()
        self.cv  = tf.keras.layers.Conv2D(filter, kernel_size=kernel_size, strides=strides, padding="same", use_bias=False)
        self.bn  = tf.keras.layers.BatchNormalization()
        self.act = tf.nn.silu

    def call(self, inputs):
        return self.act(self.bn(self.cv(inputs)))


class CSPSPP(tf.keras.layers.Layer):
    def __init__(self, filter, k=(5, 9, 13)):
        super(CSPSPP, self).__init__()
        self.cv1 = MyConv(filter, kernel_size=1, strides=1)
        self.cv2 = tf.keras.layers.Conv2D(filter, kernel_size=1, strides=1, use_bias=False)
        self.cv3 = MyConv(filter, kernel_size=3, strides=1)
        self.cv4 = MyConv(filter, kernel_size=1, strides=1)
        self.m  = [ tf.keras.layers.MaxPool2D(pool_size=x, strides=1, padding="same") for x in k ]
        self.cv5 = MyConv(filter, kernel_size=1, strides=1)
        self.cv6 = MyConv(filter, kernel_size=3, strides=1)
        self.bn  = tf.keras.layers.BatchNormalization() 
        self.act = Mish()
        self.cv7 = MyConv(filter, kernel_size=1, strides=1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(tf.concat([x1] + [m(x1) for m in self.m], axis=1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(tf.concat([y1, y2], axis=1))))