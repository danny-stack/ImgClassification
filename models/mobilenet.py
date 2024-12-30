import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DepthwiseSeparableConv(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super().__init__()
        self.depthwise = tf.keras.layers.DepthwiseConv2D(3, strides=strides, padding='same')
        self.pointwise = tf.keras.layers.Conv2D(filters, 1)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return tf.nn.relu(x)

class MobileNet(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.initial = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        self.blocks = [
            DepthwiseSeparableConv(64),
            DepthwiseSeparableConv(128, strides=2),
            DepthwiseSeparableConv(128),
            DepthwiseSeparableConv(256, strides=2),
            DepthwiseSeparableConv(256),
            DepthwiseSeparableConv(512, strides=2)
        ]
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.initial(inputs)
        for block in self.blocks:
            x = block(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
