import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Shortcut (identity or downsample)
        if strides != 1:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters, 1, strides=strides, use_bias=False),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x

    def call(self, inputs, training=False):
        shortcut = self.shortcut(inputs)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x += shortcut
        return tf.nn.relu(x)

class ResNetCIFAR(tf.keras.Model):
    # def __init__(self, num_classes=10):
    #     super(ResNetCIFAR, self).__init__()
    #     self.conv1 = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False)
    #     self.bn1 = tf.keras.layers.BatchNormalization()

    #     # ResNet layers
    #     self.blocks1 = self._build_block(16, 2, strides=1)
    #     self.blocks2 = self._build_block(32, 2, strides=2)
    #     self.blocks3 = self._build_block(64, 2, strides=2)

    #     self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
    #     self.fc = tf.keras.layers.Dense(num_classes)

    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.blocks1 = self._build_block(64, 3, strides=1)
        self.blocks2 = self._build_block(128, 3, strides=2)
        self.blocks3 = self._build_block(256, 3, strides=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def _build_block(self, filters, blocks, strides):
        layers = [ResBlock(filters, strides)]
        for _ in range(1, blocks):
            layers.append(ResBlock(filters, strides=1))
        return tf.keras.Sequential(layers)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.blocks1(x, training=training)
        x = self.blocks2(x, training=training)
        x = self.blocks3(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)
        return x