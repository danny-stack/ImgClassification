import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class WideResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, width_factor=8):
        super().__init__()
        width = filters * width_factor

        self.conv1 = tf.keras.layers.Conv2D(width, 3, strides=strides, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.conv2 = tf.keras.layers.Conv2D(width, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 修改 downsample 的逻辑
        self.downsample = None
        if strides != 1 or width != filters:  # 宽度或步幅需要调整
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.Conv2D(width, 1, strides=strides),
                tf.keras.layers.BatchNormalization()
            ])

    def call(self, inputs, training=False):
        identity = inputs

        # 主路径
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # 调整 identity 的形状
        if self.downsample is not None:
            identity = self.downsample(inputs)

        # 确保加法操作维度一致
        x += identity
        return tf.nn.relu(x)


class WideResNet(tf.keras.Model):
    def __init__(self, num_classes=10, width_factor=8, depth=16):
        super().__init__()
        n = (depth - 4) // 6

        self.conv1 = tf.keras.layers.Conv2D(16, 3, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()

        self.layer1 = tf.keras.Sequential([WideResBlock(16, 1, width_factor) for _ in range(n)])
        self.layer2 = tf.keras.Sequential([WideResBlock(32, 2, width_factor)] + [WideResBlock(32, 1, width_factor) for _ in range(n-1)])
        self.layer3 = tf.keras.Sequential([WideResBlock(64, 2, width_factor)] + [WideResBlock(64, 1, width_factor) for _ in range(n-1)])
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        
        x = tf.nn.relu(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
