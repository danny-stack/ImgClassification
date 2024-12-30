import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class DenseBlock(layers.Layer):
    def __init__(self, num_layers, growth_rate):
        super().__init__()
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                keras.Sequential([
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.Conv2D(4 * growth_rate, 1),
                    layers.BatchNormalization(),
                    layers.ReLU(),
                    layers.Conv2D(growth_rate, 3, padding='same'),
                ])
            )
    
    def call(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = tf.concat([x, new_features], axis=-1)
        return x

class TransitionLayer(layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.layers = keras.Sequential([
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, 1),
            layers.AveragePooling2D(2)
        ])
    
    def call(self, x):
        return self.layers(x)

class DenseNet(keras.Model):
    def __init__(self, num_classes, growth_rate=32):
        super().__init__()
        
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(3, strides=2, padding='same')
        
        num_channels = 64
        # Dense blocks
        self.dense1 = DenseBlock(6, growth_rate)
        num_channels += 6 * growth_rate
        self.trans1 = TransitionLayer(num_channels // 2)
        num_channels //= 2
        
        self.dense2 = DenseBlock(12, growth_rate)
        num_channels += 12 * growth_rate
        self.trans2 = TransitionLayer(num_channels // 2)
        num_channels //= 2
        
        self.dense3 = DenseBlock(24, growth_rate)
        num_channels += 24 * growth_rate
        self.trans3 = TransitionLayer(num_channels // 2)
        num_channels //= 2
        
        self.dense4 = DenseBlock(16, growth_rate)
        
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        
        x = self.avgpool(x)
        x = self.fc(x)
        return x