import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ResBlock(tf.keras.layers.Layer):
   def __init__(self, filters, strides=1):
       super().__init__()
       self.conv1 = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding='same')
       self.bn1 = tf.keras.layers.BatchNormalization()
       self.conv2 = tf.keras.layers.Conv2D(filters, 3, padding='same') 
       self.bn2 = tf.keras.layers.BatchNormalization()

       if strides != 1:
           self.downsample = tf.keras.Sequential([
               tf.keras.layers.Conv2D(filters, 1, strides=strides),
               tf.keras.layers.BatchNormalization()
           ])
       else:
           self.downsample = None

   def call(self, inputs):
       identity = inputs
       
       x = self.conv1(inputs)
       x = self.bn1(x)
       x = tf.nn.relu(x)
       
       x = self.conv2(x)
       x = self.bn2(x)

       if self.downsample is not None:
           identity = self.downsample(inputs)
           
       x += identity
       return tf.nn.relu(x)

class ResNet18(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(3, strides=2, padding='same')
        
        self.blocks1 = [ResBlock(64) for _ in range(2)]
        self.blocks2 = [ResBlock(128, strides=2)] + [ResBlock(128) for _ in range(1)]
        self.blocks3 = [ResBlock(256, strides=2)] + [ResBlock(256) for _ in range(1)]
        self.blocks4 = [ResBlock(512, strides=2)] + [ResBlock(512) for _ in range(1)]
        
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(num_classes)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        for block in self.blocks1:
            x = block(x)
        for block in self.blocks2:
            x = block(x)
        for block in self.blocks3:
            x = block(x)
        for block in self.blocks4:
            x = block(x)
        
        x = self.avgpool(x)
        x = self.fc(x)
        return x
