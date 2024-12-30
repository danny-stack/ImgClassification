import tensorflow as tf
from tensorflow.keras import layers, models

def SqueezeNet(input_shape=(32, 32, 3), num_classes=10):
    inputs = tf.keras.Input(input_shape)
    
    x = layers.Conv2D(64, 3, strides=1, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)

    # Fire modules
    def fire_module(x, squeeze_filters, expand_filters):
        squeeze = layers.Conv2D(squeeze_filters, 1, activation='relu')(x)
        expand1x1 = layers.Conv2D(expand_filters, 1, activation='relu')(squeeze)
        expand3x3 = layers.Conv2D(expand_filters, 3, padding='same', activation='relu')(squeeze)
        return layers.Concatenate()([expand1x1, expand3x3])

    x = fire_module(x, 8, 32)
    x = fire_module(x, 8, 32)
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)  # 修改 padding='same'
    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    # 移除或调整最后的池化层
    x = fire_module(x, 32, 128)
    x = fire_module(x, 32, 128)
    x = layers.GlobalAveragePooling2D()(x)  # 直接使用 GAP 替代部分池化操作
    x = layers.Dense(num_classes)(x)
    
    return models.Model(inputs, x)

# model = SqueezeNet(input_shape=(32, 32, 3), num_classes=10)
# model.summary()


# model = SqueezeNet(num_classes=10)


