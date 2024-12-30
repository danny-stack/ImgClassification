import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PatchEmbed(layers.Layer):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = layers.Conv2D(embed_dim, patch_size, strides=patch_size)
        
    def call(self, x):
        x = self.proj(x)
        return tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]])

class VisionTransformer(keras.Model):
    def __init__(
        self, 
        input_shape,
        patch_size,
        num_classes,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()
        
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        
        self.cls_token = self.add_weight(
            "cls_token", shape=[1, 1, embed_dim],
            initializer="zeros", trainable=True
        )
        self.pos_embed = self.add_weight(
            "pos_embed", shape=[1, num_patches + 1, embed_dim],
            initializer="zeros", trainable=True
        )
        
        # self.blocks = [
        #     TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
        #     for _ in range(depth)
        # ]
        self.blocks = tf.keras.Sequential([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = layers.LayerNormalization()
        self.head = layers.Dense(num_classes)
        
    def call(self, x):
        B = tf.shape(x)[0]
        x = self.patch_embed(x)
        
        cls_tokens = tf.repeat(self.cls_token, B, axis=0)
        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

class TransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0):
        super().__init__()
        self.norm1 = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads, dim//num_heads)
        self.norm2 = layers.LayerNormalization()
        self.mlp = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio)),
            layers.Activation('gelu'),
            layers.Dense(dim)
        ])
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.dropout(x)
        return x