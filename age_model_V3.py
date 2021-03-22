# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

l2 = tf.keras.regularizers.l2

class fix_conv2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, kernel_regularizer, dir,  **kwargs):
        super(fix_conv2d, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_regularizer = kernel_regularizer
        self.dir = dir
        
    def build(self, input_shape):
        input_channels = tf.TensorShape(input_shape)[-1]
        kernel_shape = (self.kernel_size, self.kernel_size) + (input_channels, self.filters)
        self.kernel = self.add_weight(
            name="fix_kernel",
            shape=kernel_shape,
            initializer="glorot_uniform",
            regularizer=self.kernel_regularizer,
            trainable=True)
        
        if self.dir == "down":      # residual block에 각각의 dir을 적용하여 학습해보자!! 기억해!! 그리고 마지막단에 일반 conv를 추가
            self.kernel.numpy()[2:4, 0:4, :, :] = 0.
        elif self.dir == "up":
            self.kernel.numpy()[0:2, 0:4, :, :] = 0.   # inception module으로적용해보는것이 더 좋을지도!?!!!
        elif self.dir == "left":
            self.kernel.numpy()[0:4, 0:2, :, :] = 0.
        elif self.dir == "right":
            self.kernel.numpy()[0:4, 2:4, :, :] = 0.
        

        self.params = self.kernel_size // 2 * self.kernel_size * input_channels * self.filters
        
    def call(self, inputs):
        h = tf.nn.conv2d(inputs, self.kernel, self.strides, self.padding)
        return h

def module_1(input, filters, kernel_size, strides, padding, weight_decay):

    bridge = tf.keras.layers.Conv2D(filters=filters,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=l2(weight_decay))(input)
    bridge = tf.keras.layers.BatchNormalization()(bridge)

    bridge_1 = tf.keras.layers.Conv2D(filters=filters // 2,
                                      kernel_size=1,
                                      strides=1,
                                      padding="same",
                                      use_bias=False,
                                      kernel_regularizer=l2(weight_decay))(input)
    bridge_1 = tf.keras.layers.BatchNormalization()(bridge_1)
    bridge_1 = tf.keras.layers.ReLU()(bridge_1)
    bridge_1 = fix_conv2d(filters=filters,
                          kernel_size=kernel_size,
                          strides=1,
                          padding="SAME",
                          kernel_regularizer=l2(weight_decay),
                          dir="down")(bridge_1)
    bridge_1 = tf.keras.layers.BatchNormalization()(bridge_1)

    bridge_2 = tf.keras.layers.Conv2D(filters=filters // 2,
                                      kernel_size=1,
                                      strides=1,
                                      padding="same",
                                      use_bias=False,
                                      kernel_regularizer=l2(weight_decay))(input)
    bridge_2 = tf.keras.layers.BatchNormalization()(bridge_2)
    bridge_2 = tf.keras.layers.ReLU()(bridge_2)
    bridge_2 = fix_conv2d(filters=filters,
                          kernel_size=kernel_size,
                          strides=1,
                          padding="SAME",
                          kernel_regularizer=l2(weight_decay),
                          dir="up")(bridge_2)
    bridge_2 = tf.keras.layers.BatchNormalization()(bridge_2)

    bridge_3 = tf.keras.layers.Conv2D(filters=filters // 2,
                                      kernel_size=1,
                                      strides=1,
                                      padding="same",
                                      use_bias=False,
                                      kernel_regularizer=l2(weight_decay))(input)
    bridge_3 = tf.keras.layers.BatchNormalization()(bridge_3)
    bridge_3 = tf.keras.layers.ReLU()(bridge_3)
    bridge_3 = fix_conv2d(filters=filters,
                          kernel_size=kernel_size,
                          strides=1,
                          padding="SAME",
                          kernel_regularizer=l2(weight_decay),
                          dir="left")(bridge_3)
    bridge_3 = tf.keras.layers.BatchNormalization()(bridge_3)

    bridge_4 = tf.keras.layers.Conv2D(filters=filters // 2,
                                      kernel_size=1,
                                      strides=1,
                                      padding="same",
                                      use_bias=False,
                                      kernel_regularizer=l2(weight_decay))(input)
    bridge_4 = tf.keras.layers.BatchNormalization()(bridge_4)
    bridge_4 = tf.keras.layers.ReLU()(bridge_4)
    bridge_4 = fix_conv2d(filters=filters,
                          kernel_size=kernel_size,
                          strides=1,
                          padding="SAME",
                          kernel_regularizer=l2(weight_decay),
                          dir="right")(bridge_4)
    bridge_4 = tf.keras.layers.BatchNormalization()(bridge_4)

    h = (bridge_1 + bridge_2 + bridge_3 + bridge_4) + bridge

    return tf.keras.layers.ReLU()(h)

def model_V3(input_shape=(256, 256, 3),
             weight_decay=0.000005,
             num_classes=98):
    c = 0   # 데이터를 얻었으니까 이 모델에 적용해보자!
    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=4,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 128]

    h = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=2, padding="same")(h)    # [128, 128, 128]

    for _ in range(1):
        h = module_1(h, 256, 4, 1, "SAME", weight_decay)

    h = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=2, padding="same")(h)    # [64, 64, 256]

    for _ in range(1):
        h = module_1(h, 512, 4, 1, "SAME", weight_decay)

    h = tf.keras.layers.MaxPool2D(pool_size=(4,4), strides=2, padding="same")(h)    # [32, 32, 512]

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [16, 16, 512]

    h = tf.keras.layers.Conv2D(filters=1024,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [8, 8, 512]

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(num_classes)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def modified_GEINet(input_shape=(128, 88, 3),
                    weight_decay=0.000005,
                    num_classes=98):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 88, 64]
    
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h) # [64, 44, 64]

    h = fix_conv2d(filters=128,
                   kernel_size=4,
                   strides=1,
                   padding="SAME",
                   kernel_regularizer=l2(weight_decay),
                   dir="left")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 44, 128]

    h = fix_conv2d(filters=128,
                   kernel_size=4,
                   strides=1,
                   padding="SAME",
                   kernel_regularizer=l2(weight_decay),
                   dir="down")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 44, 128]

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h) # [32, 22, 128]

    h = fix_conv2d(filters=256,
                   kernel_size=4,
                   strides=1,
                   padding="SAME",
                   kernel_regularizer=l2(weight_decay),
                   dir="left")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [32, 22, 256]

    h = fix_conv2d(filters=256,
                   kernel_size=4,
                   strides=1,
                   padding="SAME",
                   kernel_regularizer=l2(weight_decay),
                   dir="right")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [32, 22, 256]

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h) # [16, 11, 256]

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [16, 11, 512]

    h = tf.keras.layers.GlobalAveragePooling2D()(h)

    h = tf.keras.layers.Dense(num_classes)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

model = modified_GEINet()
model.summary()