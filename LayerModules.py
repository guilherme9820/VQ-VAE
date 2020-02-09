import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Lambda, Add


class ResnetBlock(tf.keras.layers.Layer):

    def __init__(self, out_channels, kernel_dim=3, initializer='he_normal', trainable=True, name='residual'):
        super().__init__(name=name, trainable=trainable)

        self._out_channels = out_channels
        self.initializer = initializer

        self.batch1 = BatchNormalization(name=name + '_batch1', trainable=trainable)
        self.relu1 = LeakyReLU(name=name + '_lrelu1')
        self.conv1 = Conv2D(int(out_channels // 2), kernel_size=1, name=name + '_conv1',
                            trainable=trainable, padding='same', kernel_initializer=initializer)

        self.batch2 = BatchNormalization(name=name + '_batch2', trainable=trainable)
        self.relu2 = LeakyReLU(name=name + '_lrelu2')
        self.conv2 = Conv2D(int(out_channels // 2), kernel_size=kernel_dim, name=name + '_conv2',
                            trainable=trainable, padding='same', kernel_initializer=initializer)

        self.batch3 = BatchNormalization(name=name + '_batch3', trainable=trainable)
        self.relu3 = LeakyReLU(name=name + '_lrelu3')
        self.conv3 = Conv2D(out_channels, kernel_size=1, name=name + '_conv3',
                            trainable=trainable, padding='same', kernel_initializer=initializer)

    def build(self, input_shape):

        if input_shape[-1] != self._out_channels:
            # Adjusts shortcut connection if input dimension is different from output dimension
            self.shortcut = Conv2D(self._out_channels, kernel_size=1, name=self.name + '_shortcut',
                                   trainable=self.trainable, padding='same', kernel_initializer=self.initializer)
        else:
            # Identity mapping
            self.shortcut = Lambda(lambda x: x, output_shape=input_shape, name=self.name + '_shortcut')

    def call(self, inputTensor, training=True):

        x = self.batch1(inputTensor, training=training)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.batch2(x, training=training)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.batch3(x, training=training)
        x = self.relu3(x)
        x = self.conv3(x)

        shortcut = self.shortcut(inputTensor)

        return Add()([x, shortcut])


class Encoder(tf.keras.layers.Layer):

    def __init__(self, conv_kernel, conv_stride, res_kernels=[3, 3, 3], initializer='he_normal',
                 res_channels=[32, 64, 128], trainable=True, name='encoder'):

        super().__init__(name=name, trainable=trainable)

        self.conv1 = Conv2D(res_channels[0], kernel_size=conv_kernel, strides=conv_stride, name=name + '_conv1',
                            trainable=trainable, padding='same', kernel_initializer=initializer)

        self.res_blocks = [ResnetBlock(res_channels[i], res_kernels[i], initializer=initializer,
                                       trainable=trainable, name=name + f"res_block{i+1}")
                           for i in range(len(res_channels))]

    def call(self, inputTensor):

        x = self.conv1(inputTensor)

        for residual in self.res_blocks:
            x = residual(x)

        return x
