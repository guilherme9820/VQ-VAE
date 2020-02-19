import tensorflow as tf
from tensorflow.keras.layers import Lambda, Add, Conv2DTranspose
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras import Model


class ResnetBlock(tf.keras.layers.Layer):

    def __init__(self,
                 out_channels,
                 kernel_dim=3,
                 initializer='he_normal',
                 strides=1,
                 trainable=True,
                 downsample=True,
                 name='residual'):

        super().__init__(name=name, trainable=trainable)

        if downsample:
            self._out_channels = out_channels
        else:
            self._out_channels = int(out_channels // 2)

        self.initializer = initializer

        self.batch1 = BatchNormalization(name=name + '_batch1', trainable=trainable)
        self.relu1 = LeakyReLU(name=name + '_lrelu1')

        if downsample:
            self.conv1 = Conv2D(int(out_channels // 2), kernel_size=1, name=name + '_conv1', strides=1,
                                trainable=trainable, padding='same', kernel_initializer=initializer)
            self.conv2 = Conv2D(int(out_channels // 2), kernel_size=kernel_dim, name=name + '_conv2', strides=strides,
                                trainable=trainable, padding='same', kernel_initializer=initializer)
            self.conv3 = Conv2D(out_channels, kernel_size=1, name=name + '_conv3', strides=1,
                                trainable=trainable, padding='same', kernel_initializer=initializer)
        else:
            self.conv1 = Conv2DTranspose(out_channels, kernel_size=1, name=name + '_conv1', strides=1,
                                         trainable=trainable, padding='same', kernel_initializer=initializer)
            self.conv2 = Conv2DTranspose(int(out_channels // 2), kernel_size=kernel_dim, name=name + '_conv2', strides=strides,
                                         trainable=trainable, padding='same', kernel_initializer=initializer)
            self.conv3 = Conv2DTranspose(int(out_channels // 2), kernel_size=1, name=name + '_conv3', strides=1,
                                         trainable=trainable, padding='same', kernel_initializer=initializer)

        self.batch2 = BatchNormalization(name=name + '_batch2', trainable=trainable)
        self.relu2 = LeakyReLU(name=name + '_lrelu2')

        self.batch3 = BatchNormalization(name=name + '_batch3', trainable=trainable)
        self.relu3 = LeakyReLU(name=name + '_lrelu3')

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

    def __init__(self,
                 conv_kernel,
                 conv_stride,
                 res_kernels=[3, 3, 3],
                 res_channels=[32, 64, 128],
                 res_strides=[1, 1, 1],
                 trainable=True,
                 initializer='he_normal',
                 name='encoder'):

        super().__init__(name=name, trainable=trainable)

        self.input_conv = Conv2D(res_channels[0], kernel_size=conv_kernel, strides=conv_stride, name=name + '_input_conv',
                                 trainable=trainable, padding='same', kernel_initializer=initializer)

        self.res_blocks = [ResnetBlock(res_channels[i], res_kernels[i], strides=res_strides[i], initializer=initializer,
                                       trainable=trainable, name=name + f"res_block{i+1}")
                           for i in range(len(res_channels))]

    def call(self, inputTensor):

        x = self.input_conv(inputTensor)

        for residual in self.res_blocks:
            x = residual(x)

        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 outputs,
                 conv_kernel,
                 conv_stride,
                 res_kernels=[3, 3, 3],
                 res_channels=[128, 64, 32],
                 res_strides=[1, 1, 1],
                 trainable=True,
                 initializer='he_normal',
                 name='decoder'):

        super().__init__(name=name, trainable=trainable)

        self.res_blocks = [ResnetBlock(res_channels[i], res_kernels[i], strides=res_strides[i], initializer=initializer,
                                       trainable=trainable, downsample=False, name=name + f"res_block{i+1}")
                           for i in range(len(res_channels))]

        self.output_conv = Conv2DTranspose(outputs, kernel_size=conv_kernel, strides=conv_stride, name=name + '_output_conv',
                                           trainable=trainable, padding='same', kernel_initializer=initializer)

    def call(self, inputTensor):

        x = inputTensor

        for residual in self.res_blocks:
            x = residual(x)

        return self.output_conv(x)


if __name__ == "__main__":

    from tensorflow.keras.layers import Input
    import numpy as np

    encoder = Encoder(3, 1, [3, 3, 3], [32, 64, 128], [1, 1, 1], initializer='he_normal')
    decoder = Decoder(1, 3, 1, [3, 3, 3], [128, 64, 32], [1, 1, 1], initializer='he_normal')

    inputTensor = Input(shape=(28, 28, 1))

    x = encoder(inputTensor)

    x = decoder(x)

    model = Model(inputs=inputTensor, outputs=x)

    model.compile(loss='mse', optimizer='sgd')

    model.summary()
