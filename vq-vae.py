import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, Add, Flatten
from tensorflow.keras.layers import Embedding, BatchNormalization
import numpy as np
import os
import sys
import time


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, random_gen=False, only_lookup=False, name='vq', trainable=True):
        super(VectorQuantizer, self).__init__(name=name, trainable=trainable)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        # Instantiate embedding table with initialized weights
        self._embedding = Embedding(num_embeddings, embedding_dim)
        # self._random_gen = random_gen
        # self._only_lookup = only_lookup

    def build(self, input_shape):
        super().build(input_shape)

        self.emb_weights = self._embedding.weights[0]

    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            result = tf.nn.embedding_lookup(self.emb_weights, encoding_indices)

        return result

    def call(self, inputTensor, inputs_indices=None):
        '''
        Note: 
            shape_of_inputs=[batch_size, ?, ?, embedding_dim]
        '''
        # Assert last dimension of inputs is same as embedding_dim
        assert_dim = tf.assert_equal(tf.shape(inputTensor)[-1], self._embedding_dim)
        with tf.control_dependencies([assert_dim]):
            flat_inputs = tf.reshape(inputTensor, [-1, self._embedding_dim])

        # if (self._only_lookup == False):
        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                     - 2 * tf.matmul(flat_inputs, tf.transpose(self.emb_weights))
                     + tf.reduce_sum(tf.transpose(self.emb_weights)**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])  # shape=[batch_size, ?, ?]

        quantized = self.quantize(encoding_indices)

        # else:
        #     encoding_indices = tf.cast(inputs_indices, tf.int32)
        #     encodings = tf.one_hot(tf.reshape(encoding_indices, [-1,]), self._num_embeddings)

        # Important Note:
        #   quantized is differentiable w.r.t. tf.transpose(self.emb_vq),
        #   but is not differentiable w.r.t. encoding_indices.

        inp_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
        emb_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs))**2)
        loss = emb_latent_loss + self._commitment_cost * inp_latent_loss  # used to optimize self.emb_vq only!

        self.add_loss(loss, inputs=True)

        quantized = inputs + tf.stop_gradient(quantized - inputs)
        # Important Note:
        #   This step is used to copy the gradient from inputs to quantized.

        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))
        # The perplexity is the exponentiation of the entropy,
        # indicating how many codes are 'active' on average.
        # We hope the perplexity is larger.

        return {'quantize': quantized,
                'loss': loss,
                'perplexity': perplexity,
                'encodings': encodings,
                'encoding_indices': encoding_indices}

        # if (self._random_gen == False):
        #     return quantized, loss, perplexity, encodings, encoding_indices
        # else:
        #     rand_encoding_indices = tf.random_uniform(tf.shape(encoding_indices), minval=0, maxval=1)
        #     rand_encoding_indices = tf.floor(rand_encoding_indices * self._num_embeddings)
        #     rand_encoding_indices = tf.clip_by_value(rand_encoding_indices, 0, self._num_embeddings - 1)
        #     rand_encoding_indices = tf.cast(rand_encoding_indices, tf.int32)

        #     rand_quantized = tf.nn.embedding_lookup(tf.transpose(self.emb_vq), rand_encoding_indices)

        #     near_encoding_indices = tf.cast(encoding_indices, tf.float32) + tf.random_uniform(tf.shape(encoding_indices), minval=-1, maxval=1)
        #     near_encoding_indices = tf.clip_by_value(near_encoding_indices, 0, self._num_embeddings - 1)
        #     near_encoding_indices = tf.rint(near_encoding_indices)
        #     near_encoding_indices = tf.cast(near_encoding_indices, tf.int32)

        #     near_quantized = tf.nn.embedding_lookup(tf.transpose(self.emb_vq), near_encoding_indices)

        #     return quantized, loss, perplexity, encodings, encoding_indices, rand_quantized, near_quantized


class VectorQuantizerEMA(tf.keras.Model):  # NOT FINISHED YET
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5, random_gen=False, only_lookup=False):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

        self._ema_cluster_size = tf.zeros(num_embeddings)

    def __call__(self, inputs, training=True):
        '''
        Note: 
            shape_of_inputs=[batch_size, ?, ?, embedding_dim]
        '''
        # Assert last dimension of inputs is same as embedding_dim
        assert_dim = tf.assert_equal(tf.shape(inputs)[-1], embedding_dim)
        with tf.control_dependencies([assert_dim]):
            flat_inputs = tf.reshape(inputs, [-1, embedding_dim])

        if (self._only_lookup == False):
            distances = tf.reduce_sum(flat_inputs**2, 1, keepdims=True) - 2 * tf.matmul(flat_inputs, emb_vq) + tf.reduce_sum(emb_vq**2, 0, keepdims=True)
            encoding_indices = tf.argmax(-distances, 1)
            encodings = tf.one_hot(encoding_indices, num_embeddings)
            encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])  # shape=[batch_size, ?, ?]
        else:
            inputs_indices = tf.cast(inputs_indices, tf.int32)
            encoding_indices = inputs_indices
            encodings = tf.one_hot(tf.reshape(encoding_indices, [-1, ]), num_embeddings)

        # Use EMA to update the embedding vectors
        if training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * tf.reduce_sum(encodings, axis=0)

            # Laplace smoothing of the cluster size
            n = tf.reduce_sum(self._ema_cluster_size)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            # dw = torch.matmul(encodings.t(), flat_input)
            dw = tf.nn.embedding_lookup(tf.transpose(flat_input), tf.transpose(encoding_indices))
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        quantized = tf.nn.embedding_lookup(tf.transpose(emb_vq), encoding_indices)

        # Quantize and unflatten
        # quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class ResnetIdentityBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_dim=3, name='residual', trainable=True):
        super().__init__(name=name, trainable=trainable)
        self._out_channels = out_channels

        # conv1 = Conv2D(int(out_channels // 2), kernel_size=1, name=name + '_conv1', trainable=trainable, padding='same')
        self.batch1 = BatchNormalization(name=name + '_batch1', trainable=trainable)
        self.relu1 = LeakyReLU(name=name + '_lrelu1')

        self.conv2 = Conv2D(int(out_channels // 2), kernel_size=kernel_dim, name=name + '_conv2', trainable=trainable, padding='same', kernel_initializer='he_normal')
        self.batch2 = BatchNormalization(name=name + '_batch2', trainable=trainable)
        self.relu2 = LeakyReLU(name=name + '_lrelu2')

        self.conv3 = Conv2D(out_channels, kernel_size=1, name=name + '_conv3', trainable=trainable, padding='same', kernel_initializer='he_normal')
        self.batch3 = BatchNormalization(name=name + '_batch3', trainable=trainable)

        self.relu3 = LeakyReLU(name=name + '_lrelu3')

    def call(self, inputTensor, training=True):
        """Create a Residual Block with two conv layers"""

        # x = conv1(inputTensor)
        x = self.batch1(inputTensor, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.batch3(x, training=training)

        # What type of shortcut connection to use
        if inputTensor.shape[-1] != self._out_channels:
            shortcut = tf.pad(inputTensor, [[0, 0], [0, 0], [0, 0], [0, self._out_channels - inputTensor.shape[-1]]])
        else:
            # Identity mapping.
            shortcut = inputTensor

        added = Add()([x, shortcut])

        return self.relu3(added)


class Encoder(tf.keras.Model):

    def __init__(self, res_kernels=[3, 3, 3], res_channels=[32, 64, 128], conv_kernels=[3, 4, 5], conv_strides=[1, 2, 3], trainable=True, name='encoder'):
        super().__init__(name=name, trainable=trainable)

        self._res_blocks = []
        self._downscalings = []

        for i, (channel, res_k, conv_k, stride) in enumerate(zip(res_channels, res_kernels, conv_kernels, conv_strides)):
            self._res_blocks.append(ResnetIdentityBlock(channel, res_k, name=name + f"res_block{i + 1}", trainable=trainable))
            self._downscalings.append(Conv2D(channel, kernel_size=conv_k, strides=stride, name=name + f"downscaling{i + 1}", trainable=trainable, padding='valid', kernel_initializer='he_normal'))

    def call(self, inputTensor):

        x = inputTensor

        for res, down in zip(self._res_blocks, self._downscalings):
            x = res(x)
            x = down(x)

        return x


if __name__ == "__main__":
    encoder = Encoder(res_kernels=[3, 3], res_channels=[32, 64], conv_kernels=[3, 4], conv_strides=[1, 2], trainable=True)

    encoder.build((None, 28, 28, 1))

    inputT = np.random.rand(1, 28, 28, 1)

    print(encoder.summary(), encoder(inputT).shape)
