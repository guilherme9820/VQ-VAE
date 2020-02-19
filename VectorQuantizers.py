import tensorflow as tf
import numpy as np


class VectorQuantizer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, trainable=True, name='vq'):
        super().__init__(name=name, trainable=trainable)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        # Instantiate embedding table with initialized weights

        initializer = tf.keras.initializers.he_normal()

        self._emb_weights = tf.Variable(initializer((num_embeddings, embedding_dim)), trainable=trainable, name=name + '_embed')

    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            result = tf.nn.embedding_lookup(self._emb_weights.read_value(), encoding_indices)

        return result

    def call(self, inputTensor):
        '''
        Note: 
            shape_of_inputs=[batch_size, ?, ?, embedding_dim]
        '''
        # Assert last dimension of inputs is same as embedding_dim
        with tf.control_dependencies([inputTensor]):
            emb_w = self._emb_weights.read_value()

        assert_dim = tf.assert_equal(tf.shape(inputTensor)[-1], self._embedding_dim)
        with tf.control_dependencies([assert_dim]):
            flat_inputs = tf.reshape(inputTensor, [-1, self._embedding_dim])

        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                     - 2 * tf.matmul(flat_inputs, tf.transpose(emb_w))
                     + tf.reduce_sum(tf.transpose(emb_w)**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputTensor)[:-1])  # shape=[batch_size, ?, ?]

        quantized = self.quantize(encoding_indices)

        # Important Note:
        #   quantized is differentiable w.r.t. tf.transpose(self.emb_vq),
        #   but is not differentiable w.r.t. encoding_indices.

        inp_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputTensor)**2)
        emb_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputTensor))**2)
        loss = emb_latent_loss + self._commitment_cost * inp_latent_loss  # used to optimize self.emb_vq only!

        self.add_loss(loss, inputs=True)

        quantized = inputTensor + tf.stop_gradient(quantized - inputTensor)
        # Important Note:
        #   This step is used to copy the gradient from inputs to quantized.

        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))
        # The perplexity is the exponentiation of the entropy,
        # indicating how many codes are 'active' on average.
        # We hope the perplexity is larger.

        return loss, quantized, perplexity, encodings

        # return {'quantize': quantized,
        #         'loss': loss,
        #         'perplexity': perplexity,
        #         'encodings': encodings,
        #         'encoding_indices': encoding_indices}


class VectorQuantizerEMA(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay, epsilon=1e-5, trainable=True, name='vq_ema'):
        super().__init__(name=name, trainable=trainable)

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        initializer = tf.keras.initializers.he_normal()

        self._emb_weights = tf.Variable(initializer((num_embeddings, embedding_dim)), trainable=trainable, name=name + '_embed')
        self._ema_w = tf.Variable(initializer((num_embeddings, embedding_dim)), trainable=trainable, name=name + '_ema_w')
        self._ema_cluster_size = tf.Variable(tf.zeros((num_embeddings)), trainable=trainable, name=name + '_ema_cluster_size')
        # self._ema = tf.train.ExponentialMovingAverage(decay=decay)

    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            result = tf.nn.embedding_lookup(self._emb_weights.read_value(), encoding_indices)

        return result

    def call(self, inputTensor, training=True):
        '''
        Note: 
            shape_of_inputs=[batch_size, ?, ?, embedding_dim]
        '''
        # Assert last dimension of inputs is same as embedding_dim
        with tf.control_dependencies([inputTensor]):
            emb_w = self._emb_weights.read_value()

        assert_dim = tf.assert_equal(tf.shape(inputTensor)[-1], self._embedding_dim)
        with tf.control_dependencies([assert_dim]):
            flat_inputs = tf.reshape(inputTensor, [-1, self._embedding_dim])

        # if (self._only_lookup == False):
        distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True)
                     - 2 * tf.matmul(flat_inputs, tf.transpose(emb_w))
                     + tf.reduce_sum(tf.transpose(emb_w)**2, 0, keepdims=True))

        encoding_indices = tf.argmax(-distances, 1)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputTensor)[:-1])  # shape=[batch_size, ?, ?]
        quantized = self.quantize(encoding_indices)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputTensor) ** 2)

        if training:

            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * tf.reduce_sum(encodings, 0)

            dw = tf.matmul(flat_inputs, encodings, transpose_a=True)

            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * tf.transpose(dw)

            n = tf.reduce_sum(self._ema_cluster_size)
            updated_ema_cluster_size = ((self._ema_cluster_size + self._epsilon)
                                        / (n + self._num_embeddings * self._epsilon) * n)

            normalised_updated_ema_w = (self._ema_w
                                        / tf.reshape(updated_ema_cluster_size, [-1, 1]))

            with tf.control_dependencies([e_latent_loss]):
                update_w = self._emb_weights.assign(normalised_updated_ema_w)

            with tf.control_dependencies([update_w]):
                loss = self._commitment_cost * e_latent_loss

        else:
            loss = self._commitment_cost * e_latent_loss

        quantized = inputTensor + tf.stop_gradient(quantized - inputTensor)
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

        # return {'quantize': quantized,
        #         'loss': loss,
        #         'perplexity': perplexity,
        #         'encodings': encodings,
        #         'encoding_indices': encoding_indices}


# if __name__ == "__main__":
#     embedding_dim = 64
#     num_embeddings = 512

#     commitment_cost = 0.25

#     decay = 0.99

#     learning_rate = 1e-3

#     inputTensor = np.random.rand(1, 28, 28, embedding_dim).astype(np.float32)

#     vq = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost, trainable=True, name='vq')

#     vq_ema = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay, epsilon=1e-5)

#     print(vq(inputTensor))
