import tensorflow as tf
import numpy as np


def lrelu(x, trainbable=None):
    alpha = 0.2
    return tf.maximum(alpha * x, x)


def prelu(x, trainable=True):
    alpha = tf.get_variable(
        name='alpha', 
        shape=x.get_shape()[-1],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def conv_layer(x, filter_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight', 
        shape=filter_shape,
        dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv2d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, 1],
        padding='SAME')


def deconv_layer(x, filter_shape, output_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv2d_transpose(
        value=x,
        filter=filter_,
        output_shape=output_shape,
        strides=[1, stride, stride, 1])


def max_pooling_layer(x, size, stride):
    return tf.nn.max_pool(
        value=x,
        ksize=[1, size, size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')


def avg_pooling_layer(x, size, stride):
    return tf.nn.avg_pool(
        value=x,
        ksize=[1, size, size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')


def full_connection_layer(x, out_dim, trainable=True):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.add(tf.matmul(x, W), b)


def batch_normalize(x, is_training, decay=0.99, epsilon=0.001, trainable=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(
            x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var', 
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)
    return tf.cond(is_training, bn_train, bn_inference)


def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def pixel_shuffle_layer(x, r, n_split):
    def PS(x, r):
        bs, a, b, c = x.get_shape().as_list()
        x = tf.reshape(x, (bs, a, b, r, r))
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.split(x, a, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        x = tf.split(x, b, 1)
        x = tf.concat([tf.squeeze(x_) for x_ in x], 2)
        return tf.reshape(x, (bs, a*r, b*r, 1))

    xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)
    

class VGG19:
    def __init__(self, x, t, is_training):
        if x is None: return
        self.out, self.phi = self.build_model(x, is_training)
        self.loss = self.inference_loss(self.out, t)

    def build_model(self, x, is_training, reuse=False):
        with tf.variable_scope('vgg19', reuse=reuse):
            phi = []
            with tf.variable_scope('conv1a'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv1b'):
                x = conv_layer(x, [3, 3, 64, 64], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv2a'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv2b'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv3a'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv3b'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv3c'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv3d'):
                x = conv_layer(x, [3, 3, 256, 256], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv4a'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv4b'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv4c'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv4d'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            with tf.variable_scope('conv5a'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv5b'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv5c'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            with tf.variable_scope('conv5d'):
                x = conv_layer(x, [3, 3, 512, 512], 1)
                x = batch_normalize(x, is_training)
                x = lrelu(x)
            phi.append(x)

            x = max_pooling_layer(x, 2, 2)
            x = flatten_layer(x)
            with tf.variable_scope('fc1'):
                x = full_connection_layer(x, 4096)
                x = lrelu(x)
            with tf.variable_scope('fc2'):
                x = full_connection_layer(x, 4096)
                x = lrelu(x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 100)

            return x, phi


    def inference_loss(self, out, t):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(t, 100),
            logits=out)
        return tf.reduce_mean(cross_entropy)

    
    
class SRGAN:
    def __init__(self, x, is_training, batch_size, infer=False):
        self.batch_size = batch_size
        self.vgg = VGG19(None, None, None)
        self.downscaled = self.downscale(x)
        if infer is False:
            self.imitation = self.generator(self.downscaled, is_training, False)
            self.real_output = self.discriminator(x, is_training, False)
            self.fake_output = self.discriminator(self.imitation, is_training, True)
            self.g_loss, self.d_loss = self.inference_losses(
                x, self.imitation, self.real_output, self.fake_output)
        else:
            self.imitation = self.generator(x, is_training, False)  # so the model can get a new downscaled instance and never see the real version
        
        
    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                x = deconv_layer(
                    x, [3, 3, 64, 3], [self.batch_size, 24, 24, 64], 1)
            x = tf.nn.relu(x)
            shortcut = x
            for i in range(5):
                mid = x
                with tf.variable_scope('block{}a'.format(i+1)):
                    x = deconv_layer(
                        x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('block{}b'.format(i+1)):
                    x = deconv_layer(
                        x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                    x = batch_normalize(x, is_training)
                x = tf.add(x, mid)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(
                    x, [3, 3, 64, 64], [self.batch_size, 24, 24, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.add(x, shortcut)
            with tf.variable_scope('deconv3'):
                x = deconv_layer(
                    x, [3, 3, 256, 64], [self.batch_size, 24, 24, 256], 1)
                x = pixel_shuffle_layer(x, 2, 64) # n_split = 256 / 2 ** 2
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv4'):
                x = deconv_layer(
                    x, [3, 3, 64, 64], [self.batch_size, 48, 48, 64], 1)
                x = pixel_shuffle_layer(x, 2, 16)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv5'):
                x = deconv_layer(
                    x, [3, 3, 3, 16], [self.batch_size, 96, 96, 3], 1)

        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x


    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1'):
                x = conv_layer(x, [3, 3, 3, 64], 1)
                x = lrelu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 64, 64], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv3'):
                x = conv_layer(x, [3, 3, 64, 128], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv5'):
                x = conv_layer(x, [3, 3, 128, 256], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 256, 256], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv7'):
                x = conv_layer(x, [3, 3, 256, 512], 1)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 512, 512], 2)
                x = lrelu(x)
                x = batch_normalize(x, is_training)
            x = flatten_layer(x)
            with tf.variable_scope('fc'):
                x = full_connection_layer(x, 1024)
                x = lrelu(x)
            with tf.variable_scope('softmax'):
                x = full_connection_layer(x, 1)
                
        self.d_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return x


    def downscale(self, x):
        K = 4
        arr = np.zeros([K, K, 3, 3])
        arr[:, :, 0, 0] = 1.0 / K ** 2
        arr[:, :, 1, 1] = 1.0 / K ** 2
        arr[:, :, 2, 2] = 1.0 / K ** 2
        weight = tf.constant(arr, dtype=tf.float32)
        downscaled = tf.nn.conv2d(
            x, weight, strides=[1, K, K, 1], padding='SAME')
        return downscaled


    def inference_losses(self, x, imitation, true_output, fake_output):
        def inference_content_loss(x, imitation):
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), False) # First
            _, imitation_phi = self.vgg.build_model(
                imitation, tf.constant(False), True) # Second

            content_loss = None
            for i in range(len(x_phi)):
                l2_loss = tf.nn.l2_loss(x_phi[i] - imitation_phi[i])
                if content_loss is None:
                    content_loss = l2_loss
                else:
                    content_loss = content_loss + l2_loss
            return tf.reduce_mean(content_loss)

        def inference_adversarial_loss(real_output, fake_output):
            alpha = 1e-5
            g_loss = tf.reduce_mean(
                tf.nn.l2_loss(fake_output - tf.ones_like(fake_output)))
            d_loss_real = tf.reduce_mean(
                tf.nn.l2_loss(real_output - tf.ones_like(true_output)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.l2_loss(fake_output + tf.zeros_like(fake_output)))
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        def inference_adversarial_loss_with_sigmoid(real_output, fake_output):
            alpha = 1e-3
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_output),
                logits=fake_output)
            d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_output),
                logits=real_output)
            d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_output),
                logits=fake_output)
            d_loss = d_loss_real + d_loss_fake
            return (g_loss * alpha, d_loss * alpha)

        content_loss = inference_content_loss(x, imitation)
        generator_loss, discriminator_loss = (
            inference_adversarial_loss(true_output, fake_output))
        g_loss = content_loss + generator_loss
        d_loss = discriminator_loss
        return (g_loss, d_loss)

