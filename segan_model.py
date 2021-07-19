import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Flatten, Dense
from tensorflow.keras.layers import PReLU, Concatenate, LeakyReLU
from tensorflow.keras import Model
import os
import numpy as np

class Generator(Model):
    def __init__(self, default_float='float32'):
        super(Generator, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        self.enc_filters = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.enc = [Conv1D(filters=i, kernel_size=31, strides=2, padding='same') for i in self.enc_filters]

        self.dec_filters = [512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 1]
        self.dec = [Conv1DTranspose(filters=i, kernel_size=31, strides=2, padding='same') for i in self.dec_filters]

        self.concat = Concatenate(axis=2)
        self.enc_prelu = [PReLU() for i in range(len(self.enc_filters))]
        self.dec_prelu = [PReLU() for i in range(len(self.dec_filters)-1)]

    def call(self, x):
        temp_x = x
        if len(temp_x.shape) == 2:
            temp_x = tf.reshape(temp_x, [temp_x.shape[0], temp_x.shape[1], -1])
        elif len(temp_x.shape) == 1:
            temp_x = tf.reshape(temp_x, [1, temp_x.shape[0], 1])

        # pass the encoder
        enc_out = temp_x
        save_enc_out = []
        for i in range(len(self.enc_filters)):
            enc_out = self.enc_prelu[i](self.enc[i](enc_out))
            save_enc_out.append(enc_out)

        # generate z [z ~ N(0,1)] and concat c with z
        c = enc_out
        z = tf.random.normal(shape=c.shape, mean=0.0, stddev=1.0)
        dec_input = self.concat([c, z])

        # pass the decoder
        l = len(self.dec_filters)
        dec_out = dec_input
        for i in range(l):
            dec_out = self.dec[i](dec_out)
            if i != 0 and i != l-1:
                dec_out = self.concat([dec_out, save_enc_out[l-2-i]])
            if i != l-1:
                dec_out = self.dec_prelu[i](dec_out)

        dec_out = tf.keras.activations.tanh(dec_out)
        return tf.squeeze(dec_out)

    def save_optimizer_state(self, optimizer, save_path, save_name):

        # Create folder if it does not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save weights
        np.save(os.path.join(save_path, save_name), optimizer.get_weights())

        return


    def load_optimizer_state(self, optimizer, load_path, load_name):

        opt_weights = np.load(os.path.join(load_path, load_name) + '.npy', allow_pickle=True)

        optimizer.set_weights(opt_weights)

        return




class Discriminator(Model):
    def __init__(self, leaky_alpha, default_float='float32'):
        super(Discriminator, self).__init__()
        tf.keras.backend.set_floatx(default_float)

        self.dis_filters = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.dis = [Conv1D(filters=i, kernel_size=31, strides=2, padding='same') for i in self.dis_filters]

        self.one_conv = Conv1D(filters=1, kernel_size=1, strides=1, padding='same')
        self.flatten = Flatten()
        self.dis_fully = Dense(units=1)

        self.concat = Concatenate(axis=2)
        self.leakyrelu = LeakyReLU(alpha=leaky_alpha)


    def call(self, x, y):
        temp_x = x
        if len(temp_x.shape) == 2:
            temp_x = tf.reshape(temp_x, [temp_x.shape[0], temp_x.shape[1], -1])
        elif len(temp_x.shape) == 1:
            temp_x = tf.reshape(temp_x, [1, temp_x.shape[0], 1])

        temp_y = y
        if len(temp_y.shape) == 2:
            temp_y = tf.reshape(temp_y, [temp_y.shape[0], temp_y.shape[1], -1])
        elif len(temp_y.shape) == 1:
            temp_y = tf.reshape(temp_y, [1, temp_y.shape[0], 1])

        dis_input = self.concat([temp_x, temp_y])

        # pass the discriminator
        dis_out = dis_input
        for dis_filter in self.dis:
            dis_out = self.leakyrelu(dis_filter(dis_out))
        dis_out = self.flatten(self.leakyrelu(self.one_conv(dis_out)))
        logits = self.dis_fully(dis_out)

        return logits

    def save_optimizer_state(self, optimizer, save_path, save_name):

        # Create folder if it does not exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save weights
        np.save(os.path.join(save_path, save_name), optimizer.get_weights())

        return


    def load_optimizer_state(self, optimizer, load_path, load_name):

        opt_weights = np.load(os.path.join(load_path, load_name) + '.npy', allow_pickle=True)

        optimizer.set_weights(opt_weights)

        return


