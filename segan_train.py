import numpy as np
import segan_model
import tensorflow as tf
import customfunction as cf
import wav
import json
import time
import datetime
import math
import os

# prevent GPU overflow
gpu_config = tf.compat.v1.ConfigProto()
gpu_config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.InteractiveSession(config=gpu_config)

# read config file
with open("config.json", "r") as f_json:
    config = json.load(f_json)

default_float = config['default_float']
pre_emphasis = config['pre_emphasis']

input_size = config['input_size']
shift_size = config['shift_size']
batch_size = config['batch_size']
epochs = config['epochs']
lamda = config['lamda']


training_source_path = config['training_source_path']
training_target_path = config['training_target_path']

save_generator_check_point_name = config['save_generator_check_point_name']
load_generator_check_point_name = config['load_generator_check_point_name']
save_discriminator_check_point_name = config['save_discriminator_check_point_name']
load_discriminator_check_point_name = config['load_discriminator_check_point_name']

# training_path is path or file?
target_path_isdir = os.path.isdir(training_target_path)
source_path_isdir = os.path.isdir(training_source_path)
if target_path_isdir != source_path_isdir:
    raise Exception("ERROR: Target and source path is incorrect")
if target_path_isdir:
    if not cf.compare_path_list(training_target_path, training_source_path, 'wav'):
        raise Exception("ERROR: Target and source file list is not same")
    training_target_file_list = cf.read_path_list(training_target_path, "wav")
    training_source_file_list = cf.read_path_list(training_source_path, "wav")
else:
    training_target_file_list = [training_target_path]
    training_source_file_list = [training_source_path]

x_signal, y_signal = [], []
num_of_total_frame = 0
for i in range(len(training_target_file_list)):
    # read train data file
    target_signal, target_sample_rate = wav.read_wav(training_target_file_list[i])
    source_signal, source_sample_rate = wav.read_wav(training_source_file_list[i])

    target_signal = np.array(target_signal)
    source_signal = np.array(source_signal)
    size_of_target = target_signal.size
    size_of_source = source_signal.size

    target_signal = np.append(target_signal[0], target_signal[1:] - pre_emphasis * target_signal[:-1])
    source_signal = np.append(source_signal[0], source_signal[1:] - pre_emphasis * source_signal[:-1])

    # source & target file incorrect
    if size_of_source != size_of_target:
        raise Exception("ERROR: Input, output size mismatch")
    if shift_size <= 0:
        raise Exception("ERROR: Shift size is smaller or same with 0")

    # padding
    mod = (shift_size - (size_of_source % shift_size)) % shift_size
    target_signal_padded = np.concatenate([target_signal, np.zeros(mod)]).astype(default_float)
    source_signal_padded = np.concatenate([source_signal, np.zeros(mod)]).astype(default_float)

    # make dataset
    number_of_frames = math.ceil(size_of_source/shift_size) - 1
    num_of_total_frame += number_of_frames
    for j in range(number_of_frames):
        x_signal.append(source_signal_padded[j*shift_size:(j*shift_size) + input_size])
        y_signal.append(target_signal_padded[j*shift_size:(j*shift_size) + input_size])

train_dataset = tf.data.Dataset.from_tensor_slices((x_signal, y_signal)).shuffle(number_of_frames).batch(batch_size)


# make model
Generator_model = segan_model.Generator(default_float=default_float)
generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])

Discriminator_model = segan_model.Discriminator(leaky_alpha=config['relu_alpha'], default_float=default_float)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate'])

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()
generator_train_loss = tf.keras.metrics.Mean(name='generator_train_loss')
discriminator_train_loss = tf.keras.metrics.Mean(name='discriminator_train_loss')

train_f_loss = tf.keras.metrics.Mean(name='train_f_loss')
train_l1_norm = tf.keras.metrics.Mean(name='train_l1_norm')

@tf.function
def d_loss(r_logit, f_logit):
    # r_loss = mse(r_logit, tf.ones_like(r_logit))      # E[(D(x, x_c) - 1) ** 2]/2
    r_loss = tf.math.reduce_mean(tf.math.squared_difference(r_logit, 1.))
    # f_loss = mse(f_logit, tf.zeros_like(f_logit))     # E[(D(G(z, x_c), x_c)) ** 2]/2
    f_loss = tf.math.reduce_mean(tf.math.squared_difference(f_logit, 0.))
    return r_loss + f_loss

@tf.function
def g_loss(f_logit, G_logits, x):
    # f_loss = mse(f_logit, tf.ones_like(f_logit))     # E[(D(G(z, x_c), x_c) - 1) ** 2]/2
    f_loss = tf.math.reduce_mean(tf.math.squared_difference(f_logit, 1.))
    # norm = lamda * mae(G_logits, tf.squeeze(x))
    norm = lamda * tf.math.reduce_mean(tf.math.abs(tf.math.subtract(G_logits, tf.squeeze(x))))
    return f_loss + norm, f_loss, norm

@tf.function
def d_train_step(real_audio, noise_audio):
    with tf.GradientTape() as disc_tape:
        D_real_logits = Discriminator_model(real_audio, noise_audio)
        enhanced_audio = Generator_model(noise_audio)
        D_fake_logits = Discriminator_model(enhanced_audio, noise_audio)
        discriminator_loss = d_loss(D_real_logits, D_fake_logits)
    discriminator_gradients = disc_tape.gradient(discriminator_loss, Discriminator_model.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, Discriminator_model.trainable_variables))
    discriminator_train_loss(discriminator_loss)


@tf.function
def g_train_step(real_audio, noise_audio):
    with tf.GradientTape() as gen_tape:
        enhanced_audio = Generator_model(noise_audio)
        D_fake_logits = Discriminator_model(enhanced_audio, noise_audio)
        generator_loss, f_loss, l1_norm_loss = g_loss(D_fake_logits, enhanced_audio, real_audio)
    generator_gradients = gen_tape.gradient(generator_loss, Generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(generator_gradients, Generator_model.trainable_variables))
    generator_train_loss(generator_loss)
    train_f_loss(f_loss)
    train_l1_norm(l1_norm_loss)

# load model
if load_generator_check_point_name != "" and load_discriminator_check_point_name != "":
    saved_epoch = int(load_generator_check_point_name.split('_')[-1])
    for y, x in train_dataset:
        g_train_step(x, y)
        d_train_step(x, y)
        break
    Generator_model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), load_generator_check_point_name))
    Generator_model.load_optimizer_state(generator_optimizer, '{}/checkpoint/{}'.format(cf.load_path(), load_generator_check_point_name), 'generator_optimizer')
    generator_train_loss.reset_states()

    Discriminator_model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), load_discriminator_check_point_name))
    Discriminator_model.load_optimizer_state(discriminator_optimizer, '{}/checkpoint/{}'.format(cf.load_path(), load_discriminator_check_point_name), 'discriminator_optimizer')
    discriminator_train_loss.reset_states()

else:
    cf.clear_plot_file('{}/{}'.format(cf.load_path(), config['generator_plot_file']))
    cf.clear_plot_file('{}/{}'.format(cf.load_path(), config['discriminator_plot_file']))
    cf.clear_plot_file('{}/{}'.format(cf.load_path(), config['generator_adv_plot_file']))
    cf.clear_plot_file('{}/{}'.format(cf.load_path(), config['generator_l1_plot_file']))
    saved_epoch = 0


# train run
for epoch in range(saved_epoch, saved_epoch+epochs):
    i = 0
    start = time.time()
    flag, count = 0, 0
    for noise_audio, clean_audio in train_dataset:
        print("\rTrain : epoch {}/{}, training {}/{}".format(epoch + 1, saved_epoch+epochs, i + 1, math.ceil(num_of_total_frame / batch_size)), end='')
        d_train_step(clean_audio, noise_audio)
        g_train_step(clean_audio, noise_audio)
        i += 1

    print(" | loss : {}".format(generator_train_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start), end='')
    print(" | loss : {}".format(discriminator_train_loss.result()), " | Processing time :", datetime.timedelta(seconds=time.time() - start))

    if ((epoch + 1) % config['save_check_point_period'] == 0) or (epoch + 1 == 1):
        cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_path(), save_generator_check_point_name, epoch+1))
        Generator_model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_path(), save_generator_check_point_name, epoch+1))
        Generator_model.save_optimizer_state(generator_optimizer, '{}/checkpoint/{}_{}'.format(cf.load_path(), save_generator_check_point_name, epoch+1), 'generator_optimizer')
        cf.createFolder("{}/checkpoint/{}_{}".format(cf.load_path(), save_discriminator_check_point_name, epoch + 1))
        Discriminator_model.save_weights('{}/checkpoint/{}_{}/data.ckpt'.format(cf.load_path(), save_discriminator_check_point_name, epoch + 1))
        Discriminator_model.save_optimizer_state(discriminator_optimizer, '{}/checkpoint/{}_{}'.format(cf.load_path(), save_discriminator_check_point_name, epoch + 1), 'discriminator_optimizer')

    # write plot file
    cf.write_plot_file('{}/{}'.format(cf.load_path(), config['generator_plot_file']), epoch+1, generator_train_loss.result())
    cf.write_plot_file('{}/{}'.format(cf.load_path(), config['discriminator_plot_file']), epoch+1, discriminator_train_loss.result())
    cf.write_plot_file('{}/{}'.format(cf.load_path(), config['generator_adv_plot_file']), epoch + 1, train_f_loss.result())
    cf.write_plot_file('{}/{}'.format(cf.load_path(), config['generator_l1_plot_file']), epoch + 1, train_l1_norm.result())


    generator_train_loss.reset_states()
    discriminator_train_loss.reset_states()
    train_f_loss.reset_states()
    train_l1_norm.reset_states()