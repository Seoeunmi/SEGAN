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

test_source_path = config['test_source_path']
load_generator_check_point_name = config['load_generator_check_point_name']

# training_source_path is path or file?
source_path_isdir = os.path.isdir(test_source_path)

# make model
Generator_model = segan_model.Generator(default_float=default_float)

# load model
if load_generator_check_point_name != "":
    Generator_model.load_weights('{}/checkpoint/{}/data.ckpt'.format(cf.load_path(), load_generator_check_point_name))
else:
    raise Exception("ERROR: 'load_check_point_name' is empty. Test need check point.")

# test function
@tf.function
def test_step(x):
    y_pred = Generator_model(x)
    return y_pred

def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x

# make test data
if source_path_isdir:
    test_source_file_list = cf.read_path_list(test_source_path, "wav")
else:
    test_source_file_list = [test_source_path]

for i in range(len(test_source_file_list)):
    test_source_signal, test_source_sample_rate = wav.read_wav(test_source_file_list[i])

    test_source_signal = np.array(test_source_signal)
    test_size_of_source = test_source_signal.size

    source_signal = np.append(test_source_signal[0], test_source_signal[1:] - pre_emphasis * test_source_signal[:-1])
    size_of_source = test_source_signal.size

    # padding
    mod = (input_size - (size_of_source % input_size)) % input_size
    test_source_signal_padded = np.concatenate([source_signal, np.zeros(mod)]).astype(default_float)

    # test run
    result = []
    frame = 0
    sample = 0
    start = time.time()

    while sample < test_size_of_source:
        print("\rTest({}) : frame {}/{}".format(test_source_file_list[i], frame + 1, math.ceil(test_size_of_source / input_size)), end='')
        y_pred = test_step(test_source_signal_padded[sample:sample + input_size])
        y_pred = np.array(y_pred, dtype=default_float)
        y_pred = de_emph(y_pred)
        y_pred = y_pred.tolist()
        result.extend(y_pred)
        sample += input_size
        frame += 1
    print(" | Processing time :", datetime.timedelta(seconds=time.time() - start))

    # save output
    result_path = "{}/test_result/{}/result/{}".format(cf.load_path(), load_generator_check_point_name, os.path.dirname(test_source_file_list[i].replace(test_source_path, "")))
    file_name = os.path.basename(test_source_file_list[i])
    cf.createFolder(result_path)
    wav.write_wav(result[:len(result) - mod], "{}/{}".format(result_path, file_name), test_source_sample_rate)
