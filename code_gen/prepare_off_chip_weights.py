from dataclasses import replace
from posixpath import split
import numpy as np
import utils
import code_generation_constants as cgc
import os

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

weights_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/'+ \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(cgc.MODEL_NAME)

biases_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/'+ \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/biases/'.format(cgc.MODEL_NAME)

general_specs_file = '/media/SSD2TB/fareed/wd/cnn_layers/model_components/basic_defs/general_specs.h'

weights_file_format = weights_files_location + 'weights_{}.txt'
fc_weights_file_format = weights_files_location + 'weights_{}.txt'
fc_biases_file_format = biases_files_location + 'biases_{}.txt'

ofms_parallelism_key = 'pw_conv_parallelism_out'

off_chip_fc_weights_file = '../off_chip_weights/{}_fc_weights.txt'
off_chip_fc_weight_sums_file = '../off_chip_weights/{}_fc_weight_sums.txt'
off_chip_fc_biases_file = '../off_chip_weights/{}_fc_biases.txt'

pipeline_len = 0
if cgc.PIPELINE == True:
    pipeline_len = cgc.PIPELINE_LEN

off_chip_weights_file = '../off_chip_weights/{}_off_chip_weights_pipe_{}.txt'
off_chip_weights_offsets_file = '../off_chip_weights/{}_off_chip_weights_offsets_pipe_{}.txt'
num_of_pw_weights_file = '../off_chip_weights/{}_num_of_pw_weights_pipe_{}.txt'

model_dag = utils.read_model_dag()

def get_ofms_parallelism(parallelism_file):
    ofms_parallelism = 1
    with open(parallelism_file, 'r') as f:
        for line in f:
            line = line.replace(' ', '').replace('\n', '').replace('\t', '')
            if (ofms_parallelism_key + '=') in line:
                ofms_parallelism = int( ( line.split(';')[0] ).split('=')[-1] )
    
    return ofms_parallelism

def get_layer_index_from_file_name(file_name):
    splits = file_name.split('_')
    for split in splits:
        if split.isnumeric():
            return int(split)

layers_weights = {}
num_s_pw_layers_so_far = 0
num_conv_layers_so_far = 0
fc_layer_index = 0
fc_weights_shape = []
all_pw_s_weights = 0
first_layer = True
for layer_index in range(len(model_dag)):
    layer_specs = model_dag[layer_index]

    if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES:
        num_conv_layers_so_far += 1

    if (num_conv_layers_so_far <= cgc.PIPELINE_LEN and cgc.PIPELINE == True) or first_layer:
        first_layer = False
        continue
    
    if 'type' in layer_specs and (layer_specs['type'] == 'pw' or layer_specs['type'] == 's'):
        weights_file = weights_file_format.format(layer_index)
        weights = np.loadtxt(weights_file).astype(np.int8)
        layers_weights[layer_index] = weights
        num_s_pw_layers_so_far += 1
        all_pw_s_weights += weights.size
    elif 'type' in layer_specs and layer_specs['type'] == 'fc':
        fc_layer_index = layer_index
        fc_weights_shape = layer_specs['weights_shape']
        fc_weights_file = fc_weights_file_format.format(layer_index)
        fc_biases_file = fc_biases_file_format.format(layer_index)
        weights = np.loadtxt(fc_weights_file).astype(np.int8)
        biases = np.loadtxt(fc_biases_file).astype(np.int32)
        np.savetxt(off_chip_fc_weights_file.format(cgc.MODEL_NAME), weights, fmt='%i')
        np.savetxt(off_chip_fc_biases_file.format(cgc.MODEL_NAME), biases, fmt='%i')
        weights = np.reshape(weights, fc_weights_shape)
        weight_sums = np.sum(weights, axis=1)
        np.savetxt(off_chip_fc_weight_sums_file.format(cgc.MODEL_NAME), weight_sums, fmt='%i')

layers_indices = list(layers_weights.keys())
layers_indices.sort()
last_layer_index = layers_indices[-1]
layers_weights_offsets = [0] * (last_layer_index + 1)
weights_combined_so_far = 0
combined_weights = []

for layer_index in layers_indices:
    combined_weights.append(layers_weights[layer_index])
    layers_weights_offsets[layer_index] = weights_combined_so_far
    weights_combined_so_far += layers_weights[layer_index].size

np.savetxt(off_chip_weights_file.format(cgc.MODEL_NAME, pipeline_len), 
           np.concatenate(combined_weights, 0), fmt='%i')
np.savetxt(off_chip_weights_offsets_file.format(cgc.MODEL_NAME, pipeline_len),
            np.array(layers_weights_offsets), fmt='%i')


if cgc.PIPELINE == True:
    num_of_pw_weights_file = num_of_pw_weights_file.format(cgc.MODEL_NAME, cgc.PIPELINE_LEN)
else:
    num_of_pw_weights_file = num_of_pw_weights_file.format(cgc.MODEL_NAME)

with open(num_of_pw_weights_file, 'w') as f:
    f.write(str(weights_combined_so_far))

replacement_string = ''
with open(general_specs_file, 'r') as f:
    for line in f:
        if 'const int all_pw_s_weights =' in line:
            replacement_string += 'const int all_pw_s_weights = {} / weights_group_items;\n'.format(
                all_pw_s_weights)
        else:
            replacement_string += line

with open(general_specs_file, 'w') as f:
    f.write(replacement_string)
