from posixpath import split
import numpy as np
import utils
import code_generation_constants as cgc
import os

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/'+ \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(cgc.MODEL_NAME)

weights_file_format = weights_files_location + 'weights_{}_pw.txt'

parallelism_file = '../model_components/basic_defs/parallelism_and_tiling.h'
ofms_parallelism_key = 'pw_conv_parallelism_out'

off_chip_weights_file = '../off_chip_weights/off_chip_weights.txt'
off_chip_weights_offsets_file = '../off_chip_weights/off_chip_weights_offsets.txt'
num_of_pw_weights_file = '../off_chip_weights/num_of_pw_weights_file.txt'

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
num_pw_layer = 0
for file in os.scandir(weights_files_location):
    file_name = file.name
    if file_name.startswith('conv2d_') and file_name.endswith('_weights.txt') and 'pw' in file_name:
        layer_index = get_layer_index_from_file_name(file_name)
        weights = np.loadtxt(file.path).astype(np.int8)
        layers_weights[layer_index] = weights
        num_pw_layer += 1

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

np.savetxt(off_chip_weights_file, np.concatenate(combined_weights, 0), fmt='%i')
np.savetxt(off_chip_weights_offsets_file, np.array(layers_weights_offsets), fmt='%i')
with open(num_of_pw_weights_file, 'w') as f:
    f.write(str(weights_combined_so_far))