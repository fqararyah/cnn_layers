from ctypes import util
import numpy as np
import utils

weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/'+ \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/'

weights_file_format = weights_files_location + 'weights_{}.txt'

parallelism_file = '/media/SSD2TB/wd/cnn_layers/basic_defs/parallelism_and_tiling.h'
ofms_parallelism_key = 'pw_conv_parallelism_out'

off_chip_weights_file = '/media/SSD2TB/wd/cnn_layers/off_chip_weights.txt'

first_off_chip_layer = 2
last_off_chip_layer = 10 # len(layer_types)

def get_ofms_parallelism(parallelism_file):
    ofms_parallelism = 1
    with open(parallelism_file, 'r') as f:
        for line in f:
            line = line.replace(' ', '').replace(';','').replace('\n', '').replace('\t', '')
            if (ofms_parallelism_key + '=') in line:
                ofms_parallelism = int(line.split('=')[-1])
    
    return ofms_parallelism


layer_types = utils.read_layers_types()
layers_weights_shapes = utils.read_layers_weight_shapes(layer_types)
expansion_projection = utils.read_expansion_projection()

formated_weights_all_layers = np.array()
for i in range(first_off_chip_layer, last_off_chip_layer):
    if layer_types[i] == 'dw' or (layer_types[i] == 'pw' and expansion_projection[i] == 0):
        continue
    weights = np.loadtxt(weights_file_format.format(i)).astype(np.int8)
    weights = np.reshape(weights, \
            (layers_weights_shapes[i].num_of_filters, layers_weights_shapes[i].depth, \
                layers_weights_shapes[i].height, layers_weights_shapes[i].width))
    
    ofms_parallelism = get_ofms_parallelism(parallelism_file)

    num_filters = weights.shape[0]
    if num_filters % ofms_parallelism != 0:
        weights = np.append(weights, np.zeros(( int(ofms_parallelism - (num_filters % ofms_parallelism) ), \
             weights.shape[1], weights.shape[2], weights.shape[3])), 0).astype(np.int8)

    splitted_weights = np.split(weights, int(weights.shape[0] / 3), 0)
    splitted_weights = [np.transpose(splitted_weights[i], (1,2,3,0)) for i in range(len(splitted_weights))]
    combined_weights = np.concatenate(splitted_weights, 0)
    combined_weights = combined_weights.reshape(combined_weights.size) 
    formated_weights_all_layers = np.append(formated_weights_all_layers, combined_weights)

np.savetxt()