import numpy as np
import utils
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/'+ \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(cgc.MODEL_NAME)

weights_file_format = weights_files_location + 'weights_{}.txt'

parallelism_file = '../model_components/basic_defs/parallelism_and_tiling.h'
ofms_parallelism_key = 'pw_conv_parallelism_out'

off_chip_weights_file = '../off_chip_weights/{}_off_chip_weights_fpga.txt'

model_dag = utils.read_model_dag()

first_off_chip_layer = 0
last_off_chip_layer = len(model_dag)

def get_ofms_parallelism(parallelism_file):
    ofms_parallelism = 1
    with open(parallelism_file, 'r') as f:
        for line in f:
            line = line.replace(' ', '').replace('\n', '').replace('\t', '')
            if (ofms_parallelism_key + '=') in line:
                ofms_parallelism = int( ( line.split(';')[0] ).split('=')[-1] )
    
    return ofms_parallelism

formated_weights_all_layers = []
for i in range(first_off_chip_layer, last_off_chip_layer):
    layer_specs = model_dag[i]
    if 'type' not in layer_specs or (layer_specs['type'] != 'pw' and layer_specs['type'] != 's'):
        continue
    
    layer_weights_shape = layer_specs['weights_shape']
    num_of_filters = layer_weights_shape[0]
    filter_size = layer_weights_shape[1]
    if layer_specs['type'] != 'pw':
        filter_size *= layer_weights_shape[2] * layer_weights_shape[3]
    weights = np.loadtxt(weights_file_format.format(i)).astype(np.int8)
    weights = np.reshape(weights, \
            (num_of_filters, filter_size))
    
    ofms_parallelism = get_ofms_parallelism(parallelism_file)
    # print("get_ofms_parallelism", ofms_parallelism)

    num_filters = weights.shape[0]
    if num_filters % ofms_parallelism != 0:
        weights = np.append(weights, np.zeros(( int(ofms_parallelism - (num_filters % ofms_parallelism) ), \
             weights.shape[1], weights.shape[2], weights.shape[3])), 0).astype(np.int8)

    splitted_weights = np.split(weights, int(weights.shape[0] / ofms_parallelism), 0)
    splitted_weights = [np.transpose(splitted_weights[i], (1,0)) for i in range(len(splitted_weights))]
    combined_weights = np.concatenate(splitted_weights, 0)
    combined_weights = combined_weights.reshape(combined_weights.size) 
    formated_weights_all_layers.append(combined_weights)

np.savetxt(off_chip_weights_file.format(cgc.MODEL_NAME), np.concatenate(formated_weights_all_layers, 0), fmt='%i')