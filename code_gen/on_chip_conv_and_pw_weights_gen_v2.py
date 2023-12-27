
import utils
from os.path import exists
import numpy as np
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

on_chip_conv_and_layers = cgc.PIPELINE_LEN if cgc.PIPELINE else 1
weights_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(
    cgc.MODEL_NAME)
weights_file_format = 'weights_{}.txt'
# './out/dw_weights.h'
on_chip_weights_header_file = '../model_components/model/headers/{}_on_chip_weights_v2.h'.format(cgc.MODEL_NAME)

if cgc.PIPELINE == True:
    on_chip_weights_file = '../on_chip_weights/{}_on_chip_weights_pipe_{}.txt'.format(cgc.MODEL_NAME, cgc.PIPELINE_LEN)
else:
    on_chip_weights_file = '../on_chip_weights/{}_on_chip_weights_pipe.txt'.format(cgc.MODEL_NAME, 0)

general_specs_file = '/media/SSD2TB/fareed/wd/cnn_layers/model_components/basic_defs/general_specs.h'

first_layer_weights_declaration_string = 'const static layer_0_weights_dt first_layer_weights[first_conv_layer_num_fils]' + \
    '[first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim]{\n'

model_dag = utils.read_model_dag()

with open(on_chip_weights_header_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write('#include "' + cgc.MODEL_NAME + '_layers_specs.h"\n')
    f.write("#if FIRST_PART_IMPLEMENTATION ==" + cgc.PIPELINED_ENGINES_MODE + "\n")
    f.write("#ifndef ON_CHIP_WEIGHTS\n")
    f.write("#define ON_CHIP_WEIGHTS\n")

def write_first_layer_weights(layer_weights_shape, weights, on_chip_weights_header_file):
    num_of_filters = layer_weights_shape[0]
    filter_depth = layer_weights_shape[1]
    filter_h = layer_weights_shape[2]
    filter_w = layer_weights_shape[3]
    with open(on_chip_weights_header_file, 'a') as f:
        f.write(first_layer_weights_declaration_string)
        for i in range(num_of_filters):
            f.write('{\n')
            for j in range(filter_depth):
                if filter_h > 1:
                    f.write('{\n')
                for k in range(filter_h):
                    if filter_w > 1:
                        f.write('{')
                    for l in range(filter_w):
                        f.write(str(weights[i][j][k][l]))
                        if(l < filter_w - 1) or \
                            (layer_type == 'pw' and j < filter_depth - 1):
                            f.write(', ')
                    if filter_w > 1:
                        f.write('},\n')
                if filter_h > 1:
                    f.write('},\n')
            f.write('},\n')
        f.write('};\n')

        assert(all_on_chip_pw_s_weights % cgc.ON_CHIP_WEIGHTS_PORTS == 0)
        f.write('#endif\n')
        f.write('#endif\n')


if cgc.PIPELINE == False:
    exit(0)

first_layer = True
num_of_generated_layers = 0
all_on_chip_pw_s_weights = 0
formated_weights_all_layers = []

for ii in range(len(model_dag)):
    if num_of_generated_layers >= on_chip_conv_and_layers:
        break
    layer_specs = model_dag[ii]
    layer_type = ''
    if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES and layer_specs['type'] != 'dw':
        layer_type = layer_specs['type']
    else:
        continue

    layer_weights_shape = layer_specs['weights_shape']
    num_of_generated_layers += 1

    weights_file = weights_files_location + \
        weights_file_format.format(str(ii))
    
    weights = np.loadtxt(weights_file).astype(np.int8)

    all_on_chip_pw_s_weights += weights.size

    if first_layer:
        first_layer = False
        weights = np.reshape(weights, \
            (layer_weights_shape[0], layer_weights_shape[1], layer_weights_shape[2], layer_weights_shape[3]))
        write_first_layer_weights(layer_weights_shape, weights, on_chip_weights_header_file)
        continue

    num_of_filters = layer_weights_shape[0]
    filter_size = layer_weights_shape[1]
    if layer_specs['type'] != 'pw':
        filter_size *= layer_weights_shape[2] * layer_weights_shape[3]

    weights = np.reshape(weights, \
            (num_of_filters, filter_size))
    
    num_filters = weights.shape[0]
    assert(num_filters % cgc.ON_CHIP_WEIGHTS_PORTS == 0)
    # if num_filters % ofms_parallelism != 0:
    #     weights = np.append(weights, np.zeros(( int(ofms_parallelism - (num_filters % ofms_parallelism) ), \
    #          weights.shape[1], weights.shape[2], weights.shape[3])), 0).astype(np.int8)

    splitted_weights = np.split(weights, int(weights.shape[0] / cgc.ON_CHIP_WEIGHTS_PORTS), 0)
    splitted_weights = [np.transpose(splitted_weights[i], (1,0)) for i in range(len(splitted_weights))]
    combined_weights = np.concatenate(splitted_weights, 0)
    combined_weights = combined_weights.reshape(combined_weights.size) 
    formated_weights_all_layers.append(combined_weights)

np.savetxt(on_chip_weights_file, np.concatenate(formated_weights_all_layers, 0), fmt='%i')    

replacement_string = ''
with open(general_specs_file, 'r') as f:
    for line in f:
        if 'const int all_on_chip_pw_s_weights =' in line:
            replacement_string += 'const int all_on_chip_pw_s_weights = ' + str(all_on_chip_pw_s_weights) + ';\n'
        else:
            replacement_string += line

with open(general_specs_file, 'w') as f:
    f.write(replacement_string)