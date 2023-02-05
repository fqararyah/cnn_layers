
import utils
from os.path import exists
import numpy as np
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

bit_width = 8
from_files = True
on_chip_conv_and_layers = cgc.PILELINE_LEN if cgc.PIPELINE else 1
weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(cgc.MODEL_NAME)
weights_file_format = {'s':'conv2d_{}_s_weights.txt', 'pw': 'conv2d_{}_pw_weights.txt'}
conv_pw_weights_h_file = '../model_components/model/headers/on_chip_conv_pw_weights.h' #'./out/dw_weights.h'

weights_declaration_string = {'s':'const static layer_0_weights_dt weights_*i*[layer_*i*_s_num_fils][layer_*i*_s_depth]'+\
    '[layer_*i*_s_filter_dim][layer_*i*_s_filter_dim]' \
        ,'pw': 'const static weights_dt pw_weights_*i*[layer_*i*_pw_num_fils][layer_*i*_pw_depth]'}

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weight_shapes(layers_types)

num_of_generated_layers = 0
with open(conv_pw_weights_h_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write('#include "layers_specs.h"\n')
    f.write("#ifndef CONV_PW_WEIGHTS\n")
    f.write("#define CONV_PW_WEIGHTS\n")

    for ii in range(len(layers_weights)):
        if ii >= on_chip_conv_and_layers:
            break
        if layers_types[ii] == 'dw':
            continue
        
        num_of_generated_layers += 1
        weights_file = weights_files_location +  weights_file_format[layers_types[ii]].format(str(ii))

        f.write(weights_declaration_string[layers_types[ii]].replace(
            '*i*', str(ii)) + '= {\n')
        
        if from_files:
            weights = np.loadtxt(weights_file).astype(np.int8)
            weights = np.reshape(weights, \
                (layers_weights[ii].num_of_filters, layers_weights[ii].depth, layers_weights[ii].height, layers_weights[ii].width))
            for i in range(layers_weights[ii].num_of_filters):
                f.write('{\n')
                for j in range(layers_weights[ii].depth):
                    if layers_weights[ii].height > 1:
                        f.write('{\n')
                    for k in range(layers_weights[ii].height):
                        if layers_weights[ii].width > 1:
                            f.write('{')
                        for l in range(layers_weights[ii].width):
                            f.write(str(weights[i][j][k][l]))
                            if(l < layers_weights[ii].width - 1) or \
                                (layers_types[ii] == 'pw'  and j < layers_weights[ii].depth -1):
                                f.write(', ')
                        if layers_weights[ii].width > 1:
                            f.write('},\n')
                    if layers_weights[ii].height > 1:
                        f.write('},\n')
                f.write('},\n')
            f.write('};\n')
        else:
            for i in range(layers_weights[ii].num_of_filters):
                f.write('{')
                for j in range(layers_weights[ii].depth):
                    f.write('{')
                    for k in range(layers_weights[ii].height):
                        f.write('{')
                        for l in range(layers_weights[ii].width):
                            f.write(str(-1 + (i*j*k % (2**(bit_width - 1)))))
                            if(k < layers_weights[ii].width - 1):
                                f.write(', ')
                    f.write('},\n')
                f.write('},\n')
            f.write('};\n')

    f.write('#endif\n')
