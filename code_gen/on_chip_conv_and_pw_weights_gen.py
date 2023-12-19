
import utils
from os.path import exists
import numpy as np
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

on_chip_conv_and_layers = cgc.PIPELINE_LEN if cgc.PIPELINE else 1
weights_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(cgc.MODEL_NAME)
weights_file_format = 'weights_{}.txt'
conv_pw_weights_h_file = '../model_components/model/headers/{}_on_chip_weights.h'.format(cgc.MODEL_NAME) #'./out/dw_weights.h'

weights_declaration_string = {'s':'const static weights_dt weights_*i*[layer_*i*_s_num_fils][layer_*i*_s_depth]'+\
    '[layer_*i*_s_filter_dim][layer_*i*_s_filter_dim]' \
        ,'pw': 'const static weights_dt pw_weights_*i*[layer_*i*_pw_num_fils][layer_*i*_pw_depth]'}

first_layer_weights_declaration_string = 'const static layer_0_weights_dt first_layer_weights[first_conv_layer_num_fils]' + \
    '[first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim]'

model_dag = utils.read_model_dag()

num_of_generated_layers = 0
with open(conv_pw_weights_h_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write('#include "{}_layers_specs.h"\n'.format(cgc.MODEL_NAME))
    f.write("#if FIRST_PART_IMPLEMENTATION == " + cgc.BOTTLENECK_CHAIN_MODE + \
             " && MODEL_ID == " + cgc.MODEL_NAME.upper() + "\n")
    f.write("#ifndef CONV_PW_WEIGHTS\n")
    f.write("#define CONV_PW_WEIGHTS\n")

    first_layer = True
    for ii in range(len(model_dag)):
        if num_of_generated_layers >= on_chip_conv_and_layers:
            break
        layer_specs = model_dag[ii]
        layer_type = ''
        if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES and layer_specs['type'] != 'dw':
            layer_type = layer_specs['type']
        else:
            continue
        
        num_of_generated_layers += 1
        weights_file = weights_files_location +  weights_file_format.format(str(ii))
        
        layer_weights_shape = layer_specs['weights_shape'] 

        if first_layer:
            f.write(first_layer_weights_declaration_string + '= {\n')
            first_layer = False
        else:
            f.write(weights_declaration_string[layer_type].replace(
                '*i*', str(ii)) + '= {\n')
        
        weights = np.loadtxt(weights_file).astype(np.int8)
        num_of_filters = layer_weights_shape[0]
        filter_depth = layer_weights_shape[1]
        filter_h = layer_weights_shape[2] if len(layer_weights_shape) > 2 else 1 
        filter_w = layer_weights_shape[3] if len(layer_weights_shape) > 3 else 1 
        weights = np.reshape(weights, \
            (num_of_filters, filter_depth, filter_w, filter_h))
                
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
                            (layer_type == 'pw'  and j < filter_depth -1):
                            f.write(', ')
                    if filter_w > 1:
                        f.write('},\n')
                if filter_h > 1:
                    f.write('},\n')
            f.write('},\n')
        f.write('};\n')

    f.write('#endif\n')
    f.write('#endif\n')
