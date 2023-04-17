
import utils
from os.path import exists
import numpy as np
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

on_chip_conv_and_layers = cgc.PIPELINE_LEN if cgc.PIPELINE else 1
weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(
    cgc.MODEL_NAME)
weights_file_format = 'weights_{}.txt'
# './out/dw_weights.h'
conv_weights_h_file_3x3 = '../model_components/model/headers/on_chip_conv_weights_v2.h'
pw_weights_h_file = '../model_components/model/headers/on_chip_pw_weights_v2.h'

s_3x3_conv_weights_declaration_string = 'const static weights_dt s_3x3_weights[][9] ={\n'
pw_weights_declaration_string = 'const static weights_dt pw_weights[] = {\n'

s_3x3_conv_weights = None
pw_weights = None

model_dag = utils.read_model_dag()

num_of_generated_layers = 0
with open(conv_weights_h_file_3x3, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write('#include "layers_specs.h"\n')
    f.write("#if FIBHA_VERSION == 2\n")
    f.write("#ifndef CONV_WEIGHTS_3x3\n")
    f.write("#define CONV_WEIGHTS_3x3\n")

with open(pw_weights_h_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write('#include "layers_specs.h"\n')
    f.write("#if FIBHA_VERSION == 2\n")
    f.write("#ifndef PW_WEIGHTS\n")
    f.write("#define PW_WEIGHTS\n")

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
    filter_dim = 1
    if layer_type != 'pw':
        filter_dim = layer_weights_shape[-1]
        weights = np.reshape(
            weights, ( int(weights.size / (filter_dim**2)), filter_dim**2))
        if filter_dim == 3:
            if s_3x3_conv_weights is not None:
                s_3x3_conv_weights = np.concatenate(s_3x3_conv_weights, weights)
            else:
                s_3x3_conv_weights = weights
    else:
        if pw_weights is not None:
            pw_weights = np.concatenate((pw_weights, weights))
        else:
            pw_weights = weights

with open(conv_weights_h_file_3x3, 'a') as f:
    f.write(s_3x3_conv_weights_declaration_string)
    for i in range(s_3x3_conv_weights.shape[0]):
        f.write('{')
        f.write(str(s_3x3_conv_weights[i][0]))
        for j in range(1, s_3x3_conv_weights.shape[1]):
            f.write(', ')
            f.write(str(s_3x3_conv_weights[i][j]))
        f.write('},\n')
    f.write('};\n')

    f.write('#endif\n')
    f.write('#endif\n')

with open(pw_weights_h_file, 'a') as f:
    f.write(pw_weights_declaration_string)
    f.write(str(pw_weights[0]))
    for i in range(1, pw_weights.shape[0]):
        f.write(', ')
        f.write(str(pw_weights[i]))
        if i % 64 == 0:
            f.write('\n')
    f.write('};\n')

    f.write('#endif\n')
    f.write('#endif\n')