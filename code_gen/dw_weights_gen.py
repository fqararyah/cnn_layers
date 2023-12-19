
import code_generation_constants as cgc
import utils
import numpy as np

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

bit_width = 8
from_files = True
weights_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(
    cgc.MODEL_NAME)
reading_weights_file_format = 'weights_{}.txt'
# './out/dw_weights.h'
dw_weights_h_file = '../model_components/model/headers/{}_dw_weights_{}.h'.format(
    cgc.MODEL_NAME, cgc.FIRST_PART_IMPLEMENTATION)
general_specs_file = '/media/SSD2TB/fareed/wd/cnn_layers/model_components/basic_defs/general_specs.h'

if cgc.PIPELINE:
    dw_off_chip_weights_file = '../off_chip_weights/{}_off_chip_dw_weights_pipeline_{}.txt'.format(cgc.MODEL_NAME, cgc.PIPELINE_LEN)
else:
    dw_off_chip_weights_file = '../off_chip_weights/{}_off_chip_dw_weights.txt'.format(cgc.MODEL_NAME)

dw_off_chip_weights = []

pipe_dw_weights_declaration_string = 'const static dw_weights_dt dw_weights_*i*[layer_*i*_dw_depth][layer_*i*_dw_filter_dim * layer_*i*_dw_filter_dim]'
pipe_dw_weights_declaration_string_v2 = 'const static dw_weights_dt pipe_dw_weights_3x3[][9] = {\n'
if cgc.DW_WEIGHTS_OFF_CHIP:
    seml_dw_weights_declaration_string = 'static dw_weights_dt seml_dw_weights_3x3[{}][9];\n'
else:
    seml_dw_weights_declaration_string = 'const static dw_weights_dt seml_dw_weights_3x3[][9] = {\n'
dw_layers_weights_offsets_declaration_string = 'const static int dw_layers_weights_offsets[] ={'
model_dag = utils.read_model_dag()
dw_layers_weights_offsets = [0] * (len(model_dag) + 1)

pipe_dw_weights_v2 = None


last_layer = cgc.LAST_LAYER_TO_GENERATE + 1 if \
    cgc.LAST_LAYER_TO_GENERATE != -1 else len(model_dag)
first_layer = cgc.FIRST_LAYER_TO_GENERATE

last_pipeline_layer = cgc.PIPELINE_LEN if cgc.PIPELINE == True else 0
first_pipeline_layer = 0



current_index = 0
num_of_layers_generated_for = 0
with open(dw_weights_h_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write("#if FIRST_PART_IMPLEMENTATION == " + str(cgc.FIRST_PART_IMPLEMENTATION) + \
            " && MODEL_ID == " + cgc.MODEL_NAME.upper() + "\n")
    f.write("#ifndef DW_WEIGHTS\n")
    f.write("#define DW_WEIGHTS\n")

    for ii in range(len(model_dag)):
        layer_specs = model_dag[ii]
        current_index += 1
        layer_type = ''

        dw_layers_weights_offsets[ii + 1] = dw_layers_weights_offsets[ii]
        if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES:
            num_of_layers_generated_for += 1

        if num_of_layers_generated_for > last_pipeline_layer:
            break

        if 'type' not in layer_specs or layer_specs['type'] != 'dw':
            continue

        weights_file = weights_files_location + \
            reading_weights_file_format.format(str(ii))

        layer_weights_shape = layer_specs['weights_shape']
        num_of_filters = layer_weights_shape[0]
        filter_height = layer_weights_shape[1]
        filter_width = layer_weights_shape[2]

        if cgc.FIRST_PART_IMPLEMENTATION == cgc.BOTTLENECK_CHAIN_MODE:
            f.write(pipe_dw_weights_declaration_string.replace(
                '*i*', str(ii)) + '= {\n')
            with open(weights_file, 'r') as f2:
                for i in range(num_of_filters):
                    f.write('{')
                    for j in range(filter_height):
                        for k in range(filter_width):
                            f.write(f2.readline().replace(
                                ' ', '').replace('\n', ''))
                            if(k < filter_width - 1):
                                f.write(', ')
                        if j != filter_height - 1:
                            f.write(',\n')
                    f.write('},\n')
                f.write('};\n')
        elif cgc.FIRST_PART_IMPLEMENTATION == cgc.PIPELINED_ENGINES_MODE :
            dw_layers_weights_offsets[ii + 1] += num_of_filters * filter_height * filter_width
            current_weights = np.loadtxt(weights_file).astype(np.int8)
            filter_dim = layer_weights_shape[-1]
            current_weights = np.reshape(current_weights, (int(
                current_weights.size / (filter_dim**2)), filter_dim**2))
            if pipe_dw_weights_v2 is not None:
                pipe_dw_weights_v2 = np.concatenate(
                    (pipe_dw_weights_v2, current_weights))
            else:
                pipe_dw_weights_v2 = current_weights

    if cgc.FIRST_PART_IMPLEMENTATION == cgc.PIPELINED_ENGINES_MODE and pipe_dw_weights_v2 is not None:
        f.write(pipe_dw_weights_declaration_string_v2)
        for i in range(pipe_dw_weights_v2.shape[0]):
            f.write('{')
            f.write(str(pipe_dw_weights_v2[i][0]))
            for j in range(1, pipe_dw_weights_v2.shape[1]):
                f.write(', ')
                f.write(str(pipe_dw_weights_v2[i][j]))
            f.write('},\n')
        f.write('};\n')

    if cgc.PIPELINE == False or last_layer > last_pipeline_layer:
        max_dw_num_filters = 0
        if cgc.DW_WEIGHTS_OFF_CHIP == False:
            f.write(seml_dw_weights_declaration_string)
            
        for ii in range(current_index, last_layer):
            layer_specs = model_dag[ii]
            layer_type = ''
            if ii == current_index:
                dw_layers_weights_offsets[ii] = 0
            if ii > 0:
                dw_layers_weights_offsets[ii +
                                          1] = dw_layers_weights_offsets[ii]
            if 'type' not in layer_specs or layer_specs['type'] != 'dw':
                continue

            layer_weights_shape = layer_specs['weights_shape']
            num_of_filters = layer_weights_shape[0]
            filter_height = layer_weights_shape[1]
            filter_width = layer_weights_shape[2]

            dw_layers_weights_offsets[ii + 1] += num_of_filters * filter_height * filter_width
            max_dw_num_filters = max(max_dw_num_filters, num_of_filters)
            weights_file = weights_files_location + \
                reading_weights_file_format.format(str(ii))
            
            if cgc.DW_WEIGHTS_OFF_CHIP == False:
                with open(weights_file, 'r') as f2:
                    for i in range(num_of_filters):
                        f.write('{')
                        for j in range(filter_height):
                            for k in range(filter_width):
                                f.write(f2.readline().replace(
                                    ' ', '').replace('\n', ''))
                                if(k < filter_width - 1):
                                    f.write(', ')
                            if j != filter_height - 1:
                                f.write(',\n')
                        f.write('},\n')
            else:
                with open(weights_file, 'r') as f2:
                    for i in range(num_of_filters):
                        for j in range(filter_height):
                            for k in range(filter_width):
                                dw_off_chip_weights.append(f2.readline().replace(
                                    ' ', '').replace('\n', ''))

        if cgc.DW_WEIGHTS_OFF_CHIP == False:
            f.write('};\n')
        else:
            f.write(seml_dw_weights_declaration_string.format(max_dw_num_filters))

        f.write(dw_layers_weights_offsets_declaration_string +
                str(dw_layers_weights_offsets).replace('[', '').replace(']', '') + '};\n')
    f.write('#endif\n')
    f.write('#endif\n')

with open(dw_off_chip_weights_file, 'w') as f:
    for i in range(len(dw_off_chip_weights)):
        f.write(dw_off_chip_weights[i] + '\n')


replacement_string = ''
with open(general_specs_file, 'r') as f:
    for line in f:
        if 'const int all_dw_off_chip_weights =' in line:
            replacement_string += 'const int all_dw_off_chip_weights = {};\n'.format(len(dw_off_chip_weights))
        elif 'const int MAX_DW_LAYER_D' in line:
            replacement_string += 'const int MAX_DW_LAYER_D = ' + str(max_dw_num_filters) + ';\n'
        else:
            replacement_string += line

with open(general_specs_file, 'w') as f:
    f.write(replacement_string)