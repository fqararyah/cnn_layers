
import code_generation_constants as cgc
import utils

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

bit_width = 8
from_files = True
weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(
    cgc.MODEL_NAME)
reading_weights_file_format = 'weights_{}_dw.txt'
# './out/dw_weights.h'
dw_weights_h_file = '../model_components/model/headers/dw_weights.h'

dw_weights_declaration_string = 'const static dw_weights_dt dw_weights_*i*[layer_*i*_dw_depth][layer_*i*_dw_filter_size * layer_*i*_dw_filter_size]'

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weight_shapes(layers_types)

last_layer = cgc.LAST_LAYER_TO_GENERATE + 1 if \
    cgc.LAST_LAYER_TO_GENERATE != -1 else len(layers_types)
first_layer = cgc.FIRST_LAYER_TO_GENERATE

if cgc.PIPELINE:
    last_layer = max(last_layer, cgc.PILELINE_LEN + 1)
    first_layer = 0

current_index = 0
with open(dw_weights_h_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef DW_WEIGHTS\n")
    f.write("#define DW_WEIGHTS\n")

    for ii in range(first_layer, last_layer):
        if layers_types[ii] != 'dw':
            continue
        weights_file = weights_files_location + \
            reading_weights_file_format.format(str(ii))
        f.write(dw_weights_declaration_string.replace(
            '*i*', str(ii)) + '= {\n')

        if from_files:
            with open(weights_file, 'r') as f2:
                for i in range(layers_weights[ii].num_of_filters):
                    f.write('{')
                    for j in range(layers_weights[ii].height):
                        for k in range(layers_weights[ii].width):
                            f.write(f2.readline().replace(
                                ' ', '').replace('\n', ''))
                            if(k < layers_weights[ii].width - 1):
                                f.write(', ')
                        if j != layers_weights[ii].height - 1:
                            f.write(',\n')
                    f.write('},\n')
                f.write('};\n')
        else:
            for i in range(layers_weights[ii].num_of_filters):
                f.write('{')
                for j in range(layers_weights[ii].height):
                    f.write('{')
                    for k in range(layers_weights[ii].width):
                        f.write(str(-1 + (i*j*k % (2**(bit_width - 1)))))
                        if(k < layers_weights[ii].width - 1):
                            f.write(', ')
                    f.write('},\n')
                f.write('},\n')
            f.write('};\n')

    f.write('#endif\n')
