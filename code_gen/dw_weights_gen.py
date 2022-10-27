
import utils

utils.set_globals('mob_v2', 'mobilenetv2')

bit_width = 8
from_files = True
weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/'
weights_file_format = 'weights_{}_dw.txt'
dw_weights_h_file = '../client/dw_weights.h' #'./out/dw_weights.h'

dw_weights_declaration_string = 'const static dw_weights_dt dw_weights_*i*[layer_*i*_dw_depth][layer_*i*_dw_filter_size][layer_*i*_dw_filter_size]'

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weight_shapes(layers_types)
first_gen_layer = 2
last_gen_layer = len(layers_weights)

current_index = 0
with open(dw_weights_h_file, 'w') as f:
    f.write('#include "../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef DW_WEIGHTS\n")
    f.write("#define DW_WEIGHTS\n")

    for ii in range(first_gen_layer, last_gen_layer):
        if layers_types[ii] != 'dw':
            continue
        weights_file = weights_files_location +  weights_file_format.format(str(ii))
        f.write(dw_weights_declaration_string.replace(
            '*i*', str(ii)) + '= {\n')
        
        if from_files:
            with open(weights_file, 'r') as f2:
                for i in range(layers_weights[ii].num_of_filters):
                    f.write('{')
                    for j in range(layers_weights[ii].height):
                        f.write('{')
                        for k in range(layers_weights[ii].width):
                            f.write(f2.readline().replace(' ','').replace('\n', ''))
                            if(k < layers_weights[ii].width - 1):
                                f.write(', ')
                        f.write('},\n')
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
