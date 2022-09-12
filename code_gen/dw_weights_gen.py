
import utils

utils.set_globals('mob_v2', 'mobilenetv2')

bit_width = 8
dw_weights_h_file = './out/dw_weights.h'

dw_weights_declaration_string = 'const static dw_weights_dt dw_weights_*i*[layer_*i*_dw_depth][layer_*i*_dw_filter_size][layer_*i*_dw_filter_size]'

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weights(layers_types)

current_index = 0
with open(dw_weights_h_file, 'w') as f:
    f.write('#include "../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef DW_WEIGHTS\n")
    f.write("#define DW_WEIGHTS\n")

    for ii in range(len(layers_weights)):
        if layers_types[ii] != 'dw':
            if ii > 0:
                current_index += 1
            continue
        f.write(dw_weights_declaration_string.replace(
            '*i*', str(current_index)) + '= {\n')
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
