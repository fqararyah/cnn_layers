
from multiprocessing.spawn import old_main_modules
from numpy import iinfo
import utils

utils.set_globals('mob_v2', 'mobilenetv2')

bit_width = 8
on_chip_layers = 7
pw_weights_h_file = '../client/sesl_pw_weights.h' #'./out/pw_weights.h'

pw_weights_declaration_string = 'const static weights_dt pw_weights_*i*[layer_*i*_pw_num_fils][layer_*i*_pw_depth]'

layers_types = utils.read_layers_types()
expansion_projection = utils.read_expansion_projection()
layers_weights = utils.read_layers_weights(layers_types)

current_index = 0
with open(pw_weights_h_file, 'w') as f:
    f.write('#include "../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef PW_WEIGHTS\n")
    f.write("#define PW_WEIGHTS\n")

    ii = 0
    omitted_layers = 0
    while ii - omitted_layers < on_chip_layers:
        if layers_types[ii] != 'pw' or expansion_projection[ii] == 0:
            if layers_types[ii] == 'pw':
                omitted_layers += 1
            ii += 1
            continue
        f.write(pw_weights_declaration_string.replace(
            '*i*', str(ii)) + '= {\n')
        for i in range(layers_weights[ii].num_of_filters):
            f.write('{')
            for j in range(layers_weights[ii].depth):
                f.write(str(-1 + (i*j % (2**(bit_width - 1)))))
                if(j < layers_weights[ii].depth - 1):
                    f.write(', ')
            f.write('},\n')
        f.write('};\n')

        ii += 1

    f.write('#endif\n')