
import utils
from os.path import exists

utils.set_globals('mob_v2', 'mobilenetv2')

bit_width = 8
from_files = True
on_chip_conv_and_layers = 2
weights_files_location = '/media/fareed/D/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/'
weights_file_format = {'c':'weights_{}_c.txt', 'pw': 'weights_{}_pw.txt'}
dw_weights_h_file = '../client/conv_pw_weights.h' #'./out/dw_weights.h'

dw_weights_declaration_string = {'c':'const static layer_0_weights_dt weights_*i*[layer_*i*_num_fils][layer_*i*_depth]\
    [layer_*i*_filter_size][layer_*i*_filter_size]' \
        ,'pw': 'const static weights_dt weights_*i*_pw[layer_*i*_num_fils][layer_*i*_depth]'}

layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weights(layers_types)

num_of_generated_layers = 0
with open(dw_weights_h_file, 'w') as f:
    f.write('#include "../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef DW_WEIGHTS\n")
    f.write("#define DW_WEIGHTS\n")

    for ii in range(len(layers_weights)):
        if layers_types[ii] == 'dw':
            continue

        if num_of_generated_layers > on_chip_conv_and_layers:
            break

        num_of_generated_layers += 1
        weights_file = weights_files_location +  weights_file_format[layers_types[ii]].format(str(ii))

        if not exists(weights_file):
            continue
            
        f.write(dw_weights_declaration_string[layers_types[ii]].replace(
            '*i*', str(ii)) + '= {\n')
        
        if from_files:
            with open(weights_file, 'r') as f2:
                for i in range(layers_weights[ii].num_of_filters):
                    f.write('{\n')
                    for j in range(layers_weights[ii].depth):
                        if layers_weights[ii].height > 1:
                            f.write('{\n')
                        for k in range(layers_weights[ii].height):
                            if layers_weights[ii].width > 1:
                                f.write('{')
                            for l in range(layers_weights[ii].width):
                                f.write(f2.readline().replace(' ','').replace('\n', ''))
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
