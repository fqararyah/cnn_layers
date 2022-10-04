import utils

utils.set_globals('mob_v2', 'mobilenetv2')

scales_bit_width = 18
scales_integer_part_width = 0
biases_bit_width = 32

from_files = True
weights_scales_biases_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/'+ \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/weights/'
fms_scales_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/'

fms_scales_file_format = fms_scales_files_location + 'fms_{}_scales.txt'
fms_zero_points_file_format = fms_scales_files_location + 'fms_{}_zero_points.txt'
weights_scales_file_format = weights_scales_biases_files_location + 'weights_{}_scales.txt'
weights_zero_points_file_format = weights_scales_biases_files_location + 'weights_{}_zero_points.txt'
biases_file_format = weights_scales_biases_files_location + 'weights_{}_biases.txt'

layers_types = utils.read_layers_types()
layers_weights_shapes = utils.read_layers_weight_shapes(layers_types)

h_file = '../client/quantization_and_biases.h' #'./out/dw_weights.h'

skip_connections_indices = utils.read_skip_connections_indices()

conv_fms_scales_declaration_string = 'const static scales_dt conv_fms_scales[{}] = '.format(len(layers_weights_shapes)) + '{'
add_fms_scales_declaration_string = 'const static scales_dt add_fms_scales[{}] = '.format(len(skip_connections_indices)) + '{'
conv_fms_zero_points_declaration_string = 'const static fms_dt conv_fms_scales[{}] = '.format(len(layers_weights_shapes)) + '{'
add_fms_zero_points_declaration_string = 'const static fms_dt add_fms_scales[{}] = '.format(len(skip_connections_indices)) + '{'

weights_scales_declaration_string ='const static scales_dt weights_scales[] = {'
weights_zero_points_declaration_string ='const static fms_dt weights_zero_points[] = {'
biases_declaration_string = 'const static biases_dt biases[] = {'

overall_fms_scales = len(skip_connections_indices) + len(layers_weights_shapes)
expansion_projection = utils.read_expansion_projection()
skip_connection_current_index = 0
with open(h_file, 'w') as wf:
    wf.write('#include "../basic_defs/basic_defs_glue.h"\n')
    wf.write("#ifndef BIAS_QUANT\n")
    wf.write("#define BIAS_QUANT\n")
    fms_file_index = 1
    #writing fms scales and zero_points
    for layer_index in range(overall_fms_scales):
        if layers_types[layer_index - skip_connection_current_index] == 'pw' \
        and expansion_projection[layer_index - skip_connection_current_index] == 0:
            continue
            
        with open(fms_scales_file_format.format(fms_file_index), 'r') as f:
            scale = f.readline().replace(' ', '').replace('\n', '')
        with open(fms_zero_points_file_format.format(fms_file_index), 'r') as f:
            zero_point = f.readline().replace(' ', '').replace('\n', '')

        if layer_index - skip_connection_current_index in skip_connections_indices and \
             skip_connections_indices[layer_index - skip_connection_current_index] == 1:
            skip_connections_indices[skip_connection_current_index] = 0
            add_fms_scales_declaration_string += scale
            add_fms_zero_points_declaration_string += zero_point
            if skip_connection_current_index < len(skip_connections_indices) - 1:
                add_fms_scales_declaration_string += ', '
                add_fms_zero_points_declaration_string += ', '
                
            skip_connection_current_index += 1 
        else:
            conv_fms_scales_declaration_string += scale
            conv_fms_zero_points_declaration_string += zero_point
            if layer_index - skip_connection_current_index < len(layers_weights_shapes) - 1:
                conv_fms_scales_declaration_string += ', '
                conv_fms_zero_points_declaration_string += ', '

        fms_file_index += 1 

    add_fms_scales_declaration_string += '};\n'
    add_fms_zero_points_declaration_string += '};\n'
    conv_fms_scales_declaration_string += '};\n'
    conv_fms_zero_points_declaration_string += '};\n'

    wf.write(add_fms_scales_declaration_string)
    wf.write(add_fms_zero_points_declaration_string)
    wf.write(conv_fms_scales_declaration_string)
    wf.write(conv_fms_zero_points_declaration_string)

    #writing weights scales and zero_points
    for layer_index in range(len(layers_weights_shapes)):
        if layers_types[layer_index] == 'pw' and expansion_projection[layer_index] == 0:
            continue
        with open(weights_scales_file_format.format(layer_index), 'r') as f:
            for line in f:
                weights_scales_declaration_string += f.readline().replace(' ', '').replace('\n', '') + ', '
        weights_scales_declaration_string += '\n'
        
        with open(weights_zero_points_file_format.format(layer_index), 'r') as f:
            for line in f:
                weights_zero_points_declaration_string += f.readline().replace(' ', '').replace('\n', '') + ', '
        weights_zero_points_declaration_string += '\n'

        with open(biases_file_format.format(layer_index), 'r') as f:
            for line in f:
                biases_declaration_string += f.readline().replace(' ', '').replace('\n', '') + ', '
        biases_declaration_string += '\n'

    weights_scales_declaration_string += '};\n'
    weights_zero_points_declaration_string += '};\n'
    biases_declaration_string += '};\n'

    wf.write(weights_scales_declaration_string)
    wf.write(weights_zero_points_declaration_string)
    wf.write(biases_declaration_string)
    wf.write("#endif\n")