import utils

utils.set_globals('mob_v2', 'mobilenetv2')
import numpy as np

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

conv_fms_scales = []
conv_fms_scales_declaration_string = 'const static scales_dt conv_fms_scales[] = {'
add_fms_scales_declaration_string = 'const static scales_dt add_fms_scales[] = {'
conv_fms_zero_points = []
fused_zero_points = []
biases = []
layers_fused_parameters_offsets = []
conv_fms_zero_pointsdeclaration_string = 'const static fms_dt conv_fms_zero_points[] = {'
add_fms_zero_points_declaration_string = 'const static fms_dt add_fms_zero_points[] = {'
fused_zero_points_declaration_string = 'const static biases_dt fused_zero_points[] = {\n' 
layers_fused_parameters_offsets_declaration_string = 'const static int layers_fused_parameters_offsets[] = {0, \n' 

fused_scales_declaration_string ='const static scales_dt fused_scales[] = {'
weights_zero_points_declaration_string ='const static fms_dt weights_zero_points[] = {'
biases_declaration_string = 'const static biases_dt biases[] = {'

overall_fms_scales = len(skip_connections_indices) + len(layers_weights_shapes) + 1
expansion_projection = utils.read_expansion_projection()
skip_connection_current_index = 0
with open(h_file, 'w') as wf:
    wf.write('#include "../basic_defs/basic_defs_glue.h"\n')
    wf.write("#ifndef BIAS_QUANT\n")
    wf.write("#define BIAS_QUANT\n")
    fms_file_index = 1
    #writing fms scales and zero_points
    for layer_index in range(overall_fms_scales):
        if layer_index < overall_fms_scales - 1 and layers_types[layer_index - skip_connection_current_index] == 'pw' \
        and expansion_projection[layer_index - skip_connection_current_index] == 0:
            conv_fms_scales.append(0)
            conv_fms_zero_points.append(0)
            continue
            
        with open(fms_scales_file_format.format(fms_file_index), 'r') as f:
            scale = f.readline().replace(' ', '').replace('\n', '')
        with open(fms_zero_points_file_format.format(fms_file_index), 'r') as f:
            zero_point = f.readline().replace(' ', '').replace('\n', '')

        if layer_index - skip_connection_current_index in skip_connections_indices and \
             skip_connections_indices[layer_index - skip_connection_current_index] == 1:
            skip_connections_indices[layer_index - skip_connection_current_index] = 0
            add_fms_scales_declaration_string += scale
            add_fms_zero_points_declaration_string += zero_point
            if skip_connection_current_index < len(skip_connections_indices) - 1:
                add_fms_scales_declaration_string += ', '
                add_fms_zero_points_declaration_string += ', '
                
            skip_connection_current_index += 1 
        else:
            conv_fms_scales.append(float(scale))
            conv_fms_zero_points.append(int(zero_point))
            #if layer_index - skip_connection_current_index < len(layers_weights_shapes) - 1:
            #    conv_fms_scales += ', '
            #    conv_fms_zero_points += ', '

        fms_file_index += 1 

    add_fms_scales_declaration_string += '};\n'
    add_fms_zero_points_declaration_string += '};\n'
    #conv_fms_scales += '};\n'
    #conv_fms_zero_points += '};\n'

    wf.write(add_fms_scales_declaration_string)
    wf.write(add_fms_zero_points_declaration_string)
    #wf.write(conv_fms_scales)
    wf.write(conv_fms_zero_pointsdeclaration_string + str(conv_fms_zero_points).replace('[', '').replace(']', '};\n') )
    wf.write(conv_fms_scales_declaration_string + str(conv_fms_scales).replace('[', '').replace(']', '};\n') )
    #writing weights scales and zero_points
    for layer_index in range(len(layers_weights_shapes)):
        if layers_types[layer_index] == 'pw' and expansion_projection[layer_index] == 0:
            continue

        weights_file = weights_scales_biases_files_location + 'weights_{}_{}.txt'.format(layer_index, layers_types[layer_index])
        weights = np.loadtxt(weights_file).astype(np.int8)

        with open(biases_file_format.format(layer_index), 'r') as f:
            for line in f:
                bias = line.replace(' ', '').replace('\n', '')
                biases_declaration_string += bias + ','
                biases.append(int(bias))
        biases_declaration_string += '\n'

        weights = np.reshape(weights, \
            (layers_weights_shapes[layer_index].num_of_filters, layers_weights_shapes[layer_index].depth, \
                layers_weights_shapes[layer_index].height, layers_weights_shapes[layer_index].width))
        for i in range(layers_weights_shapes[layer_index].num_of_filters):
            fused_zero_points.append(np.sum(weights[i,:,:,:]) * -conv_fms_zero_points[layer_index] + biases[i])
                    
        layers_fused_parameters_offsets.append(layers_weights_shapes[layer_index].num_of_filters)

        #print(len(conv_fms_scales))
        with open(weights_scales_file_format.format(layer_index), 'r') as f:
            for line in f:
                if layers_types[layer_index] == 'pw' and expansion_projection[layer_index] == 0:
                    continue
                weight_scale = float(line.replace(' ', '').replace('\n', ''))
                fused_scales_declaration_string += str( weight_scale \
                    * conv_fms_scales[layer_index]) + ', '

                # if len(fused_scales_declaration_string) < 100:
                #     print(fused_scales_declaration_string)
                #     print( weight_scale \
                #     , conv_fms_scales[layer_index], (conv_fms_scales[layer_index + 1] if conv_fms_scales[layer_index + 1] != 0 else \
                #        conv_fms_scales[layer_index + 2] )) 


        fused_scales_declaration_string += '\n'

        with open(weights_zero_points_file_format.format(layer_index), 'r') as f:
            for line in f:
                weights_zero_points_declaration_string += line.replace(' ', '').replace('\n', '') + ', '
        weights_zero_points_declaration_string += '\n'

    fused_zero_points_declaration_string += str(fused_zero_points).replace('[', '').replace(']', '') +'};\n'
    fused_scales_declaration_string += '};\n'
    weights_zero_points_declaration_string += '};\n'
    biases_declaration_string += '};\n'

    wf.write(layers_fused_parameters_offsets_declaration_string + str(layers_fused_parameters_offsets).replace('[', '').replace(']', '};\n') )
    wf.write(fused_zero_points_declaration_string)
    wf.write(fused_scales_declaration_string)
    #wf.write(weights_zero_points_declaration_string)
    wf.write(biases_declaration_string)
    wf.write("#endif\n")