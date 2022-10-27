from unittest import skip
import utils

utils.set_globals('mob_v2', 'mobilenetv2')

DEBUGGING = True

in_out_file = '../client/seml.cpp'
in_out_header_file = '../client/seml.h'
ofms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/'
ifms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/fms/'

ifms_file_format = 'fms_{}_{}_{}_{}.txt'

debugging_includes_block = '#include "../tests/test_utils.h"\n'

expansion_projection_block = 'pw_conv(off_chip_weights, channels, result2, *i*, layer_*i*_pw_depth,\n\
    layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
    layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
    layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
    layer_*i*_pw_num_of_weight_groups_for_one_pass,\n\
    *DIRECTION*, layer_*i*_pw_weights_offset, layer_*i*_relu);\n'

dw_block = 'fill_dw_layer_weights(dw_weights_*i*, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
    dw_conv_3x3(dw_weights_buffer, channels, result2, *i*, layer_*i*_dw_depth,\n\
    layer_*i*_dw_ifm_width, layer_*i*_dw_ifm_height, layer_*i*_dw_num_of_tiles_in_d,\n\
    layer_*i*_dw_num_of_tiles_h, layer_*i*_dw_num_of_tiles_w,\n\
    layer_*i*_dw_strides, layer_*i*_dw_padding_left,layer_*i*_dw_padding_top,\n\
    *DIRECTION*);\n'

# projection_block = 'pw_conv(off_chip_weights, channels, result2, *i*, layer_*i*_pw_depth,\n\
#     layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
#     layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
#     layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
#     layer_*i*_pw_num_of_weight_groups_for_one_pass,\n\
#     *DIRECTION*, layer_*i*_pw_weights_offset, layer_*i*_relu);\n'

debugging_dump_ofms_block = 'dumb_layer_output("{}",\n {}, {}, {}, {});\n'
debugging_fill_layer_input_block = 'fill_layer_input("{}",\n {}, {}, {});\n'
debugging_verify_fill_layer_input_block = 'verify_fill_layer_input("{}",\n {}, {}, {}, {});\n'

#layers_to_debug = [2, 12, 13, 20, 21, 22, 23, 24,25,26,27,28,29,30,31, 32,33,34, 35, 36, 37, 38, 39, 40, 41, 
#layers_to_debug = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
layers_to_debug = [2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30, 33, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

layers_types = utils.read_layers_types()
layers_strides = utils.read_layers_strides()
expansion_projection = utils.read_expansion_projection()
layers_output_shapes = utils.read_layers_output_shapes()
layers_inputs_shapes = utils.read_layers_input_shapes()
skip_connections_indices = utils.read_skip_connections_indices()

tf_lite_to_my_cnn_layer_ifms_mapping = {0: 1}
skip_connections_so_far = 0
for layer_index in range(1, len(layers_output_shapes)):
    if layer_index in skip_connections_indices:
        skip_connections_so_far += 1
    tf_lite_to_my_cnn_layer_ifms_mapping[layer_index] = layer_index + skip_connections_so_far

def replace(replacement_dic, block):
    for key, val in replacement_dic.items():
        block = block.replace(key, str(val))

    return block

defines_str = ''
file_replacement = ''
if DEBUGGING:
    file_replacement += debugging_includes_block
with open(in_out_header_file, 'r') as f:
    for line in f:
        if '#ifndef' in line or '#define ' in line:
            defines_str = defines_str + line
        else:
            if len(line.replace(' ', '').replace('\t', '').replace('\n', '')) > 1 and line in debugging_includes_block:
                continue
            file_replacement += line

with open(in_out_header_file, 'w') as f:
    f.write(defines_str + file_replacement)

file_replacement = ''
in_a_code_gen_area = False
insert_index = -1
layers_to_generate = [1, len(layers_types)]

with open(in_out_file, 'r') as f:
    for line in f:
        if not in_a_code_gen_area:
            file_replacement += line
        if utils.START_CODE_GENERATION_SIGNAL in line:
            insert_index = len(file_replacement)
            in_a_code_gen_area = True
            if '[' in line and ']' in line and ':' in line:
                layers_to_generate_start_end = line.replace(' ', '').split('[')[1].split(']')[
                            0].replace(' ', '').split(':')
            
            if 2 == len(layers_to_generate_start_end) and layers_to_generate_start_end[-1] == '':
                layers_to_generate_start_end[-1] = len(layers_inputs_shapes)
            elif 1 == len(layers_to_generate_start_end):
                layers_to_generate_start_end.append(len(layers_inputs_shapes))
            
            
            layers_to_generate = [int(x) for x in (
                int(layers_to_generate_start_end[0]), int(layers_to_generate_start_end[1])
            )]
        elif utils.END_CODE_GENERATION_SIGNAL in line:
            in_a_code_gen_area = False
            file_replacement += line

direction = 1
code_to_insert = ''
skip_connections_depth = 3

skip_connections_indices = utils.read_skip_connections_indices()
for layer_indx in range(layers_to_generate[0], layers_to_generate[1]):
    target_block = ''
    replacement_dict = {}
    replacement_dict['*i*'] = layer_indx
    replacement_dict['*DIRECTION*'] = direction
    read_write = 0

    if layers_types[layer_indx] == 'pw' and expansion_projection[layer_indx]:
        target_block = expansion_projection_block
        if layer_indx + skip_connections_depth + 1 in skip_connections_indices:
            read_write += 2
        if layer_indx + 1 in skip_connections_indices:
            read_write += 1

    elif layers_types[layer_indx] == 'dw':
        target_block = dw_block

    replacement_dict['*RW*'] = read_write
    if DEBUGGING and layer_indx == layers_to_debug[0]:
        # file_name
        ifms_file = ifms_file_format.format(tf_lite_to_my_cnn_layer_ifms_mapping[layer_indx], layers_inputs_shapes[layer_indx].depth,
                                            layers_inputs_shapes[layer_indx].height, layers_inputs_shapes[layer_indx].width)
        # insert func call
        code_to_insert += debugging_fill_layer_input_block.format(ifms_file_path + ifms_file,
                                                                    'channels' if direction == 0 else 'result2',
                                                                    str(
                                                                        layers_inputs_shapes[layer_indx].height),
                                                                    str(layers_inputs_shapes[layer_indx].width))
        code_to_insert += debugging_verify_fill_layer_input_block.format(ofms_file_path + 'verify_' + str(layer_indx)+'.txt',
                                                        'channels' if direction == 0 else 'result2',
                                                        layers_inputs_shapes[layer_indx].depth * layers_inputs_shapes[layer_indx].height *
                                                        layers_inputs_shapes[layer_indx].width, str(
                                                            layers_inputs_shapes[layer_indx].height),
                                                        str(layers_inputs_shapes[layer_indx].width))

    code_to_insert += replace(replacement_dict, target_block)

    if DEBUGGING and layer_indx in layers_to_debug:
        code_to_insert += debugging_dump_ofms_block.format(ofms_file_path + 'ofms_' + str(layer_indx)+'.txt',
                                                           'result2' if direction == 0 else 'channels',
                                                           layers_output_shapes[layer_indx].depth * layers_output_shapes[layer_indx].height *
                                                           layers_output_shapes[layer_indx].width, str(
                                                               layers_output_shapes[layer_indx].height),
                                                           str(layers_output_shapes[layer_indx].width))
    
    if expansion_projection[layer_indx] or layers_types[layer_indx] != 'pw':
        direction = 1 - direction

file_replacement = file_replacement[:insert_index] + \
    code_to_insert + file_replacement[insert_index:]

with open(in_out_file, 'w') as f:
    f.write(file_replacement)
