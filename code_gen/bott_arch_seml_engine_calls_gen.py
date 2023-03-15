import code_generation_constants as cgc
import utils


utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)


in_out_file = '../model_components/model/SEML/imp/seml.cpp'
in_out_header_file = '../model_components/model/SEML/headers/seml.h'
ofms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/'
ifms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/fms/'.format(
    cgc.MODEL_NAME)

ifms_file_format = 'fms_conv2d_{}_{}_{}_{}.txt'

debugging_includes_block = '#include "../../../../tests/test_utils.h"\n'

# fill_quantization_parameters_block = 'fill_fused_scales_and_zero_points(fused_scales,fused_scales, \n\
#     fused_scales_log_2_shifts, fused_scales_log_2_shifts, relu_6_fused_scales,\n\
#      relu_6_fused_scales, fused_zero_points,\n\
#     fused_zero_points, layer_*i*_*TYPE*_num_fils);\n'

layer_0_s_block = 'layer_0_s_3x3(weights_0, input_image, result);\n'
expansion_projection_block = 'pw_conv(off_chip_weights, *CHANNELS*, result, *i*, layer_*i*_pw_depth,\n\
    layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
    layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
    layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
    layer_*i*_pw_num_of_weight_groups_for_one_pass,\n\
    *DIRECTION*, layer_*i*_pw_weights_offset, layer_*i*_activation,\n\
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,\n\
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);\n'

# 'fill_dw_layer_weights(seml_dw_weights_3x3, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
dw_block_0 = \
    'dw_conv_3x3(seml_dw_weights_3x3, channels, result, *i*, layer_*i*_dw_depth,\n\
    layer_*i*_dw_ifm_width, layer_*i*_dw_ifm_height, layer_*i*_dw_num_of_tiles_in_d,\n\
    layer_*i*_dw_num_of_tiles_h, layer_*i*_dw_num_of_tiles_w,\n\
    layer_*i*_dw_strides, layer_*i*_dw_padding_left, layer_*i*_dw_padding_right, layer_*i*_dw_padding_top,\n\
    *DIRECTION*, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,\n\
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);\n'

# 'fill_dw_layer_weights(seml_dw_weights_3x3, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
dw_block_1 = \
    'dw_conv_3x3(seml_dw_weights_3x3, result, channels, *i*, layer_*i*_dw_depth,\n\
    layer_*i*_dw_ifm_width, layer_*i*_dw_ifm_height, layer_*i*_dw_num_of_tiles_in_d,\n\
    layer_*i*_dw_num_of_tiles_h, layer_*i*_dw_num_of_tiles_w,\n\
    layer_*i*_dw_strides, layer_*i*_dw_padding_left, layer_*i*_dw_padding_right, layer_*i*_dw_padding_top,\n\
    *DIRECTION*, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,\n\
        fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);\n'

# projection_block = 'pw_conv(off_chip_weights, channels, result2, *i*, layer_*i*_pw_depth,\n\
#     layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
#     layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
#     layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
#     layer_*i*_pw_num_of_weight_groups_for_one_pass,\n\
#     *DIRECTION*, layer_*i*_pw_weights_offset, layer_*i*_activation);\n'

debugging_dump_ofms_block = '#if DEBUGGING\n dump_layer_output("{}",\n {}, {}, {}, {});\n#endif\n'
debugging_fill_layer_input_block = '#if DEBUGGING\n fill_layer_input("{}",\n {}, {}, {});\n#endif\n'
debugging_verify_fill_layer_input_block = '#if DEBUGGING\n verify_fill_layer_input("{}",\n {}, {}, {}, {});\n#endif\n'

# layers_to_debug = [2, 12, 13, 20, 21, 22, 23, 24,25,26,27,28,29,30,31, 32,33,34, 35, 36, 37, 38, 39, 40, 41,
#layers_to_debug = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
#layers_to_debug = [2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30, 33, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

layers_types = utils.read_layers_types()
layers_strides = utils.read_layers_strides()
layers_output_shapes = utils.read_layers_output_shapes()
layers_inputs_shapes = utils.read_layers_input_shapes()
skip_connections_indices = utils.read_skip_connections_indices()
layers_execution_sequence = utils.read_layers_execution_sequence()


def replace(replacement_dic, block):
    for key, val in replacement_dic.items():
        block = block.replace(key, str(val))

    return block


defines_str = ''
file_replacement = ''
if cgc.DEBUGGING:
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
layers_to_generate = [cgc.FIRST_LAYER_TO_GENERATE,
                      cgc.LAST_LAYER_TO_GENERATE + 1 if cgc.LAST_LAYER_TO_GENERATE != -1 else len(layers_types)]

with open(in_out_file, 'r') as f:
    for line in f:
        if not in_a_code_gen_area:
            file_replacement += line
        if cgc.START_CODE_GENERATION_SIGNAL in line:
            insert_index = len(file_replacement)
            in_a_code_gen_area = True
        elif cgc.END_CODE_GENERATION_SIGNAL in line:
            in_a_code_gen_area = False
            file_replacement += line

direction = 0
code_to_insert = ''
skip_connections_depths = {'mob_v1': 0, 'mob_v2': 3, 'eff_b0': 5,
                           'mnas': 3, 'prox': 3, 'mob_v1_0_5': 0, 'mob_v2_0_5': 3}
skip_connections_depth = skip_connections_depths[cgc.MODEL_NAME]
max_fms_size_in_seml = layers_inputs_shapes[layers_to_generate[0]].width * layers_inputs_shapes[layers_to_generate[0]].width *\
    layers_inputs_shapes[layers_to_generate[0]].depth
max_fms_size_in_seml_layer_index = layers_to_generate[0]
for i in range(layers_to_generate[0], layers_to_generate[1]):
    current_fms_size = layers_inputs_shapes[i].width * \
        layers_inputs_shapes[i].width * layers_inputs_shapes[i].depth
    if(current_fms_size > max_fms_size_in_seml):
        max_fms_size_in_seml = current_fms_size
        max_fms_size_in_seml_layer_index = i

print(max_fms_size_in_seml, max_fms_size_in_seml_layer_index)


def get_layer_index_in_execution_sequence(layers_execution_sequence, layer_index):
    index_in_execution = 0
    conv2d_layers_count = 0
    if layer_index <= 0:
        return 0
    while conv2d_layers_count <= layer_index:
        if 'conv2d' in layers_execution_sequence[index_in_execution]:
            conv2d_layers_count += 1
        index_in_execution += 1

    return index_in_execution - 1


def get_secondary_input_layer(layers_execution_sequence, prev_conv2d_layer_index):
    secondary_input_layer = ''
    print(prev_conv2d_layer_index)
    layer_index = prev_conv2d_layer_index + 1
    conv2d_layers_count = 0
    while conv2d_layers_count == 0:
        if 'conv2d' in layers_execution_sequence[layer_index]:
            conv2d_layers_count += 1
        else:
            layer_index += 1

    if 'pad' not in layers_execution_sequence[layer_index - 1]:
        secondary_input_layer = '_' + \
            layers_execution_sequence[layer_index - 1]
    else:
        secondary_input_layer = '_' + \
            layers_execution_sequence[layer_index - 2]

    if 'conv2d' in secondary_input_layer:
        secondary_input_layer = ''

    return secondary_input_layer, layer_index


prev_layer_index = get_layer_index_in_execution_sequence(
    layers_execution_sequence, cgc.FIRST_LAYER_TO_GENERATE - 1)
for layer_index in range(layers_to_generate[0], layers_to_generate[1]):
    target_block = ''  # fill_quantization_parameters_block
    replacement_dict = {}
    replacement_dict['*i*'] = layer_index
    replacement_dict['*DIRECTION*'] = direction

    read_write = 0
    if layer_index == 0:
        target_block += layer_0_s_block
        replacement_dict['*TYPE*_'] = ''
    if layers_types[layer_index] == 'pw':
        replacement_dict['*TYPE*'] = 'pw'
        replacement_dict['*CHANNELS*'] = 'tmp_channels' if (layer_index - skip_connections_depth - 1 in skip_connections_indices
                                                            or layer_index - 1 in skip_connections_indices or layer_index + skip_connections_depth - 1 in skip_connections_indices)\
            and layer_index == cgc.PILELINE_LEN\
            else 'channels'
        target_block += expansion_projection_block
        if layer_index + skip_connections_depth in skip_connections_indices:
            read_write += 2
        if layer_index in skip_connections_indices:
            read_write += 1

    elif layers_types[layer_index] == 'dw':
        if direction == 0:
            target_block += dw_block_0
        else:
            target_block += dw_block_1
        replacement_dict['*TYPE*'] = 'dw'

    replacement_dict['*RW*'] = read_write
    if cgc.DEBUGGING and layer_index == cgc.LAYERS_TO_DEBUG[0] and layer_index != 0:
        # file_name
        secondary_layer, prev_layer_index = get_secondary_input_layer(
            layers_execution_sequence, prev_layer_index)
        ifms_file = ifms_file_format.format(str(layer_index) + secondary_layer, layers_inputs_shapes[layer_index].depth,
                                            layers_inputs_shapes[layer_index].height, layers_inputs_shapes[layer_index].width)
        # insert func call
        code_to_insert += debugging_fill_layer_input_block.format(ifms_file_path + ifms_file,
                                                                  'channels' if direction == 0 else 'result',
                                                                  str(
                                                                      layers_inputs_shapes[layer_index].height),
                                                                  str(layers_inputs_shapes[layer_index].width))
        code_to_insert += debugging_verify_fill_layer_input_block.format(ofms_file_path + 'verify_' + str(layer_index)+'.txt',
                                                                         'channels' if direction == 0 else 'result',
                                                                         layers_inputs_shapes[layer_index].depth * layers_inputs_shapes[layer_index].height *
                                                                         layers_inputs_shapes[layer_index].width, str(
            layers_inputs_shapes[layer_index].height),
            str(layers_inputs_shapes[layer_index].width))

    code_to_insert += replace(replacement_dict, target_block)

    if layer_index in cgc.LAYERS_TO_DEBUG:
        code_to_insert += debugging_dump_ofms_block.format(ofms_file_path + 'ofms_' + str(layer_index)+'.txt',
                                                           'result' if direction == 0 else 'channels',
                                                           layers_output_shapes[layer_index].depth * layers_output_shapes[layer_index].height *
                                                           layers_output_shapes[layer_index].width, str(
                                                               layers_output_shapes[layer_index].height),
                                                           str(layers_output_shapes[layer_index].width))

    direction = 1 - direction

file_replacement = file_replacement[:insert_index] + \
    code_to_insert + file_replacement[insert_index:]

with open(in_out_file, 'w') as f:
    f.write(file_replacement)
