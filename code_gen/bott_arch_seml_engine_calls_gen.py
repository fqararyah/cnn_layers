import code_generation_constants as cgc
import utils


utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)


in_out_file = '../model_components/model/SEML/imp/seml.cpp'
in_out_header_file = '../model_components/model/SEML/headers/seml.h'
ofms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/'
ifms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/fms/'.format(
    cgc.MODEL_NAME)

ifms_file_format = 'fms_{}_{}_{}_{}.txt'

debugging_includes_block = '#include "../../../../tests/test_utils.h"\n'

# fill_quantization_parameters_block = 'fill_fused_scales_and_zero_points(layer_*i*_fused_scales,fused_scales, \n\
#     layer_*i*_fused_scales_log_2_shifts, fused_scales_log_2_shifts, layer_*i*_relu_6_fused_scales,\n\
#      relu_6_fused_scales, layer_*i*_fused_zero_points,\n\
#     fused_zero_points, layer_*i*_*TYPE*_num_fils);\n'

layer_0_block = 'layer_0_3x3(weights_0, input_image, result2, layer_0_fused_scales, layer_0_fused_scales_log_2_shifts, layer_0_relu_6_fused_scales, layer_0_fused_zero_points);\n'

expansion_projection_block = 'pw_conv(off_chip_weights, channels, result2, *i*, layer_*i*_pw_depth,\n\
    layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
    layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
    layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
    layer_*i*_pw_num_of_weight_groups_for_one_pass,\n\
    *DIRECTION*, layer_*i*_pw_weights_offset, layer_*i*_relu, layer_*i*_fused_scales, layer_*i*_fused_scales_log_2_shifts, layer_*i*_relu_6_fused_scales, layer_*i*_fused_zero_points);\n'

dw_block_0 = 'fill_dw_layer_weights(dw_weights_*i*, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
    dw_conv_3x3(dw_weights_buffer, channels, result2, *i*, layer_*i*_dw_depth,\n\
    layer_*i*_dw_ifm_width, layer_*i*_dw_ifm_height, layer_*i*_dw_num_of_tiles_in_d,\n\
    layer_*i*_dw_num_of_tiles_h, layer_*i*_dw_num_of_tiles_w,\n\
    layer_*i*_dw_strides, layer_*i*_dw_padding_left, layer_*i*_dw_padding_right, layer_*i*_dw_padding_top,\n\
    *DIRECTION*, layer_*i*_fused_scales, layer_*i*_fused_scales_log_2_shifts, layer_*i*_relu_6_fused_scales, layer_*i*_fused_zero_points);\n'

dw_block_1 = 'fill_dw_layer_weights(dw_weights_*i*, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
    dw_conv_3x3(dw_weights_buffer, result2, channels, *i*, layer_*i*_dw_depth,\n\
    layer_*i*_dw_ifm_width, layer_*i*_dw_ifm_height, layer_*i*_dw_num_of_tiles_in_d,\n\
    layer_*i*_dw_num_of_tiles_h, layer_*i*_dw_num_of_tiles_w,\n\
    layer_*i*_dw_strides, layer_*i*_dw_padding_left, layer_*i*_dw_padding_right, layer_*i*_dw_padding_top,\n\
    *DIRECTION*, layer_*i*_fused_scales, layer_*i*_fused_scales_log_2_shifts, layer_*i*_relu_6_fused_scales, layer_*i*_fused_zero_points);\n'

# projection_block = 'pw_conv(off_chip_weights, channels, result2, *i*, layer_*i*_pw_depth,\n\
#     layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
#     layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
#     layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
#     layer_*i*_pw_num_of_weight_groups_for_one_pass,\n\
#     *DIRECTION*, layer_*i*_pw_weights_offset, layer_*i*_relu);\n'

debugging_dump_ofms_block = 'dump_layer_output("{}",\n {}, {}, {}, {});\n'
debugging_fill_layer_input_block = 'fill_layer_input("{}",\n {}, {}, {});\n'
debugging_verify_fill_layer_input_block = 'verify_fill_layer_input("{}",\n {}, {}, {}, {});\n'

# layers_to_debug = [2, 12, 13, 20, 21, 22, 23, 24,25,26,27,28,29,30,31, 32,33,34, 35, 36, 37, 38, 39, 40, 41,
#layers_to_debug = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
#layers_to_debug = [2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30, 33, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

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
    tf_lite_to_my_cnn_layer_ifms_mapping[layer_index] = layer_index + \
        skip_connections_so_far


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
skip_connections_depth = 3

for i in range(layers_to_generate[0] + 1):
    if layers_types[layer_index] == 'pw' and not expansion_projection[layer_index]:
        continue
    direction = 1-direction


skip_connections_indices = utils.read_skip_connections_indices()

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

for layer_index in range(layers_to_generate[0], layers_to_generate[1]):
    target_block = ''#fill_quantization_parameters_block
    replacement_dict = {}
    replacement_dict['*i*'] = layer_index
    replacement_dict['*DIRECTION*'] = direction

    read_write = 0
    if layers_types[layer_index] == 'pw' and expansion_projection[layer_index] == 0:
        continue

    if layer_index == 0:
        target_block += layer_0_block
        replacement_dict['*TYPE*_'] = ''
    if layers_types[layer_index] == 'pw':
        replacement_dict['*TYPE*'] = 'pw'
        target_block += expansion_projection_block
        if layer_index + skip_connections_depth + 1 in skip_connections_indices:
            read_write += 2
        if layer_index + 1 in skip_connections_indices:
            read_write += 1

    elif layers_types[layer_index] == 'dw':
        if direction == 0:
            target_block += dw_block_0
        else:
            target_block += dw_block_1
        replacement_dict['*TYPE*'] = 'dw'

    replacement_dict['*RW*'] = read_write
    if cgc.DEBUGGING and layer_index == cgc.LAYERS_TO_DEBUG[0]:
        # file_name
        ifms_file = ifms_file_format.format(tf_lite_to_my_cnn_layer_ifms_mapping[layer_index], layers_inputs_shapes[layer_index].depth,
                                            layers_inputs_shapes[layer_index].height, layers_inputs_shapes[layer_index].width)
        # insert func call
        code_to_insert += debugging_fill_layer_input_block.format(ifms_file_path + ifms_file,
                                                                  'channels' if direction == 0 else 'result2',
                                                                  str(
                                                                      layers_inputs_shapes[layer_index].height),
                                                                  str(layers_inputs_shapes[layer_index].width))
        code_to_insert += debugging_verify_fill_layer_input_block.format(ofms_file_path + 'verify_' + str(layer_index)+'.txt',
                                                                         'channels' if direction == 0 else 'result2',
                                                                         layers_inputs_shapes[layer_index].depth * layers_inputs_shapes[layer_index].height *
                                                                         layers_inputs_shapes[layer_index].width, str(
            layers_inputs_shapes[layer_index].height),
            str(layers_inputs_shapes[layer_index].width))

    code_to_insert += replace(replacement_dict, target_block)

    if cgc.DEBUGGING and layer_index in cgc.LAYERS_TO_DEBUG:
        code_to_insert += debugging_dump_ofms_block.format(ofms_file_path + 'ofms_' + str(layer_index)+'.txt',
                                                           'result2' if direction == 0 else 'channels',
                                                           layers_output_shapes[layer_index].depth * layers_output_shapes[layer_index].height *
                                                           layers_output_shapes[layer_index].width, str(
                                                               layers_output_shapes[layer_index].height),
                                                           str(layers_output_shapes[layer_index].width))

    if expansion_projection[layer_index] or layers_types[layer_index] != 'pw':
        direction = 1 - direction

file_replacement = file_replacement[:insert_index] + \
    code_to_insert + file_replacement[insert_index:]

with open(in_out_file, 'w') as f:
    f.write(file_replacement)
