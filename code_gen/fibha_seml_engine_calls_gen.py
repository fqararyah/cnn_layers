import code_generation_constants as cgc
import utils


utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

in_out_file = '../model_components/model/SEML/imp/{}_seml{}.cpp'.format(cgc.MODEL_NAME,
                                                                        cgc.FIBHA_VERSION_POSTFIX)
constants_header_file = '../model_components/basic_defs/simulation_constants.h'
in_out_header_file = '../model_components/model/SEML/headers/seml.h'
ofms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/'
ifms_file_path = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/fms/'.format(
    cgc.MODEL_NAME)

ifms_file_format = 'ifms_{}.txt'

debugging_includes_block = '#include "../../../../tests/test_utils.h"\n'

layer_0_s_block = 'layer_0_s_3x3(weights_1, input_image, result);\n'

s_pw_block = 'pw_and_conv(off_chip_weights, {} , {}, tmp_channels, *i*, layer_*i*_s_specs,\n\
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,\n\
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);\n'

pw_block = 'pw_conv(off_chip_weights, {} , {}, tmp_channels, *i*, layer_*i*_pw_specs,\n\
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,\n\
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);\n'

# 'fill_dw_layer_weights(seml_dw_weights_3x3, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
dw_block = \
    'seml_engines::dw_conv_3x3(seml_dw_weights_3x3, {}, {}, *i*,layer_*i*_dw_specs,\n\
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,\n\
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);\n'

debugging_dump_ofms_block = '#if DEBUGGING\n dump_layer_output("{}",\n {}, {});\n#endif\n'
debugging_fill_layer_input_block = '#if DEBUGGING\n fill_layer_input("{}",\n {}, {});\n#endif\n'
debugging_verify_fill_layer_input_block = '#if DEBUGGING\n verify_fill_layer_input("{}",\n {}, {});\n#endif\n'

# layers_to_debug = [2, 12, 13, 20, 21, 22, 23, 24,25,26,27,28,29,30,31, 32,33,34, 35, 36, 37, 38, 39, 40, 41,
#layers_to_debug = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
#layers_to_debug = [2,3,4,5,6,7,8,9,10, 11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29, 30, 33, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

model_dag = utils.read_model_dag()


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

first_seml_layer_to_generate = 0
conv_layers_so_far = 0
while conv_layers_so_far < cgc.FIRST_LAYER_TO_GENERATE:
    if 'type' in model_dag[first_seml_layer_to_generate] and model_dag[first_seml_layer_to_generate]['type'] in cgc.CONV_LAYER_TYPES:
        conv_layers_so_far += 1

    first_seml_layer_to_generate += 1

layers_to_generate = [first_seml_layer_to_generate,
                      cgc.LAST_LAYER_TO_GENERATE + 1 if cgc.LAST_LAYER_TO_GENERATE != -1 else len(model_dag)]

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

direction = 0  # assumes a topological ordering
code_to_insert = ''

for layer_index in range(layers_to_generate[0], layers_to_generate[1]):
    layer_specs = model_dag[layer_index]
    layer_type = ''
    if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES:
        layer_type = layer_specs['type']
    else:
        continue

    target_block = ''
    if layer_index == 0:
        target_block = layer_0_s_block
    if layer_type == 'dw':
        target_block = dw_block
    elif layer_type == 'pw':
        target_block = pw_block
    elif layer_type == 's':
        target_block = s_pw_block

    target_block = target_block.replace('*i*', str(layer_index))

    if direction == 0:
        target_block = target_block.format('channels', 'result')
    else:
        target_block = target_block.format('result', 'channels')

    if cgc.DEBUGGING and layer_index == cgc.LAYERS_TO_DEBUG[0] and layer_index != 0:
        # file_name
        ifms_file = ifms_file_format.format(layer_index)
        # insert func call
        layer_specs_str = 'layer_' + \
            str(layer_index) + '_' + layer_type + '_specs'
        if len(model_dag[layer_specs['parents'][0]]['children']) > 1:
            code_to_insert += debugging_fill_layer_input_block.format(ifms_file_path + ifms_file, 'tmp_channels',
                                                                      layer_specs_str)
        code_to_insert += debugging_fill_layer_input_block.format(ifms_file_path + ifms_file,
                                                                  'channels' if direction == 0 else 'result',
                                                                  layer_specs_str
                                                                  )
        code_to_insert += debugging_verify_fill_layer_input_block.format(ofms_file_path + 'verify_' + str(layer_index)+'.txt',
                                                                         'channels' if direction == 0 else 'result',
                                                                         layer_specs_str
                                                                         )

    code_to_insert += target_block

    if layer_index in cgc.LAYERS_TO_DEBUG:
        code_to_insert += debugging_dump_ofms_block.format(ofms_file_path + 'ofms_' + str(layer_index)+'.txt',
                                                           'result' if direction == 0 else 'channels',
                                                           'layer_' +
                                                           str(layer_index) + '_' +
                                                           layer_type + '_specs'
                                                           )

    current_layer_parent_children = model_dag[layer_specs['parents'][0]]['children']
    last_non_add_child = len(current_layer_parent_children) - 1
    while last_non_add_child >= 0 and model_dag[current_layer_parent_children[last_non_add_child]]['name'] == 'add':
        last_non_add_child -= 1

    if layer_index == current_layer_parent_children[last_non_add_child]:
        direction = 1 - direction

file_replacement = file_replacement[:insert_index] + \
    code_to_insert + file_replacement[insert_index:]

with open(in_out_file, 'w') as f:
    f.write(file_replacement)

file_replacement = ''
with open(constants_header_file, 'r') as f:
    for line in f:
        if '#define CHAIN_LENGTH' in line:
            file_replacement += '#define CHAIN_LENGTH ' + \
                str(cgc.PIPELINE_LEN) + '\n'
        else:
            file_replacement += line

with open(constants_header_file, 'w') as f:
    f.write(file_replacement)
