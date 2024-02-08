import code_generation_constants as cgc
import utils


utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

pipeline_len_str = 'pipe_' + str(cgc.PIPELINE_LEN)
if not cgc.PIPELINE:
    pipeline_len_str = 'pipe_0'

in_out_file = '../model_components/model/SEML/imp/{}_seml{}_{}.cpp'.format(cgc.MODEL_NAME,
                                                                        cgc.FIBHA_VERSION_POSTFIX, pipeline_len_str)

if 'uniform' in cgc.MODEL_NAME:
    in_out_file = '../model_components/model/SEML/imp/{}_seml{}_{}.cpp'.format('mob_v2',
                                                                            cgc.FIBHA_VERSION_POSTFIX, pipeline_len_str)

constants_header_file = '../model_components/basic_defs/simulation_constants.h'
in_out_header_file = '../model_components/model/SEML/headers/seml.h'
ofms_file_path = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/'
ifms_file_path = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/fms/'.format(
    cgc.MODEL_NAME)

ifms_file_format = 'ifms_{}.txt'

debugging_includes_block = '#include "../../../../tests/test_utils.h"\n'

first_conv_block = '//layer_0_s_3x3(weights_1, input_image, result);\n'

# first_conv_block = 'pw_and_conv(off_chip_weights, channels , result, tmp_channels, *i*, layer_*i*_s_specs,\n\
#     first_conv_layer_fused_scales, first_conv_layer_relu_6_fused_scales, first_conv_layer_fused_zero_points, model_configs_list);\n'

s_pw_block = 'pw_and_conv(off_chip_weights, {} , {}, tmp_channels, *i*, layer_*i*_s_specs,\n\
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer, model_configs_list);\n'

fill_scales_block = 'seml_engines::fill_fused_scales(off_chip_fused_scales,\n\
                                     seml_fused_scales_buffer,\n\
                                     layers_fused_parameters_offsets[{}],\n\
                                     layer_{}_{}_specs.layer_num_fils);\n'

fill_zps_block = 'seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, \n\
                                seml_fused_zero_points_buffer, \n\
                                layers_fused_parameters_offsets[{}], \n\
                                layer_{}_{}_specs.layer_num_fils);\n'

pw_block = 'pw_conv(off_chip_weights, {}, {}, tmp_channels, *i*, layer_*i*_pw_specs,\n\
                            seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,\n\
                            model_configs_list, layer_*i*_pw_specs.layer_ofm_height);\n\
                            even_odd = 1 - even_odd;\n'

# 'fill_dw_layer_weights(seml_dw_weights_3x3, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
fill_dw_weights_block = 'seml_engines::fill_layer_dw_weights_off_chip\n\
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[{}], layer_{}_dw_specs.layer_depth);\n'
dw_block = \
    'seml_engines::dw_conv_3x3(seml_dw_weights_3x3, {}, {}, *i*, layer_*i*_dw_specs,\n\
                                              seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,\n\
                                              model_configs_list,\n\
                                            layer_*i*_dw_specs.layer_ofm_height, 0);\n\
                                            even_odd = 1 - even_odd;\n'

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
first_conv_layer = (not cgc.PIPELINE) or (not cgc.PIPELINE_LEN == 0)
for layer_index in range(layers_to_generate[0], layers_to_generate[1]):
    layer_specs = model_dag[layer_index]
    layer_type = ''
    if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES:
        layer_type = layer_specs['type']
    else:
        continue

    target_block = ''
    if layer_type == 's' and first_conv_layer:
        first_conv_layer = False
        continue
    elif layer_type == 'dw':
        if cgc.DW_WEIGHTS_OFF_CHIP:
            target_block = fill_dw_weights_block.format(layer_index, layer_index)
        
        target_block += fill_scales_block.format(layer_index, layer_index, 'dw')
        target_block += fill_zps_block.format(layer_index, layer_index, 'dw')
        target_block += dw_block
    elif layer_type == 'pw':
        target_block += fill_scales_block.format(layer_index, layer_index, 'pw')
        target_block += fill_zps_block.format(layer_index, layer_index, 'pw')
        target_block += pw_block
        target_block += '\n'
    elif layer_type == 's':
        target_block += fill_scales_block.format(layer_index, layer_index, 's')
        target_block += fill_zps_block.format(layer_index, layer_index, 's')
        target_block += s_pw_block
        target_block += '\n'


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
        if '#define PIPELINE_LENGTH' in line:
            if cgc.PIPELINE:
                file_replacement += '#define PIPELINE_LENGTH ' + \
                    str(cgc.PIPELINE_LEN) + '\n'
            else:
               file_replacement += '#define PIPELINE_LENGTH 0\n' 
        else:
            file_replacement += line

with open(constants_header_file, 'w') as f:
    f.write(file_replacement)
