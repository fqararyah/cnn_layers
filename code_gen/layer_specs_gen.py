
import utils
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

# './out/layers_specs.h'
out_file = '../model_components/model/headers/{}_layers_specs_pipe_{}.h'

pipeline_len = 0
if cgc.PIPELINE:
    pipeline_len = cgc.PIPELINE_LEN

model_config_file = '../model_config/{}.txt'.format(cgc.MODEL_NAME)

specs_struct = 'const layer_specs {} = {}\n\
                {},//layer_index;\n\
                {},//conv_layer_type;; \n\
                {},//layer_num_fils \n\
                {},//strides;\n\
                {},//filter_size;\n\
                {},//padding_left;\n\
                {},//padding_right;\n\
                {},//padding_top;\n\
                {},//padding_bottom;\n\
                {},//layer_depth;\n\
                {},//layer_ifm_height;\n\
                {},//layer_ifm_width;\n\
                {},//layer_ofm_height;\n\
                {},//layer_ofm_width;\n\
                {},//layer_activation;\n\
                {},//layer_num_of_tiles_in_d;\n\
                {},//layer_num_of_tiles_out_d;\n\
                {},//layer_num_of_ifm_tiles_h;\n\
                {},//layer_num_of_ifm_tiles_w;\n\
                {},//layer_num_of_ofm_tiles_h;\n\
                {},//layer_num_of_ofm_tiles_w;\n\
                {},//layer_num_of_weight_groups_for_one_pass;\n\
                {},//layer_weights_offset;\n\
                {},//layer_weights_offset_on_chip;\n\
                {},//dw_ifms_cumulative_width_offset;\n\
                {},//bool write_to_result_or_channels;\n\
                {},//bool write_to_tmp;\n\
                {},//bool fused_with_add;\n\
                {},//fms_dt layer_ifms_zero_point;\n\
                {},//,fms_dt layer_ofms_scale;\n\
                {},//fms_dt layer_ofms_zero_point;\n\
                {},//rec_scales_dt add_layer_scale_reciprocal;\n\
                {},//biases_dt add_layer_zero_point;\n\
                {},//scales_dt skip_connection_other_layer_scale;\n\
                {}//biases_dt skip_connection_other_layer_zero_point;\n\
                {};\n'

pooling_specs_struct = 'const pooling_layer_specs layer_{}_specs = {}\n\
                {},//const pooling_fused_scales_dt fused_scale; \n\
                {},//const biases_dt ifms_zero_point;\n\
                {},//const biases_dt ofms_zero_point;\n\
                {};\n'

quantize_specs_struct = 'const Quantization_layer_specs quantize_layer_specs = {}\n\
                {},//const pooling_fused_scales_dt fused_scale; \n\
                {},//const biases_dt ifms_zero_point;\n\
                {}//const biases_dt ofms_zero_point;\n\
                {};\n'

fc_specs_struct = 'const fc_layer_specs layer_{}_specs = {}\n\
                {},//const fms_dt ifm_zero_point\n\
                {};\n'

specs_block = "//****************************\n \
const int layer_{}_{}_num_fils = {};\n\
const int layer_{}_{}_depth = {};\n\
const int layer_{}_{}_filter_dim = {};\n \
const int layer_{}_{}_ifm_width = {};\n \
//****************************\n"

first_layer_specs_block = "//****************************\n \
const int first_conv_layer_num_fils = {};\n\
const int first_conv_layer_depth = {};\n\
const int first_conv_layer_filter_dim = {};\n \
const int first_conv_layer_strides = {};\n \
const int first_conv_layer_padding_left = {};\n \
const int first_conv_layer_padding_right = {};\n \
const int first_conv_layer_ifm_width = {};\n \
//****************************\n"

model_dag = utils.read_model_dag()
model_configs_list = [0] * 2 * len(model_dag)
current_block_indx = 0
cumulative_s_pw_weights = 0
cumulative_s_pw_weights_on_chip = 0
cumulative_dw_weights = 0
dw_ifms_cumulative_width_offset = 0
num_conv_layers_so_far = 0
first_conv_layer = True
first_conv_layer_index = 0
with open(out_file.format(cgc.MODEL_NAME, pipeline_len), 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write('#if PIPELINE_LENGTH == ' + str(pipeline_len) + '\n')
    f.write("#ifndef LAYERS_SPECS\n")
    f.write("#define LAYERS_SPECS\n")

    for layer_index in range(len(model_dag)):
        layer_specs = model_dag[layer_index]
        layer_type = ''
        replacement_list = []
        if 'type' in layer_specs:
            layer_type = layer_specs['type']
        if layer_type in cgc.CONV_LAYER_TYPES:
            num_conv_layers_so_far += 1
            layer_weights_shape = layer_specs['weights_shape']
            layer_weights_size = 1
            for i in layer_weights_shape:
                layer_weights_size *= i
            layer_filter_dim = 1
            layer_ifms_depth = layer_weights_shape[1]
            layer_num_fils = layer_weights_shape[0]
            if layer_type != 'pw':
                layer_filter_dim = layer_weights_shape[-1]
            if layer_type == 'dw':
                layer_ifms_depth = layer_num_fils

            # replacement_dic['*PREV*'] = layers_types[i-1]
            strides = layer_specs['strides']
            filter_dim = layer_filter_dim
            num_of_filters = layer_num_fils
            if first_conv_layer:
                replacement_list.append('first_conv_layer_specs')
                first_conv_layer_index = layer_index
            else:
                replacement_list.append(
                    'layer_' + str(layer_index) + '_' + layer_type + '_specs')
            replacement_list.append('{')
            replacement_list.append(layer_index)
            if layer_type == 'pw':
                replacement_list.append('PW_CONV')
            elif layer_type == 'dw':
                replacement_list.append('DW_CONV')
            else:
                replacement_list.append('S_CONV')
            replacement_list.append(num_of_filters)
            replacement_list.append(strides)
            replacement_list.append(filter_dim)
            padding_left = 0
            padding_right = 0
            padding_top = 0
            padding_bottom = 0
            if layer_type != 'pw':
                padding = int(filter_dim - strides)
                if strides == 1:
                    padding_left = int(padding / 2)
                    padding_right = int(padding / 2)
                    padding_top = int(padding / 2)
                    padding_bottom = int(padding / 2)
                else:
                    padding_right = padding
                    padding_bottom = padding

            replacement_list.append(padding_left)
            replacement_list.append(padding_right)
            replacement_list.append(padding_top)
            replacement_list.append(padding_bottom)

            layer_ifms_shape = layer_specs['ifms_shape']
            layer_depth = layer_ifms_shape[0]
            layer_height = layer_ifms_shape[1]
            layer_width = layer_ifms_shape[2]

            if model_dag[layer_index - 1]['name'] == 'pad':
                layer_width -= padding_right
                layer_height -= padding_bottom

            replacement_list.append(layer_depth)
            replacement_list.append(layer_height)
            replacement_list.append(layer_width)

            replacement_list.append(int(layer_height / strides))
            replacement_list.append(int(layer_width / strides))

            layer_activation = ''
            if layer_specs['activation'] != '':
                layer_activation = layer_specs['activation']
            else:
                layer_activation = '0'

            replacement_list.append(layer_activation)

            replacement_list.append(
                '(' + str(layer_depth) + ' + pw_tile_d - 1) / pw_tile_d')
            replacement_list.append('(' + str(num_of_filters) +
                                    ' + pw_conv_parallelism_out - 1) / pw_conv_parallelism_out')

            replacement_list.append(
                '(' + str(layer_height) + ' + pw_tile_h - 1) / pw_tile_h')
            replacement_list.append(
                '(' + str(layer_width) + ' + pw_tile_w - 1) / pw_tile_w')

            replacement_list.append(
                '(' + str(int(layer_height / strides)) + ' + pw_tile_h - 1) / pw_tile_h')
            replacement_list.append(
                '(' + str(int(layer_width / strides)) + ' + pw_tile_w - 1) / pw_tile_w')

            replacement_list.append( '(' +
                str(layer_depth) + ' * pw_conv_parallelism_out ) / weights_group_items')

            if (layer_type == 'pw' or layer_type == 's') and not first_conv_layer:
                replacement_list.append(
                    str(cumulative_s_pw_weights) + ' / weights_group_items')
                replacement_list.append(cumulative_s_pw_weights_on_chip)
                replacement_list.append(0)
                if num_conv_layers_so_far > cgc.PIPELINE_LEN or cgc.PIPELINE == False:
                    cumulative_s_pw_weights += layer_weights_size
                cumulative_s_pw_weights_on_chip += int(layer_weights_size)
            elif layer_type == 'dw':
                replacement_list.append(cumulative_dw_weights)
                replacement_list.append(0)
                replacement_list.append(dw_ifms_cumulative_width_offset)
                dw_ifms_cumulative_width_offset += int(
                    layer_width) * layer_depth * (layer_filter_dim - strides)
                cumulative_dw_weights += int(
                    layer_depth)
            else:
                if layer_type == 'pw' or layer_type == 's':
                    first_conv_layer = False
                replacement_list.append(0)
                replacement_list.append(0)
                replacement_list.append(0)

            write_to_tmp = 0
            write_to_result_or_channels = 1
            fused_with_add = 0
            add_layer_scale_reciprocal = 1
            add_layer_zero_point = 0
            skip_connection_other_layer_scale = 1
            skip_connection_other_layer_zero_point = 0

            layer_children = layer_specs['children']
            for i in range(len(layer_children)):
                if layer_children[i] - i != layer_index + 1:
                    write_to_tmp = 1

            if model_dag[layer_children[0]]['name'] == 'add' and model_dag[layer_children[0]]['id'] == layer_index + 1:
                add_layer_specs = model_dag[layer_children[0]]
                the_other_conv_layer_specs = model_dag[add_layer_specs['parents'][0]]
                fused_with_add = 1
                add_layer_scale_reciprocal = 1 / add_layer_specs['ofms_scales']
                add_layer_zero_point = add_layer_specs['ofms_zero_points']
                skip_connection_other_layer_scale = the_other_conv_layer_specs['ofms_scales']
                skip_connection_other_layer_zero_point = the_other_conv_layer_specs[
                    'ofms_zero_points']
                if len(add_layer_specs['children']) > 1:
                    # if the fused add layer is a beginning of a branch
                    write_to_tmp = 1

            if write_to_tmp == 1 and fused_with_add == 0 and len(layer_children) <= 1:
                write_to_result_or_channels = 0

            replacement_list.append(write_to_result_or_channels)
            replacement_list.append(write_to_tmp)
            replacement_list.append(fused_with_add)

            replacement_list.append(layer_specs['ifms_zero_points'])
            replacement_list.append(layer_specs['ofms_scales'])
            replacement_list.append(layer_specs['ofms_zero_points'])

            replacement_list.append(add_layer_scale_reciprocal)
            replacement_list.append(add_layer_zero_point)
            replacement_list.append(skip_connection_other_layer_scale)
            replacement_list.append(skip_connection_other_layer_zero_point)

            replacement_list.append('}')

            if current_block_indx > 0:
                to_write_specs_block = specs_block.format(layer_index, layer_type, layer_num_fils,
                                                          layer_index, layer_type, layer_ifms_depth,
                                                          layer_index, layer_type, layer_filter_dim,
                                                          layer_index, layer_type, layer_width)
            else:
                current_block_indx += 1
                to_write_specs_block = first_layer_specs_block.format(layer_num_fils, layer_ifms_depth, layer_filter_dim,
                                                                      strides, padding_left, padding_right, layer_width)
            f.write(to_write_specs_block)

            f.write(specs_struct.format(*replacement_list))

            model_configs_list[2 * layer_index] = layer_depth
            model_configs_list[2 * layer_index + 1] = num_of_filters

        elif 'type' in layer_specs:
            layer_type = layer_specs['type']
            if layer_type == 'avgpool':
                replacement_list.append(str(layer_index) + '_' + layer_type)
                replacement_list.append('{')
                pooling_ifms_scale = layer_specs['ifms_scales']
                pooling_ofms_scale = layer_specs['ofms_scales']
                replacement_list.append(
                    pooling_ifms_scale / pooling_ofms_scale)

                replacement_list.append(layer_specs['ifms_zero_points'])
                replacement_list.append(layer_specs['ofms_zero_points'])
                replacement_list.append('}')
                f.write(pooling_specs_struct.format(*replacement_list))

            elif layer_type == 'fc':
                replacement_list.append(str(layer_index) + '_' + layer_type)
                replacement_list.append('{')
                replacement_list.append(layer_specs['ifms_zero_points'])
                replacement_list.append('}')
                f.write(fc_specs_struct.format(*replacement_list))
        elif layer_specs['name'] == 'quantize' and first_conv_layer:
            replacement_list.append('{')
            ofms_scale = layer_specs['ofms_scales']
            ifms_scale = layer_specs['ifms_scales']
            replacement_list.append(
                ifms_scale / ofms_scale)

            replacement_list.append(layer_specs['ifms_zero_points'])
            replacement_list.append(layer_specs['ofms_zero_points'])
            replacement_list.append('}')
            f.write(quantize_specs_struct.format(*replacement_list))

    f.write('const layer_specs layer_' + str(first_conv_layer_index) + '_s_specs = first_conv_layer_specs;\n')
    f.write('#endif\n')
    f.write('#endif\n')

with open(model_config_file, 'w') as f:
    for i in model_configs_list:
        f.write(str(i) + '\n')
