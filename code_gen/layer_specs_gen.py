import utils
import prepare_off_chip_weights

import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

out_file = '../model_components/model/headers/layers_specs.h'  # './out/layers_specs.h'

weights_group_items = 64

specs_struct = 'const layer_specs layer_{}_specs = {}\n\
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

specs_block = "//****************************\n \
const int layer_{}_{}_num_fils = {} / alpha;\n\
const int layer_{}_{}_depth = {};\n\
const int layer_{}_{}_filter_dim = {};\n \
const int layer_{}_{}_ifm_width = {};\n \
//****************************\n"

model_dag = utils.read_model_dag()

current_block_indx = 0
cumulative_pw_weights = 0
with open(out_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef LAYERS_SPECS\n")
    f.write("#define LAYERS_SPECS\n")
    for layer_index in range(len(model_dag)):
        layer_specs = model_dag[layer_index]
        layer_type = ''
        if 'type' in layer_specs:
            layer_type = layer_specs['type']
        if layer_type not in cgc.CONV_LAYER_TYPES:
            continue

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

        replacement_list = []
        #replacement_dic['*PREV*'] = layers_types[i-1]
        strides = layer_specs['strides']
        filter_dim = layer_filter_dim
        num_of_filters = layer_num_fils
        replacement_list.append(str(layer_index) + '_' + layer_type)
        replacement_list.append('{')
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

        replacement_list.append(layer_depth)
        replacement_list.append(layer_height)
        replacement_list.append(layer_width)

        replacement_list.append(int(layer_height / strides))
        replacement_list.append(int(layer_width / strides))

        layer_activation = ''
        if layer_specs['activation'] == 'relu6':
            layer_activation = '6'
        elif layer_specs['activation'] == 'sigmoid':
            layer_activation = '2'
        else:
            layer_activation = '0'

        replacement_list.append(layer_activation)

        replacement_list.append(
            '(' + str(layer_depth) + ' + pw_tile_d - 1) / pw_tile_d')
        replacement_list.append('(' + str(num_of_filters) +
                                ' + pw_conv_parallelism_out) / pw_conv_parallelism_out')

        replacement_list.append(
            '(' + str(layer_height) + ' + pw_tile_h - 1) / pw_tile_h')
        replacement_list.append(
            '(' + str(layer_width) + ' + pw_tile_w - 1) / pw_tile_w')

        replacement_list.append(
            '(' + str(int(layer_height / strides)) + ' + pw_tile_h - 1) / pw_tile_h')
        replacement_list.append(
            '(' + str(int(layer_width / strides)) + ' + pw_tile_w - 1) / pw_tile_w')

        replacement_list.append(
            str(layer_depth) + ' * pw_conv_parallelism_out / weights_group_items')

        if layer_type == 'pw' and i > 0:
            replacement_list.append(cumulative_pw_weights)
            cumulative_pw_weights += int(
                layer_weights_size / weights_group_items)
        else:
            replacement_list.append(0)

        write_to_tmp = 0
        fused_with_add = 0
        add_layer_scale_reciprocal = 1
        add_layer_zero_point = 0
        skip_connection_other_layer_scale = 1
        skip_connection_other_layer_zero_point = 0

        layer_children = layer_specs['children']
        if len(layer_children) > 1:
            write_to_tmp = 1

        if model_dag[layer_children[0]]['name'] == 'add':
            add_layer_specs = model_dag[layer_children[0]]
            the_other_conv_layer_specs = model_dag[add_layer_specs['parents'][0]]
            fused_with_add = 1
            add_layer_scale_reciprocal = 1 / add_layer_specs['ofms_scales']
            add_layer_zero_point = add_layer_specs['ofms_zero_points']
            skip_connection_other_layer_scale = the_other_conv_layer_specs['ofms_scales']
            skip_connection_other_layer_zero_point = the_other_conv_layer_specs['ofms_zero_points']
            if len(add_layer_specs['children']) > 1:
                #if the fused add layer is a beginning of a branch
                write_to_tmp = 1

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

        to_write_specs_block = specs_block.format(layer_index, layer_type, layer_num_fils,
                           layer_index, layer_type, layer_ifms_depth,
                           layer_index, layer_type, layer_filter_dim,
                           layer_index, layer_type, layer_width)
        f.write(to_write_specs_block)

        f.write(specs_struct.format(*replacement_list))

    f.write('#endif\n')
