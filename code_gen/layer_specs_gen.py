import utils
import prepare_off_chip_weights

import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

out_file = '../model_components/model/headers/layers_specs.h'  # './out/layers_specs.h'

to_replace = ['*LNF*', '*LD*', '*LW*', '*LH*', '*LST*', '*LPL*',
              '*LPR*', '*LFS*', '*i*', '*i-1*', '*i-1_pw*', '*LWOF*']

weights_group_items = 64

specs_block = 'const layer_specs layer_{}_specs = {}\n\
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
            {};\n'

block_0 = "//****************************\n \
const int layer_0_s_num_fils = *LNF* / alpha;\n\
const int layer_0_s_depth = input_image_depth;\n\
const int layer_0_s_ifm_height = input_image_height;\n\
const int layer_0_s_ifm_width = input_image_width;\n\
const int layer_0_s_strides = *LST*;\n\
const int layer_0_s_ofm_height = layer_0_s_ifm_height / layer_0_s_strides;\n\
const int layer_0_s_ofm_width = layer_0_s_ifm_width / layer_0_s_strides;\n\
const int layer_0_s_num_of_tiles_out_d = int(0.99 + ((float) layer_0_s_num_fils) / pw_conv_parallelism_out);\n\
const int layer_0_s_padding_left = *LPL*;\n\
const int layer_0_s_padding_right = *LPR*;\n\
const int layer_0_s_padding_top = *LPT*;\n \
const int layer_0_s_padding_bottom = *LPB*;\n \
const int layer_0_s_filter_dim = *LFS*;\n \
const int layer_0_s_num_of_tiles_w = layer_0_s_ofm_width / pw_tile_w; \n \
const int layer_0_s_num_of_tiles_h = layer_0_s_ofm_height / pw_tile_h; \n \
const int layer_0_s_num_of_tiles_d_in = layer_0_s_depth / pw_tile_d; \n \
//****************************\n"


dw_block = "const int layer_*i*_dw_num_fils = layer_*i-1*_*PREV*_num_fils / alpha;\n \
const int layer_*i*_dw_depth = layer_*i*_dw_num_fils;\n \
const int layer_*i*_dw_strides = *LST*;\n \
const int layer_*i*_dw_ifm_height = layer_*i-1*_*PREV*_ofm_height;\n \
const int layer_*i*_dw_ifm_width = layer_*i-1*_*PREV*_ofm_width;\n \
const int layer_*i*_dw_ofm_height = layer_*i*_dw_ifm_height / layer_*i*_dw_strides;\n \
const int layer_*i*_dw_ofm_width = layer_*i*_dw_ifm_width / layer_*i*_dw_strides;\n \
const int layer_*i*_dw_padding_left = *LPL*;\n \
const int layer_*i*_dw_padding_right = *LPR*;\n \
const int layer_*i*_dw_padding_top = *LPT*;\n \
const int layer_*i*_dw_padding_bottom = *LPB*;\n \
const int layer_*i*_dw_filter_size = *LFS*;\n \
const int layer_*i*_dw_num_of_tiles_in_d = (int)(0.99 + (float)layer_*i*_dw_depth / dw_tile_d);\n \
const int layer_*i*_dw_ifm_num_of_tiles_w = (int)(0.99 + (float)layer_*i*_dw_ifm_width / dw_tile_w); \n \
const int layer_*i*_dw_ifm_num_of_tiles_h = (int)(0.99 + (float)layer_*i*_dw_ifm_height / dw_tile_h); \n \
const int layer_*i*_dw_num_of_tiles_w = (int)(0.99 + (float)layer_*i*_dw_ofm_width / dw_tile_w); \n \
const int layer_*i*_dw_num_of_tiles_h = (int)(0.99 + (float)layer_*i*_dw_ofm_height / dw_tile_h); \n \
//****************************\n"

pw_block = "//****************************\n \
const int layer_*i*_pw_num_fils = *LNF* / alpha;\n \
const int layer_*i*_pw_depth = layer_*i-1*_*PREV*_num_fils;\n \
const int layer_*i*_pw_ifm_height = layer_*i-1*_*PREV*_ofm_height;\n \
const int layer_*i*_pw_ifm_width = layer_*i-1*_*PREV*_ofm_width;\n \
const int layer_*i*_pw_ofm_height = layer_*i*_pw_ifm_height;\n \
const int layer_*i*_pw_ofm_width = layer_*i*_pw_ifm_width;\n \
const int layer_*i*_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_*i*_pw_depth / pw_tile_d);\n \
const int layer_*i*_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_*i*_pw_num_fils / pw_conv_parallelism_out);\n \
const int layer_*i*_pw_num_of_tiles_w = (int)(0.99 + (float)layer_*i*_pw_ofm_width / pw_tile_w); \n \
const int layer_*i*_pw_num_of_tiles_h = (int)(0.99 + (float)layer_*i*_pw_ofm_height / pw_tile_h); \n \
const int layer_*i*_pw_num_of_weight_groups_for_one_pass = layer_*i*_pw_depth * pw_conv_parallelism_out / weights_group_items; \n \
const int layer_*i*_pw_weights_offset = *LWOF*; \n \
const int layer_*i*_activation = *LA*;\n\
//****************************\n"


layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weight_shapes(layers_types)
layers_inputs = utils.read_layers_input_shapes()
layers_outputs = utils.read_layers_output_shapes()
layers_strides = utils.read_layers_strides()
layers_activations = utils.read_layers_activations()


def replace(replacement_dic, block):
    for key, val in replacement_dic.items():
        block = block.replace(key, str(val))

    return block


current_block_indx = 0
cumulative_pw_weights = 0
target_block = ''
with open(out_file, 'w') as f:
    f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef LAYERS_SPECS\n")
    f.write("#define LAYERS_SPECS\n")
    for i in range(len(layers_types)):
        replacement_dic = {}
        replacement_dic['*PREV*'] = layers_types[i-1]
        if layers_types[i] == 'pw':
            replacement_dic['*LWOF*'] = cumulative_pw_weights
            print(i, layers_weights[i].get_size(), layers_inputs[i].height)
            assert layers_weights[i].get_size() % weights_group_items == 0 or layers_outputs[i].height == 1 or cgc.MODEL_NAME != 'mob_v2'
            cumulative_pw_weights += int(
                layers_weights[i].get_size() / weights_group_items)
        if layers_types[i] in ['pw', 's']:
            replacement_dic['*LNF*'] = layers_weights[i].num_of_filters
        if layers_types[i] in ['dw', 's']:
            replacement_dic['*LST*'] = layers_strides[i]
            padding = int(layers_weights[i].width - layers_strides[i])
            if padding % 2 == 0:
                replacement_dic['*LPL*'] = int(padding / 2)
                replacement_dic['*LPR*'] = int(padding / 2)
                replacement_dic['*LPT*'] = int(padding / 2)
                replacement_dic['*LPB*'] = int(padding / 2)
            else:
                replacement_dic['*LPL*'] = 0
                replacement_dic['*LPR*'] = padding
                replacement_dic['*LPT*'] = 0
                replacement_dic['*LPB*'] = padding
            replacement_dic['*LFS*'] = layers_weights[i].width

        replacement_dic['*i*'] = i
        replacement_dic['*i-1*'] = i - 1
        current_layer_activation = ''
        if layers_activations[i] == 'relu6':
            current_layer_activation = '6'
        elif layers_activations[i] == 'sigmoid':
            current_layer_activation = '2'
        else:
            current_layer_activation = '0'
        replacement_dic['*LA*'] = current_layer_activation
        if i == 0:
            target_block = block_0
        elif layers_types[i] == 'pw':
            target_block = pw_block
        else:
            target_block = dw_block

        target_block = replace(replacement_dic, target_block)
        f.write(target_block)
        
        replacement_list = []
        #replacement_dic['*PREV*'] = layers_types[i-1]
        strides = layers_strides[i]
        filter_dim = layers_weights[i].width
        num_of_filters = layers_weights[i].num_of_filters
        replacement_list.append(str(i) + '_' + layers_types[i])
        replacement_list.append('{')
        replacement_list.append(num_of_filters)
        replacement_list.append(strides)
        replacement_list.append(filter_dim)
        padding_left = 0
        padding_right = 0
        padding_top = 0
        padding_bottom = 0
        if layers_types[i] != 'pw':
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

        layer_depth = layers_inputs[i].depth
        layer_height = layers_inputs[i].height
        layer_width = layers_inputs[i].width

        replacement_list.append(layer_depth)
        replacement_list.append(layer_height)
        replacement_list.append(layer_width)

        replacement_list.append( int(layer_height / strides) )
        replacement_list.append( int(layer_width / strides) )

        layer_activation = ''
        if layers_activations[i] == 'relu6':
            layer_activation = '6'
        elif layers_activations[i] == 'sigmoid':
            layer_activation = '2'
        else:
            layer_activation = '0'

        replacement_list.append(layer_activation)

        replacement_list.append('(' + str(layer_depth) + ' + pw_tile_d) / pw_tile_d')
        replacement_list.append('(' + str(num_of_filters) +
                                 ' + pw_conv_parallelism_out) / pw_conv_parallelism_out')

        replacement_list.append('(' + str(layer_height) + ' + pw_tile_h) / pw_tile_h')
        replacement_list.append('(' + str(layer_width) + ' + pw_tile_w) / pw_tile_w')

        replacement_list.append('(' + str( int(layer_height / strides)) + ' + pw_tile_h) / pw_tile_h')
        replacement_list.append('(' + str(int(layer_width / strides)) + ' + pw_tile_w) / pw_tile_w')

        replacement_list.append( str(layer_depth) + ' * pw_conv_parallelism_out / weights_group_items')

        if layers_types[i] != 'dw' and i > 0:
            replacement_list.append(cumulative_pw_weights)
            cumulative_pw_weights += int(layers_weights[i].get_size() / weights_group_items)
        else:
            replacement_list.append(0)

        replacement_list.append('}')

        f.write(specs_block.format(*replacement_list))

    f.write('#endif\n')
