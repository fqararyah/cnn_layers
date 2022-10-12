import utils

utils.set_globals('mob_v2', 'mobilenetv2')

out_file = '../model/layers_specs.h' #'./out/layers_specs.h'

to_replace = ['*LNF*', '*LD*', '*LW*', '*LH*', '*LST*', '*LPL*', '*LPR*', '*LFS*', '*i*', '*i-1*', '*i-1_pw*', '*LWOF*']

block_0 = "//****************************\n \
const int layer_0_num_fils = *LNF* / alpha;\n\
const int layer_0_depth = input_image_depth;\n\
const int layer_0_ifm_height = input_image_height;\n\
const int layer_0_ifm_width = input_image_width;\n\
const int layer_0_strides = *LST*;\n\
const int layer_0_ofm_height = layer_0_ifm_height / layer_0_strides;\n\
const int layer_0_ofm_width = layer_0_ifm_width / layer_0_strides;\n\
const int layer_0_num_of_tiles_out_d = int(0.99 + ((float) layer_0_num_fils) / pw_conv_parallelism_out);\n\
const int layer_0_padding_left = *LPL*;\n\
const int layer_0_padding_right = *LPR*;\n\
const int layer_0_filter_size = *LFS*;\n \
const int layer_0_num_of_tiles_w = layer_0_ofm_width / pw_tile_w; \n \
const int layer_0_num_of_tiles_h = layer_0_ofm_height / pw_tile_h; \n \
const int layer_0_num_of_tiles_d_in = layer_0_depth / pw_tile_d; \n \
//****************************\n"


expansion_block = "//****************************\n \
const int layer_*i*_pw_num_fils = *LNF* / alpha;\n \
const int layer_*i*_pw_depth = layer_*i-1_pw*_num_fils;\n \
const int layer_*i*_pw_ifm_height = layer_*i-1_pw*_ofm_height;\n \
const int layer_*i*_pw_ifm_width = layer_*i-1_pw*_ofm_width;\n \
const int layer_*i*_pw_ofm_height = layer_*i*_pw_ifm_height;\n \
const int layer_*i*_pw_ofm_width = layer_*i*_pw_ifm_width;\n \
const int layer_*i*_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_*i*_pw_depth / pw_tile_d);\n \
const int layer_*i*_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_*i*_pw_num_fils / pw_conv_parallelism_out);\n \
const int layer_*i*_pw_num_of_tiles_w = (int)(0.99 + (float)layer_*i*_pw_ofm_width / pw_tile_w); \n \
const int layer_*i*_pw_num_of_tiles_h = (int)(0.99 + (float)layer_*i*_pw_ofm_height / pw_tile_h); \n \
const int layer_*i*_pw_num_of_weight_groups_in_depth = layer_*i*_pw_depth / weights_group_items; \n \
const int layer_*i*_pw_weights_offset = *LWOF*; \n \
const int layer_*i*_relu = 6;\n\
//****************************\n"

dw_block = "const int layer_*i*_dw_num_fils = layer_*i-1*_pw_num_fils / alpha;\n \
const int layer_*i*_dw_depth = layer_*i*_dw_num_fils;\n \
const int layer_*i*_dw_strides = *LST*;\n \
const int layer_*i*_dw_ifm_height = layer_*i-1*_pw_ofm_height;\n \
const int layer_*i*_dw_ifm_width = layer_*i-1*_pw_ofm_width;\n \
const int layer_*i*_dw_ofm_height = layer_*i*_dw_ifm_height / layer_*i*_dw_strides;\n \
const int layer_*i*_dw_ofm_width = layer_*i*_dw_ifm_width / layer_*i*_dw_strides;\n \
const int layer_*i*_dw_padding_left = *LPL*;\n \
const int layer_*i*_dw_padding_right = *LPR*;\n \
const int layer_*i*_dw_filter_size = *LFS*;\n \
const int layer_*i*_dw_num_of_tiles_in_d = (int)(((float)layer_*i*_dw_depth / dw_tile_d) + 0.5);\n \
const int layer_*i*_dw_num_of_tiles_w = layer_*i*_dw_ofm_width / dw_tile_w; \n \
const int layer_*i*_dw_num_of_tiles_h = layer_*i*_dw_ofm_height / dw_tile_h; \n \
//****************************\n"

projection_block = "//****************************\n \
const int layer_*i*_pw_num_fils = *LNF* / alpha;\n \
const int layer_*i*_pw_depth = layer_*i-1*_dw_depth;\n \
const int layer_*i*_pw_ifm_height = layer_*i-1*_dw_ofm_height;\n \
const int layer_*i*_pw_ifm_width = layer_*i-1*_dw_ofm_width;\n \
const int layer_*i*_pw_ofm_height = layer_*i*_pw_ifm_height;\n \
const int layer_*i*_pw_ofm_width = layer_*i*_pw_ifm_width;\n \
const int layer_*i*_pw_num_of_tiles_in_d = (int)(0.99 + (float)layer_*i*_pw_depth / pw_tile_d);\n \
const int layer_*i*_pw_num_of_tiles_out_d = (int)(0.99 + (float)layer_*i*_pw_num_fils / pw_conv_parallelism_out);\n \
const int layer_*i*_pw_num_of_tiles_w = (int)(0.99 + (float)layer_*i*_pw_ofm_width / pw_tile_w); \n \
const int layer_*i*_pw_num_of_tiles_h = (int)(0.99 + (float)layer_*i*_pw_ofm_height / pw_tile_h); \n \
const int layer_*i*_pw_num_of_weight_groups_in_depth = layer_*i*_pw_depth / weights_group_items; \n \
const int layer_*i*_pw_weights_offset = *LWOF*; \n \
const int layer_*i*_relu = 0;\n\
//****************************\n"


layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weight_shapes(layers_types)
layers_inputs = utils.read_layers_input_shapes()
layers_outputs = utils.read_layers_output_shapes()
layers_strides = utils.read_layers_strides()
expansion_projection = utils.read_expansion_projection()

def replace(replacement_dic, block):
    for key, val in replacement_dic.items():
        block = block.replace(key, str(val))
    
    return block

current_block_indx = 0
cumulative_pw_weights = 0
target_block = ''
with open(out_file, 'w') as f:
    f.write('#include "../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef LAYERS_SPECS\n")
    f.write("#define LAYERS_SPECS\n")
    for i in range(len(layers_types)):
        replacement_dic = {}
        there_is_expansion_or_projection = expansion_projection[i] != 0
        if layers_types[i] == 'pw':
            replacement_dic['*LWOF*'] = cumulative_pw_weights
            if there_is_expansion_or_projection:
                cumulative_pw_weights += layers_weights[i].get_size()
        if layers_types[i] in ['pw', 'c']:
            replacement_dic['*LNF*'] = layers_weights[i].num_of_filters 
        if layers_types[i] in ['dw', 'c']:
            replacement_dic['*LST*'] = layers_strides[i]
            replacement_dic['*LPL*'] = (layers_weights[i].width - 1) /2
            replacement_dic['*LPR*'] = (layers_weights[i].width - 1) /2 if layers_strides[i] == 1 else 0
            replacement_dic['*LFS*'] = layers_weights[i].width
        
        replacement_dic['*i*'] = i
        replacement_dic['*i-1*'] = i - 1
        if i == 0:
            target_block = block_0
        elif i % 3 == 1:
            target_block = expansion_block
            replacement_dic['*i-1_pw*'] = i - 1 if i == 1 else str(i - 1) + '_pw'
        elif i % 3 == 2:
            target_block = dw_block
        else:
            target_block = projection_block
    
        target_block = replace(replacement_dic, target_block)
        f.write(target_block)
    
    f.write('#endif\n')