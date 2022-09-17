import utils

utils.set_globals('mob_v2', 'mobilenetv2')

out_file = './out/layers_specs.h'

to_replace = ['*LNF*', '*LD*', '*LW*', '*LH*', '*LST*', '*LPL*', '*LPR*', '*LFS*', '*i*', '*i-1*', '*i-1_pw*', '*LWOF*']

block_0 = "//****************************\n \
const int layer_0_num_fils = *LNF* / alpha;\n\
const int layer_0_depth = input_image_depth;\n\
const int layer_0_ifm_height = input_image_height;\n\
const int layer_0_ifm_width = input_image_width;\n\
const int layer_0_strides = *LST*;\n\
const int layer_0_ofm_height = layer_0_ifm_height / layer_0_strides;\n\
const int layer_0_ofm_width = layer_0_ifm_width / layer_0_strides;\n\
const int layer_0_padding_left = *LPL*;\n\
const int layer_0_padding_right = *LPR*;\n\
const int layer_0_filter_size = *LFS*;\n \
const int layer_0_num_of_tiles_w = layer_0_ofm_width / pw_tile_w; \n \
const int layer_0_num_of_tiles_h = layer_0_ofm_height / pw_tile_h; \n \
const int layer_0_num_of_tiles_d_in = layer_0_depth / pw_tile_d; \n \
const normalization_scheme layers_0_normalization = {0.0, 1.0}; \n \
//****************************\n"
#first and second, fourth and fifth, seventh and eigth, ... 3n + 1, 3n + 2
block_1_2 = "//****************************\n \
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
const normalization_scheme layer_*i*_pw_normalization = {0.0, 1.0}; \n \
const int layer_*i*_dw_num_fils = layer_*i*_pw_num_fils / alpha;\n \
const int layer_*i*_dw_depth = layer_*i*_dw_num_fils;\n \
const int layer_*i*_dw_strides = *LST*;\n \
const int layer_*i*_dw_ifm_height = layer_*i*_pw_ofm_height;\n \
const int layer_*i*_dw_ifm_width = layer_*i*_pw_ofm_width;\n \
const int layer_*i*_dw_ofm_height = layer_*i*_dw_ifm_height / layer_*i*_dw_strides;\n \
const int layer_*i*_dw_ofm_width = layer_*i*_dw_ifm_width / layer_*i*_dw_strides;\n \
const int layer_*i*_dw_padding_left = *LPL*;\n \
const int layer_*i*_dw_padding_right = *LPR*;\n \
const int layer_*i*_dw_filter_size = *LFS*;\n \
const int layer_*i*_dw_num_of_tiles_in_d = (int)(((float)layer_*i*_dw_depth / dw_tile_d) + 0.5);\n \
const int layer_*i*_dw_num_of_tiles_w = layer_*i*_dw_ofm_width / dw_tile_w; \n \
const int layer_*i*_dw_num_of_tiles_h = layer_*i*_dw_ofm_height / dw_tile_h; \n \
const normalization_scheme layer_*i*_dw_normalization = {0.0, 1.0}; \n \
//****************************\n"

#third, sixth, ninth, ... 3n
block_3 = "//****************************\n \
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
const normalization_scheme layer_*i*_pw_normalization = {0.0, 1.0}; \n \
const int layer_*i*_pw_weights_offset = *LWOF*; \n \
//****************************\n"


layers_types = utils.read_layers_types()
layers_weights = utils.read_layers_weights(layers_types)
layers_inputs = utils.read_layers_inputs()
layers_outputs = utils.read_layers_outputs()
layers_strides = utils.read_layers_strides()
expansion_projection = utils.read_expansion_projection()

def replace(replacement_dic, block):
    for key, val in replacement_dic.items():
        block = block.replace(key, str(val))
    
    return block

target_block = ''
current_block_indx = 0
cumulative_pw_weights = 0
with open(out_file, 'w') as f:
    f.write('#include "../basic_defs/basic_defs_glue.h"\n')
    f.write("#ifndef LAYERS_SPECS\n")
    f.write("#define LAYERS_SPECS\n")
    for i in range(len(layers_types)):
        replacement_dic = {}
        if layers_types[i] == 'pw':
            replacement_dic['*LWOF*'] = cumulative_pw_weights
            if expansion_projection[i] != 0:
                cumulative_pw_weights += layers_weights[i].get_size()
        if layers_types[i] in ['pw', 'c']:
            replacement_dic['*LNF*'] = layers_weights[i].num_of_filters 
        if layers_types[i] in ['dw', 'c']:
            replacement_dic['*LST*'] = layers_strides[i]
            replacement_dic['*LPL*'] = (layers_weights[i].width - 1) /2
            replacement_dic['*LPR*'] = (layers_weights[i].width - 1) /2 if layers_strides[i] == 1 else 0
            replacement_dic['*LFS*'] = layers_weights[i].width
        
        if i == 0:
            target_block = block_0
            current_block_indx += 1
        elif i % 3 == 1:
            target_block = block_1_2
        elif i % 3 == 2:
            replacement_dic['*i*'] = current_block_indx
            replacement_dic['*i-1*'] = current_block_indx - 1
            replacement_dic['*i-1_pw*'] = current_block_indx - 1 if i == 2 else str(current_block_indx - 1) + '_pw' 
            current_block_indx += 1
        else:
            target_block = block_3
            replacement_dic['*i*'] = current_block_indx
            replacement_dic['*i-1*'] = current_block_indx - 1
            current_block_indx += 1

        target_block = replace(replacement_dic, target_block)
        if i % 3 != 1:
            f.write(target_block)
    
    f.write('#endif\n')