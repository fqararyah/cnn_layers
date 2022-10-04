from code_gen.utils import read_skip_connections_indices
import utils

utils.set_globals('mob_v2', 'mobilenetv2')

in_out_file = '../client/seml.hpp'

expansion_block= 'pw_conv(off_chip_weights, channels, result2, *i*, layer_*i*_pw_depth,\n\
    layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
    layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
    layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
    layer_*i*_pw_num_of_weight_groups_in_depth,\n\
    layer_*i*_pw_normalization, *DIRECTION*, layer_*i*_pw_weights_offset);\n'

dw_block = 'fill_dw_layer_weights(dw_weights_*i*, dw_weights_buffer, layer_*i*_dw_depth, layer_*i*_dw_filter_size, layer_*i*_dw_filter_size);\n\
    dw_conv_3x3(dw_weights_buffer, channels, result2, *i*, layer_*i*_dw_depth,\n\
    layer_*i*_dw_ifm_width, layer_*i*_dw_ifm_height, layer_*i*_dw_num_of_tiles_in_d,\n\
    layer_*i*_dw_num_of_tiles_h, layer_*i*_dw_num_of_tiles_w,\n\
    layer_*i*_dw_strides, layer_*i*_dw_padding_left,\n\
    layer_*i*_dw_normalization, *DIRECTION*);\n'

projection_block ='pw_conv(off_chip_weights, channels, result2, *i*, layer_*i*_pw_depth,\n\
    layer_*i*_pw_num_fils, layer_*i*_pw_num_of_tiles_in_d,\n\
    layer_*i*_pw_num_of_tiles_out_d, layer_*i*_pw_num_of_tiles_h,\n\
    layer_*i*_pw_num_of_tiles_w, tmp_channels, *RW*,\n\
    layer_*i*_pw_num_of_weight_groups_in_depth,\n\
    layer_*i*_pw_normalization, *DIRECTION*, layer_*i*_pw_weights_offset);\n'


layers_types = utils.read_layers_types()
layers_strides = utils.read_layers_strides()
expansion_projection = utils.read_expansion_projection()

def replace(replacement_dic, block):
    for key, val in replacement_dic.items():
        block = block.replace(key, str(val))
    
    return block

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
            if '[' in line and ']' in line:
                layers_to_generate = [int(x) for x in (
                    line.replace(' ', '').split('[')[1].split(']')[0].replace(' ', '').split(':')
                )]  
        elif utils.END_CODE_GENERATION_SIGNAL in line:
            in_a_code_gen_area = False
            file_replacement += line

direction = 1
code_to_insert = ''

skip_connections_indices = utils.read_skip_connections_indices()
for  layer_indx in range(layers_to_generate[0], layers_to_generate[1]):
    target_block = ''
    replacement_dict = {}
    replacement_dict['*i*'] = layer_indx
    replacement_dict['*DIRECTION*'] = direction
    read_write = 0
    if expansion_projection:
        direction = 1 - direction

    if layer_indx % 3 == 1 and expansion_projection[layer_indx]:
        target_block = expansion_block
        if layer_indx + 3 in skip_connections_indices:
            read_write = 2
    elif layer_indx % 3 == 2:
        target_block = dw_block
    elif layer_indx % 3 == 0 and expansion_projection[layer_indx]:
        target_block = projection_block
        if layer_indx + 1 in skip_connections_indices:
            read_write = 1
    
    replacement_dict['*RW*'] = read_write
    code_to_insert += replace(replacement_dict, target_block)

file_replacement = file_replacement[:insert_index] + code_to_insert + file_replacement[insert_index:]

with open(in_out_file, 'w') as f:
    f.write(file_replacement)