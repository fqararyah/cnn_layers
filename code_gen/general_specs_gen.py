import utils
import code_generation_constants as cgc
from class_model_general_specs import general_model_specs

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)
model_dag = utils.read_model_dag()

# './out/layers_specs.h'
out_file = '../model_components/basic_defs/{}_general_specs_pipe_{}.h'

pipeline_len = 0
if cgc.PIPELINE:
    pipeline_len = cgc.PIPELINE_LEN

model_config_file = '../model_config/{}.txt'.format(cgc.MODEL_NAME)

general_model_specs_inst = general_model_specs()

max_fms_size = 0
max_tmp_fms_size = 0
max_dw_layer_d = 0
all_on_chip_pw_s_weights = 0
all_dw_off_chip_weights = 0
all_off_chip_fused_scales_zps = 0
all_off_chip_pw_s_weights = 0

layer_index = 0
num_conv_layers_so_far = 0
first_conv_layer = False
last_pipeline_layer = 0

while num_conv_layers_so_far < pipeline_len or not first_conv_layer:
    layer_specs = model_dag[layer_index]
    layer_type = ''
    if 'type' in layer_specs:
        layer_type = layer_specs['type']
    if layer_type in cgc.CONV_LAYER_TYPES:
        num_conv_layers_so_far += 1
        first_conv_layer = True
        
        layer_weights_shape = layer_specs['weights_shape']
        layer_weights_size = 1
        for i in range(len(layer_weights_shape)):
            layer_weights_size *= layer_weights_shape[i] 
        
        if layer_type != 'dw':
            all_on_chip_pw_s_weights += layer_weights_size
        
        last_pipeline_layer = layer_index
        
    layer_index += 1


layer_specs = model_dag[last_pipeline_layer]
ofms_shape = layer_specs['ofms_shape']
num_filters = ofms_shape[0]
ofms_height = ofms_shape[1]
ofms_width = ofms_shape[2]
ofms_size = num_filters * ofms_height * ofms_width

write_to_tmp = 0
layer_children = layer_specs['children']
if len(layer_children) > 1:
    write_to_tmp = 1
if model_dag[layer_children[0]]['name'] == 'add' and model_dag[layer_children[0]]['id'] == layer_index + 1:
    add_layer_specs = model_dag[layer_children[0]]
    if len(add_layer_specs['children']) > 1:
        write_to_tmp = 1

if write_to_tmp:
    max_tmp_fms_size = ofms_size
    general_model_specs_inst.max_tmp_fms_shape = ofms_shape
         
while layer_index < len(model_dag):
    layer_specs = model_dag[layer_index]
    layer_type = ''
    replacement_list = []
    if 'type' in layer_specs:
        layer_type = layer_specs['type']
    if layer_type in cgc.CONV_LAYER_TYPES:
        ifms_shape = layer_specs['ifms_shape']
        ofms_shape = layer_specs['ofms_shape']
        layer_weights_shape = layer_specs['weights_shape']
        
        ifms_depth = ifms_shape[0]
        ifms_height = ifms_shape[1]
        ifms_width = ifms_shape[2]

        num_filters = ofms_shape[0]
        ofms_height = ofms_shape[1]
        ofms_width = ofms_shape[2]
        ofms_size = num_filters * ofms_height * ofms_width
        
        all_off_chip_fused_scales_zps += num_filters
        strides = layer_specs['strides']
        
        filter_hw_dim = 1        
        padding_left = 0
        padding_right = 0
        padding_top = 0
        padding_bottom = 0
        padding = 0
        if layer_type != 'pw':
            filter_hw_dim = layer_weights_shape[-1]
            padding = int(filter_hw_dim - strides)
            if strides == 1:
                padding_left = int(padding / 2)
                padding_right = int(padding / 2)
                padding_top = int(padding / 2)
                padding_bottom = int(padding / 2)
            else:
                padding_right = padding
                padding_bottom = padding

        if model_dag[layer_index - 1]['name'] == 'pad':
            ifms_width -= padding_right
            ifms_height -= padding_bottom
        
        ifms_size = ifms_depth * ifms_height * ifms_width

        write_to_tmp = 0
        layer_children = layer_specs['children']
        if len(layer_children) > 1:
            write_to_tmp = 1
        if model_dag[layer_children[0]]['name'] == 'add' and model_dag[layer_children[0]]['id'] == layer_index + 1:
            add_layer_specs = model_dag[layer_children[0]]
            if len(add_layer_specs['children']) > 1:
                write_to_tmp = 1
                        
        if max_fms_size <= ifms_size:
            max_fms_size = ifms_size
            general_model_specs_inst.max_fms_shape = [ifms_depth, ifms_height, ifms_width]

        if write_to_tmp == 1 and max_tmp_fms_size <= ofms_size:
            max_tmp_fms_size = ofms_size
            general_model_specs_inst.max_tmp_fms_shape = ofms_shape

        layer_weights_size = 1
        for i in range(len(layer_weights_shape)):
            layer_weights_size *= layer_weights_shape[i] 

        if layer_type == 'dw':
            general_model_specs_inst.max_filter_hw_dim = max(general_model_specs_inst.max_filter_hw_dim, filter_hw_dim)
            general_model_specs_inst.max_dw_layer_d = max(general_model_specs_inst.max_dw_layer_d, ifms_depth)
            all_dw_off_chip_weights += layer_weights_size
            
        else:
            all_off_chip_pw_s_weights += layer_weights_size
            if layer_type == 's':
                general_model_specs_inst.max_std_conv_filter_hw_dim = \
                    max(general_model_specs_inst.max_std_conv_filter_hw_dim, filter_hw_dim)
                
        if strides == 1:
            general_model_specs_inst.max_filter_dim_stride_1 = max(
                general_model_specs_inst.max_filter_dim_stride_1, filter_hw_dim
            )
        elif strides == 2:
            general_model_specs_inst.max_filter_dim_stride_2 = max(
                general_model_specs_inst.max_filter_dim_stride_2, filter_hw_dim
            )
            
        general_model_specs_inst.max_conv_d = max(general_model_specs_inst.max_conv_d, ifms_depth)
        general_model_specs_inst.max_padding_lr = max(
            general_model_specs_inst.max_padding_lr, padding_left + padding_right
        )
    
    elif layer_type == 'fc':
        general_model_specs_inst.fc_layer_input_size = layer_specs['ifms_shape'][-1]
        
    layer_index += 1

general_model_specs_inst.all_on_chip_pw_s_weights = all_on_chip_pw_s_weights
general_model_specs_inst.all_dw_off_chip_weights = all_dw_off_chip_weights
general_model_specs_inst.all_off_chip_fused_scales_zps = all_off_chip_fused_scales_zps
general_model_specs_inst.all_off_chip_pw_s_weights = all_off_chip_pw_s_weights

general_model_specs_inst.write_specs_to_header(out_file.format(cgc.MODEL_NAME, pipeline_len), cgc.MODEL_NAME, pipeline_len)