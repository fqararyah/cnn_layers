
import utils
import code_generation_constants as cgc
import os


# './out/layers_specs.h'
model_arch_dir = '/media/SSD2TB/fareed/wd/models/codesign/batch1_model_dags/'
out_file = '/media/SSD2TB/fareed/wd/models/codesign/batch1_models_schemes/{}_configs.txt'

for model_file_name in os.listdir(model_arch_dir):
    model_dag = utils.read_model_dag(
        os.path.join(model_arch_dir, model_file_name))

    model_file_name = model_file_name[0:model_file_name.find('.')]
    print(model_file_name)

    model_configs_list = [0] * 2 * len(model_dag)

    model_config_file = out_file.format(model_file_name)
    with open(model_config_file, 'w') as f:

        f.write('#include "../../basic_defs/basic_defs_glue.h"\n')
        f.write("#ifndef LAYERS_SPECS\n")
        f.write("#define LAYERS_SPECS\n")

        for layer_index in range(len(model_dag)):
            layer_specs = model_dag[layer_index]
            layer_type = ''
            replacement_list = []
            if 'type' in layer_specs:
                layer_type = layer_specs['type']
            if layer_type in cgc.CONV_LAYER_TYPES:
                layer_weights_shape = layer_specs['weights_shape']
                num_of_filters = layer_weights_shape[0]

                layer_ifms_shape = layer_specs['ifms_shape']
                layer_depth = layer_ifms_shape[0]

                model_configs_list[2 * layer_index] = layer_depth
                model_configs_list[2 * layer_index + 1] = num_of_filters

    with open(model_config_file, 'w') as f:
        for i in model_configs_list:
            f.write(str(i) + '\n')
