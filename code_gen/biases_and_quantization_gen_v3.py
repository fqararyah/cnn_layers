from random import uniform
import numpy as np
import utils
import math
import code_generation_constants as cgc
import os

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

from_files = True
#########################################################################
biases_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/biases/'.format(
        cgc.MODEL_NAME)
weights_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(
        cgc.MODEL_NAME)
weights_scales_files_location = weights_files_location
fms_scales_files_location = '/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/fms/'.format(
        cgc.MODEL_NAME)
general_specs_file = '/media/SSD2TB/fareed/wd/cnn_layers/model_components/basic_defs/general_specs.h'

if cgc.PIPELINE:
    off_chip_fused_scales_file = '../off_chip_weights/{}_fused_scales_pipeline_{}.txt'.format(cgc.MODEL_NAME, cgc.PIPELINE_LEN)
    dw_off_chip_fused_zps_file = '../off_chip_weights/{}_fused_zps_pipeline_{}.txt'.format(cgc.MODEL_NAME, cgc.PIPELINE_LEN)
else:
    off_chip_fused_scales_file = '../off_chip_weights/{}_fused_scales.txt'.format(cgc.MODEL_NAME)
    dw_off_chip_fused_zps_file = '../off_chip_weights/{}_fused_zps.txt'.format(cgc.MODEL_NAME)

#########################################################################

#########################################################################
model_dag = utils.read_model_dag()

model_activation = '0'
add_layers_activation = '0'

overall_quantization_arrays_num_of_elements = 0
for layer_specs in model_dag:
    if 'ifms_shape' in layer_specs and len(layer_specs['ifms_shape']) > 0:
        overall_quantization_arrays_num_of_elements += layer_specs['ifms_shape'][0]
print(overall_quantization_arrays_num_of_elements)
first_quantization_arrays_elements_threshold = int(
    overall_quantization_arrays_num_of_elements / 2)
first_quantization_arrays_num_of_elements = 0
#########################################################################

#########################################################################
secondary_layer_fms_scales_files_formats = {}
secondary_layer_fms_zero_points_files_formats = {}


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


weights_file_format = 'weights_{}.txt'
weights_scales_file_format = weights_scales_files_location + \
    'weights_{}_scales.txt'
weights_zero_points_file_format = weights_scales_files_location + \
    'weights_{}_zps.txt'
biases_file_format = biases_files_location + 'biases_{}.txt'
#########################################################################
max_layer_d = 0

# './out/dw_weights.h'
h_file = '../model_components/model/headers/{}_quantization_and_biases_{}.h'.format(cgc.MODEL_NAME,
                                                                                   cgc.FIRST_PART_IMPLEMENTATION)

layers_fused_parameters_offsets = [0] * (len(model_dag) + 1)
pipe_layers_fused_parameters_offsets = [0] * (len(model_dag) + 1)
layers_fused_parameters_offsets_declaration_string = 'const static int layers_fused_parameters_offsets[] = { \n'
pipe_layers_fused_parameters_offsets_declaration_string = 'const static int pipe_layers_fused_parameters_offsets[] = { \n'

weights_zero_points_declaration_string = 'const static fms_dt weights_zero_points[] = {'
last_secondary_type_after_a_conv = {}
with open(h_file, 'w') as wf:
    wf.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    wf.write('#if FIRST_PART_IMPLEMENTATION ==' + str(cgc.FIRST_PART_IMPLEMENTATION) +
             " && MODEL_ID == " + cgc.MODEL_NAME.upper() + "\n")
    wf.write("#ifndef BIAS_QUANT\n")
    wf.write("#define BIAS_QUANT\n")

    # for now, I am getting the average pooling quantization manually from netron
    # wf.write('const pooling_fused_scales_dt pooling_fused_scale = ' +
    #          str(0.0235294122248888 / 0.020379824563860893) + ';\n')
    # assert(0.0235294122248888 / 0.020379824563860893 < 1)
    # assert(0.0235294122248888 / 0.020379824563860893 > 0.1)
    # wf.write('const biases_dt pooling_ifms_zero_point = -128;\n')
    # wf.write('const biases_dt pooling_ofms_zero_point = -128;\n')

    # writing weights scales and zero_points
    seml_fused_zero_points = []
    seml_fused_scales = []
    seml_fused_scales_log_2_shifts = []
    seml_relu_6_fused_scales = [0] * len(model_dag)
    pipe_fused_zero_points = []
    pipe_fused_scales = []
    pipe_fused_scales_log_2_shifts = []
    pipe_relu_6_fused_scales = [0] * len(model_dag)
    to_generate_for_layers = cgc.LAST_LAYER_TO_GENERATE if cgc.LAST_LAYER_TO_GENERATE > 0 else len(
        model_dag)
    num_of_generated_for_layers = 0

    first_conv_layer = True
    for layer_index in range(to_generate_for_layers):
        layers_fused_parameters_offsets[layer_index +
                                        1] = layers_fused_parameters_offsets[layer_index]
        pipe_layers_fused_parameters_offsets[layer_index +
                                             1] = pipe_layers_fused_parameters_offsets[layer_index]
        layer_specs = model_dag[layer_index]
        layer_type = ''

        if 'name' in layer_specs and layer_specs['name'] == 'add':
            if 'activation' in layer_specs and layer_specs['activation'] != '':
                add_layers_activation = layer_specs['activation']

        if 'type' in layer_specs and layer_specs['type'] in cgc.CONV_LAYER_TYPES:
            layer_type = layer_specs['type']
            if 'activation' in layer_specs and layer_specs['activation'] not in ['', '0']:
                model_activation = layer_specs['activation']
        else:
            continue

        fused_zero_points = []
        fused_scales = []
        fused_scales_log_2_shifts = []
        relu_6_fused_scales = []
        biases = []

        weights_file = weights_file_format.format(layer_index)

        weights = np.loadtxt(weights_files_location +
                             weights_file).astype(np.int8)

        if os.path.isfile(biases_file_format.format(layer_index)):
            with open(biases_file_format.format(layer_index), 'r') as f:
                for line in f:
                    bias = line.replace(' ', '').replace('\n', '')
                    assert(int(bias) < 2**31-1)
                    biases.append(int(bias))
        else:
            print(bcolors.WARNING +
                  biases_file_format.format(layer_index) + ' does not exist!!!')

        ifms_zero_point = layer_specs['ifms_zero_points']
        layer_weight_shape = layer_specs['weights_shape']

        if layer_type == 'pw':
            weights = np.reshape(
                weights, (layer_weight_shape[0], layer_weight_shape[1]))
        elif layer_type == 'dw':
            weights = np.reshape(
                weights, (layer_weight_shape[0], layer_weight_shape[1], layer_weight_shape[2]))
        else:
            weights = np.reshape(
                weights, (layer_weight_shape[0], layer_weight_shape[1], layer_weight_shape[2], layer_weight_shape[3]))

        if layer_weight_shape[0] > max_layer_d:
            max_layer_d = layer_weight_shape[0]

        for i in range(layer_weight_shape[0]):
            filter_weights_sum = 0
            if layer_type == 'pw':
                filter_weights_sum = np.sum(weights[i, :])
            elif layer_type == 'dw':
                filter_weights_sum = np.sum(weights[i, :, :])
            else:
                filter_weights_sum = np.sum(weights[i, :, :, :])

            fused_zero_point = filter_weights_sum * -ifms_zero_point \
                + biases[i]
            assert(fused_zero_point < 2 ** 31 -
                   1 and fused_zero_point > - 2**31)
            fused_zero_points.append(fused_zero_point)

        if (cgc.PIPELINE == False or num_of_generated_for_layers >= cgc.PIPELINE_LEN):
            layers_fused_parameters_offsets[layer_index +
                                            1] += layer_weight_shape[0]
        else:
            pipe_layers_fused_parameters_offsets[layer_index +
                                                 1] += layer_weight_shape[0]

        with open(weights_scales_file_format.format(layer_index), 'r') as f:
            ifms_scale = layer_specs['ifms_scales']
            ofms_scale = layer_specs['ofms_scales']

            if utils.NET_PREFIX not in ['eff', 'eff_b0']:
                    relu_6_fused_scale = round(6 / ofms_scale)

            if layer_index > 0 and relu_6_fused_scale > 2**32 - 1:
                        relu_6_fused_scale = 2**32 - 1
            assert(relu_6_fused_scale <= 2**32 - 1 or (layer_index ==
                                                        0 and relu_6_fused_scale <= 2**38 - 1))
            
            assert(relu_6_fused_scale < 256)
            relu_6_fused_scales.append(relu_6_fused_scale)

            for line in f:
                weight_scale = float(line.replace(' ', '').replace('\n', ''))
                ifm_weight_fused_scale = weight_scale * ifms_scale
                assert(ifm_weight_fused_scale < 0.5)
                ofm_ifm_weigh_fused_scale = ifm_weight_fused_scale / \
                    ofms_scale
                fused_scales.append(ofm_ifm_weigh_fused_scale)
                assert(ofm_ifm_weigh_fused_scale <
                       1) or 'mob_v1' in cgc.MODEL_NAME or 'mob_v2_0_' in cgc.MODEL_NAME \
                       or 'uniform' in cgc.MODEL_NAME
                assert(ofm_ifm_weigh_fused_scale > 0)
                current_log = math.log2(fused_scales[-1]) + 1
                abs_current_log_int = abs(int(current_log))
                decomposed_val = fused_scales[-1] / (2 ** -abs_current_log_int)
                assert(abs_current_log_int >= 0 and abs_current_log_int < 32)
                assert(decomposed_val > 0.1 and decomposed_val <
                       1) or 'mob_v1' in cgc.MODEL_NAME or 'mob_v2_0_5' in cgc.MODEL_NAME \
                       or 'uniform' in cgc.MODEL_NAME
                assert(fused_scales[-1] == decomposed_val *
                       (2 ** -abs_current_log_int))
                fused_scales_log_2_shifts.append(abs_current_log_int)
                #fused_scales[-1] = decomposed_val
                
                    # if layer_index == 3:
                    #     print('>>>>>>>>',ofms_scale)

                    # assert(relu_6_fused_scales[-1] > 256) or \
                    #     utils.NET_PREFIX in [
                    #         'mnas', 'prox', 'mob_v1_0_5', 'mob_v2_0_5', 'mob_v2_0_75'] or 'uniform' in utils.NET_PREFIX

        if ((cgc.PIPELINE == True and num_of_generated_for_layers < cgc.PIPELINE_LEN)
                or num_of_generated_for_layers == 0):
            if cgc.FIRST_PART_IMPLEMENTATION == cgc.BOTTLENECK_CHAIN_MODE or num_of_generated_for_layers == 0:
                fused_zero_points_declaration_string = 'const static biases_dt layer_{}_{}_fused_zero_points[] = \n'.format(
                    layer_index, layer_type)  if not first_conv_layer else 'const static biases_dt first_conv_layer_fused_zero_points[] ='
                fused_zero_points_declaration_string += '{ ' + str(
                    fused_zero_points).replace('[', '').replace(']', '') + '};\n'

                fused_scales_declaration_string = 'const static fused_scales_dt layer_{}_{}_fused_scales[] ='.format(
                    layer_index, layer_type) if not first_conv_layer else 'const static fused_scales_dt first_conv_layer_fused_scales[] = '
                fused_scales_declaration_string += '{ ' + str(
                    fused_scales).replace('[', '').replace(']', '') + '};\n'

                fused_scales_log_2_shifts_declaration_string = 'const static fused_scales_log_2_shifts_dt layer_{}_{}_fused_scales_log_2_shifts[] ='.format(
                    layer_index, layer_type)  if not first_conv_layer  else 'const static fused_scales_log_2_shifts_dt first_conv_layer_fused_scales_log_2_shifts[] ='
                fused_scales_log_2_shifts_declaration_string += '{ ' + str(
                    fused_scales_log_2_shifts).replace('[', '').replace(']', '') + '};\n'

                relu_6_fused_scales_declaration_string = 'const static relu_6_fused_scales_dt layer_{}_{}_relu_6_fused_scales[] ='.format(
                    layer_index, layer_type) if not first_conv_layer else 'const static layer_0_relu_6_fused_scales_dt first_conv_layer_relu_6_fused_scales[] ='
                relu_6_fused_scales_declaration_string += '{ ' + str(
                    relu_6_fused_scales).replace('[', '').replace(']', '') + '};\n'
                
                

                wf.write(fused_zero_points_declaration_string)
                wf.write(fused_scales_declaration_string)
                wf.write(fused_scales_log_2_shifts_declaration_string)
                wf.write(relu_6_fused_scales_declaration_string)
        
            elif cgc.FIRST_PART_IMPLEMENTATION == cgc.PIPELINED_ENGINES_MODE and num_of_generated_for_layers < cgc.PIPELINE_LEN:
                pipe_fused_scales.extend(fused_scales)
                #seml_fused_scales_log_2_shifts.extend(fused_scales_log_2_shifts)
                pipe_relu_6_fused_scales[layer_index] = relu_6_fused_scales[0]
                pipe_fused_zero_points.extend(fused_zero_points)    

        else:
            seml_fused_scales.extend(fused_scales)
            #seml_fused_scales_log_2_shifts.extend(fused_scales_log_2_shifts)
            seml_relu_6_fused_scales[layer_index] = relu_6_fused_scales[0]
            seml_fused_zero_points.extend(fused_zero_points)

        num_of_generated_for_layers += 1
        first_conv_layer = False

    print(first_quantization_arrays_num_of_elements)

    wf.write(layers_fused_parameters_offsets_declaration_string +
             str(layers_fused_parameters_offsets).replace('[', '').replace(']', '};\n'))
    wf.write(pipe_layers_fused_parameters_offsets_declaration_string +
             str(pipe_layers_fused_parameters_offsets).replace('[', '').replace(']', '};\n'))
########################################################################################################
    if cgc.FIRST_PART_IMPLEMENTATION == cgc.PIPELINED_ENGINES_MODE and cgc.PIPELINE_LEN > 0:
        pipe_fused_zero_points_declaration_string = 'const static biases_dt pipe_fused_zero_points[] = \n'

        pipe_fused_zero_points_declaration_string += '{ ' + str(
            pipe_fused_zero_points[0:first_quantization_arrays_num_of_elements]).replace('[', '').replace(']', '') + '};\n'

        pipe_fused_scales_declaration_string = 'const static fused_scales_dt pipe_fused_scales[] ='
        pipe_fused_scales_declaration_string += '{ ' + str(
            pipe_fused_scales[0:first_quantization_arrays_num_of_elements]).replace('[', '').replace(']', '') + '};\n'

        pipe_fused_scales_log_2_shifts_declaration_string = 'const static fused_scales_log_2_shifts_dt pipe_fused_scales_log_2_shifts[] ='
        pipe_fused_scales_log_2_shifts_declaration_string += '{ ' + str(
            pipe_fused_scales_log_2_shifts[0:first_quantization_arrays_num_of_elements]).replace('[', '').replace(']', '') + '};\n'

        pipe_relu_6_fused_scales_declaration_string = 'const static relu_6_fused_scales_dt pipe_relu_6_fused_scales[] ='
        pipe_relu_6_fused_scales_declaration_string += '{ ' + str(
            pipe_relu_6_fused_scales[0:first_quantization_arrays_num_of_elements]).replace('[', '').replace(']', '') + '};\n'

        if cgc.LAST_LAYER_TO_GENERATE >= cgc.PIPELINE_LEN or cgc.PIPELINE == False or cgc.LAST_LAYER_TO_GENERATE == -1:
            wf.write(pipe_fused_zero_points_declaration_string)
            wf.write(pipe_fused_scales_declaration_string)
            wf.write(pipe_fused_scales_log_2_shifts_declaration_string)
            wf.write(pipe_relu_6_fused_scales_declaration_string)
########################################################################################################
    seml_fused_zero_points_declaration_string = 'static biases_dt seml_fused_zero_points_buffer[{}];\n'.format(max_layer_d)

    seml_fused_scales_declaration_string = 'static fused_scales_dt seml_fused_scales_buffer[{}];\n'.format(max_layer_d)

    seml_fused_scales_log_2_shifts_declaration_string = 'const static fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[] ='
    seml_fused_scales_log_2_shifts_declaration_string += '{ ' + str(
        seml_fused_scales_log_2_shifts).replace('[', '').replace(']', '') + '};\n'

    seml_relu_6_fused_scales_declaration_string = 'const static relu_6_fused_scales_dt relu_6_fused_scales[] ='
    seml_relu_6_fused_scales_declaration_string += '{ ' + str(
        seml_relu_6_fused_scales).replace('[', '').replace(']', '') + '};\n'

    if cgc.LAST_LAYER_TO_GENERATE >= cgc.PIPELINE_LEN or cgc.PIPELINE == False or cgc.LAST_LAYER_TO_GENERATE == -1:
        wf.write(seml_fused_zero_points_declaration_string)
        wf.write(seml_fused_scales_declaration_string)
        wf.write(seml_fused_scales_log_2_shifts_declaration_string)
        wf.write(seml_relu_6_fused_scales_declaration_string)
########################################################################################################
    wf.write("#endif\n")
    wf.write("#endif\n")

with open(off_chip_fused_scales_file, 'w') as f:
    for scale in seml_fused_scales:
        f.write(str(scale) + '\n')

with open(dw_off_chip_fused_zps_file, 'w') as f:
    for zp in seml_fused_zero_points:
        f.write(str(zp) + '\n')

# for i in range(len(add_layers_fms_scales_rec)):
#     print(i, add_layers_fms_scales[i], add_layers_fms_scales_rec[i])

replacement_string = ''
with open(general_specs_file, 'r') as f:
    for line in f:
        if 'const int all_off_chip_fused_scales_zps =' in line:
            replacement_string += 'const int all_off_chip_fused_scales_zps = {};\n'.format(len(seml_fused_scales))
        elif 'const int first_quantization_arrays_num_elements' in line:
            replacement_string += 'const int first_quantization_arrays_num_elements = ' + \
                str(first_quantization_arrays_num_of_elements) + ';\n'
        elif '#define MODEL_ACTIVATION' in line:
            replacement_string += '#define MODEL_ACTIVATION ' + model_activation + '\n'
        elif '#define ADD_LAYER_ACTIVATION' in line:
            replacement_string += '#define ADD_LAYER_ACTIVATION ' + add_layers_activation + '\n'
        elif 'const int MAX_LAYER_D' in line:
            replacement_string += 'const int MAX_LAYER_D = ' + str(max_layer_d) + ';\n'
        else:
            replacement_string += line

with open(general_specs_file, 'w') as f:
    f.write(replacement_string)
