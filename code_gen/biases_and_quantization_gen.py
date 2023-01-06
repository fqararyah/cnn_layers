import numpy as np
import utils
import math
import code_generation_constants as cgc
import os

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

scales_bit_width = 18
scales_integer_part_width = 1
biases_bit_width = 32
from_files = True
#########################################################################
biases_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/biases/'.format(
        cgc.MODEL_NAME)
weights_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(
        cgc.MODEL_NAME)
weights_scales_files_location = weights_files_location
fms_scales_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/fms/'.format(
        cgc.MODEL_NAME)
#########################################################################

#########################################################################
layers_types = utils.read_layers_types()
secondary_layers_types = utils.read_secondary_layers_types()
layers_weights_shapes = utils.read_layers_weight_shapes(layers_types)
#########################################################################

#########################################################################
conv_fms_scales_file_format = 'fms_conv2d_{}_scales.txt'
conv_fms_zero_points_file_format = 'fms_conv2d_{}_zero_points.txt'

secondary_layer_fms_scales_files_formats = {}
secondary_layer_fms_zero_points_files_formats = {}

all_fms_scales_and_zps_directory_files = os.listdir(fms_scales_files_location)
for secondary_layer_type in secondary_layers_types:
    secondary_layer_fms_scales_files_formats[secondary_layer_type] = 'fms_conv2d_{}_' + \
        secondary_layer_type+'_{}_scales.txt'
    secondary_layer_fms_zero_points_files_formats[secondary_layer_type] = 'fms_conv2d_{}_' + \
        secondary_layer_type+'_{}_zero_points.txt'

all_possible_conv_fms_scales_files = []
all_possible_conv_fms_zps_files = []
all_possible_secondary_fms_scales_files = []
all_possible_secondary_fms_zps_files = []
for i in range(len(all_fms_scales_and_zps_directory_files)):
    all_possible_conv_fms_scales_files.append(
        conv_fms_scales_file_format.format(i))
    all_possible_conv_fms_zps_files.append(
        conv_fms_zero_points_file_format.format(i))

for i in range(len(all_fms_scales_and_zps_directory_files)):
    all_possible_secondary_fms_scales_files.append({})
    all_possible_secondary_fms_zps_files.append({})
    for key, val in secondary_layer_fms_scales_files_formats.items():
        all_possible_secondary_fms_scales_files[-1][key] = []
        for j in range(5):
            all_possible_secondary_fms_scales_files[-1][key].append(
                val.format(i, j))
    for key, val in secondary_layer_fms_zero_points_files_formats.items():
        all_possible_secondary_fms_zps_files[-1][key] = []
        for j in range(5):
            all_possible_secondary_fms_zps_files[-1][key].append(
                val.format(i, j))

weights_file_formats = ['conv2d_{}_s_weights.txt', 'conv2d_{}_pw_weights.txt', 'conv2d_{}_dw_weights.txt']
weights_scales_file_format = weights_scales_files_location + \
    'conv2d_{}_scales.txt'
weights_zero_points_file_format = weights_scales_files_location + \
    'conv2d_{}_zero_points.txt'
biases_file_format = biases_files_location + 'conv2d_{}_biases.txt'
#########################################################################


# './out/dw_weights.h'
h_file = '../model_components/model/headers/quantization_and_biases.h'

layers_fused_parameters_offsets = [0]
layers_fused_parameters_offsets_declaration_string = 'const static int layers_fused_parameters_offsets[] = { \n'

conv_fms_scales_rec = []
conv_fms_scales = []
conv_fms_zero_points = []
conv_fms_scales_declaration_string = 'const static scales_dt conv_fms_scales[] = {'
conv_fms_scales_rec_declaration_string = 'const static rec_scales_dt conv_fms_scales_rec[] = {'
conv_fms_zero_pointsdeclaration_string = 'const static fms_dt conv_fms_zero_points[] = {'

secondary_layers_fms_scales_rec = {}
secondary_layers_fms_scales = {}
secondary_layers_fms_zero_points = {}

secondary_layers_declaration_strings = {}
secondary_layers_fms_scales_rec_declaration_string = 'const static rec_scales_dt {}_layers_fms_scales_rec[] ='
secondary_layers_fms_scales_declaration_string = 'const static scales_dt {}_layers_fms_scales[] ='
secondary_layers_fms_zero_points_declaration_string = 'const static fms_dt {}_layers_fms_zero_points[] ='

for secondary_layer_type in secondary_layers_types:
    secondary_layers_fms_scales_rec[secondary_layer_type] = []
    secondary_layers_fms_scales[secondary_layer_type] = []
    secondary_layers_fms_zero_points[secondary_layer_type] = []

    secondary_layers_declaration_strings[secondary_layer_type] = {}
    secondary_layers_declaration_strings[secondary_layer_type]['scale_rec'] = \
        secondary_layers_fms_scales_rec_declaration_string.format(
            secondary_layer_type) + ' {'
    secondary_layers_declaration_strings[secondary_layer_type]['scale'] = \
        secondary_layers_fms_scales_declaration_string.format(
            secondary_layer_type) + ' {'
    secondary_layers_declaration_strings[secondary_layer_type]['zp'] = \
        secondary_layers_fms_zero_points_declaration_string.format(
            secondary_layer_type) + ' {'

weights_zero_points_declaration_string = 'const static fms_dt weights_zero_points[] = {'
last_secondary_type_after_a_conv = {}
with open(h_file, 'w') as wf:
    wf.write('#include "../../basic_defs/basic_defs_glue.h"\n')
    wf.write("#ifndef BIAS_QUANT\n")
    wf.write("#define BIAS_QUANT\n")

    # for now, I am getting the average pooling quantization manually from netron
    wf.write('const pooling_fused_scales_dt pooling_fused_scale = ' +
             str(0.0235294122248888 / 0.020379824563860893) + ';\n')
    #assert(0.0235294122248888 / 0.020379824563860893 < 1)
    #assert(0.0235294122248888 / 0.020379824563860893 > 0.1)
    wf.write('const biases_dt pooling_ifms_zero_point = -128;\n')
    wf.write('const biases_dt pooling_ofms_zero_point = -128;\n')

    # writing fms scales and zero_points
    for fms_file_index in range(len(all_possible_conv_fms_scales_files)):
        scales_file_path = fms_scales_files_location + \
            all_possible_conv_fms_scales_files[fms_file_index]
        zps_file_path = fms_scales_files_location + \
            all_possible_conv_fms_zps_files[fms_file_index]
        if os.path.exists(scales_file_path):
            with open(scales_file_path, 'r') as f:
                scale = f.readline().replace(' ', '').replace('\n', '')
            with open(zps_file_path, 'r') as f:
                zero_point = f.readline().replace(' ', '').replace('\n', '')

            conv_fms_scales_rec.append(1.0/float(scale))
            assert(1.0/float(scale) < 255)
            conv_fms_scales.append(float(scale))
            assert(float(scale) < 1 and conv_fms_scales[-1] > 0.001)
            conv_fms_zero_points.append(int(zero_point))

            for scales in secondary_layers_fms_scales.values():
                scales.append(0)
            for scales in secondary_layers_fms_zero_points.values():
                scales.append(0)
            for scales in secondary_layers_fms_scales_rec.values():
                scales.append(0)

    for fms_file_index in range(len(all_possible_secondary_fms_scales_files)):
        for secondary_layer_type, files_names in all_possible_secondary_fms_scales_files[fms_file_index].items():
            for internal_file_index in range(len(files_names)):
                scales_file_path = fms_scales_files_location + \
                    files_names[internal_file_index]
                if os.path.exists(scales_file_path):
                    zps_file_path = fms_scales_files_location + \
                        all_possible_secondary_fms_zps_files[fms_file_index][secondary_layer_type][internal_file_index]
                    with open(scales_file_path, 'r') as f:
                        scale = f.readline().replace(' ', '').replace('\n', '')
                    with open(zps_file_path, 'r') as f:
                        zero_point = f.readline().replace(' ', '').replace('\n', '')
                    secondary_layers_fms_scales_rec[secondary_layer_type][fms_file_index] = (
                        1.0/float(scale))
                    assert(1.0/float(scale) < 255)
                    secondary_layers_fms_scales[secondary_layer_type][fms_file_index] = (
                        float(scale))
                    assert(float(scale) < 1 and conv_fms_scales[-1] > 0.001)
                    secondary_layers_fms_zero_points[secondary_layer_type][fms_file_index] = (
                        int(zero_point))
                    last_secondary_type_after_a_conv[fms_file_index] = secondary_layer_type

    for secondary_layer_type, declaration_strings in secondary_layers_declaration_strings.items():
        wf.write(declaration_strings['scale_rec'] + str(
            secondary_layers_fms_scales_rec[secondary_layer_type]).replace('[', '').replace(']', '};\n'))
        wf.write(declaration_strings['scale'] + str(
            secondary_layers_fms_scales[secondary_layer_type]).replace('[', '').replace(']', '};\n'))
        wf.write(declaration_strings['zp'] + str(
            secondary_layers_fms_zero_points[secondary_layer_type]).replace('[', '').replace(']', '};\n'))

    wf.write(conv_fms_zero_pointsdeclaration_string +
             str(conv_fms_zero_points).replace('[', '').replace(']', '};\n'))
    wf.write(conv_fms_scales_declaration_string +
             str(conv_fms_scales).replace('[', '').replace(']', '};\n'))
    wf.write(conv_fms_scales_rec_declaration_string +
             str(conv_fms_scales_rec).replace('[', '').replace(']', '};\n'))

    #writing weights scales and zero_points
    seml_fused_zero_points = []
    seml_fused_scales = []
    seml_fused_scales_log_2_shifts = []
    seml_relu_6_fused_scales = []
    for layer_index in range(len(layers_weights_shapes)):
        fused_zero_points = []
        fused_scales = []
        fused_scales_log_2_shifts = []
        relu_6_fused_scales = []
        biases = []

        weights_file = ''
        for format in weights_file_formats:
            weights_file = format.format(
                layer_index, layers_types[layer_index])
            if os.path.exists(weights_files_location + weights_file):
                break
            
        weights = np.loadtxt(weights_files_location + weights_file).astype(np.int8)

        with open(biases_file_format.format(layer_index), 'r') as f:
            for line in f:
                bias = line.replace(' ', '').replace('\n', '')
                assert(int(bias) < 2**31-1)
                biases.append(int(bias))

        ifms_zero_point = conv_fms_zero_points[layer_index]
        if layer_index in last_secondary_type_after_a_conv:
            ifms_zero_point = secondary_layers_fms_zero_points[last_secondary_type_after_a_conv[layer_index]][layer_index]
            #print(layer_index, ifms_zero_point)
        weights = np.reshape(weights,
                             (layers_weights_shapes[layer_index].num_of_filters, layers_weights_shapes[layer_index].depth,
                              layers_weights_shapes[layer_index].height, layers_weights_shapes[layer_index].width))
        for i in range(layers_weights_shapes[layer_index].num_of_filters):
            fused_zero_point = np.sum(weights[i, :, :, :]) * -ifms_zero_point \
                + biases[i]
            assert(fused_zero_point < 2 ** 31 -
                   1 and fused_zero_point > - 2**31)
            fused_zero_points.append(fused_zero_point)
            
        if (cgc.PIPELINE == True and layer_index < cgc.PILELINE_LEN) or layer_index == 0:
            layers_fused_parameters_offsets.append(0)
        else:
            layers_fused_parameters_offsets.append(layers_weights_shapes[layer_index].num_of_filters +
                                                   layers_fused_parameters_offsets[-1])

        with open(weights_scales_file_format.format(layer_index), 'r') as f:
            ifms_scale = conv_fms_scales[layer_index]
            if layer_index in last_secondary_type_after_a_conv:
                ifms_scale = secondary_layers_fms_scales[last_secondary_type_after_a_conv[layer_index]][layer_index]
                #print(layer_index, ifms_scale)
            for line in f:
                weight_scale = float(line.replace(' ', '').replace('\n', ''))
                ifm_weight_fused_scale = weight_scale * ifms_scale
                assert(ifm_weight_fused_scale < 0.02)
                ofm_ifm_weigh_fused_scale = ifm_weight_fused_scale / \
                    conv_fms_scales[layer_index + 1 if layer_index > 0 else 2]
                fused_scales.append(ofm_ifm_weigh_fused_scale)
                assert(ofm_ifm_weigh_fused_scale < 1)
                assert(ofm_ifm_weigh_fused_scale > 0)
                current_log = math.log2(fused_scales[-1]) + 1
                abs_current_log_int = abs(int(current_log))
                decomposed_val = fused_scales[-1] / (2 ** -abs_current_log_int)
                assert(abs_current_log_int >= 0 and abs_current_log_int < 32)
                assert(decomposed_val > 0.1 and decomposed_val < 1)
                assert(fused_scales[-1] == decomposed_val *
                       (2 ** -abs_current_log_int))
                fused_scales_log_2_shifts.append(abs_current_log_int)
                fused_scales[-1] = decomposed_val
                relu_6_fused_scale = round(6 / ifm_weight_fused_scale)
                assert(relu_6_fused_scale < 2**32 - 1 or (layer_index ==
                       0 and relu_6_fused_scale < 2**38 - 1))
                relu_6_fused_scales.append(relu_6_fused_scale)

                assert(relu_6_fused_scales[-1] > 256)

        if cgc.PIPELINE == True and layer_index < cgc.PILELINE_LEN:
            fused_zero_points_declaration_string = 'const static biases_dt layer_{}_{}_fused_zero_points[] = \n'.format(
                layer_index, layers_types[layer_index])
            fused_zero_points_declaration_string += '{ ' + str(
                fused_zero_points).replace('[', '').replace(']', '') + '};\n'

            fused_scales_declaration_string = 'const static fused_scales_dt layer_{}_{}_fused_scales[] ='.format(
                layer_index, layers_types[layer_index])
            fused_scales_declaration_string += '{ ' + str(
                fused_scales).replace('[', '').replace(']', '') + '};\n'

            fused_scales_log_2_shifts_declaration_string = 'const static fused_scales_log_2_shifts_dt layer_{}_{}_fused_scales_log_2_shifts[] ='.format(
                layer_index, layers_types[layer_index])
            fused_scales_log_2_shifts_declaration_string += '{ ' + str(
                fused_scales_log_2_shifts).replace('[', '').replace(']', '') + '};\n'

            relu_6_fused_scales_declaration_string = 'const static relu_6_fused_scales_dt layer_{}_{}_relu_6_fused_scales[] ='.format(
                layer_index, layers_types[layer_index]) if layer_index != 0 else 'const static layer_0_relu_6_fused_scales_dt layer_0_s_relu_6_fused_scales[] ='
            relu_6_fused_scales_declaration_string += '{ ' + str(
                relu_6_fused_scales).replace('[', '').replace(']', '') + '};\n'

            wf.write(fused_zero_points_declaration_string)
            wf.write(fused_scales_declaration_string)
            wf.write(fused_scales_log_2_shifts_declaration_string)
            wf.write(relu_6_fused_scales_declaration_string)
        else:
            seml_fused_scales.append(fused_scales)
            seml_fused_scales_log_2_shifts.append(fused_scales_log_2_shifts)
            seml_relu_6_fused_scales.append(relu_6_fused_scales)
            seml_fused_zero_points.append(fused_zero_points)

    wf.write(layers_fused_parameters_offsets_declaration_string +
             str(layers_fused_parameters_offsets).replace('[', '').replace(']', '};\n'))

    seml_fused_zero_points_declaration_string = 'const static biases_dt fused_zero_points[] = \n'.format(
        layer_index)
    seml_fused_zero_points_declaration_string += '{ ' + str(
        seml_fused_zero_points).replace('[', '').replace(']', '') + '};\n'

    seml_fused_scales_declaration_string = 'const static fused_scales_dt fused_scales[] ='.format(
        layer_index)
    seml_fused_scales_declaration_string += '{ ' + str(
        seml_fused_scales).replace('[', '').replace(']', '') + '};\n'

    seml_fused_scales_log_2_shifts_declaration_string = 'const static fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[] ='.format(
        layer_index)
    seml_fused_scales_log_2_shifts_declaration_string += '{ ' + str(
        seml_fused_scales_log_2_shifts).replace('[', '').replace(']', '') + '};\n'

    seml_relu_6_fused_scales_declaration_string = 'const static relu_6_fused_scales_dt relu_6_fused_scales[] ='.format(
        layer_index) if layer_index != 0 else 'const static layer_0_relu_6_fused_scales_dt layer_0_relu_6_fused_scales[] ='
    seml_relu_6_fused_scales_declaration_string += '{ ' + str(
        seml_relu_6_fused_scales).replace('[', '').replace(']', '') + '};\n'

    if cgc.LAST_LAYER_TO_GENERATE >= cgc.PILELINE_LEN or cgc.PIPELINE == False:
        wf.write(seml_fused_zero_points_declaration_string)
        wf.write(seml_fused_scales_declaration_string)
        wf.write(seml_fused_scales_log_2_shifts_declaration_string)
        wf.write(seml_relu_6_fused_scales_declaration_string)

    wf.write("#endif\n")


# for i in range(len(add_layers_fms_scales_rec)):
#     print(i, add_layers_fms_scales[i], add_layers_fms_scales_rec[i])
