import numpy as np
import utils
import math
import code_generation_constants as cgc

utils.set_globals(cgc.MODEL_NAME, cgc.MODEL_NAME)

scales_bit_width = 18
scales_integer_part_width = 1
biases_bit_width = 32

from_files = True
weights_scales_biases_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/weights/'.format(cgc.MODEL_NAME)
fms_scales_files_location = '/media/SSD2TB/wd/my_repos/DL_Benchmarking/' + \
    'tflite_scripts_imgnt_accuracy_and_weight_extraction/{}/fms/'.format(cgc.MODEL_NAME)

fms_scales_file_format = fms_scales_files_location + 'fms_{}_scales.txt'
fms_zero_points_file_format = fms_scales_files_location + 'fms_{}_zero_points.txt'
weights_scales_file_format = weights_scales_biases_files_location + \
    'weights_{}_scales.txt'
weights_zero_points_file_format = weights_scales_biases_files_location + \
    'weights_{}_zero_points.txt'
biases_file_format = weights_scales_biases_files_location + 'weights_{}_biases.txt'

layers_types = utils.read_layers_types()
layers_weights_shapes = utils.read_layers_weight_shapes(layers_types)

# './out/dw_weights.h'
h_file = '../model_components/model/headers/quantization_and_biases.h'

skip_connections_indices = utils.read_skip_connections_indices()

conv_fms_scales_rec = []
conv_fms_scales = []
conv_fms_scales_declaration_string = 'const static scales_dt conv_fms_scales[] = {'
conv_fms_scales_rec_declaration_string = 'const static rec_scales_dt conv_fms_scales_rec[] = {'
add_layers_fms_scales_rec = [0] * len(layers_types)
add_layers_fms_scales = [0] * len(layers_types)
add_layers_fms_scales_rec_declaration_string = 'const static rec_scales_dt add_layers_fms_scales_rec[] = {'
add_layers_fms_scales_declaration_string = 'const static scales_dt add_layers_fms_scales[] = {'
conv_fms_zero_points = []

layers_fused_parameters_offsets = []
conv_fms_zero_pointsdeclaration_string = 'const static fms_dt conv_fms_zero_points[] = {'
add_layers_fms_zero_points = [0] * len(layers_types)
add_layers_fms_zero_points_declaration_string = 'const static fms_dt add_layers_fms_zero_points[] = {'
layers_fused_parameters_offsets_declaration_string = 'const static int layers_fused_parameters_offsets[] = {0, \n'

weights_zero_points_declaration_string = 'const static fms_dt weights_zero_points[] = {'

overall_fms_scales = len(skip_connections_indices) + \
    len(layers_weights_shapes) + 1  # n layers -> n+ 1 fms files
expansion_projection = utils.read_expansion_projection()
skip_connection_current_index = 0
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

    fms_file_index = 1
    # writing fms scales and zero_points
    for layer_index in range(overall_fms_scales):
        if layer_index < overall_fms_scales - 1 and layers_types[layer_index - skip_connection_current_index] == 'pw' \
                and expansion_projection[layer_index - skip_connection_current_index] == 0:
            conv_fms_scales_rec.append(0)
            conv_fms_scales.append(0)
            conv_fms_zero_points.append(0)
            continue

        with open(fms_scales_file_format.format(fms_file_index), 'r') as f:
            scale = f.readline().replace(' ', '').replace('\n', '')
        with open(fms_zero_points_file_format.format(fms_file_index), 'r') as f:
            zero_point = f.readline().replace(' ', '').replace('\n', '')

        if layer_index - skip_connection_current_index - 1 in skip_connections_indices \
                and skip_connections_indices[layer_index - skip_connection_current_index - 1] == 1:
            # the first -1 is because ofm of a layer is stored in file fms_layer_index+1
            # if layer_index == 22:
            #     print(scale)
            #     print(zero_point)
            skip_connections_indices[layer_index -
                                     skip_connection_current_index - 1] = 0
            add_layers_fms_scales_rec[layer_index -
                                      skip_connection_current_index - 1] = 1 / float(scale)
            assert(add_layers_fms_scales_rec[layer_index -
                                             skip_connection_current_index - 1] < 255)
            add_layers_fms_scales[layer_index -
                                  skip_connection_current_index - 1] = float(scale)
            assert(add_layers_fms_scales[layer_index -
                                         skip_connection_current_index - 1] < 1 and \
                                             add_layers_fms_scales[layer_index -
                                         skip_connection_current_index - 1] > 0.001)
            add_layers_fms_zero_points[layer_index -
                                       skip_connection_current_index - 1] = int(zero_point)

            skip_connection_current_index += 1
        else:
            conv_fms_scales_rec.append(1.0/float(scale))
            assert(conv_fms_scales_rec[-1] < 255)
            conv_fms_scales.append(float(scale))
            assert(conv_fms_scales[-1] < 1)
            assert(conv_fms_scales[-1] > 0.001)
            conv_fms_zero_points.append(int(zero_point))

        fms_file_index += 1

    wf.write(add_layers_fms_scales_rec_declaration_string +
             str(add_layers_fms_scales_rec).replace('[', '').replace(']', '};\n'))
    wf.write(add_layers_fms_scales_declaration_string +
             str(add_layers_fms_scales).replace('[', '').replace(']', '};\n'))
    wf.write(add_layers_fms_zero_points_declaration_string +
             str(add_layers_fms_zero_points).replace('[', '').replace(']', '};\n'))

    wf.write(conv_fms_zero_pointsdeclaration_string +
             str(conv_fms_zero_points).replace('[', '').replace(']', '};\n'))
    wf.write(conv_fms_scales_declaration_string +
             str(conv_fms_scales).replace('[', '').replace(']', '};\n'))
    wf.write(conv_fms_scales_rec_declaration_string +
             str(conv_fms_scales_rec).replace('[', '').replace(']', '};\n'))
    # writing weights scales and zero_points
    for layer_index in range(len(layers_weights_shapes)):
        fused_zero_points = []
        fused_scales = []
        fused_scales_log_2_shifts = []
        relu_6_fused_scales = []
        biases = []
        if layers_types[layer_index] == 'pw' and expansion_projection[layer_index] == 0:
            layers_fused_parameters_offsets.append(
                layers_fused_parameters_offsets[-1])
            continue

        weights_file = weights_scales_biases_files_location + \
            'weights_{}_{}.txt'.format(layer_index, layers_types[layer_index])
        weights = np.loadtxt(weights_file).astype(np.int8)

        with open(biases_file_format.format(layer_index), 'r') as f:
            for line in f:
                bias = line.replace(' ', '').replace('\n', '')
                assert(int(bias) < 2**31-1)
                biases.append(int(bias))

        weights = np.reshape(weights,
                             (layers_weights_shapes[layer_index].num_of_filters, layers_weights_shapes[layer_index].depth,
                              layers_weights_shapes[layer_index].height, layers_weights_shapes[layer_index].width))
        for i in range(layers_weights_shapes[layer_index].num_of_filters):
            # if layer_index == 2:
            #     print(weights[i, :, :, :])
            #     print('*****')
            fused_zero_point = np.sum(weights[i, :, :, :]) * \
                                     (-conv_fms_zero_points[layer_index] if layer_index not in skip_connections_indices
                                      else -add_layers_fms_zero_points[layer_index]) \
                                     + biases[i]
            assert(fused_zero_point < 2** 31 -1 and fused_zero_point > - 2**31)
            fused_zero_points.append(fused_zero_point)
            # if fused_zero_points[-1] < -2**29 or fused_zero_points[-1] >= 2**29:
            #     print(layer_index, 'XXXXXXXXXXXXXXXXXXXX', fused_zero_points[-1])
            # if layer_index == 3:
            #     print(np.sum(weights[i,:,:,:]) * -conv_fms_zero_points[layer_index] , biases[i])
        if len(layers_fused_parameters_offsets) == 0:
            layers_fused_parameters_offsets.append(
                layers_weights_shapes[layer_index].num_of_filters)
        else:
            layers_fused_parameters_offsets.append(layers_weights_shapes[layer_index].num_of_filters +
                                                   layers_fused_parameters_offsets[-1])

        with open(weights_scales_file_format.format(layer_index), 'r') as f:
            for line in f:
                weight_scale = float(line.replace(' ', '').replace('\n', ''))
                ifm_weight_fused_scale = weight_scale * \
                    (conv_fms_scales[layer_index]
                     if layer_index not in skip_connections_indices else add_layers_fms_scales[layer_index])
                assert(ifm_weight_fused_scale < 0.02)
                ofm_ifm_weigh_fused_scale = ifm_weight_fused_scale / conv_fms_scales[layer_index + 1 if layer_index > 0 else 2]
                fused_scales.append(ofm_ifm_weigh_fused_scale)
                assert(ofm_ifm_weigh_fused_scale < 1)
                assert(ofm_ifm_weigh_fused_scale > 0)
                # assert(fused_scales[-1] <= 1)
                # if layer_index == 0:
                current_log = math.log2(fused_scales[-1]) + 1
                abs_current_log_int = abs(int(current_log))
                decomposed_val = fused_scales[-1] / (2 ** -abs_current_log_int)
                # #print(fused_scales[-1], decomposed_val * (2 ** -abs_current_log_int))
                assert(abs_current_log_int >= 0 and abs_current_log_int < 32)
                assert(decomposed_val > 0.1 and decomposed_val < 1)
                assert(fused_scales[-1] == decomposed_val * (2 ** -abs_current_log_int))
                fused_scales_log_2_shifts.append(abs_current_log_int)
                fused_scales[-1] = decomposed_val
                # if(6 / ifm_weight_fused_scale > 2**31 -1):
                #     print(layer_index, 6 / ifm_weight_fused_scale)
                relu_6_fused_scale = round(6 / ifm_weight_fused_scale)
                assert(relu_6_fused_scale < 2**32 -1 or (layer_index == 0 and relu_6_fused_scale < 2**38 -1) )
                relu_6_fused_scales.append(relu_6_fused_scale)
                
                assert(relu_6_fused_scales[-1] > 256)   

        with open(weights_zero_points_file_format.format(layer_index), 'r') as f:
            for line in f:
                weights_zero_points_declaration_string += line.replace(
                    ' ', '').replace('\n', '') + ', '

        fused_zero_points_declaration_string = 'const static biases_dt layer_{}_fused_zero_points[] = \n'.format(
            layer_index)
        fused_zero_points_declaration_string += '{ ' + str(
            fused_zero_points).replace('[', '').replace(']', '') + '};\n'

        fused_scales_declaration_string = 'const static fused_scales_dt layer_{}_fused_scales[] ='.format(
            layer_index)
        fused_scales_declaration_string += '{ ' + str(
            fused_scales).replace('[', '').replace(']', '') + '};\n'
        fused_scales_log_2_shifts_declaration_string = 'const static fused_scales_log_2_shifts_dt layer_{}_fused_scales_log_2_shifts[] ='.format(
            layer_index)
        fused_scales_log_2_shifts_declaration_string += '{ ' + str(
            fused_scales_log_2_shifts).replace('[', '').replace(']', '') + '};\n'

        relu_6_fused_scales_declaration_string = 'const static relu_6_fused_scales_dt layer_{}_relu_6_fused_scales[] ='.format(
            layer_index) if layer_index != 0 else 'const static layer_0_relu_6_fused_scales_dt layer_0_relu_6_fused_scales[] ='
        relu_6_fused_scales_declaration_string += '{ ' + str(
            relu_6_fused_scales).replace('[', '').replace(']', '') + '};\n'

        wf.write(fused_zero_points_declaration_string)
        wf.write(fused_scales_declaration_string)
        wf.write(fused_scales_log_2_shifts_declaration_string)
        wf.write(relu_6_fused_scales_declaration_string)

    weights_zero_points_declaration_string += '};\n'

    wf.write(layers_fused_parameters_offsets_declaration_string +
             str(layers_fused_parameters_offsets).replace('[', '').replace(']', '};\n'))
    # wf.write(weights_zero_points_declaration_string)
    wf.write("#endif\n")


for i in range(len(add_layers_fms_scales_rec)):
    print(i, add_layers_fms_scales[i], add_layers_fms_scales_rec[i])
