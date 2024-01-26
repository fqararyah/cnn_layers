#include "../headers/seml.h"

#if HW == CPU
#include "../../../../tests/test_utils.h"
#endif

using namespace seml_engines;

#if FIBHA_VERSION == 2 && MODEL_ID == MOB_V2 && PIPELINE_LENGTH == 0

void seml(weights_grp_dt off_chip_weights[all_off_chip_pw_s_weights],
          weights_dt off_chip_dw_weights[all_dw_off_chip_weights],
          fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps],
          biases_dt off_chip_fused_zero_points[all_off_chip_fused_scales_zps],
          fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
          fms_dt result[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
          fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
          fms_dt fc_input[fc_layer_input_size],
          int model_configs_list[2 * max_conv_layers])
{
#pragma HLS INLINE off
    //		for(int i=0;i<max_fms_size;i++){
    //			result[i] = i % 127;
    //		}
    //		begin_code_generation
#if DEBUGGING
 fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_2.txt",
 channels, layer_2_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_2.txt",
 channels, layer_2_dw_specs);
#endif
seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[2], layer_2_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[2],
                                     layer_2_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[2], 
                                layer_2_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 2,layer_2_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_2.txt",
 result, layer_2_dw_specs);
#endif
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[3],
                                     layer_3_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[3], 
                                layer_3_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 3, layer_3_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_3.txt",
 channels, layer_3_pw_specs);
#endif
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[4],
                                     layer_4_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[4], 
                                layer_4_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 4, layer_4_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_4.txt",
 result, layer_4_pw_specs);
#endif
seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[6], layer_6_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[6],
                                     layer_6_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[6], 
                                layer_6_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 6,layer_6_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_6.txt",
 channels, layer_6_dw_specs);
#endif
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[7],
                                     layer_7_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[7], 
                                layer_7_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 7, layer_7_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

#if DEBUGGING
 dump_layer_output("/media/SSD2TB/fareed/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
 result, layer_7_pw_specs);
#endif
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[8],
                                     layer_8_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[8], 
                                layer_8_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 8, layer_8_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[9], layer_9_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[9],
                                     layer_9_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[9], 
                                layer_9_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 9,layer_9_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[10],
                                     layer_10_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[10], 
                                layer_10_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 10, layer_10_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[12],
                                     layer_12_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[12], 
                                layer_12_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 12, layer_12_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[14], layer_14_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[14],
                                     layer_14_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[14], 
                                layer_14_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 14,layer_14_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[15],
                                     layer_15_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[15], 
                                layer_15_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 15, layer_15_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[16],
                                     layer_16_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[16], 
                                layer_16_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 16, layer_16_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[17], layer_17_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[17],
                                     layer_17_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[17], 
                                layer_17_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 17,layer_17_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[18],
                                     layer_18_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[18], 
                                layer_18_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 18, layer_18_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[20],
                                     layer_20_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[20], 
                                layer_20_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 20, layer_20_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[21], layer_21_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[21],
                                     layer_21_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[21], 
                                layer_21_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 21,layer_21_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[22],
                                     layer_22_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[22], 
                                layer_22_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 22, layer_22_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[24],
                                     layer_24_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[24], 
                                layer_24_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[26], layer_26_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[26],
                                     layer_26_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[26], 
                                layer_26_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 26,layer_26_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[27],
                                     layer_27_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[27], 
                                layer_27_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 27, layer_27_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[28],
                                     layer_28_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[28], 
                                layer_28_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 28, layer_28_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[29], layer_29_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[29],
                                     layer_29_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[29], 
                                layer_29_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 29,layer_29_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[30],
                                     layer_30_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[30], 
                                layer_30_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 30, layer_30_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[32],
                                     layer_32_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[32], 
                                layer_32_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 32, layer_32_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[33], layer_33_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[33],
                                     layer_33_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[33], 
                                layer_33_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 33,layer_33_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[34],
                                     layer_34_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[34], 
                                layer_34_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 34, layer_34_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[36],
                                     layer_36_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[36], 
                                layer_36_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 36, layer_36_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[37], layer_37_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[37],
                                     layer_37_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[37], 
                                layer_37_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 37,layer_37_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[38],
                                     layer_38_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[38], 
                                layer_38_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 38, layer_38_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[40],
                                     layer_40_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[40], 
                                layer_40_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 40, layer_40_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[41], layer_41_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[41],
                                     layer_41_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[41], 
                                layer_41_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 41,layer_41_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[42],
                                     layer_42_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[42], 
                                layer_42_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 42, layer_42_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[43],
                                     layer_43_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[43], 
                                layer_43_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 43, layer_43_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[44], layer_44_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[44],
                                     layer_44_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[44], 
                                layer_44_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 44,layer_44_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[45],
                                     layer_45_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[45], 
                                layer_45_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 45, layer_45_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[47],
                                     layer_47_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[47], 
                                layer_47_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 47, layer_47_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[48], layer_48_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[48],
                                     layer_48_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[48], 
                                layer_48_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 48,layer_48_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[49],
                                     layer_49_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[49], 
                                layer_49_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 49, layer_49_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[51],
                                     layer_51_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[51], 
                                layer_51_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 51, layer_51_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[53], layer_53_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[53],
                                     layer_53_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[53], 
                                layer_53_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 53,layer_53_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[54],
                                     layer_54_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[54], 
                                layer_54_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 54, layer_54_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[55],
                                     layer_55_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[55], 
                                layer_55_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 55, layer_55_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[56], layer_56_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[56],
                                     layer_56_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[56], 
                                layer_56_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 56,layer_56_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[57],
                                     layer_57_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[57], 
                                layer_57_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 57, layer_57_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[59],
                                     layer_59_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[59], 
                                layer_59_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 59, layer_59_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[60], layer_60_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[60],
                                     layer_60_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[60], 
                                layer_60_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 60,layer_60_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[61],
                                     layer_61_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[61], 
                                layer_61_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 61, layer_61_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[63],
                                     layer_63_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[63], 
                                layer_63_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 63, layer_63_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_layer_dw_weights_off_chip
    (off_chip_dw_weights, seml_dw_weights_3x3, dw_layers_weights_offsets[64], layer_64_dw_specs.layer_depth);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[64],
                                     layer_64_dw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[64], 
                                layer_64_dw_specs.layer_num_fils);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 64,layer_64_dw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
        model_configs_list);
seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[65],
                                     layer_65_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[65], 
                                layer_65_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, result , channels, tmp_channels, 65, layer_65_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

seml_engines::fill_fused_scales(off_chip_fused_scales,
                                     seml_fused_scales_buffer,
                                     layers_fused_parameters_offsets[66],
                                     layer_66_pw_specs.layer_num_fils);
seml_engines::fill_fused_zero_points(off_chip_fused_zero_points, 
                                seml_fused_zero_points_buffer, 
                                layers_fused_parameters_offsets[66], 
                                layer_66_pw_specs.layer_num_fils);
pw_conv(off_chip_weights, channels , result, tmp_channels, 66, layer_66_pw_specs,
    seml_fused_scales_buffer, relu_6_fused_scales, seml_fused_zero_points_buffer,
    model_configs_list);

// 	//end_code_generation
#if MODEL_ID == RESNET50
    avgpool(channels, fc_input, layer_73_avgpool_specs);
#elif MODEL_ID == MOB_V2
    avgpool(result, fc_input, layer_67_avgpool_specs);
#endif
    // fc_layer(fc_weights, fc_input, fc_output);
}

#endif