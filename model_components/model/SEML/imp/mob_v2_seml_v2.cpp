#include "../headers/seml.h"

#if HW == CPU
#include "../../../../tests/test_utils.h"
#endif

using namespace seml_engines;

#if FIBHA_VERSION == 2  && MODEL_ID == MOB_V2

void seml(weights_grp_dt off_chip_weights[all_pw_weights],
          fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt fc_input[fc_layer_input_size])
{
#pragma HLS INLINE off
    //		for(int i=0;i<max_fms_size;i++){
    //			result[i] = i % 127;
    //		}
    //		begin_code_generation
pw_conv(off_chip_weights, channels , result, tmp_channels, 20, layer_20_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_24.txt",
 channels, layer_24_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 26,layer_26_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_26.txt",
 result, layer_26_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 27, layer_27_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_27.txt",
 channels, layer_27_pw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 28, layer_28_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_28.txt",
 result, layer_28_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 29,layer_29_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_29.txt",
 channels, layer_29_dw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 30, layer_30_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_30.txt",
 result, layer_30_pw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 32, layer_32_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_32.txt",
 channels, layer_32_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, channels, result, 33,layer_33_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_33.txt",
 result, layer_33_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 34, layer_34_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_34.txt",
 channels, layer_34_pw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 36, layer_36_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_36.txt",
 result, layer_36_pw_specs);
#endif
seml_engines::dw_conv_3x3(seml_dw_weights_3x3, result, channels, 37,layer_37_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_37.txt",
 channels, layer_37_dw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 38, layer_38_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 40, layer_40_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
// 	//end_code_generation
#if MODEL_ID == RESNET50
    avgpool(channels, fc_input, layer_73_avgpool_specs);
#elif MODEL_ID == MOB_V2
    avgpool(channels, fc_input, layer_67_avgpool_specs);
#endif
    // fc_layer(fc_weights, fc_input, fc_output);
}

#endif