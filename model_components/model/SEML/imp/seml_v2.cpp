#include "../headers/seml.h"

#include "../../../../tests/test_utils.h"

#if FIBHA_VERSION == 2

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
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 22, layer_22_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 24, layer_24_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 26,layer_26_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 27, layer_27_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_27.txt",
 result, layer_27_pw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 28, layer_28_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_28.txt",
 channels, layer_28_pw_specs);
#endif
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 29,layer_29_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_29.txt",
 result, layer_29_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 30, layer_30_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_30.txt",
 channels, layer_30_pw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 32, layer_32_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_32.txt",
 result, layer_32_pw_specs);
#endif
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 33,layer_33_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 34, layer_34_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 36, layer_36_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 37,layer_37_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 38, layer_38_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 40, layer_40_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_40.txt",
 result, layer_40_pw_specs);
#endif
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 41,layer_41_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_41.txt",
 channels, layer_41_dw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 42, layer_42_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_42.txt",
 result, layer_42_pw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 43, layer_43_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_43.txt",
 channels, layer_43_pw_specs);
#endif
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 44,layer_44_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_44.txt",
 result, layer_44_dw_specs);
#endif
pw_conv(off_chip_weights, result , channels, tmp_channels, 45, layer_45_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 47, layer_47_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_47.txt",
 result, layer_47_pw_specs);
#endif
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 48,layer_48_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 49, layer_49_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 51, layer_51_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 53,layer_53_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 54, layer_54_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 55, layer_55_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 56,layer_56_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 57, layer_57_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 59, layer_59_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 60,layer_60_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 61, layer_61_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 63, layer_63_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 64,layer_64_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 65, layer_65_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 66, layer_66_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_66.txt",
 channels, layer_66_pw_specs);
#endif
    // 	//end_code_generation
    avgpool(channels, fc_input);
    // fc_layer(fc_weights, fc_input, fc_output);
}

#endif