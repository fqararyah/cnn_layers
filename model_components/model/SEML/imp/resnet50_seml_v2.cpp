#include "../headers/seml.h"

#if HW == CPU
#include "../../../../tests/test_utils.h"
#endif

using namespace seml_engines;

#if FIBHA_VERSION == 2 && MODEL_ID == RESNET50

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
#if DEBUGGING
 fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/resnet50/fms/ifms_23.txt",
 channels, layer_23_pw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_23.txt",
 channels, layer_23_pw_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 23, layer_23_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_23.txt",
 result, layer_23_pw_specs);
#endif
pw_and_conv(off_chip_weights, result , channels, tmp_channels, 24, layer_24_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 25, layer_25_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 27, layer_27_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, channels , result, tmp_channels, 28, layer_28_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 29, layer_29_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 31, layer_31_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, result , channels, tmp_channels, 32, layer_32_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_32.txt",
 channels, layer_32_s_specs);
#endif
pw_conv(off_chip_weights, channels , result, tmp_channels, 33, layer_33_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 35, layer_35_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 36, layer_36_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, result , channels, tmp_channels, 37, layer_37_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 38, layer_38_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 40, layer_40_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, channels , result, tmp_channels, 41, layer_41_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 42, layer_42_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 44, layer_44_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, result , channels, tmp_channels, 45, layer_45_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 46, layer_46_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 48, layer_48_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, channels , result, tmp_channels, 49, layer_49_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 50, layer_50_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 52, layer_52_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, result , channels, tmp_channels, 53, layer_53_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 54, layer_54_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 56, layer_56_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, channels , result, tmp_channels, 57, layer_57_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 58, layer_58_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 60, layer_60_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 61, layer_61_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, channels , result, tmp_channels, 62, layer_62_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 63, layer_63_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 65, layer_65_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, result , channels, tmp_channels, 66, layer_66_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, channels , result, tmp_channels, 67, layer_67_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 69, layer_69_pw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_and_conv(off_chip_weights, channels , result, tmp_channels, 70, layer_70_s_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
pw_conv(off_chip_weights, result , channels, tmp_channels, 71, layer_71_pw_specs,
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