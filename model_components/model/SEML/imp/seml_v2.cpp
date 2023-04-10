#include "../headers/seml.h"

#include "../../../../tests/test_utils.h"

#if FIBHA_VERSION == 2

void seml(weights_grp_dt off_chip_weights[all_pw_weights],
          fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          const layer_0_weights_dt weights_1[layer_1_s_num_fils][layer_1_s_depth][layer_1_s_filter_dim][layer_1_s_filter_dim],
          fms_dt fc_input[fc_layer_input_size])
{
#pragma HLS INLINE off
    //		for(int i=0;i<max_fms_size;i++){
    //			result[i] = i % 127;
    //		}
    //		begin_code_generation
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 17,layer_17_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 21,layer_21_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 26,layer_26_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/ifms_29.txt",
 result, layer_29_dw_specs);
#endif
#if DEBUGGING
 verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_29.txt",
 result, layer_29_dw_specs);
#endif
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 29,layer_29_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_29.txt",
 channels, layer_29_dw_specs);
#endif
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 33,layer_33_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 37,layer_37_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 41,layer_41_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 44,layer_44_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 48,layer_48_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 53,layer_53_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 56,layer_56_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 60,layer_60_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 64,layer_64_dw_specs,
    fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
    // 	//end_code_generation
    avgpool(channels, fc_input);
    // fc_layer(fc_weights, fc_input, fc_output);
}

#endif