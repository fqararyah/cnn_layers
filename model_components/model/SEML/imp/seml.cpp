#include "../headers/seml.h"

#include "../../../../tests/test_utils.h"

void seml(weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt tmp_channels[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
		fms_dt fc_input[fc_layer_input_size]) {
#pragma HLS INLINE off
//		for(int i=0;i<max_fms_size;i++){
//			result[i] = i % 127;
//		}
//		begin_code_generation
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/fms_conv2d_6_24_56_56.txt",
 channels, 56, 56);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_6.txt",
 channels, 75264, 56, 56);
pw_conv(off_chip_weights, tmp_channels, result, 6, layer_6_pw_depth,
    layer_6_pw_num_fils, layer_6_pw_num_of_tiles_in_d,
    layer_6_pw_num_of_tiles_out_d, layer_6_pw_num_of_tiles_h,
    layer_6_pw_num_of_tiles_w, tmp_channels, 0,
    layer_6_pw_num_of_weight_groups_for_one_pass,
    0, layer_6_pw_weights_offset, layer_6_activation,
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_6.txt",
 result, 451584, 56, 56);
#endif
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 7, layer_7_dw_depth,
    layer_7_dw_ifm_width, layer_7_dw_ifm_height, layer_7_dw_num_of_tiles_in_d,
    layer_7_dw_num_of_tiles_h, layer_7_dw_num_of_tiles_w,
    layer_7_dw_strides, layer_7_dw_padding_left, layer_7_dw_padding_right, layer_7_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
        fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
 channels, 451584, 56, 56);
#endif
pw_conv(off_chip_weights, channels, result, 8, layer_8_pw_depth,
    layer_8_pw_num_fils, layer_8_pw_num_of_tiles_in_d,
    layer_8_pw_num_of_tiles_out_d, layer_8_pw_num_of_tiles_h,
    layer_8_pw_num_of_tiles_w, tmp_channels, 1,
    layer_8_pw_num_of_weight_groups_for_one_pass,
    0, layer_8_pw_weights_offset, layer_8_activation,
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
 result, 75264, 56, 56);
#endif
pw_conv(off_chip_weights, channels, result, 9, layer_9_pw_depth,
    layer_9_pw_num_fils, layer_9_pw_num_of_tiles_in_d,
    layer_9_pw_num_of_tiles_out_d, layer_9_pw_num_of_tiles_h,
    layer_9_pw_num_of_tiles_w, tmp_channels, 0,
    layer_9_pw_num_of_weight_groups_for_one_pass,
    1, layer_9_pw_weights_offset, layer_9_activation,
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_9.txt",
 channels, 451584, 56, 56);
#endif
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 10, layer_10_dw_depth,
    layer_10_dw_ifm_width, layer_10_dw_ifm_height, layer_10_dw_num_of_tiles_in_d,
    layer_10_dw_num_of_tiles_h, layer_10_dw_num_of_tiles_w,
    layer_10_dw_strides, layer_10_dw_padding_left, layer_10_dw_padding_right, layer_10_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
    fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_10.txt",
 result, 112896, 28, 28);
#endif
pw_conv(off_chip_weights, channels, result, 11, layer_11_pw_depth,
    layer_11_pw_num_fils, layer_11_pw_num_of_tiles_in_d,
    layer_11_pw_num_of_tiles_out_d, layer_11_pw_num_of_tiles_h,
    layer_11_pw_num_of_tiles_w, tmp_channels, 2,
    layer_11_pw_num_of_weight_groups_for_one_pass,
    1, layer_11_pw_weights_offset, layer_11_activation,
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_11.txt",
 channels, 25088, 28, 28);
#endif
pw_conv(off_chip_weights, channels, result, 12, layer_12_pw_depth,
    layer_12_pw_num_fils, layer_12_pw_num_of_tiles_in_d,
    layer_12_pw_num_of_tiles_out_d, layer_12_pw_num_of_tiles_h,
    layer_12_pw_num_of_tiles_w, tmp_channels, 0,
    layer_12_pw_num_of_weight_groups_for_one_pass,
    0, layer_12_pw_weights_offset, layer_12_activation,
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_12.txt",
 result, 150528, 28, 28);
#endif
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 13, layer_13_dw_depth,
    layer_13_dw_ifm_width, layer_13_dw_ifm_height, layer_13_dw_num_of_tiles_in_d,
    layer_13_dw_num_of_tiles_h, layer_13_dw_num_of_tiles_w,
    layer_13_dw_strides, layer_13_dw_padding_left, layer_13_dw_padding_right, layer_13_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
        fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_13.txt",
 channels, 150528, 28, 28);
#endif
pw_conv(off_chip_weights, channels, result, 14, layer_14_pw_depth,
    layer_14_pw_num_fils, layer_14_pw_num_of_tiles_in_d,
    layer_14_pw_num_of_tiles_out_d, layer_14_pw_num_of_tiles_h,
    layer_14_pw_num_of_tiles_w, tmp_channels, 3,
    layer_14_pw_num_of_weight_groups_for_one_pass,
    0, layer_14_pw_weights_offset, layer_14_activation,
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_14.txt",
 result, 25088, 28, 28);
#endif
pw_conv(off_chip_weights, channels, result, 15, layer_15_pw_depth,
    layer_15_pw_num_fils, layer_15_pw_num_of_tiles_in_d,
    layer_15_pw_num_of_tiles_out_d, layer_15_pw_num_of_tiles_h,
    layer_15_pw_num_of_tiles_w, tmp_channels, 0,
    layer_15_pw_num_of_weight_groups_for_one_pass,
    1, layer_15_pw_weights_offset, layer_15_activation,
         fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points,
         fused_scales_part2, fused_scales_log_2_shifts_part2, relu_6_fused_scales_part2, fused_zero_points_part2);
#if DEBUGGING
 dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_15.txt",
 channels, 150528, 28, 28);
#endif
// 	//end_code_generation
		avgpool(channels, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
