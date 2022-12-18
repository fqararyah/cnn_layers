#include "../headers/seml.h"

#include "../../../../tests/test_utils.h"

void seml(weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result2[max_fms_size],
		fms_dt tmp_channels[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h*max_conv_w],
		fms_dt fc_input[fc_layer_input_size]) {
#pragma HLS INLINE off
//		for(int i=0;i<max_fms_size;i++){
//			result2[i] = i % 127;
//		}
//		begin_code_generation
//fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/fms_8_144_56_56.txt",
// result2, 56, 56);
//verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_8.txt",
// result2, 451584, 56, 56);
fill_dw_layer_weights(dw_weights_8, dw_weights_buffer, layer_8_dw_depth, layer_8_dw_filter_size, layer_8_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, result2, channels, 8, layer_8_dw_depth,
    layer_8_dw_ifm_width, layer_8_dw_ifm_height, layer_8_dw_num_of_tiles_in_d,
    layer_8_dw_num_of_tiles_h, layer_8_dw_num_of_tiles_w,
    layer_8_dw_strides, layer_8_dw_padding_left, layer_8_dw_padding_right, layer_8_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
 channels, 451584, 56, 56);
pw_conv(off_chip_weights, channels, result2, 9, layer_9_pw_depth,
    layer_9_pw_num_fils, layer_9_pw_num_of_tiles_in_d,
    layer_9_pw_num_of_tiles_out_d, layer_9_pw_num_of_tiles_h,
    layer_9_pw_num_of_tiles_w, tmp_channels, 1,
    layer_9_pw_num_of_weight_groups_for_one_pass,
    0, layer_9_pw_weights_offset, layer_9_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_9.txt",
 result2, 75264, 56, 56);
pw_conv(off_chip_weights, channels, result2, 10, layer_10_pw_depth,
    layer_10_pw_num_fils, layer_10_pw_num_of_tiles_in_d,
    layer_10_pw_num_of_tiles_out_d, layer_10_pw_num_of_tiles_h,
    layer_10_pw_num_of_tiles_w, tmp_channels, 0,
    layer_10_pw_num_of_weight_groups_for_one_pass,
    1, layer_10_pw_weights_offset, layer_10_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_10.txt",
 channels, 451584, 56, 56);
fill_dw_layer_weights(dw_weights_11, dw_weights_buffer, layer_11_dw_depth, layer_11_dw_filter_size, layer_11_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 11, layer_11_dw_depth,
    layer_11_dw_ifm_width, layer_11_dw_ifm_height, layer_11_dw_num_of_tiles_in_d,
    layer_11_dw_num_of_tiles_h, layer_11_dw_num_of_tiles_w,
    layer_11_dw_strides, layer_11_dw_padding_left, layer_11_dw_padding_right, layer_11_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_11.txt",
 result2, 112896, 28, 28);
pw_conv(off_chip_weights, channels, result2, 12, layer_12_pw_depth,
    layer_12_pw_num_fils, layer_12_pw_num_of_tiles_in_d,
    layer_12_pw_num_of_tiles_out_d, layer_12_pw_num_of_tiles_h,
    layer_12_pw_num_of_tiles_w, tmp_channels, 2,
    layer_12_pw_num_of_weight_groups_for_one_pass,
    1, layer_12_pw_weights_offset, layer_12_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_12.txt",
 channels, 25088, 28, 28);
pw_conv(off_chip_weights, channels, result2, 13, layer_13_pw_depth,
    layer_13_pw_num_fils, layer_13_pw_num_of_tiles_in_d,
    layer_13_pw_num_of_tiles_out_d, layer_13_pw_num_of_tiles_h,
    layer_13_pw_num_of_tiles_w, tmp_channels, 0,
    layer_13_pw_num_of_weight_groups_for_one_pass,
    0, layer_13_pw_weights_offset, layer_13_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_13.txt",
 result2, 150528, 28, 28);
fill_dw_layer_weights(dw_weights_14, dw_weights_buffer, layer_14_dw_depth, layer_14_dw_filter_size, layer_14_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, result2, channels, 14, layer_14_dw_depth,
    layer_14_dw_ifm_width, layer_14_dw_ifm_height, layer_14_dw_num_of_tiles_in_d,
    layer_14_dw_num_of_tiles_h, layer_14_dw_num_of_tiles_w,
    layer_14_dw_strides, layer_14_dw_padding_left, layer_14_dw_padding_right, layer_14_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 15, layer_15_pw_depth,
    layer_15_pw_num_fils, layer_15_pw_num_of_tiles_in_d,
    layer_15_pw_num_of_tiles_out_d, layer_15_pw_num_of_tiles_h,
    layer_15_pw_num_of_tiles_w, tmp_channels, 3,
    layer_15_pw_num_of_weight_groups_for_one_pass,
    0, layer_15_pw_weights_offset, layer_15_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 16, layer_16_pw_depth,
    layer_16_pw_num_fils, layer_16_pw_num_of_tiles_in_d,
    layer_16_pw_num_of_tiles_out_d, layer_16_pw_num_of_tiles_h,
    layer_16_pw_num_of_tiles_w, tmp_channels, 0,
    layer_16_pw_num_of_weight_groups_for_one_pass,
    1, layer_16_pw_weights_offset, layer_16_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
fill_dw_layer_weights(dw_weights_17, dw_weights_buffer, layer_17_dw_depth, layer_17_dw_filter_size, layer_17_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 17, layer_17_dw_depth,
    layer_17_dw_ifm_width, layer_17_dw_ifm_height, layer_17_dw_num_of_tiles_in_d,
    layer_17_dw_num_of_tiles_h, layer_17_dw_num_of_tiles_w,
    layer_17_dw_strides, layer_17_dw_padding_left, layer_17_dw_padding_right, layer_17_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 18, layer_18_pw_depth,
    layer_18_pw_num_fils, layer_18_pw_num_of_tiles_in_d,
    layer_18_pw_num_of_tiles_out_d, layer_18_pw_num_of_tiles_h,
    layer_18_pw_num_of_tiles_w, tmp_channels, 1,
    layer_18_pw_num_of_weight_groups_for_one_pass,
    1, layer_18_pw_weights_offset, layer_18_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 19, layer_19_pw_depth,
    layer_19_pw_num_fils, layer_19_pw_num_of_tiles_in_d,
    layer_19_pw_num_of_tiles_out_d, layer_19_pw_num_of_tiles_h,
    layer_19_pw_num_of_tiles_w, tmp_channels, 0,
    layer_19_pw_num_of_weight_groups_for_one_pass,
    0, layer_19_pw_weights_offset, layer_19_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
fill_dw_layer_weights(dw_weights_20, dw_weights_buffer, layer_20_dw_depth, layer_20_dw_filter_size, layer_20_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, result2, channels, 20, layer_20_dw_depth,
    layer_20_dw_ifm_width, layer_20_dw_ifm_height, layer_20_dw_num_of_tiles_in_d,
    layer_20_dw_num_of_tiles_h, layer_20_dw_num_of_tiles_w,
    layer_20_dw_strides, layer_20_dw_padding_left, layer_20_dw_padding_right, layer_20_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_20.txt",
 channels, 37632, 14, 14);
// 	//end_code_generation
		avgpool(channels, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
