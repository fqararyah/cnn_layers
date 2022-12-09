#include "../headers/seml.h"

#include "../../../../tests/test_utils.h"

void seml(weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result2[max_fms_size],
		fms_dt tmp_channels[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h*max_conv_w],
		fms_dt fc_input[fc_layer_input_size]) {
#pragma HLS INLINE off
		fused_scales_dt fused_scales[max_conv_d];
    fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[max_conv_d];
		relu_6_fused_scales_dt relu_6_fused_scales[max_conv_d];
		biases_dt fused_zero_points[max_conv_d];
		//begin_code_generation
fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/fms_5_96_112_112.txt",
 channels, 112, 112);
verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_5.txt",
 channels, 1204224, 112, 112);
fill_fused_scales_and_zero_points(layer_5_fused_scales,fused_scales, 
    layer_5_fused_scales_log_2_shifts, fused_scales_log_2_shifts, layer_5_relu_6_fused_scales,
     relu_6_fused_scales, layer_5_fused_zero_points,
    fused_zero_points, layer_5_dw_num_fils);
fill_dw_layer_weights(dw_weights_5, dw_weights_buffer, layer_5_dw_depth, layer_5_dw_filter_size, layer_5_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 5, layer_5_dw_depth,
    layer_5_dw_ifm_width, layer_5_dw_ifm_height, layer_5_dw_num_of_tiles_in_d,
    layer_5_dw_num_of_tiles_h, layer_5_dw_num_of_tiles_w,
    layer_5_dw_strides, layer_5_dw_padding_left, layer_5_dw_padding_right, layer_5_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_5.txt",
 result2, 301056, 56, 56);
fill_fused_scales_and_zero_points(layer_6_fused_scales,fused_scales, 
    layer_6_fused_scales_log_2_shifts, fused_scales_log_2_shifts, layer_6_relu_6_fused_scales,
     relu_6_fused_scales, layer_6_fused_zero_points,
    fused_zero_points, layer_6_pw_num_fils);
pw_conv(off_chip_weights, channels, result2, 6, layer_6_pw_depth,
    layer_6_pw_num_fils, layer_6_pw_num_of_tiles_in_d,
    layer_6_pw_num_of_tiles_out_d, layer_6_pw_num_of_tiles_h,
    layer_6_pw_num_of_tiles_w, tmp_channels, 2,
    layer_6_pw_num_of_weight_groups_for_one_pass,
    1, layer_6_pw_weights_offset, layer_6_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_6.txt",
 channels, 75264, 56, 56);
fill_fused_scales_and_zero_points(layer_7_fused_scales,fused_scales, 
    layer_7_fused_scales_log_2_shifts, fused_scales_log_2_shifts, layer_7_relu_6_fused_scales,
     relu_6_fused_scales, layer_7_fused_zero_points,
    fused_zero_points, layer_7_pw_num_fils);
pw_conv(off_chip_weights, channels, result2, 7, layer_7_pw_depth,
    layer_7_pw_num_fils, layer_7_pw_num_of_tiles_in_d,
    layer_7_pw_num_of_tiles_out_d, layer_7_pw_num_of_tiles_h,
    layer_7_pw_num_of_tiles_w, tmp_channels, 0,
    layer_7_pw_num_of_weight_groups_for_one_pass,
    0, layer_7_pw_weights_offset, layer_7_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_7.txt",
 result2, 451584, 56, 56);
fill_fused_scales_and_zero_points(layer_8_fused_scales,fused_scales, 
    layer_8_fused_scales_log_2_shifts, fused_scales_log_2_shifts, layer_8_relu_6_fused_scales,
     relu_6_fused_scales, layer_8_fused_zero_points,
    fused_zero_points, layer_8_dw_num_fils);
fill_dw_layer_weights(dw_weights_8, dw_weights_buffer, layer_8_dw_depth, layer_8_dw_filter_size, layer_8_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, result2, channels, 8, layer_8_dw_depth,
    layer_8_dw_ifm_width, layer_8_dw_ifm_height, layer_8_dw_num_of_tiles_in_d,
    layer_8_dw_num_of_tiles_h, layer_8_dw_num_of_tiles_w,
    layer_8_dw_strides, layer_8_dw_padding_left, layer_8_dw_padding_right, layer_8_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_8.txt",
 channels, 451584, 56, 56);
// 	//end_code_generation
		avgpool(result2, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
