#include "../headers/seml.h"

#include "../../../../tests/test_utils.h"

void seml(weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result2[max_fms_size],
		fms_dt tmp_channels[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w],
		fms_dt fc_input[fc_layer_input_size]) {
#pragma HLS INLINE off
		fused_scales_dt fused_scales[max_conv_d];
		relu_6_fused_scales_dt relu_6_fused_scales[max_conv_d];
		biases_dt fused_zero_points[max_conv_d];
		//begin_code_generation
fill_fused_scales_and_zero_points(layer_2_fused_scales,
  fused_scales, layer_2_relu_6_fused_scales, relu_6_fused_scales, layer_2_fused_zero_points,
  fused_zero_points, layer_2_dw_num_fils);
fill_dw_layer_weights(dw_weights_2, dw_weights_buffer, layer_2_dw_depth, layer_2_dw_filter_size, layer_2_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 2, layer_2_dw_depth,
    layer_2_dw_ifm_width, layer_2_dw_ifm_height, layer_2_dw_num_of_tiles_in_d,
    layer_2_dw_num_of_tiles_h, layer_2_dw_num_of_tiles_w,
    layer_2_dw_strides, layer_2_dw_padding_left,layer_2_dw_padding_top,
    1, fused_scales, relu_6_fused_scales, fused_zero_points);
fill_fused_scales_and_zero_points(layer_3_fused_scales,
  fused_scales, layer_3_relu_6_fused_scales, relu_6_fused_scales, layer_3_fused_zero_points,
  fused_zero_points, layer_3_pw_num_fils);
pw_conv(off_chip_weights, channels, result2, 3, layer_3_pw_depth,
    layer_3_pw_num_fils, layer_3_pw_num_of_tiles_in_d,
    layer_3_pw_num_of_tiles_out_d, layer_3_pw_num_of_tiles_h,
    layer_3_pw_num_of_tiles_w, tmp_channels, 0,
    layer_3_pw_num_of_weight_groups_for_one_pass,
    0, layer_3_pw_weights_offset, layer_3_relu, fused_scales, relu_6_fused_scales, fused_zero_points);
//fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/mob_v2/fms/fms_4_16_112_112.txt",
// result2, 112, 112);
//verify_fill_layer_input("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/verify_4.txt",
// result2, 200704, 112, 112);
//fill_fused_scales_and_zero_points(layer_4_fused_scales,
//  fused_scales, layer_4_relu_6_fused_scales, relu_6_fused_scales, layer_4_fused_zero_points,
//  fused_zero_points, layer_4_pw_num_fils);
//pw_conv(off_chip_weights, channels, result2, 4, layer_4_pw_depth,
//    layer_4_pw_num_fils, layer_4_pw_num_of_tiles_in_d,
//    layer_4_pw_num_of_tiles_out_d, layer_4_pw_num_of_tiles_h,
//    layer_4_pw_num_of_tiles_w, tmp_channels, 0,
//    layer_4_pw_num_of_weight_groups_for_one_pass,
//    1, layer_4_pw_weights_offset, layer_4_relu, fused_scales, relu_6_fused_scales, fused_zero_points);
dump_layer_output("/media/SSD2TB/wd/my_repos/DL_Benchmarking/tflite_scripts_imgnt_accuracy_and_weight_extraction/scratch_out/ofms_4.txt",
 channels, 1204224, 112, 112);
fill_fused_scales_and_zero_points(layer_5_fused_scales,
  fused_scales, layer_5_relu_6_fused_scales, relu_6_fused_scales, layer_5_fused_zero_points,
  fused_zero_points, layer_5_dw_num_fils);
fill_dw_layer_weights(dw_weights_5, dw_weights_buffer, layer_5_dw_depth, layer_5_dw_filter_size, layer_5_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 5, layer_5_dw_depth,
    layer_5_dw_ifm_width, layer_5_dw_ifm_height, layer_5_dw_num_of_tiles_in_d,
    layer_5_dw_num_of_tiles_h, layer_5_dw_num_of_tiles_w,
    layer_5_dw_strides, layer_5_dw_padding_left,layer_5_dw_padding_top,
    0, fused_scales, relu_6_fused_scales, fused_zero_points);
// 	//end_code_generation
		avgpool(channels, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
