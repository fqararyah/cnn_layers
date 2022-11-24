#include "../headers/seml.h"

#include "../../../../tests/test_utils.h"

void seml(
		fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
		weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt result2[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size],
		fms_dt tmp_channels_2[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w],
		fms_dt fc_input[fc_layer_input_size]) {

	scales_dt fused_scales[max_conv_d];
	biases_dt fused_zero_points[max_conv_d];
	//begin_code_generation
fill_fused_scales_and_zero_points(layer_8_fused_scales,
  fused_scales, layer_8_fused_zero_points,
  fused_zero_points, layer_8_dw_num_fils);
fill_dw_layer_weights(dw_weights_8, dw_weights_buffer, layer_8_dw_depth, layer_8_dw_filter_size, layer_8_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 8, layer_8_dw_depth,
    layer_8_dw_ifm_width, layer_8_dw_ifm_height, layer_8_dw_num_of_tiles_in_d,
    layer_8_dw_num_of_tiles_h, layer_8_dw_num_of_tiles_w,
    layer_8_dw_strides, layer_8_dw_padding_left,layer_8_dw_padding_top,
    1, fused_scales, fused_zero_points);
fill_fused_scales_and_zero_points(layer_9_fused_scales,
  fused_scales, layer_9_fused_zero_points,
  fused_zero_points, layer_9_pw_num_fils);
pw_conv(off_chip_weights, channels, result2, 9, layer_9_pw_depth,
    layer_9_pw_num_fils, layer_9_pw_num_of_tiles_in_d,
    layer_9_pw_num_of_tiles_out_d, layer_9_pw_num_of_tiles_h,
    layer_9_pw_num_of_tiles_w, tmp_channels, 1,
    layer_9_pw_num_of_weight_groups_for_one_pass,
    0, layer_9_pw_weights_offset, layer_9_relu, fused_scales, fused_zero_points);
fill_fused_scales_and_zero_points(layer_10_fused_scales,
  fused_scales, layer_10_fused_zero_points,
  fused_zero_points, layer_10_pw_num_fils);
pw_conv(off_chip_weights, channels, result2, 10, layer_10_pw_depth,
    layer_10_pw_num_fils, layer_10_pw_num_of_tiles_in_d,
    layer_10_pw_num_of_tiles_out_d, layer_10_pw_num_of_tiles_h,
    layer_10_pw_num_of_tiles_w, tmp_channels, 0,
    layer_10_pw_num_of_weight_groups_for_one_pass,
    1, layer_10_pw_weights_offset, layer_10_relu, fused_scales, fused_zero_points);
// 	//end_code_generation
 	avgpool(channels, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);
}
