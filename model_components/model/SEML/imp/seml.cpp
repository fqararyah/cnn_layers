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
//		for(int i=0;i<max_fms_size;i++){
//			result2[i] = i % 127;
//		}
//		begin_code_generation
fill_dw_layer_weights(dw_weights_44, dw_weights_buffer, layer_44_dw_depth, layer_44_dw_filter_size, layer_44_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, result2, channels, 44, layer_44_dw_depth,
    layer_44_dw_ifm_width, layer_44_dw_ifm_height, layer_44_dw_num_of_tiles_in_d,
    layer_44_dw_num_of_tiles_h, layer_44_dw_num_of_tiles_w,
    layer_44_dw_strides, layer_44_dw_padding_left, layer_44_dw_padding_right, layer_44_dw_padding_top,
    1, layer_44_fused_scales, layer_44_fused_scales_log_2_shifts, layer_44_relu_6_fused_scales, layer_44_fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 45, layer_45_pw_depth,
    layer_45_pw_num_fils, layer_45_pw_num_of_tiles_in_d,
    layer_45_pw_num_of_tiles_out_d, layer_45_pw_num_of_tiles_h,
    layer_45_pw_num_of_tiles_w, tmp_channels, 3,
    layer_45_pw_num_of_weight_groups_for_one_pass,
    0, layer_45_pw_weights_offset, layer_45_relu, layer_45_fused_scales, layer_45_fused_scales_log_2_shifts, layer_45_relu_6_fused_scales, layer_45_fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 46, layer_46_pw_depth,
    layer_46_pw_num_fils, layer_46_pw_num_of_tiles_in_d,
    layer_46_pw_num_of_tiles_out_d, layer_46_pw_num_of_tiles_h,
    layer_46_pw_num_of_tiles_w, tmp_channels, 0,
    layer_46_pw_num_of_weight_groups_for_one_pass,
    1, layer_46_pw_weights_offset, layer_46_relu, layer_46_fused_scales, layer_46_fused_scales_log_2_shifts, layer_46_relu_6_fused_scales, layer_46_fused_zero_points);
fill_dw_layer_weights(dw_weights_47, dw_weights_buffer, layer_47_dw_depth, layer_47_dw_filter_size, layer_47_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 47, layer_47_dw_depth,
    layer_47_dw_ifm_width, layer_47_dw_ifm_height, layer_47_dw_num_of_tiles_in_d,
    layer_47_dw_num_of_tiles_h, layer_47_dw_num_of_tiles_w,
    layer_47_dw_strides, layer_47_dw_padding_left, layer_47_dw_padding_right, layer_47_dw_padding_top,
    0, layer_47_fused_scales, layer_47_fused_scales_log_2_shifts, layer_47_relu_6_fused_scales, layer_47_fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 48, layer_48_pw_depth,
    layer_48_pw_num_fils, layer_48_pw_num_of_tiles_in_d,
    layer_48_pw_num_of_tiles_out_d, layer_48_pw_num_of_tiles_h,
    layer_48_pw_num_of_tiles_w, tmp_channels, 1,
    layer_48_pw_num_of_weight_groups_for_one_pass,
    1, layer_48_pw_weights_offset, layer_48_relu, layer_48_fused_scales, layer_48_fused_scales_log_2_shifts, layer_48_relu_6_fused_scales, layer_48_fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 49, layer_49_pw_depth,
    layer_49_pw_num_fils, layer_49_pw_num_of_tiles_in_d,
    layer_49_pw_num_of_tiles_out_d, layer_49_pw_num_of_tiles_h,
    layer_49_pw_num_of_tiles_w, tmp_channels, 0,
    layer_49_pw_num_of_weight_groups_for_one_pass,
    0, layer_49_pw_weights_offset, layer_49_relu, layer_49_fused_scales, layer_49_fused_scales_log_2_shifts, layer_49_relu_6_fused_scales, layer_49_fused_zero_points);
fill_dw_layer_weights(dw_weights_50, dw_weights_buffer, layer_50_dw_depth, layer_50_dw_filter_size, layer_50_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, result2, channels, 50, layer_50_dw_depth,
    layer_50_dw_ifm_width, layer_50_dw_ifm_height, layer_50_dw_num_of_tiles_in_d,
    layer_50_dw_num_of_tiles_h, layer_50_dw_num_of_tiles_w,
    layer_50_dw_strides, layer_50_dw_padding_left, layer_50_dw_padding_right, layer_50_dw_padding_top,
    1, layer_50_fused_scales, layer_50_fused_scales_log_2_shifts, layer_50_relu_6_fused_scales, layer_50_fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 51, layer_51_pw_depth,
    layer_51_pw_num_fils, layer_51_pw_num_of_tiles_in_d,
    layer_51_pw_num_of_tiles_out_d, layer_51_pw_num_of_tiles_h,
    layer_51_pw_num_of_tiles_w, tmp_channels, 0,
    layer_51_pw_num_of_weight_groups_for_one_pass,
    0, layer_51_pw_weights_offset, layer_51_relu, layer_51_fused_scales, layer_51_fused_scales_log_2_shifts, layer_51_relu_6_fused_scales, layer_51_fused_zero_points);
pw_conv(off_chip_weights, channels, result2, 52, layer_52_pw_depth,
    layer_52_pw_num_fils, layer_52_pw_num_of_tiles_in_d,
    layer_52_pw_num_of_tiles_out_d, layer_52_pw_num_of_tiles_h,
    layer_52_pw_num_of_tiles_w, tmp_channels, 0,
    layer_52_pw_num_of_weight_groups_for_one_pass,
    1, layer_52_pw_weights_offset, layer_52_relu, layer_52_fused_scales, layer_52_fused_scales_log_2_shifts, layer_52_relu_6_fused_scales, layer_52_fused_zero_points);
// 	//end_code_generation
		avgpool(channels, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
