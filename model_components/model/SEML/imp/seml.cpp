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
pw_conv(off_chip_weights, tmp_channels, result, 6, layer_6_pw_depth,
    layer_6_pw_num_fils, layer_6_pw_num_of_tiles_in_d,
    layer_6_pw_num_of_tiles_out_d, layer_6_pw_num_of_tiles_h,
    layer_6_pw_num_of_tiles_w, tmp_channels, 0,
    layer_6_pw_num_of_weight_groups_for_one_pass,
    0, layer_6_pw_weights_offset, layer_6_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 7, layer_7_dw_depth,
    layer_7_dw_ifm_width, layer_7_dw_ifm_height, layer_7_dw_num_of_tiles_in_d,
    layer_7_dw_num_of_tiles_h, layer_7_dw_num_of_tiles_w,
    layer_7_dw_strides, layer_7_dw_padding_left, layer_7_dw_padding_right, layer_7_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 8, layer_8_pw_depth,
    layer_8_pw_num_fils, layer_8_pw_num_of_tiles_in_d,
    layer_8_pw_num_of_tiles_out_d, layer_8_pw_num_of_tiles_h,
    layer_8_pw_num_of_tiles_w, tmp_channels, 1,
    layer_8_pw_num_of_weight_groups_for_one_pass,
    0, layer_8_pw_weights_offset, layer_8_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 9, layer_9_pw_depth,
    layer_9_pw_num_fils, layer_9_pw_num_of_tiles_in_d,
    layer_9_pw_num_of_tiles_out_d, layer_9_pw_num_of_tiles_h,
    layer_9_pw_num_of_tiles_w, tmp_channels, 0,
    layer_9_pw_num_of_weight_groups_for_one_pass,
    1, layer_9_pw_weights_offset, layer_9_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 10, layer_10_dw_depth,
    layer_10_dw_ifm_width, layer_10_dw_ifm_height, layer_10_dw_num_of_tiles_in_d,
    layer_10_dw_num_of_tiles_h, layer_10_dw_num_of_tiles_w,
    layer_10_dw_strides, layer_10_dw_padding_left, layer_10_dw_padding_right, layer_10_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 11, layer_11_pw_depth,
    layer_11_pw_num_fils, layer_11_pw_num_of_tiles_in_d,
    layer_11_pw_num_of_tiles_out_d, layer_11_pw_num_of_tiles_h,
    layer_11_pw_num_of_tiles_w, tmp_channels, 2,
    layer_11_pw_num_of_weight_groups_for_one_pass,
    1, layer_11_pw_weights_offset, layer_11_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 12, layer_12_pw_depth,
    layer_12_pw_num_fils, layer_12_pw_num_of_tiles_in_d,
    layer_12_pw_num_of_tiles_out_d, layer_12_pw_num_of_tiles_h,
    layer_12_pw_num_of_tiles_w, tmp_channels, 0,
    layer_12_pw_num_of_weight_groups_for_one_pass,
    0, layer_12_pw_weights_offset, layer_12_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 13, layer_13_dw_depth,
    layer_13_dw_ifm_width, layer_13_dw_ifm_height, layer_13_dw_num_of_tiles_in_d,
    layer_13_dw_num_of_tiles_h, layer_13_dw_num_of_tiles_w,
    layer_13_dw_strides, layer_13_dw_padding_left, layer_13_dw_padding_right, layer_13_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 14, layer_14_pw_depth,
    layer_14_pw_num_fils, layer_14_pw_num_of_tiles_in_d,
    layer_14_pw_num_of_tiles_out_d, layer_14_pw_num_of_tiles_h,
    layer_14_pw_num_of_tiles_w, tmp_channels, 3,
    layer_14_pw_num_of_weight_groups_for_one_pass,
    0, layer_14_pw_weights_offset, layer_14_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 15, layer_15_pw_depth,
    layer_15_pw_num_fils, layer_15_pw_num_of_tiles_in_d,
    layer_15_pw_num_of_tiles_out_d, layer_15_pw_num_of_tiles_h,
    layer_15_pw_num_of_tiles_w, tmp_channels, 0,
    layer_15_pw_num_of_weight_groups_for_one_pass,
    1, layer_15_pw_weights_offset, layer_15_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 16, layer_16_dw_depth,
    layer_16_dw_ifm_width, layer_16_dw_ifm_height, layer_16_dw_num_of_tiles_in_d,
    layer_16_dw_num_of_tiles_h, layer_16_dw_num_of_tiles_w,
    layer_16_dw_strides, layer_16_dw_padding_left, layer_16_dw_padding_right, layer_16_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 17, layer_17_pw_depth,
    layer_17_pw_num_fils, layer_17_pw_num_of_tiles_in_d,
    layer_17_pw_num_of_tiles_out_d, layer_17_pw_num_of_tiles_h,
    layer_17_pw_num_of_tiles_w, tmp_channels, 1,
    layer_17_pw_num_of_weight_groups_for_one_pass,
    1, layer_17_pw_weights_offset, layer_17_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 18, layer_18_pw_depth,
    layer_18_pw_num_fils, layer_18_pw_num_of_tiles_in_d,
    layer_18_pw_num_of_tiles_out_d, layer_18_pw_num_of_tiles_h,
    layer_18_pw_num_of_tiles_w, tmp_channels, 0,
    layer_18_pw_num_of_weight_groups_for_one_pass,
    0, layer_18_pw_weights_offset, layer_18_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 19, layer_19_dw_depth,
    layer_19_dw_ifm_width, layer_19_dw_ifm_height, layer_19_dw_num_of_tiles_in_d,
    layer_19_dw_num_of_tiles_h, layer_19_dw_num_of_tiles_w,
    layer_19_dw_strides, layer_19_dw_padding_left, layer_19_dw_padding_right, layer_19_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 20, layer_20_pw_depth,
    layer_20_pw_num_fils, layer_20_pw_num_of_tiles_in_d,
    layer_20_pw_num_of_tiles_out_d, layer_20_pw_num_of_tiles_h,
    layer_20_pw_num_of_tiles_w, tmp_channels, 2,
    layer_20_pw_num_of_weight_groups_for_one_pass,
    0, layer_20_pw_weights_offset, layer_20_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 21, layer_21_pw_depth,
    layer_21_pw_num_fils, layer_21_pw_num_of_tiles_in_d,
    layer_21_pw_num_of_tiles_out_d, layer_21_pw_num_of_tiles_h,
    layer_21_pw_num_of_tiles_w, tmp_channels, 0,
    layer_21_pw_num_of_weight_groups_for_one_pass,
    1, layer_21_pw_weights_offset, layer_21_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 22, layer_22_dw_depth,
    layer_22_dw_ifm_width, layer_22_dw_ifm_height, layer_22_dw_num_of_tiles_in_d,
    layer_22_dw_num_of_tiles_h, layer_22_dw_num_of_tiles_w,
    layer_22_dw_strides, layer_22_dw_padding_left, layer_22_dw_padding_right, layer_22_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 23, layer_23_pw_depth,
    layer_23_pw_num_fils, layer_23_pw_num_of_tiles_in_d,
    layer_23_pw_num_of_tiles_out_d, layer_23_pw_num_of_tiles_h,
    layer_23_pw_num_of_tiles_w, tmp_channels, 3,
    layer_23_pw_num_of_weight_groups_for_one_pass,
    1, layer_23_pw_weights_offset, layer_23_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 24, layer_24_pw_depth,
    layer_24_pw_num_fils, layer_24_pw_num_of_tiles_in_d,
    layer_24_pw_num_of_tiles_out_d, layer_24_pw_num_of_tiles_h,
    layer_24_pw_num_of_tiles_w, tmp_channels, 0,
    layer_24_pw_num_of_weight_groups_for_one_pass,
    0, layer_24_pw_weights_offset, layer_24_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 25, layer_25_dw_depth,
    layer_25_dw_ifm_width, layer_25_dw_ifm_height, layer_25_dw_num_of_tiles_in_d,
    layer_25_dw_num_of_tiles_h, layer_25_dw_num_of_tiles_w,
    layer_25_dw_strides, layer_25_dw_padding_left, layer_25_dw_padding_right, layer_25_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 26, layer_26_pw_depth,
    layer_26_pw_num_fils, layer_26_pw_num_of_tiles_in_d,
    layer_26_pw_num_of_tiles_out_d, layer_26_pw_num_of_tiles_h,
    layer_26_pw_num_of_tiles_w, tmp_channels, 3,
    layer_26_pw_num_of_weight_groups_for_one_pass,
    0, layer_26_pw_weights_offset, layer_26_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 27, layer_27_pw_depth,
    layer_27_pw_num_fils, layer_27_pw_num_of_tiles_in_d,
    layer_27_pw_num_of_tiles_out_d, layer_27_pw_num_of_tiles_h,
    layer_27_pw_num_of_tiles_w, tmp_channels, 0,
    layer_27_pw_num_of_weight_groups_for_one_pass,
    1, layer_27_pw_weights_offset, layer_27_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 28, layer_28_dw_depth,
    layer_28_dw_ifm_width, layer_28_dw_ifm_height, layer_28_dw_num_of_tiles_in_d,
    layer_28_dw_num_of_tiles_h, layer_28_dw_num_of_tiles_w,
    layer_28_dw_strides, layer_28_dw_padding_left, layer_28_dw_padding_right, layer_28_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 29, layer_29_pw_depth,
    layer_29_pw_num_fils, layer_29_pw_num_of_tiles_in_d,
    layer_29_pw_num_of_tiles_out_d, layer_29_pw_num_of_tiles_h,
    layer_29_pw_num_of_tiles_w, tmp_channels, 1,
    layer_29_pw_num_of_weight_groups_for_one_pass,
    1, layer_29_pw_weights_offset, layer_29_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 30, layer_30_pw_depth,
    layer_30_pw_num_fils, layer_30_pw_num_of_tiles_in_d,
    layer_30_pw_num_of_tiles_out_d, layer_30_pw_num_of_tiles_h,
    layer_30_pw_num_of_tiles_w, tmp_channels, 0,
    layer_30_pw_num_of_weight_groups_for_one_pass,
    0, layer_30_pw_weights_offset, layer_30_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 31, layer_31_dw_depth,
    layer_31_dw_ifm_width, layer_31_dw_ifm_height, layer_31_dw_num_of_tiles_in_d,
    layer_31_dw_num_of_tiles_h, layer_31_dw_num_of_tiles_w,
    layer_31_dw_strides, layer_31_dw_padding_left, layer_31_dw_padding_right, layer_31_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 32, layer_32_pw_depth,
    layer_32_pw_num_fils, layer_32_pw_num_of_tiles_in_d,
    layer_32_pw_num_of_tiles_out_d, layer_32_pw_num_of_tiles_h,
    layer_32_pw_num_of_tiles_w, tmp_channels, 2,
    layer_32_pw_num_of_weight_groups_for_one_pass,
    0, layer_32_pw_weights_offset, layer_32_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 33, layer_33_pw_depth,
    layer_33_pw_num_fils, layer_33_pw_num_of_tiles_in_d,
    layer_33_pw_num_of_tiles_out_d, layer_33_pw_num_of_tiles_h,
    layer_33_pw_num_of_tiles_w, tmp_channels, 0,
    layer_33_pw_num_of_weight_groups_for_one_pass,
    1, layer_33_pw_weights_offset, layer_33_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 34, layer_34_dw_depth,
    layer_34_dw_ifm_width, layer_34_dw_ifm_height, layer_34_dw_num_of_tiles_in_d,
    layer_34_dw_num_of_tiles_h, layer_34_dw_num_of_tiles_w,
    layer_34_dw_strides, layer_34_dw_padding_left, layer_34_dw_padding_right, layer_34_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 35, layer_35_pw_depth,
    layer_35_pw_num_fils, layer_35_pw_num_of_tiles_in_d,
    layer_35_pw_num_of_tiles_out_d, layer_35_pw_num_of_tiles_h,
    layer_35_pw_num_of_tiles_w, tmp_channels, 3,
    layer_35_pw_num_of_weight_groups_for_one_pass,
    1, layer_35_pw_weights_offset, layer_35_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 36, layer_36_pw_depth,
    layer_36_pw_num_fils, layer_36_pw_num_of_tiles_in_d,
    layer_36_pw_num_of_tiles_out_d, layer_36_pw_num_of_tiles_h,
    layer_36_pw_num_of_tiles_w, tmp_channels, 0,
    layer_36_pw_num_of_weight_groups_for_one_pass,
    0, layer_36_pw_weights_offset, layer_36_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 37, layer_37_dw_depth,
    layer_37_dw_ifm_width, layer_37_dw_ifm_height, layer_37_dw_num_of_tiles_in_d,
    layer_37_dw_num_of_tiles_h, layer_37_dw_num_of_tiles_w,
    layer_37_dw_strides, layer_37_dw_padding_left, layer_37_dw_padding_right, layer_37_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 38, layer_38_pw_depth,
    layer_38_pw_num_fils, layer_38_pw_num_of_tiles_in_d,
    layer_38_pw_num_of_tiles_out_d, layer_38_pw_num_of_tiles_h,
    layer_38_pw_num_of_tiles_w, tmp_channels, 1,
    layer_38_pw_num_of_weight_groups_for_one_pass,
    0, layer_38_pw_weights_offset, layer_38_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 39, layer_39_pw_depth,
    layer_39_pw_num_fils, layer_39_pw_num_of_tiles_in_d,
    layer_39_pw_num_of_tiles_out_d, layer_39_pw_num_of_tiles_h,
    layer_39_pw_num_of_tiles_w, tmp_channels, 0,
    layer_39_pw_num_of_weight_groups_for_one_pass,
    1, layer_39_pw_weights_offset, layer_39_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 40, layer_40_dw_depth,
    layer_40_dw_ifm_width, layer_40_dw_ifm_height, layer_40_dw_num_of_tiles_in_d,
    layer_40_dw_num_of_tiles_h, layer_40_dw_num_of_tiles_w,
    layer_40_dw_strides, layer_40_dw_padding_left, layer_40_dw_padding_right, layer_40_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 41, layer_41_pw_depth,
    layer_41_pw_num_fils, layer_41_pw_num_of_tiles_in_d,
    layer_41_pw_num_of_tiles_out_d, layer_41_pw_num_of_tiles_h,
    layer_41_pw_num_of_tiles_w, tmp_channels, 2,
    layer_41_pw_num_of_weight_groups_for_one_pass,
    1, layer_41_pw_weights_offset, layer_41_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 42, layer_42_pw_depth,
    layer_42_pw_num_fils, layer_42_pw_num_of_tiles_in_d,
    layer_42_pw_num_of_tiles_out_d, layer_42_pw_num_of_tiles_h,
    layer_42_pw_num_of_tiles_w, tmp_channels, 0,
    layer_42_pw_num_of_weight_groups_for_one_pass,
    0, layer_42_pw_weights_offset, layer_42_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 43, layer_43_dw_depth,
    layer_43_dw_ifm_width, layer_43_dw_ifm_height, layer_43_dw_num_of_tiles_in_d,
    layer_43_dw_num_of_tiles_h, layer_43_dw_num_of_tiles_w,
    layer_43_dw_strides, layer_43_dw_padding_left, layer_43_dw_padding_right, layer_43_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 44, layer_44_pw_depth,
    layer_44_pw_num_fils, layer_44_pw_num_of_tiles_in_d,
    layer_44_pw_num_of_tiles_out_d, layer_44_pw_num_of_tiles_h,
    layer_44_pw_num_of_tiles_w, tmp_channels, 3,
    layer_44_pw_num_of_weight_groups_for_one_pass,
    0, layer_44_pw_weights_offset, layer_44_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 45, layer_45_pw_depth,
    layer_45_pw_num_fils, layer_45_pw_num_of_tiles_in_d,
    layer_45_pw_num_of_tiles_out_d, layer_45_pw_num_of_tiles_h,
    layer_45_pw_num_of_tiles_w, tmp_channels, 0,
    layer_45_pw_num_of_weight_groups_for_one_pass,
    1, layer_45_pw_weights_offset, layer_45_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 46, layer_46_dw_depth,
    layer_46_dw_ifm_width, layer_46_dw_ifm_height, layer_46_dw_num_of_tiles_in_d,
    layer_46_dw_num_of_tiles_h, layer_46_dw_num_of_tiles_w,
    layer_46_dw_strides, layer_46_dw_padding_left, layer_46_dw_padding_right, layer_46_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 47, layer_47_pw_depth,
    layer_47_pw_num_fils, layer_47_pw_num_of_tiles_in_d,
    layer_47_pw_num_of_tiles_out_d, layer_47_pw_num_of_tiles_h,
    layer_47_pw_num_of_tiles_w, tmp_channels, 1,
    layer_47_pw_num_of_weight_groups_for_one_pass,
    1, layer_47_pw_weights_offset, layer_47_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 48, layer_48_pw_depth,
    layer_48_pw_num_fils, layer_48_pw_num_of_tiles_in_d,
    layer_48_pw_num_of_tiles_out_d, layer_48_pw_num_of_tiles_h,
    layer_48_pw_num_of_tiles_w, tmp_channels, 0,
    layer_48_pw_num_of_weight_groups_for_one_pass,
    0, layer_48_pw_weights_offset, layer_48_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
dw_conv_3x3(seml_dw_weights_3x3, result, channels, 49, layer_49_dw_depth,
    layer_49_dw_ifm_width, layer_49_dw_ifm_height, layer_49_dw_num_of_tiles_in_d,
    layer_49_dw_num_of_tiles_h, layer_49_dw_num_of_tiles_w,
    layer_49_dw_strides, layer_49_dw_padding_left, layer_49_dw_padding_right, layer_49_dw_padding_top,
    1, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 50, layer_50_pw_depth,
    layer_50_pw_num_fils, layer_50_pw_num_of_tiles_in_d,
    layer_50_pw_num_of_tiles_out_d, layer_50_pw_num_of_tiles_h,
    layer_50_pw_num_of_tiles_w, tmp_channels, 0,
    layer_50_pw_num_of_weight_groups_for_one_pass,
    0, layer_50_pw_weights_offset, layer_50_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
pw_conv(off_chip_weights, channels, result, 51, layer_51_pw_depth,
    layer_51_pw_num_fils, layer_51_pw_num_of_tiles_in_d,
    layer_51_pw_num_of_tiles_out_d, layer_51_pw_num_of_tiles_h,
    layer_51_pw_num_of_tiles_w, tmp_channels, 0,
    layer_51_pw_num_of_weight_groups_for_one_pass,
    1, layer_51_pw_weights_offset, layer_51_activation, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
// 	//end_code_generation
		avgpool(channels, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
