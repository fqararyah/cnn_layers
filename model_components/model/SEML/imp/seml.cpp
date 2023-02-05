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
dw_conv_3x3(seml_dw_weights_3x3, channels, result, 7, layer_7_dw_depth,
    layer_7_dw_ifm_width, layer_7_dw_ifm_height, layer_7_dw_num_of_tiles_in_d,
    layer_7_dw_num_of_tiles_h, layer_7_dw_num_of_tiles_w,
    layer_7_dw_strides, layer_7_dw_padding_left, layer_7_dw_padding_right, layer_7_dw_padding_top,
    0, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
// 	//end_code_generation
		avgpool(channels, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
