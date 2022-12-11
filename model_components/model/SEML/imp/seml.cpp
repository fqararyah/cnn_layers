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
		for(int i=0;i<max_fms_size;i++){
			result2[i] = i % 127;
		}
		//begin_code_generation
fill_fused_scales_and_zero_points(layer_52_fused_scales,fused_scales, 
    layer_52_fused_scales_log_2_shifts, fused_scales_log_2_shifts, layer_52_relu_6_fused_scales,
     relu_6_fused_scales, layer_52_fused_zero_points,
    fused_zero_points, layer_52_pw_num_fils);
pw_conv(off_chip_weights, channels, result2, 52, layer_52_pw_depth,
    layer_52_pw_num_fils, layer_52_pw_num_of_tiles_in_d,
    layer_52_pw_num_of_tiles_out_d, layer_52_pw_num_of_tiles_h,
    layer_52_pw_num_of_tiles_w, tmp_channels, 0,
    layer_52_pw_num_of_weight_groups_for_one_pass,
    1, layer_52_pw_weights_offset, layer_52_relu, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
// 	//end_code_generation
		avgpool(channels, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);
}
