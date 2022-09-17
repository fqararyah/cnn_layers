#include "seml.hpp"

void top_func(fms_dt input_image[input_image_depth][input_image_height][input_image_width],
		weights_grp_dt off_chip_weights[all_pw_weights], int &result_o) {

	fms_dt channels[max_fms_size];
	fms_dt result[max_fms_size] = { 0 };
	fms_dt result2[max_fms_size] = { 0 };
	fms_dt tmp_channels[max_tmp_fms_size];
	fms_dt tmp_channels_2[max_tmp_fms_size];

#pragma HLS ARRAY_PARTITION variable = channels type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = tmp_channels type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = tmp_channels_2 type = cyclic factor = main_buffers_partitining_factor
#pragma HLS ARRAY_PARTITION variable = result2 type = cyclic factor = main_buffers_partitining_factor

	layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size];
	dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w];
	fill_layer_0_weights(weights_0);


	fms_dt fc_input[fc_layer_input_size];
	//fc_out_dt fc_output[fc_cols];

	if (perf_test_run) {
//		if (DF)
//		{
//			mobilenet_v2_pipeline_7(channels_df, channels);
//			// 		pw_conv(layer_1_weights_m, channels, result, 1, layer_s_conv_d[1],
//			// 				layer_s_num_fils[1], pw_layer_s_num_of_tiles_in_d[1],
//			// 				pw_layer_s_num_of_tiles_d_out[1], pw_layer_s_num_of_tiles_h[1],
//			// 				pw_layer_s_num_of_tiles_w[1], tmp_channels, read_write);
//			dw_conv_3x3_g(dw_weights_1, result2, channels, 1,
//						  layer_1_dw_depth, layer_1_dw_ifm_width, layer_1_dw_num_of_tiles_in_d,
//						  layer_1_dw_num_of_tiles_h, layer_1_dw_num_of_tiles_w,
//						  layer_1_dw_strides, layer_1_dw_padding_left);
//		}
//		else
//		{
//			if (start_with_pw)
//			{
//				layer_0_using_pw(weights_0, channels_df, result2, 0, 3, 32, 1,
//								 int(
//									 0.9 + ((float)layer_0_num_fils) / pw_conv_parallelism_out),
//								 224,
//								 224 / 7);
//				pw_conv(layer_1_weights_m, result2, channels, 1,
//						layer_1_pw_depth, layer_1_pw_num_fils,
//						layer_1_pw_num_of_tiles_in_d,
//						layer_1_pw_num_of_tiles_out_d,
//						layer_1_pw_num_of_tiles_h, layer_1_pw_num_of_tiles_w,
//						tmp_channels, read_write,
//						layer_1_pw_num_of_weight_groups_in_depth);
//				dw_conv_3x3_g(dw_weights_1, channels, result2, 1,
//							  layer_1_dw_depth, layer_1_dw_ifm_width, layer_1_dw_num_of_tiles_in_d,
//							  layer_1_dw_num_of_tiles_h, layer_1_dw_num_of_tiles_w,
//							  layer_1_dw_strides, layer_1_dw_padding_left);
//			}
//
//			if (!start_with_pw)
//			{
//				dw_conv_3x3_g(dw_weights_1, channels, result2, 1,
//							  layer_1_dw_depth,layer_1_dw_ifm_width, layer_1_dw_num_of_tiles_in_d,
//							  layer_1_dw_num_of_tiles_h, layer_1_dw_num_of_tiles_w,
//							  layer_1_dw_strides, layer_1_dw_padding_left);
//				pw_conv(layer_2_weights_m, result2, channels, 2,
//						layer_2_pw_depth, layer_2_pw_num_fils,
//						layer_2_pw_num_of_tiles_in_d,
//						layer_2_pw_num_of_tiles_out_d,
//						layer_2_pw_num_of_tiles_h, layer_2_pw_num_of_tiles_w,
//						tmp_channels, read_write,
//						layer_1_pw_num_of_weight_groups_in_depth);
//			}
//		}
	} else {
		seml(input_image, off_chip_weights, channels, result, result2, tmp_channels, tmp_channels_2,
		weights_0, dw_weights_buffer, fc_input);
	}

	int result_to_return = 0;
	for (int i = 0; i < max_fms_size; i++) {
#pragma HLS PIPELINE OFF
		result_to_return += (int) (channels[i] + result[i] + result2[i]);
	}
	result_o = result_to_return;
}
