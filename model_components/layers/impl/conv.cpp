#include "../headers/layers_imp_common_includes.h"
#include "../headers/conv.h"
#include "../headers/pw_conv.h"

void fill_channels_buffer_0(
		fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
		fms_dt channels_tile[input_image_depth][layer_0_filter_dim][input_image_width],
		int starting_h) {
//
//	const fms_dt current_layer_zero_point = conv_fms_zero_points[0];
//	const int filled_first_time = layer_0_filter_dim - layer_0_strides;
//	if (starting_h == 0) // first time
//			{
//		for (int d = 0; d < input_image_depth; d++) {
//			for (int h = 0; h < layer_0_filter_dim - layer_0_strides; h++) {
//				for (int w = 0; w < input_image_width; w++) {
//					channels_tile[d][h][w] = channels[d][h][w];
//				}
//			}
//		}
//	} else // shift
//	{
//		for (int d = 0; d < input_image_depth; d++) {
//			for (int h = 0; h < layer_0_filter_dim - layer_0_strides; h++) {
//				for (int w = 0; w < input_image_width; w++) {
//					channels_tile[d][h][w] =
//							channels_tile[d][h + layer_0_strides][w];
//				}
//			}
//		}
//	}
//
//	for (int d = 0; d < input_image_depth; d++) {
//		const int start_filling_h_offset = layer_0_filter_dim - layer_0_strides;
//		for (int h = start_filling_h_offset; h < layer_0_filter_dim; h++) {
//			for (int w = 0; w < input_image_width; w++) {
//				if (h + starting_h * layer_0_strides + filled_first_time
//						- start_filling_h_offset < input_image_height) {
//					channels_tile[d][h][w] = channels[d][h
//							+ starting_h * layer_0_strides + filled_first_time
//							- start_filling_h_offset][w];
//				} else {
//					channels_tile[d][h][w] = current_layer_zero_point;
//				}
//			}
//		}
//	}
}

// Note that this implementation of layer_0 is not not very optimized
void layer_0_conv_engine(
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		fms_dt channels_tile[input_image_depth][layer_0_filter_dim][input_image_width],
		fms_dt results[max_fms_size], int starting_h,  fused_scales_dt fused_scales[], relu_6_fused_scales_dt relu_6_fused_scales[], biases_dt fused_zero_points[]) {

	const biases_dt current_layer_zero_point = conv_fms_zero_points[0];
	for (int f = 0; f < layer_0_num_fils; f++) {
		fms_quantization_scheme normalization = { 0, 0, 0, 0 };
		normalization.ofm_zero_point = conv_fms_zero_points[2];
		normalization.ofm_scale_rec = conv_fms_scales_rec[2];
		normalization.ofm_scale = conv_fms_scales[2];
		normalization.fused_zero_point = fused_zero_points[f];
		normalization.fused_scales = fused_scales[f];
		normalization.relu_6_fused_scale = relu_6_fused_scales[f];
		for (int w = 0; w < layer_0_ofm_width; w++) {
			pss_dt tmp = 0;
			for (int d = 0; d < layer_0_depth; d++) {
				for (int c_h = 0; c_h < layer_0_filter_dim; c_h++) {
					for (int c_w = 0; c_w < layer_0_filter_dim; c_w++) {
						if (w * layer_0_strides + c_w < layer_0_ifm_width) {
							tmp += weights_0[f][d][c_h][c_w]
									* channels_tile[d][c_h][w * layer_0_strides
											+ c_w];
						} else {
							tmp += weights_0[f][d][c_h][c_w]
									* current_layer_zero_point;
						}
//						if (starting_h == 111 && w == 0 && f ==8) {
//							cout << weights_0[f][d][c_h][c_w] << "*"
//									<< channels_tile[d][c_h][w * layer_0_strides
//											+ c_w] << "+";
//						}
					}
//					if (starting_h == 111 && w == 0 && f ==8)
//						cout << "\n";
				}
//				if (starting_h == 111 && w == 0 && f ==8)
//					cout << "*********\n";
			}
			const int tile_in_d = f / pw_tile_d;
			const int tile_in_h = starting_h / pw_tile_h;
			const int tile_in_w = w / pw_tile_w;
			const int tile_index = tile_in_d
					* (layer_0_num_of_tiles_h * layer_0_num_of_tiles_w)
					+ tile_in_h * layer_0_num_of_tiles_w + tile_in_w;

			const int in_tile_d = f % pw_tile_d;
			const int in_tile_h = starting_h % pw_tile_h;
			const int in_tile_w = w % pw_tile_w;
			const int in_tile_index = in_tile_d * pw_tile_hw
					+ in_tile_h * pw_tile_w + in_tile_w;
//			if (starting_h == 111 && w == 0 && f ==8) {
//				cout << tmp << " " << conv_relu_norm(tmp, normalization, 6)
//						<< "\n";
//			}
			results[tile_index * pw_tile_size + in_tile_index] = conv_relu_norm(
					tmp, normalization, 6);
		}
	}
}

void layer_0_3x3(
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		fms_grp_dt channels[input_image_depth*input_image_height*input_image_width / input_image_group_items],
		fms_dt result[max_fms_size], fused_scales_dt fused_scales[], relu_6_fused_scales_dt relu_6_fused_scales[], biases_dt fused_zero_points[]) {
	fms_dt channels_tile[layer_0_depth][layer_0_filter_dim][layer_0_ifm_width];
	for (int h = 0; h < layer_0_ofm_height; h++) {
		fill_channels_buffer_0(channels, channels_tile, h);
		layer_0_conv_engine(weights_0, channels_tile, result, h,  fused_scales, relu_6_fused_scales, fused_zero_points);
	}
}