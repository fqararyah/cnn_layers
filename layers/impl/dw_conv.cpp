#include "../headers/dw_conv.h"
#include "../headers/norm_act.h"
#include "../../client/quantization_and_biases.h"
#include "../../tests/test_utils.h"

void dw_fill_channels_buffer_3x3(fms_dt channels[max_fms_size],
		fms_dt channels_tile[dw_tile_d][3][max_dw_input_width], int layer,
		int tile_indx, const int starting_h, const int strides,
		const int number_of_tiles_w, int number_of_tiles_h, int startintg_d,
		const int layer_conv_d, const int layer_ifm_height, int first_time,
		const int padding_top) {
#pragma HLS INLINE off

	const int conv_w = 3;
	const int conv_h = 3;
	int h_offset = starting_h % dw_tile_h;
	int tile_in_h = starting_h / dw_tile_h;
	const int extra_filled_first_time = conv_h - (strides + padding_top);

	if (first_time) {
		//filling the first row if stride is 2 or the second row if stride is 1
		for (int w = 0; w < number_of_tiles_w; w++) {
#pragma HLS PIPELINE
			//fill*******************************************
			for (int i_w = 0; i_w < dw_tile_w; i_w++) {
#pragma HLS UNROLL
				for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
					channels_tile[d][conv_h - strides - 1][w * pw_tile_w + i_w] =
							channels[(tile_indx + w) * dw_tile_size + i_w];
//					cout << (tile_indx + w) * dw_tile_size + i_w << " "
//							<< channels[(tile_indx + w) * dw_tile_size + i_w]
//							<< "\n";
				}
			}
		} //end filling the first row if stride is 2 or the second row if stride is 1
		if (padding_top == 1) {
			for (int w = 0; w < number_of_tiles_w; w++) {
#pragma HLS PIPELINE
				//fill*******************************************
				for (int i_w = 0; i_w < dw_tile_w; i_w++) {
#pragma HLS UNROLL
					for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
						channels_tile[d][0][w * pw_tile_w + i_w] =
								conv_fms_zero_points[layer];
					}
				}
			}
		}
	} else { //if not first row
		for (int w = 0; w < number_of_tiles_w; w++) {
#pragma HLS PIPELINE
			//shift*******************************************
			for (int i_w = 0; i_w < dw_tile_w; i_w++) {
#pragma HLS UNROLL
				for (int h = 0; h < conv_h - 1; h++) { //conv_h - 1 should be conv_h - strides, but replaced with if to avoid variable loop
#pragma HLS UNROLL
					if (h < conv_h - strides) {
						for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
							channels_tile[d][h][w * dw_tile_w + i_w] =
									channels_tile[d][h + strides][w * dw_tile_w
											+ i_w];
						}
					}
				}
			}
		}
	}
	//end shift*******************************************

	for (int w = 0; w < number_of_tiles_w; w++) {
#pragma HLS PIPELINE
		//fill*******************************************
		for (int i_w = 0; i_w < dw_tile_w; i_w++) {
#pragma HLS UNROLL
			for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
				for (int h = 1; h < conv_h; h++) // assuming the padding is on the bottom side when strides = 2
						{ //conv_h = 1 should be h = conv_h - strides, but replaced with if to avoid variable loop
#pragma HLS UNROLL
					const int current_h_offset_from_the_tile_beginning = (h
							- (conv_h - strides) + h_offset
							+ extra_filled_first_time);
					if (tile_in_h * dw_tile_h + current_h_offset_from_the_tile_beginning
									< layer_ifm_height) {
						//if not last row
						if (h >= conv_h - strides) {
							//regular filling: the last row if stride is 1 or the last two rows if stride is 2
							const int starting_indx =
									(tile_indx
											+ (current_h_offset_from_the_tile_beginning
													/ dw_tile_h)
													* number_of_tiles_w + w)
											* dw_tile_size;
							const int in_tile_index = d * dw_tile_hw
									+ (current_h_offset_from_the_tile_beginning
											% dw_tile_w) * dw_tile_h + i_w;
//							if (layer == 32 && starting_h == 0 && w == 3) {
//								cout<<starting_indx <<" = "<<"("<<tile_indx
//												<<"+ (" << current_h_offset_from_the_tile_beginning
//														<<"/"<< dw_tile_h<<")"
//														<<"*" <<number_of_tiles_w <<"+" <<w<<")"
//												<<"*"<< dw_tile_size<<"\n";
//							}
							channels_tile[d][h][w * pw_tile_w + i_w] =
									channels[starting_indx + in_tile_index];
						}
					} else {
						channels_tile[d][h][w * pw_tile_w + i_w] =
								conv_fms_zero_points[layer];
					}
				}
			}
		}
		//cout<<w<<"\n";
//		if (tile_indx == 0 && layer == 32 && starting_h == 0 && (w == 2 || w == 3) ) {
//			cout << "\n********xxx********"<<w<<"\n";
//			for (int i = 0; i < 3; i++) {
//				for (int j = w * dw_tile_w; j < w * dw_tile_w + dw_tile_w; j++) {
//					cout << channels_tile[0][i][j] << " ";
//				}
//				cout << "\n";
//			}
//			cout << "\n********xxx********\n";
//		}
	}

}

void dw_conv_eng3x3(fms_dt channels_tile[dw_tile_d][3][max_dw_input_width],
		const dw_weights_dt weights[max_conv_d][3][3],
		fms_dt result[max_fms_size], int tile_index, const int starting_h,
		int conv_depth, const int num_of_tiles_w, const int layer_ifm_width,
		const int strides, const int padding_left, int layer) {
#pragma HLS INLINE off

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };
	const int current_layer_fused_parameters_offsets =
			layers_fused_parameters_offsets[layer];

	const int h_offset = starting_h % dw_tile_h;
	const int tile_in_h = starting_h / dw_tile_h;
	const int starting_index = tile_index * pw_tile_size;

	const int current_layer_fms_zero_point = conv_fms_zero_points[layer];

	if (padding_left == 1) {
		let_most_conv: for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			dw_pss_dt tmp = current_layer_fms_zero_point
					* (weights[conv_depth + d][0][0]
							+ weights[conv_depth + d][1][0]
							+ weights[conv_depth + d][2][0]);
			for (int c_h = 0; c_h < 3; c_h++) {
#pragma HLS UNROLL
				for (int c_w = padding_left; c_w < 3; c_w++) {
#pragma HLS UNROLL
					tmp += weights[conv_depth + d][c_h][c_w]
							* channels_tile[d][c_h][c_w - padding_left];
				}
			}
			normalization.fused_scales =
					fused_scales[current_layer_fused_parameters_offsets
							+ conv_depth + d];
			normalization.fused_zero_point =
					fused_zero_points[current_layer_fused_parameters_offsets
							+ conv_depth + d];
			normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
			normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];

			fms_dt scaled_val = dw_relu_norm(tmp, normalization, 6);
			const int in_tile_index = d * dw_tile_hw + h_offset * dw_tile_w;
			result[starting_index + in_tile_index] = scaled_val;
		}
	}
	//***
	dw_conv_eng3x3_g_main: for (int w = 0; w < num_of_tiles_w; w++) {
#pragma HLS PIPELINE
#pragma HLS dependence variable = result inter false
#pragma HLS dependence variable = result intra false
		int tile_base_offset = w * dw_tile_w * strides;
		dw_conv_eng3x3_g_pipe: for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			normalization.fused_scales =
					fused_scales[current_layer_fused_parameters_offsets
							+ conv_depth + d];
			normalization.fused_zero_point =
					fused_zero_points[current_layer_fused_parameters_offsets
							+ conv_depth + d];
			normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
			normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];
			for (int i_w = 0; i_w < dw_tile_w; i_w++) {
#pragma HLS UNROLL
				dw_pss_dt tmp = 0;
				for (int c_w = 0; c_w < 3; c_w++) {
#pragma HLS UNROLL
					for (int c_h = 0; c_h < 3; c_h++) {
#pragma HLS UNROLL
						const int absolute_i_w_index = tile_base_offset + i_w * strides + c_w;
						if (absolute_i_w_index < layer_ifm_width) { // within ifm or padding right check
							tmp += weights[conv_depth + d][c_h][c_w]
									* channels_tile[d][c_h][absolute_i_w_index];
//							cout
//									<< channels_tile[d][c_h][tile_base_offset
//											+ i_w + c_w];
						} else {
							tmp += weights[conv_depth + d][c_h][c_w]
									* current_layer_fms_zero_point;
						}
					}
					//cout<<"\n";
				}
				fms_dt scaled_val = dw_relu_norm(tmp, normalization, 6);

				const int current_starting_index = (tile_index + w
						+ ((padding_left + i_w) / dw_tile_w)) * dw_tile_size;

				const int in_tile_index = d * dw_tile_hw + h_offset * dw_tile_w
						+ ((padding_left + i_w) % dw_tile_w);

				//cout<<num_of_tiles_w<<"\n";
//				if (layer == 32 && conv_depth == 0 && starting_h == 0 && w == 3) {
//					cout << "\n************\n";
//					cout << tmp << "***results_tile[t_d][t_h][t_w]***\n";
//					cout << normalization.fused_zero_point
//							<< " ***fused_zero_point***\n";
//					cout << normalization.fused_scales
//							<< " ****fused_scales**\n";
//					cout << normalization.ofm_zero_point
//							<< " ***ofm_zero_point***\n";
//					cout << normalization.ofm_scale << " ***ofm_scale***\n";
//					cout << scaled_val << "****scaled_val***\n";
//					cout << scaled_val << " " << starting_index + in_tile_index
//							<< "*****ll****";
//					cout << scaled_val << " " << starting_index + in_tile_index
//							<< " " << w << " "
//							<< (tile_index + w
//									+ ((padding_left + i_w) / dw_tile_w))
//									* dw_tile_size << "\n";
//				}
				result[current_starting_index + in_tile_index] = scaled_val;
			}
		}
	}
}

void dw_conv_3x3(dw_weights_dt weights[max_conv_d][max_conv_h][max_conv_w],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int layer_ifm_width,
		const int layer_ifm_height, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides,
		const int padding_left, const int padding_top, const int direction) {
#pragma HLS INLINE off
	fms_dt channels_tile_1[dw_tile_d][3][max_dw_input_width];
//	fms_dt channels_tile_2[dw_tile_d][3][max_dw_input_width];
	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

#pragma HLS ARRAY_PARTITION variable = channels_tile_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_tile_1 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_tile_1 cyclic factor=pw_tile_w dim = 3
//#pragma HLS ARRAY_PARTITION variable = channels_tile_2 complete dim = 1
//#pragma HLS ARRAY_PARTITION variable = channels_tile_2 complete dim = 2
//#pragma HLS ARRAY_PARTITION variable = channels_tile_2 cyclic factor=pw_tile_w dim = 3

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;
	const int num_of_input_tiles_w = layer_ifm_width % dw_tile_w == 0 ? (layer_ifm_width / dw_tile_w): 1 + (layer_ifm_width / dw_tile_w);
	const int num_of_input_tiles_h = layer_ifm_height % dw_tile_h == 0 ? (layer_ifm_height / dw_tile_h): 1 + (layer_ifm_height / dw_tile_h);
	const int num_of_input_tiles_hw = num_of_input_tiles_w * num_of_input_tiles_h;
	//int odd_even = 0;
	dw_conv_itd_loop: for (int t_in_d = 0; t_in_d < num_of_tiles_d; t_in_d++) {

		dw_conv_ith_loop: for (int t_in_h = 0; t_in_h < layer_ifm_height;
				t_in_h++) {

			int fill_tile_index = t_in_d * num_of_input_tiles_hw
					+ (t_in_h * strides / dw_tile_h) * num_of_input_tiles_w;
			int conv_tile_index = t_in_d * num_of_tiles_hw
					+ (t_in_h / dw_tile_h) * num_of_tiles_w;

//			if (direction) {
//				dw_fill_channels_buffer_3x3(result, channels_tile_1, tile_index,
//						layer, t_in_h, strides, num_of_tiles_w,
//						num_of_tiles_h, t_in_d * dw_tile_d, layer_conv_d,
//						layer_ifm_width, 1);
//			} else {
//				dw_fill_channels_buffer_3x3(channels, channels_tile_1, layer,
//						tile_index, t_in_h, strides, num_of_tiles_w,
//						num_of_tiles_h, t_in_d * dw_tile_d, layer_conv_d,
//						layer_ifm_width, 1);
//			}

//			tile_index = t_in_d * num_of_tiles_hw
//					+ (t_in_h / dw_tile_h) * num_of_tiles_w;
//			if (odd_even) {
			if (direction) {
				dw_fill_channels_buffer_3x3(result, channels_tile_1, layer,
						fill_tile_index, t_in_h * strides, strides,
						num_of_input_tiles_w, num_of_input_tiles_h,
						t_in_d * dw_tile_d, layer_conv_d, layer_ifm_height,
						t_in_h == 0, padding_top);
				dw_conv_eng3x3(channels_tile_1, weights,
						channels, //channels_tile_2
						conv_tile_index, t_in_h, t_in_d * dw_tile_d,
						num_of_tiles_w, layer_ifm_width, strides, padding_left,
						layer);
			} else {
				dw_fill_channels_buffer_3x3(channels, channels_tile_1, layer,
						fill_tile_index, t_in_h * strides, strides,
						num_of_input_tiles_w, num_of_input_tiles_h,
						t_in_d * dw_tile_d, layer_conv_d, layer_ifm_height,
						t_in_h == 0, padding_top);
				dw_conv_eng3x3(channels_tile_1, weights, result,
						conv_tile_index, //channels_tile_2
						t_in_h, t_in_d * dw_tile_d, num_of_tiles_w, layer_ifm_width,
						strides, padding_left, layer);
			}
//			} else {
//				if (direction) {
//					dw_conv_eng3x3(channels_tile_1, weights, channels,
//							tile_index, t_in_h, t_in_d * dw_tile_d,
//							num_of_tiles_w, layer_ifm_width, strides, padding_left,
//							layer);
//					dw_fill_channels_buffer_3x3(result, channels_tile_2, layer,
//							tile_index, t_in_h, strides, num_of_tiles_w,
//							num_of_tiles_h, t_in_d * dw_tile_d, layer_conv_d,
//							layer_ifm_width, 0);
//				} else {
//					dw_conv_eng3x3(channels_tile_1, weights, result, tile_index,
//							t_in_h, t_in_d * dw_tile_d, num_of_tiles_w,
//							layer_ifm_width, strides, padding_left, layer);
//					dw_fill_channels_buffer_3x3(channels, channels_tile_2,
//							layer, tile_index, t_in_h, strides,
//							num_of_tiles_w, num_of_tiles_h, t_in_d * dw_tile_d,
//							layer_conv_d, layer_ifm_width, 0);
//				}
//			}
			//odd_even = 1 - odd_even;
		} //end dw_conv_ith_loop
	}

}
//*********************************************************************************************

void dw_fill_channels_buffer_5x5(fms_dt channels[max_fms_size],
		fms_dt channels_tile[dw_tile_d][5][dw_tile_w], int tile_indx,
		int current_w, const int strides, const int number_of_tiles_w,
		int startintg_d, const int layer_conv_d, int first_time) {
#pragma HLS INLINE off
	const int starting_indx = tile_indx * pw_tile_size;
	const int channels_starting_offset = current_w % dw_tile_w;

	if (!first_time) {
		for (int t_h = 0; t_h < 5; t_h++) {
#pragma HLS PIPELINE
			for (int i = 0; i < 5 - strides; i++) {
#pragma HLS UNROLL
				for (int t_d = 0; t_d < dw_tile_d; t_d++) {
#pragma HLS UNROLL
					channels_tile[t_d][t_h][i] =
							channels_tile[t_d][t_h][dw_tile_w - (5 - strides)
									+ i];
				}
			}
		}
	}
	int left_offset = (5 - strides);
	if (first_time) {
		if (strides == 1) {
			left_offset -= 1;
		}
		for (int t_d = 0; t_d < dw_tile_d; t_d++) {
#pragma HLS UNROLL
			channels_tile[t_d][0][0] = 0;
			channels_tile[t_d][1][0] = 0;
			channels_tile[t_d][2][0] = 0;
		}
	}
	for (int h = 0; h < 5; h++) {
#pragma HLS PIPELINE
		//#pragma HLS dependence variable = channels inter false
		//#pragma HLS dependence variable = channels intra false
		for (int t_w = 0; t_w < dw_tile_w - (5 - strides + first_time); t_w++) {
#pragma HLS UNROLL
			if (current_w + t_w < number_of_tiles_w * dw_tile_w) {
				const int current_tile_index = starting_indx
						+ h * number_of_tiles_w + channels_starting_offset
						+ t_w;
				for (int t_d = 0; t_d < dw_tile_d; t_d++) {
#pragma HLS UNROLL
					if ((t_w + left_offset + 1) % 8 != 0
							&& startintg_d + t_d < layer_conv_d) {
						channels_tile[t_d][h][t_w + left_offset] =
								channels[current_tile_index * dw_tile_size
										+ t_d * dw_tile_hw + t_w];
					}
				}
			}
		}
	}
}

void dw_conv_eng5x5(fms_dt channels_tile[dw_tile_d][5][dw_tile_w],
		dw_weights_dt weights[max_conv_d][5][5], fms_dt result[max_fms_size],
		int tile_index, int conv_depth, const int layer_conv_d,
		const int strides, int starting_w) {
#pragma HLS INLINE off

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };
	const int result_base_index = tile_index * dw_tile_size;
	const int result_base_offset = starting_w % dw_tile_w;
	dw_conv_eng3x3_g_main: for (int w = 0; w < dw_tile_w - (5 - 1); w +=
			strides) {
		const int current_result_offset = result_base_index + result_base_offset
				+ w / strides;
//#pragma HLS dependence variable = result inter false
//#pragma HLS dependence variable = result intra false

		dw_conv_eng3x3_g_pipe: for (int c_h = 0; c_h < 5; c_h++) {
#pragma HLS PIPELINE
			for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
				if (conv_depth + d < layer_conv_d) {
					dw_pss_dt tmp = 0;
					for (int c_w = 0; c_w < 5; c_w++) {
#pragma HLS UNROLL
						tmp += weights[conv_depth + d][c_h][c_w]
								* channels_tile[c_h][w + c_w];
					}
					if (c_h == 0) {
						result[current_result_offset + d * dw_tile_hw] = tmp;
					}
					if (c_h > 0) {
						result[current_result_offset + d * dw_tile_hw] += tmp;
					}
				}
			}
		}
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			fms_dt scaled_val = dw_relu_norm(
					result[current_result_offset + d * dw_tile_hw],
					normalization, 6);
			result[current_result_offset + d * dw_tile_hw] = scaled_val;
		}
	}
}

void dw_conv_5x5(dw_weights_dt weights[max_conv_d][5][5],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides) {
#pragma HLS INLINE off
	fms_dt channels_tile_1[dw_tile_d][5][dw_tile_w];
	fms_dt channels_tile_2[dw_tile_d][5][dw_tile_w];
	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

#pragma HLS ARRAY_PARTITION variable = channels_tile_1 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = channels_tile_2 complete dim = 0

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;
	int odd_even = 0;
	dw_conv_itd_loop: for (int t_in_d = 0; t_in_d < num_of_tiles_d; t_in_d++) {
		dw_conv_ith_loop: for (int t_in_h = 0;
				t_in_h < num_of_tiles_h * dw_tile_h; t_in_h += strides) {
			int tile_index = t_in_d * num_of_tiles_hw + t_in_h * num_of_tiles_w;
			dw_fill_channels_buffer_5x5(channels, channels_tile_1, tile_index,
					0, strides, num_of_tiles_w, t_in_d * dw_tile_d,
					layer_conv_d, 1);
			dw_conv_itw_loop: for (int t_in_w = 5;
					t_in_w < num_of_tiles_w * dw_tile_w;
					t_in_w += (dw_tile_w - (5 - strides))) {
#pragma HLS UNROLL factor = 1
				tile_index = t_in_d * num_of_tiles_hw
						+ (t_in_h / dw_tile_h) * num_of_tiles_w
						+ (t_in_w / dw_tile_w);
				if (odd_even) {
					dw_conv_eng5x5(channels_tile_2, weights, result, tile_index,
							t_in_d * dw_tile_d, layer_conv_d, strides, t_in_w);
					dw_fill_channels_buffer_5x5(channels, channels_tile_1,
							tile_index, t_in_w, strides, num_of_tiles_w,
							t_in_d * dw_tile_d, layer_conv_d, 0);
				} else {
					dw_conv_eng5x5(channels_tile_1, weights, result, tile_index,
							t_in_d * dw_tile_d, layer_conv_d, strides, t_in_w);
					dw_fill_channels_buffer_5x5(channels, channels_tile_2,
							tile_index, t_in_w, strides, num_of_tiles_w,
							t_in_d * dw_tile_d, layer_conv_d, 0);
				}
				odd_even = 1 - odd_even;
			}
		}
	}
}

void dw_fill_channels_buffer_7x7(fms_dt channels[max_fms_size],
		fms_dt channels_tile[dw_tile_d][7][dw_tile_w], int tile_indx,
		int current_w, const int strides, const int number_of_tiles_w,
		int number_of_tiles_h, int startintg_d, const int layer_conv_d,
		int first_time) {
#pragma HLS INLINE off
	const int starting_indx = tile_indx * pw_tile_size;
	const int channels_starting_offset = current_w % dw_tile_w;

	if (!first_time) {
		for (int t_h = 0; t_h < 7; t_h++) {
#pragma HLS PIPELINE
			for (int i = 0; i < strides; i++) {
#pragma HLS UNROLL
				for (int t_d = 0; t_d < dw_tile_d; t_d++) {
#pragma HLS UNROLL
					channels_tile[t_d][t_h][i] =
							channels_tile[t_d][t_h][dw_tile_w - strides + i];
				}
			}
		}
	}
	int left_offset = (7 - strides);
	if (first_time) {
		if (strides == 1) {
			left_offset -= 1;
		}
		for (int t_d = 0; t_d < dw_tile_d; t_d++) {
#pragma HLS UNROLL
			channels_tile[t_d][0][0] = 0;
			channels_tile[t_d][1][0] = 0;
			channels_tile[t_d][2][0] = 0;
		}
	}
	for (int h = 0; h < 7; h++) {
#pragma HLS PIPELINE
		//#pragma HLS dependence variable = channels inter false
		//#pragma HLS dependence variable = channels intra false
		for (int t_w = 0; t_w < dw_tile_w - (7 - strides + first_time); t_w++) {
#pragma HLS UNROLL
			if (current_w + t_w < number_of_tiles_w * dw_tile_w) {
				const int current_tile_index = starting_indx
						+ h * number_of_tiles_w + channels_starting_offset
						+ t_w;
				for (int t_d = 0; t_d < dw_tile_d; t_d++) {
#pragma HLS UNROLL
					if ((t_w + left_offset + 1) % 8 != 0
							&& startintg_d + t_d < layer_conv_d) {
						channels_tile[t_d][h][t_w + left_offset] =
								channels[current_tile_index * dw_tile_size
										+ t_d * dw_tile_hw + t_w];
					}
				}
			}
		}
	}
}

void dw_conv_eng7x7(fms_dt channels_tile[dw_tile_d][7][dw_tile_w],
		dw_weights_dt weights[max_conv_d][7][7], fms_dt result[max_fms_size],
		int tile_index, int conv_depth, const int layer_conv_d,
		const int strides, int starting_w) {
#pragma HLS INLINE off

	fms_quantization_scheme normalization = { 0, 0, 0, 0 };
	const int result_base_index = tile_index * dw_tile_size;
	const int result_base_offset = starting_w % dw_tile_w;
	dw_conv_eng3x3_g_main: for (int w = 0; w < dw_tile_w - (7 - 1); w +=
			strides) {
		const int current_result_offset = result_base_index + result_base_offset
				+ w / strides;
//#pragma HLS dependence variable = result inter false
//#pragma HLS dependence variable = result intra false

		dw_conv_eng3x3_g_pipe: for (int c_h = 0; c_h < 7; c_h++) {
#pragma HLS PIPELINE
			for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
				if (conv_depth + d < layer_conv_d) {
					dw_pss_dt tmp = 0;
					for (int c_w = 0; c_w < 7; c_w++) {
#pragma HLS UNROLL
						tmp += weights[conv_depth + d][c_h][c_w]
								* channels_tile[c_h][w + c_w];
					}
					if (c_h == 0) {
						result[current_result_offset + d * dw_tile_hw] = tmp;
					}
					if (c_h > 0) {
						result[current_result_offset + d * dw_tile_hw] += tmp;
					}
				}
			}
		}
		for (int d = 0; d < dw_tile_d; d++) {
#pragma HLS UNROLL
			fms_dt scaled_val = dw_relu_norm(
					result[current_result_offset + d * dw_tile_hw],
					normalization, 6);
			result[current_result_offset + d * dw_tile_hw] = scaled_val;
		}
	}
}

void dw_conv_7x7(dw_weights_dt weights[max_conv_d][7][7],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides) {
#pragma HLS INLINE off
	fms_dt channels_tile_1[dw_tile_d][7][dw_tile_w];
	fms_dt channels_tile_2[dw_tile_d][7][dw_tile_w];
	fms_quantization_scheme normalization = { 0, 0, 0, 0 };

#pragma HLS ARRAY_PARTITION variable = channels_tile_1 complete dim = 0
#pragma HLS ARRAY_PARTITION variable = channels_tile_2 complete dim = 0

	const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;
	int odd_even = 0;
	dw_conv_itd_loop: for (int t_in_d = 0; t_in_d < num_of_tiles_d; t_in_d++) {
		dw_conv_ith_loop: for (int t_in_h = 0;
				t_in_h < num_of_tiles_h * dw_tile_h; t_in_h += strides) {
			int tile_index = t_in_d * num_of_tiles_hw + t_in_h * num_of_tiles_w;
			dw_fill_channels_buffer_7x7(channels, channels_tile_1, tile_index,
					0, strides, num_of_tiles_w, num_of_tiles_h,
					t_in_d * dw_tile_d, layer_conv_d, 1);
			dw_conv_itw_loop: for (int t_in_w = 5;
					t_in_w < num_of_tiles_w * dw_tile_w;
					t_in_w += (dw_tile_w - (7 - strides))) {
#pragma HLS UNROLL factor = 1
				tile_index = t_in_d * num_of_tiles_hw
						+ (t_in_h / dw_tile_h) * num_of_tiles_w
						+ (t_in_w / dw_tile_w);
				if (odd_even) {
					dw_conv_eng7x7(channels_tile_2, weights, result, tile_index,
							t_in_d * dw_tile_d, layer_conv_d, strides, t_in_w);
					dw_fill_channels_buffer_7x7(channels, channels_tile_1,
							tile_index, t_in_w, strides, num_of_tiles_w,
							num_of_tiles_h, t_in_d * dw_tile_d, layer_conv_d,
							0);
				} else {
					dw_conv_eng7x7(channels_tile_1, weights, result, tile_index,
							t_in_d * dw_tile_d, layer_conv_d, strides, t_in_w);
					dw_fill_channels_buffer_7x7(channels, channels_tile_2,
							tile_index, t_in_w, strides, num_of_tiles_w,
							num_of_tiles_h, t_in_d * dw_tile_d, layer_conv_d,
							0);
				}
				odd_even = 1 - odd_even;
			}
		}
	}
}
