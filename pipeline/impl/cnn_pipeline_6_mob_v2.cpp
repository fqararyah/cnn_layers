#include "../headers/pipeline.h"
#include <iostream>
#include <math.h>

using namespace std;

void _6_layer_0_3x3_conv(
		fms_dt channels_buffer[input_image_depth][layer_0_filter_dim
				+ (_6_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width],
		const layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		fms_dt result[layer_2_dw_depth][_6_stages_layer_0_rows_at_once][layer_2_dw_ifm_width]) {
#pragma HLS INLINE off

	fms_dt intermediate_channels_buffer[input_image_depth][layer_0_filter_dim
			+ (_6_stages_layer_0_rows_at_once - 1) * layer_0_strides][layer_2_dw_filter_size] =
			{ 0 };
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

	const int offset_left = layer_0_filter_dim - layer_0_padding_left;
	const fms_dt current_layer_zero_point = conv_fms_zero_points[0];

	// fill the intermediate_channels_buffer
	for (int d = 0; d < input_image_depth; d++) {
		for (int h = 0;
				h
						< layer_0_filter_dim
								+ (_6_stages_layer_0_rows_at_once - 1)
										* layer_0_strides; h++) {
			for (int w = 0; w < layer_0_filter_dim - layer_0_padding_left;
					w++) {
				intermediate_channels_buffer[d][h][w + layer_0_padding_left] =
						channels_buffer[d][h][w];
			}
		}
	}
	// end fill the intermediate_channels_buffer

	layer_0_ofms: for (int o_o_d = 0;
			o_o_d < layer_0_num_fils / sesl_layer_0_parallelism_ofms; o_o_d++) {
		// outer filters loop
		int o_o_d_offset = o_o_d * sesl_layer_0_parallelism_ofms; // for indexing in depth

		layer_0_pipeline: for (int w = 0; w < input_image_width; w +=
				layer_0_strides) {
#pragma HLS PIPELINE
			const int starting_fill_index = w + offset_left;
			// FMs width loop
			for (int row = 0; row <= _6_stages_layer_0_rows_at_once; row++) {
#pragma HLS UNROLL
				const int conv_start_h = row * layer_0_strides;
				layer_0_parallelized_ofms: for (int o_d = 0;
						o_d < sesl_layer_0_parallelism_ofms; o_d++) {
					first_conv_pss_dt tmp = 0;
#pragma HLS UNROLL
					// parallelized filters loop
					layer_0_d_loops: for (int d = 0; d < input_image_depth;
							d++) {
#pragma HLS UNROLL
						// parallelized depth loop
						layer_0_ch: for (int h = 0; h < layer_0_filter_dim;
								h++) {
#pragma HLS UNROLL
							// conv height loop
							layer_0_cw: for (int c_w = 0;
									c_w < layer_0_filter_dim; c_w++) {
#pragma HLS UNROLL
								// conv width loop
								tmp +=
										intermediate_channels_buffer[d][conv_start_h
												+ h][c_w]
												* weights[o_o_d_offset + o_d][d][h][c_w];
//								if (o_o_d == 0 && o_d == 0 && w == 0) {
//									cout
//											<< intermediate_channels_buffer[d][row
//													+ h][c_w] << " * "
//											<< weights[o_o_d_offset + o_d][d][h][c_w]
//											<< "\n";
//								}
							}
						}
					}
					fms_quantization_scheme normalization = { 0, 0, 0, 0 };
					normalization.ofm_zero_point = conv_fms_zero_points[2];
					normalization.ofm_scale = conv_fms_scales[2];
					normalization.fused_zero_point = fused_zero_points[o_o_d
							* sesl_layer_0_parallelism_ofms + o_d];
					normalization.fused_scales = fused_scales[o_o_d
							* sesl_layer_0_parallelism_ofms + o_d];
					result[o_o_d_offset + o_d][row][w / layer_0_strides] =
							conv_relu_norm(tmp, normalization, 6);
//					if (o_o_d == 0 && o_d == 0 && w == 0) {
//						cout << tmp << " "
//								<< conv_relu_norm(tmp, normalization, 6)
//								<< "\n";
//					}
				}
			}
			// shift and fill the intermediate_channels_buffer
			if (w < input_image_width - layer_0_strides) {
				for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
					for (int c_h = 0;
							c_h
									< layer_0_filter_dim
											+ (_6_stages_layer_0_rows_at_once
													- 1) * layer_0_strides;
							c_h++) {
#pragma HLS UNROLL
						for (int c_w = 0;
								c_w < layer_0_filter_dim - layer_0_strides;
								c_w++) {
#pragma HLS UNROLL
							intermediate_channels_buffer[d][c_h][c_w] =
									intermediate_channels_buffer[d][c_h][c_w
											+ layer_0_strides];
						}
						for (int c_w = layer_0_filter_dim - layer_0_strides;
								c_w < layer_0_filter_dim; c_w++) {
#pragma HLS UNROLL
							if (c_w - (layer_0_filter_dim - layer_0_strides)
									+ starting_fill_index < layer_0_ifm_width) {
								intermediate_channels_buffer[d][c_h][c_w] =
										channels_buffer[d][c_h][c_w
												- (layer_0_filter_dim
														- layer_0_strides)
												+ starting_fill_index];
							} else {
								intermediate_channels_buffer[d][c_h][c_w] =
										current_layer_zero_point;
							}
						}
					}
				}
			}
			// end shift and fill the intermediate_channels_buffer
		}
	}
//	cout << "\n";
//	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
//		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//			cout << result[0][h][w] << " ";
//		}
//		cout << "\n";
//	}
}

void _6_layer_2_dw(
		fms_dt channels_buffer[layer_2_dw_depth][_6_stages_layer_2_rows_at_once][layer_2_dw_ifm_width],
		const dw_weights_dt dw_weights[layer_2_dw_depth][layer_2_dw_filter_size][layer_2_dw_filter_size],
		fms_dt upper[layer_2_dw_depth][layer_2_dw_filter_size
				- layer_2_dw_strides][layer_2_dw_ifm_width],
		fms_dt lower[layer_2_dw_depth][_6_stages_layer_2_rows_at_once][layer_2_dw_ifm_width],
		fms_dt result[layer_3_pw_depth][_6_stages_layer_2_rows_at_once][layer_3_pw_ifm_width],
		int first_row) {
#pragma HLS INLINE off

	const int upper_height = layer_2_dw_filter_size - layer_2_dw_strides;
	const fms_dt current_layer_zero_point = conv_fms_zero_points[2];
	const int current_layer_fused_parameters_offsets =
			layers_fused_parameters_offsets[2];
	const int first_fill_width_offset = layer_2_dw_filter_size
			- layer_2_dw_padding_left;
	const int first_fill_top_offset = first_row * layer_2_dw_padding_top;
	const int num_cols_to_shift = layer_2_dw_filter_size - layer_2_dw_strides;

	layer_2_dw_main_loop: for (int row = 0;
			row < _6_stages_layer_2_rows_at_once; row++) {
#pragma HLS UNROLL
		// rows for next DW
		fms_dt intermediate_channels_buffer[sesl_layer_2_dw_parallelism][layer_2_dw_filter_size][layer_2_dw_filter_size];
		if (first_row) {
			for (int d = 0; d < sesl_layer_2_dw_parallelism; d++) {
				for (int h = 0; h < layer_2_dw_padding_top; h++) {
					for (int w = 0; w < layer_2_dw_filter_size; w++) {
						intermediate_channels_buffer[d][h][w] =
								current_layer_zero_point;
					}
				}
			}
		}
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

		for (int o_o_d = 0;
				o_o_d < layer_2_dw_depth / sesl_layer_2_dw_parallelism;
				o_o_d++) {
			int o_o_d_offset = o_o_d * sesl_layer_2_dw_parallelism;

			// first time fill from upper and lower:
			for (int d = 0; d < sesl_layer_2_dw_parallelism; d++) {
#pragma HLS UNROLL
				for (int h = 0;
						h < layer_2_dw_filter_size - layer_2_dw_strides - row;
						h++) {
#pragma HLS UNROLL
					// padding
					if (layer_2_dw_padding_left) {
						intermediate_channels_buffer[d][h][0] =
								current_layer_zero_point;
					}
					for (int w = layer_2_dw_padding_left;
							w < layer_2_dw_filter_size; w++) {
#pragma HLS UNROLL
						if (h + row >= first_fill_top_offset) {
							intermediate_channels_buffer[d][h][w] =
									upper[o_o_d_offset + d][h + row][w
											- layer_2_dw_padding_left];
						}
					}
				}
				const int from_upper_offset = (layer_2_dw_filter_size
						- layer_2_dw_strides - row);
				for (int h = 0; h < layer_2_dw_filter_size - from_upper_offset;
						h++) {
#pragma HLS UNROLL
					// padding left
					if (layer_2_dw_padding_left) {
						intermediate_channels_buffer[d][h + from_upper_offset][0] =
								current_layer_zero_point;
					}
					for (int w = layer_2_dw_padding_left;
							w < layer_2_dw_filter_size; w++) {
#pragma HLS UNROLL
						intermediate_channels_buffer[d][h + from_upper_offset][w] =
								channels_buffer[o_o_d_offset + d][h][w
										- layer_2_dw_padding_left];
					}
				}
			}
			//end first time fill
			layer_2_dw_pipeline: for (int w = 0; w < layer_2_dw_ofm_width;
					w++) {
#pragma HLS PIPELINE
				const int starting_fill_index = w + first_fill_width_offset;
				layer_1_pw_loops: for (int o_d = 0;
						o_d < sesl_layer_2_dw_parallelism; o_d++) {
#pragma HLS UNROLL
					//###############DW########################
					dw_pss_dt tmp = 0;
					// parallelized depth loop
					for (int c_h = 0; c_h < layer_2_dw_filter_size; c_h++) {
#pragma HLS UNROLL
						for (int c_w = 0; c_w < layer_2_dw_filter_size; c_w++) {
							// conv width loop
#pragma HLS UNROLL
							tmp += intermediate_channels_buffer[o_d][c_h][c_w]
									* dw_weights[o_o_d_offset + o_d][c_h][c_w];
							if (o_o_d == 0 && o_d == 0 && w == 3) {
								cout
										<< intermediate_channels_buffer[o_d][c_h][c_w]
										<< " * "
										<< dw_weights[o_o_d_offset + o_d][c_h][c_w]
										<< "\n";
							}
						}
					}

					fms_quantization_scheme normalization = { 0, 0, 0, 0 };
					normalization.fused_scales =
							fused_scales[current_layer_fused_parameters_offsets
									+ o_o_d_offset + o_d];
					normalization.fused_zero_point =
							fused_zero_points[current_layer_fused_parameters_offsets
									+ o_o_d_offset + o_d];
					normalization.ofm_zero_point = conv_fms_zero_points[2 + 1];
					normalization.ofm_scale = conv_fms_scales[2 + 1];
					result[o_o_d_offset + o_d][row][w] = dw_relu_norm(tmp,
							normalization, 6);

					if (o_o_d == 0 && o_d == 0 && w == 3) {
						cout << tmp << " tpm "
								<< dw_relu_norm(tmp, normalization, 6) << "\n";
					}
					//#####################end DW################
					//#####################shift and fill intermediate#################
					//shift
					for (int c_h = 0; c_h < layer_2_dw_filter_size; c_h++) {
#pragma HLS UNROLL
						for (int c_w = 0;
								c_w
										< layer_2_dw_filter_size
												- layer_2_dw_strides; c_w++) {
#pragma HLS UNROLL
							intermediate_channels_buffer[o_d][c_h][c_w] =
									intermediate_channels_buffer[o_d][c_h][c_w
											+ layer_2_dw_strides];
						}
					}
					//end shift

					//fill
					for (int h = 0;
							h
									< layer_2_dw_filter_size
											- layer_2_dw_strides - row; h++) {
#pragma HLS UNROLL
						for (int c_w = 0; c_w < layer_2_dw_strides; c_w++) {
#pragma HLS UNROLL
							if (h + row >= first_fill_top_offset) {
								// padding right
								if (c_w + starting_fill_index
										< layer_2_dw_ifm_width) {
									intermediate_channels_buffer[o_d][h][c_w
											+ num_cols_to_shift] =
											upper[o_o_d_offset + o_d][h + row][c_w
													+ starting_fill_index];
								} else {
									intermediate_channels_buffer[o_d][h][c_w
											+ num_cols_to_shift] =
											current_layer_zero_point;
								}
							}
						}
					}

					const int from_upper_offset = (layer_2_dw_filter_size
							- layer_2_dw_strides - row);
					for (int h = 0;
							h < layer_2_dw_filter_size - from_upper_offset;
							h++) {
#pragma HLS UNROLL
						for (int c_w = 0; c_w < layer_2_dw_strides; c_w++) {
#pragma HLS UNROLL
							// padding right
							if (c_w + starting_fill_index
									< layer_2_dw_ifm_width) {
								intermediate_channels_buffer[o_d][h
										+ from_upper_offset][c_w
										+ num_cols_to_shift] =
										channels_buffer[o_o_d_offset + o_d][h][c_w
												+ starting_fill_index];
								if (o_o_d == 0 && o_d == 0 && w == 2) {
									cout << h << " , "<< c_w + starting_fill_index << " >> "
											<< channels_buffer[o_o_d_offset
													+ o_d][h][c_w
													+ starting_fill_index]
											<< "\n";
								}
							} else {
								intermediate_channels_buffer[o_d][h][c_w
										+ num_cols_to_shift] =
										current_layer_zero_point;
							}
						}
					}
					//end fill
					//#####################end shift and fill intermediate#################
				}
			}
		}
	}

	cout << "\n";
	for (int h = 0; h < _6_stages_layer_2_rows_at_once; h++) {
		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
			cout << result[0][h][w] << " ";
		}
		cout << "\n";
	}

	const int num_to_be_shifted_rows = upper_height
			- _7_stages_layer_2_rows_at_once;
	layer_1_pw_dw_shift_loop: for (int o_o_d = 0;
			o_o_d < layer_1_pw_num_fils / layer_1_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_1_pw_parallelism_out;
		layer_1_shift_pipeline: for (int w = 0; w < layer_2_dw_ifm_width; w++) {
#pragma HLS UNROLL
			//###################PW#######################
			layer_1_shift_loops: for (int o_d = 0;
					o_d < layer_1_pw_parallelism_out; o_d++) {
#pragma HLS UNROLL
				//shift
				for (int h = 0; h < num_to_be_shifted_rows; h++) {
					upper[o_o_d_offset + o_d][h][w] =
							upper[o_o_d_offset + o_d][h + layer_2_dw_strides][w];
				}
				//fill
				for (int h = num_to_be_shifted_rows; h < upper_height; h++) {
					upper[o_o_d_offset + o_d][h][w] =
							channels_buffer[o_o_d_offset + o_d][h
									- num_to_be_shifted_rows][w];
				}
			}
		}
	}
}

void _6_layer_3_pw(
		fms_dt channels_buffer[layer_3_pw_depth][_6_stages_layer_3_rows_at_once][layer_3_pw_ifm_width],
		const weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
		fms_dt result[layer_4_pw_depth][_6_stages_layer_3_rows_at_once][layer_5_dw_ifm_width]) {

#pragma HLS INLINE off

// rows for next DW
	for (int o_o_d = 0;
			o_o_d < layer_3_pw_num_fils / layer_3_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_3_pw_parallelism_out;
		// filters loop
		layer_3_pw_pipeline: for (int w = 0; w < layer_3_pw_ifm_width; w++) {
#pragma HLS PIPELINE
			for (int row = 0; row < _6_stages_layer_3_rows_at_once; row++) {
				// FMs width loop
				layer_3_pw_loops: for (int o_d = 0;
						o_d < layer_3_pw_parallelism_out; o_d++) {
#pragma HLS UNROLL
					// parallelized filters loop
					pss_dt tmp = 0;
					for (int d = 0; d < layer_3_pw_parallelism_in; d++) {
#pragma HLS UNROLL
						// parallelized depth loop
						tmp += ((fms_dt) channels_buffer[d][row][w])
								* weights[o_o_d_offset + o_d][d];
					}
					fms_quantization_scheme normalization = { 0, 0, 0, 0 };
					normalization.fused_scales =
							fused_scales[layers_fused_parameters_offsets[3]
									+ o_d];
					normalization.fused_zero_point =
							fused_zero_points[layers_fused_parameters_offsets[3]
									+ o_d];
					normalization.ofm_zero_point = conv_fms_zero_points[3 + 1];
					normalization.ofm_scale = conv_fms_scales[3 + 1];
					result[o_o_d_offset + o_d][row][w] = pw_relu_norm(tmp,
							normalization, layer_3_relu);
				}
			}
		}
	}
}

void _6_layer_4_pw_5_dw(
		fms_dt channels_buffer[layer_4_pw_depth][layer_5_dw_strides][layer_5_dw_ifm_width],
		const weights_dt weights[layer_4_pw_num_fils][layer_4_pw_depth],
		const dw_weights_dt dw_weights[layer_5_dw_depth][layer_5_dw_filter_size][layer_5_dw_filter_size],
		fms_dt upper[layer_5_dw_depth][layer_5_dw_ifm_width],
		fms_dt lower[layer_5_dw_depth][layer_5_dw_strides][layer_5_dw_ifm_width],
		fms_dt result[layer_6_pw_depth][layer_6_pw_ifm_width], int active_row) {

	fms_dt intermediate_channels_buffer[layer_4_pw_parallelism_out][layer_5_dw_filter_size][layer_5_dw_filter_size];
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

#pragma HLS INLINE off
	layer_4_pw__dw_main_loop: for (int o_o_d = 0;
			o_o_d < layer_4_pw_num_fils / layer_4_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_4_pw_parallelism_out;

		// fill upper and lower (except last) rows:
		for (int d = 0; d < layer_4_pw_parallelism_out; d++) {
#pragma HLS UNROLL
			for (int h = 0; h < layer_5_dw_filter_size - layer_5_dw_strides;
					h++) {
#pragma HLS UNROLL
				// padding
				intermediate_channels_buffer[d][h][0] = 0;
				for (int w = 1; w < layer_5_dw_filter_size; w++) {
#pragma HLS UNROLL
					intermediate_channels_buffer[d][h][w] = upper[d][h][w];
				}
			}
			intermediate_channels_buffer[d][1][0] = 0;
			intermediate_channels_buffer[d][2][0] = 0;
		}

		layer_4_pw_pipeline: for (int w = 0;
				w
						< layer_5_dw_ifm_width + layer_5_dw_padding_left
								+ layer_5_dw_padding_right; w++) {
#pragma HLS PIPELINE
			//###################PW#######################
			layer_1_pw_loops: for (int o_d = 0;
					o_d < layer_4_pw_parallelism_out; o_d++) {
#pragma HLS UNROLL
				// parallelized filters loop
				if (w < layer_5_dw_ifm_width) {
					for (int row = 0; row < _6_stages_layer_3_rows_at_once;
							row++) {
#pragma HLS UNROLL
						// FMs width loop
						pss_dt tmp = 0;
						for (int d = 0; d < layer_4_pw_parallelism_in; d++) {
#pragma HLS UNROLL
							// parallelized depth loop
							tmp += ((fms_dt) channels_buffer[d][row][w])
									* weights[o_o_d_offset + o_d][d];
						}

						fms_quantization_scheme normalization = { 0, 0, 0, 0 };
						normalization.fused_scales =
								fused_scales[layers_fused_parameters_offsets[4]
										+ o_d];
						normalization.fused_zero_point =
								fused_zero_points[layers_fused_parameters_offsets[4]
										+ o_d];
						normalization.ofm_zero_point = conv_fms_zero_points[4
								+ 1];
						normalization.ofm_scale = conv_fms_scales[4 + 1];

						fms_dt scaled_val = pw_relu_norm(tmp, normalization,
								layer_3_relu);

						if (w + 1 < layer_2_dw_filter_size) {
							intermediate_channels_buffer[o_d][row
									+ layer_5_dw_filter_size
									- layer_5_dw_strides][w
									+ layer_5_dw_padding_left] = scaled_val;
						}
						lower[o_o_d_offset + o_d][row][w] = scaled_val;

					}
				}
				//###############end PW####################
				//###############DW########################
				if (w + 1 >= layer_5_dw_filter_size - layer_5_dw_padding_left
						&& active_row == layer_5_dw_strides - 1
						&& (w + 1
								- (layer_5_dw_filter_size
										- layer_5_dw_padding_left))
								% layer_5_dw_strides == 0) {
					for (int row = 0; row < layer_5_dw_strides; row++) {
						for (int c_w = layer_5_dw_filter_size
								- layer_5_dw_strides;
								c_w < layer_5_dw_filter_size; c_w++) {
							// conv width loop
#pragma HLS UNROLL
							intermediate_channels_buffer[o_d][layer_5_dw_filter_size
									- layer_5_dw_strides + row][c_w] =
									lower[o_o_d_offset + o_d][row][w + 1
											- (layer_5_dw_filter_size
													- layer_5_dw_padding_left)
											+ (c_w
													- (layer_5_dw_filter_size
															- layer_5_dw_strides))];
						}
					}
					dw_pss_dt tmp = 0;
					// parallelized depth loop
					for (int c_h = 0; c_h < layer_5_dw_filter_size; c_h++) {
#pragma HLS UNROLL
						for (int c_w = 0; c_w < layer_5_dw_filter_size; c_w++) {
							// conv width loop
#pragma HLS UNROLL
							tmp += intermediate_channels_buffer[o_d][c_h][w
									+ c_w]
									* dw_weights[o_o_d_offset + o_d][c_h][c_w];
						}
					}
					fms_quantization_scheme normalization = { 0, 0, 0, 0 };
					normalization.fused_scales =
							fused_scales[layers_fused_parameters_offsets[5]
									+ o_d];
					normalization.fused_zero_point =
							fused_zero_points[layers_fused_parameters_offsets[5]
									+ o_d];
					normalization.ofm_zero_point = conv_fms_zero_points[5 + 1];
					normalization.ofm_scale = conv_fms_scales[5 + 1];
					result[o_o_d_offset + o_d][(w - layer_5_dw_padding_left)
							/ layer_5_dw_strides] = dw_relu_norm(tmp,
							normalization, 6);
					//#####################end DW################
					//#####################shift and fill intermediate#################
					for (int c_h = 0; c_h < layer_5_dw_filter_size; c_h++) {
#pragma HLS UNROLL
						for (int c_w = 0;
								c_w
										< layer_5_dw_filter_size
												- layer_5_dw_strides; c_w++) {
#pragma HLS UNROLL
							intermediate_channels_buffer[o_d][c_h][c_w] =
									intermediate_channels_buffer[o_d][c_h][c_w
											+ layer_5_dw_strides];
						}
						if (c_h < layer_5_dw_filter_size - layer_5_dw_strides) {
							for (int c_w = layer_5_dw_filter_size
									- layer_5_dw_strides;
									c_w < layer_5_dw_filter_size; c_w++) {
#pragma HLS UNROLL
								intermediate_channels_buffer[o_d][c_h][c_w] =
										upper[o_o_d_offset + o_d][w
												+ (c_w
														- (layer_5_dw_filter_size
																- layer_5_dw_strides))
												+ layer_5_dw_padding_left];
							}
						}
					}
					//#####################end shift and fill intermediate#################
				}
			}
		}
	}

	layer_4_pw_dw_shift_loop: for (int o_o_d = 0;
			o_o_d < layer_4_pw_num_fils / layer_4_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_4_pw_parallelism_out;
		layer_3_shift_pipeline: for (int w = 0; w < layer_5_dw_ifm_width; w++) {
#pragma HLS UNROLL factor = 4
			//###################PW#######################
			layer_1_shift_loops: for (int o_d = 0;
					o_d < layer_4_pw_parallelism_out; o_d++) {
#pragma HLS UNROLL
				upper[o_o_d_offset + o_d][w] = lower[o_o_d_offset + o_d][1][w];
			}
		}
	}
}

void _6_layer_6_pw(
		fms_dt channels_buffer[layer_6_pw_depth][layer_6_pw_ifm_width],
		const weights_dt weights[layer_6_pw_num_fils][layer_6_pw_depth],
		fms_dt result[max_fms_size], int starting_h) {

#pragma HLS INLINE off

// rows for next DW
	for (int o_o_d = 0;
			o_o_d < layer_6_pw_num_fils / layer_6_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_6_pw_parallelism_out;
		// filters loop
		layer_6_pw_pipeline: for (int w = 0; w < layer_6_pw_ifm_width; w++) {
#pragma HLS PIPELINE
			// FMs width loop
			layer_6_pw_loops: for (int o_d = 0;
					o_d < layer_6_pw_parallelism_out; o_d++) {
#pragma HLS UNROLL
				// parallelized filters loop
				pss_dt tmp = 0;
				int offset_in_result =
						(o_o_d * layer_6_pw_parallelism_out + o_d)
								* switch_point_fms_height
								* switch_point_fms_width
								+ starting_h * switch_point_fms_width + w;
				for (int d = 0; d < layer_6_pw_parallelism_in; d++) {
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += ((fms_dt) channels_buffer[d][w]) * weights[o_d][d];
				}
				fms_quantization_scheme normalization = { 0, 0, 0, 0 };
				normalization.fused_scales =
						fused_scales[layers_fused_parameters_offsets[6] + o_d];
				normalization.fused_zero_point =
						fused_zero_points[layers_fused_parameters_offsets[6]
								+ o_d];
				normalization.ofm_zero_point = conv_fms_zero_points[6 + 1];
				normalization.ofm_scale = conv_fms_scales[6 + 1];
				result[offset_in_result] = pw_relu_norm(tmp, normalization,
						layer_6_relu);
			}
		}
	}
}

void _6_stages_fill_channels_buffer(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim
				+ (_6_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width],
		int starting_h) {

	const fms_dt current_layer_zero_point = conv_fms_zero_points[0];

	const int buffer_height = layer_0_filter_dim
			+ (_6_stages_layer_0_rows_at_once - 1) * layer_0_strides;
	const int rows_to_shift = layer_0_filter_dim - layer_0_strides;

	const int shift_starting_point = buffer_height - rows_to_shift;
	const int fill_starting_point = rows_to_shift;

	for (int w = 0; w < input_image_width; w++) {
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
			for (int h = 0; h < rows_to_shift; h++) {
#pragma HLS UNROLL
				if (starting_h != 0) {
					channels_buffer_0[d][h][w] = channels_buffer_0[d][h
							+ shift_starting_point][w];
				} else {
					channels_buffer_0[d][h][w] = channels[d][h][w];
				}
			}
		}
	}

	for (int w = 0; w < input_image_width; w++) {
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
			for (int h = fill_starting_point; h < buffer_height; h++) {
#pragma HLS UNROLL
				if (starting_h + h - fill_starting_point < input_image_height) {
					channels_buffer_0[d][h][w] = channels[d][starting_h + h
							- fill_starting_point][w];
				} else {
					channels_buffer_0[d][h][w] = current_layer_zero_point;
				}
			}
		}
	}
}

void cnn_pipeline_6_mob_v2(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size]) {
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

#pragma HLS ARRAY_PARTITION variable = dw_weights_1 type = complete dim = 1

//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim
			+ (_6_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
	fms_dt _6_layer_0_3x3_conv_out_0[layer_2_dw_depth][_6_stages_layer_0_rows_at_once][layer_2_dw_ifm_width] =
			{ 0 };

//##############
	fms_dt _6_layer_2_dw_upper[layer_2_dw_depth][layer_2_dw_filter_size
			- layer_2_dw_strides][layer_2_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_upper complete dim = 3

	fms_dt _6_layer_2_dw_lower[layer_2_dw_depth][_6_stages_layer_2_rows_at_once][layer_2_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_lower complete dim = 3

	fms_dt _6_layer_2_dw_out_0[layer_3_pw_depth][_6_stages_layer_2_rows_at_once][layer_3_pw_ifm_width] =
			{ 0 };
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_out_0 complete dim = 1
//##############

	fms_dt _6_layer_3_pw_out_0[layer_4_pw_depth][_6_stages_layer_3_rows_at_once][layer_5_dw_ifm_width] =
			{ 0 };

	fms_dt _6_layer_5_dw_upper[layer_5_dw_depth][layer_5_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_upper cyclic factor = layer_4_pw_parallelism_out dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_upper cyclic factor = 6 dim = 2

	fms_dt _6_layer_5_dw_lower[layer_5_dw_depth][layer_5_dw_strides][layer_5_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_lower cyclic factor = 2 dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_lower complete dim = 2
#pragma HLS ARRAY_PARTITION variable = _6_layer_5_dw_lower cyclic factor = 12 dim = 3

	fms_dt _6_layer_4_5_pw_dw_out_0[layer_6_pw_depth][layer_6_pw_ifm_width] = {
			0 };

#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_0 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _6_layer_3_pw_out_0 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_4_5_pw_dw_out_0 cyclic factor = layer_4_pw_parallelism_in/2 dim = 1
//###########################################################

//#########################odd###############################
	fms_dt channels_buffer_1[input_image_depth][layer_0_filter_dim
			+ (_6_stages_layer_2_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2

	fms_dt _6_layer_0_3x3_conv_out_1[layer_2_dw_depth][_6_stages_layer_0_rows_at_once][layer_2_dw_ifm_width] =
			{ 0 };

	fms_dt _6_layer_1_pw_out_1[layer_2_dw_depth][layer_2_dw_ifm_width] = { 0 };

//##############

	fms_dt _6_layer_2_dw_out_1[layer_3_pw_depth][_6_stages_layer_2_rows_at_once][layer_3_pw_ifm_width] =
			{ 0 };
//##############
	fms_dt _6_layer_3_pw_out_1[layer_4_pw_depth][_6_stages_layer_3_rows_at_once][layer_5_dw_ifm_width] =
			{ 0 };

	fms_dt _6_layer_4_5_pw_dw_out_1[layer_6_pw_depth][layer_6_pw_ifm_width] = {
			0 };

#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_3_pw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_4_5_pw_dw_out_1 cyclic factor = layer_4_pw_parallelism_in/2 dim = 1

//###########################################################
// pipeline filling##########################################
	_6_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
//##########
	_6_stages_fill_channels_buffer(channels, channels_buffer_1, 2);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
//##########
	_6_stages_fill_channels_buffer(channels, channels_buffer_0, 6);
	_6_layer_0_3x3_conv(channels_buffer_1, weights_0,
			_6_layer_0_3x3_conv_out_1);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 1);
//##########
	_6_stages_fill_channels_buffer(channels, channels_buffer_1, 10);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_1, 0);
//##########
	_6_stages_fill_channels_buffer(channels, channels_buffer_0, 14);
	_6_layer_0_3x3_conv(channels_buffer_1, weights_0,
			_6_layer_0_3x3_conv_out_1);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
//##########
	_6_stages_fill_channels_buffer(channels, channels_buffer_1, 18);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_1, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			1);
//##########
	int even_odd = 1;
	int h = 6;
	main_pipeline_loop: for (; h < switch_point_fms_height; h++) {
		if (even_odd) {
			_6_stages_fill_channels_buffer(channels, channels_buffer_0,
					(h * _6_stages_layer_0_rows_at_once - 1) * layer_0_strides);
			_6_layer_0_3x3_conv(channels_buffer_1, weights_0,
					_6_layer_0_3x3_conv_out_1);
			_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2,
					_6_layer_2_dw_upper, _6_layer_2_dw_lower,
					_6_layer_2_dw_out_0, 0);
			_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3,
					_6_layer_3_pw_out_1);
			_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
					_6_layer_5_dw_upper, _6_layer_5_dw_lower,
					_6_layer_4_5_pw_dw_out_0, 1);
			_6_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, result,
					h - 6);
		} else {
			_6_stages_fill_channels_buffer(channels, channels_buffer_1,
					(h * _6_stages_layer_0_rows_at_once - 1) * layer_0_strides);
			_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
					_6_layer_0_3x3_conv_out_0);
			_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2,
					_6_layer_2_dw_upper, _6_layer_2_dw_lower,
					_6_layer_2_dw_out_1, 0);
			_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3,
					_6_layer_3_pw_out_0);
			_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
					_6_layer_5_dw_upper, _6_layer_5_dw_lower,
					_6_layer_4_5_pw_dw_out_1, 1);
			_6_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, result,
					h - 6);
		}
		even_odd = 1 - even_odd;
	}
//###########################################################
// pipeline flushing##########################################
	_6_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, result,
			switch_point_fms_height - 6);
//##########
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			1);
	_6_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, result,
			switch_point_fms_height - 5);
//##########
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			1);
	_6_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, result,
			switch_point_fms_height - 4);
//##########
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			1);
	_6_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, result,
			switch_point_fms_height - 3);
//##########
	_6_layer_0_3x3_conv(channels_buffer_1, weights_0,
			_6_layer_0_3x3_conv_out_1);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_1, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			1);
	_6_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, result,
			switch_point_fms_height - 2);
//#########
	_6_stages_fill_channels_buffer(channels, channels_buffer_0,
			(switch_point_fms_height * _6_stages_layer_0_rows_at_once - 1)
					* layer_0_strides);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 0);
// padding bottom
	for (int d = 0; d < layer_3_pw_depth; d++) {
		for (int w = 0; w < layer_3_pw_ifm_width; w++) {
			_6_layer_2_dw_out_0[d][_6_stages_layer_0_rows_at_once - 1][w] = 0;
		}
	}
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			1);
	_6_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, result,
			switch_point_fms_height - 1);
}
