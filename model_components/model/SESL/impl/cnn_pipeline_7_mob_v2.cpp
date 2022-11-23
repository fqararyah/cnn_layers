#include <iostream>
#include <math.h>
#include "../headers/sesl.h"
#include "../../../../tests/test_utils.h"

using namespace std;

void _7_layer_6_pw(
		fms_dt channels_buffer[layer_6_pw_depth][layer_6_pw_ifm_width],
		const weights_dt weights[layer_6_pw_num_fils][layer_6_pw_depth],
		fms_dt result[layer_7_pw_depth][layer_7_pw_ifm_width]) {

#pragma HLS INLINE off

	const int current_layer_fused_parameters_offsets =
			layers_fused_parameters_offsets[6];

	const fms_dt current_layer_ofms_zero_point = conv_fms_zero_points[6 + 1];
	const scales_dt current_layer_ofms_scale = conv_fms_scales_rec[6 + 1];

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
				for (int d = 0; d < layer_6_pw_parallelism_in; d++) {
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += channels_buffer[d][w]
							* weights[o_o_d_offset + o_d][d];
				}
				fms_quantization_scheme normalization = { 0, 0, 0, 0 };
				normalization.fused_scales =
						fused_scales[current_layer_fused_parameters_offsets
								+ o_o_d_offset + o_d];
				normalization.fused_zero_point =
						fused_zero_points[current_layer_fused_parameters_offsets
								+ o_o_d_offset + o_d];
				normalization.ofm_zero_point = current_layer_ofms_zero_point;
				normalization.ofm_scale_rec = current_layer_ofms_scale;
				result[o_o_d_offset + o_d][w] = dw_relu_norm(tmp, normalization,
						layer_6_relu);
			}
		}
	}
}

void write_to_tmp_channel(
		fms_dt channels_buffer[layer_6_pw_num_fils][layer_6_pw_ofm_width],
		fms_dt tmp_channels[max_tmp_fms_size], int starting_h) {
#pragma HLS INLINE off

	const int num_tiles_hw = layer_6_pw_num_of_tiles_h
			* layer_6_pw_num_of_tiles_w;

	const int tile_in_h = starting_h / pw_tile_h;
	const int in_tile_h = starting_h % pw_tile_h;

	// rows for next DW
	//cout<<"\nstarting_h: "<<starting_h<<"\n";
	write_to_tmp_channel_loops: for (int o_o_d = 0;
			o_o_d < layer_6_pw_num_fils / layer_6_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_6_pw_parallelism_out;
		// filters loop
		for (int o_d = 0; o_d < layer_6_pw_parallelism_out; o_d++) {
#pragma HLS UNROLL
			const int tile_in_z = (o_o_d_offset + o_d) / pw_tile_d;
			const int in_tile_z = (o_o_d_offset + o_d) % pw_tile_d;
			layer_7_pw_pipeline: for (int w = 0; w < layer_6_pw_ofm_width;
					w++) {
#pragma HLS PIPELINE
				// FMs width loop
				// parallelized filters loop
				pss_dt tmp = 0;

				const int tile_in_w = w / pw_tile_w;
				const int tile_index = tile_in_z * num_tiles_hw
						+ tile_in_h * layer_6_pw_num_of_tiles_w + tile_in_w;

				const int in_tile_w = w % pw_tile_w;
				const int in_tile_index = in_tile_z * pw_tile_hw
						+ in_tile_h * pw_tile_w + in_tile_w;

				const int index_in_tmp_channels = tile_index * pw_tile_size
						+ in_tile_index;

				tmp_channels[index_in_tmp_channels] =
						channels_buffer[o_o_d_offset + o_d][w];
			}
		}
	}
}

void _7_layer_7_pw(
		fms_dt channels_buffer[layer_7_pw_depth][layer_7_pw_ifm_width],
		const weights_dt weights[layer_7_pw_num_fils][layer_7_pw_depth],
		fms_dt result[max_fms_size], int starting_h) {

#pragma HLS INLINE off

//	if (starting_h > 50) {
//		cout << "***7***\n";
//		for (int w = 0; w < 5; w++) {
//			//for (int d = 0; d < layer_7_pw_depth; d++)
//			cout << channels_buffer[0][w] << " ";
//			//cout << "\n";
//		}
//		cout << "******\n";
//	}

	const int current_layer_fused_parameters_offsets =
			layers_fused_parameters_offsets[7];

	const fms_dt current_layer_ofms_zero_point = conv_fms_zero_points[7 + 1];
	const scales_dt current_layer_ofms_scale = conv_fms_scales_rec[7 + 1];

	const int num_tiles_hw = layer_7_pw_num_of_tiles_h
			* layer_7_pw_num_of_tiles_w;

	const int tile_in_h = starting_h / pw_tile_h;
	const int in_tile_h = starting_h % pw_tile_h;

	// rows for next DW
	//cout<<"\nstarting_h: "<<starting_h<<"\n";
	for (int o_o_d = 0;
			o_o_d < layer_7_pw_num_fils / layer_7_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_7_pw_parallelism_out;
		// filters loop
		layer_7_pw_loops: for (int o_d = 0; o_d < layer_7_pw_parallelism_out;
				o_d++) {
#pragma HLS UNROLL
			const int tile_in_z = (o_o_d_offset + o_d) / pw_tile_d;
			const int in_tile_z = (o_o_d_offset + o_d) % pw_tile_d;
			layer_7_pw_pipeline: for (int w = 0; w < layer_7_pw_ifm_width;
					w++) {
#pragma HLS PIPELINE
				// FMs width loop
				// parallelized filters loop
				pss_dt tmp = 0;

				const int tile_in_w = w / pw_tile_w;
				const int tile_index = tile_in_z * num_tiles_hw
						+ tile_in_h * layer_7_pw_num_of_tiles_w + tile_in_w;

				const int in_tile_w = w % pw_tile_w;
				const int in_tile_index = in_tile_z * pw_tile_hw
						+ in_tile_h * pw_tile_w + in_tile_w;

				const int index_in_result = tile_index * pw_tile_size
						+ in_tile_index;

				for (int d = 0; d < layer_7_pw_parallelism_in; d++) {
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += channels_buffer[d][w]
							* weights[o_o_d_offset + o_d][d];
//					if (o_o_d + o_d == 0 && w == 0) {
//						cout << channels_buffer[d][w] << " * "
//								<< weights[o_o_d_offset + o_d][d] << " ";
//						if (d == layer_7_pw_parallelism_in - 1)
//							cout << "\n";
//					}
				}

				fms_quantization_scheme normalization = { 0, 0, 0, 0 };
				normalization.fused_scales =
						fused_scales[current_layer_fused_parameters_offsets
								+ o_o_d_offset + o_d];
				normalization.fused_zero_point =
						fused_zero_points[current_layer_fused_parameters_offsets
								+ o_o_d_offset + o_d];
				normalization.ofm_zero_point = current_layer_ofms_zero_point;
				normalization.ofm_scale_rec = current_layer_ofms_scale;
				result[index_in_result] = pw_relu_norm(tmp, normalization,
						layer_7_relu);
//				if (o_o_d_offset + o_d == 0) {
////					cout << "\n"<<tmp <<" >> "<<pw_relu_norm(tmp, normalization, layer_7_relu)
////							<< "\n";
//					cout << pw_relu_norm(tmp, normalization, layer_7_relu)
//							<< " ";
//				}
			}
		}
//		if (o_o_d == 0) {
//			cout << "\n";
//		}
	}
}

void fill_row(fms_grp_dt tmp_buffer[input_image_depth][input_image_num_fms_groups_in_width],
		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim
				+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width],
		const int input_image_num_fms_groups_in_width, int row) {

#pragma HLS INLINE OFF

	for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
		for (int o_w = 0; o_w < input_image_num_fms_groups_in_width; o_w++) {
			const int o_w_offset = o_w * input_image_group_items;
			fms_grp_dt chunck = tmp_buffer[d][o_w];
			for (int w = 0; w < input_image_group_items; w++) {
				if (o_w_offset + w < input_image_width) {
					channels_buffer_0[d][row][o_w_offset + w] = (fms_dt) chunck(
							w * fms_dt_width + fms_dt_offset, w * fms_dt_width);
				}
			}
		}
	}
}

void _7_stages_fill_channels_buffer(
		fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim
				+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width],
		int starting_h) {

	const fms_dt current_layer_zero_point = conv_fms_zero_points[0];

	const int buffer_height = layer_0_filter_dim
			+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides;
	const int rows_to_shift = layer_0_filter_dim - layer_0_strides;

	const int filling_starting_offset =
			starting_h == 0 ?
					input_image_num_fms_groups_in_width
							* rows_to_shift :
					starting_h * input_image_num_fms_groups_in_width;

	const int num_fms_groups_to_fitch = input_image_num_fms_groups_in_width
			* (buffer_height - rows_to_shift);

	const int shift_starting_point =
			starting_h >= buffer_height - rows_to_shift ?
					buffer_height - rows_to_shift : starting_h - 1;
	const int fill_starting_row = rows_to_shift;
	const int first_time_offset =
			starting_h == 0 ? (layer_0_filter_dim - layer_0_strides) : 0;

	fms_grp_dt tmp_buffer[input_image_depth][input_image_num_fms_groups_in_width];

//shift
	if (starting_h != 0) {
		for (int w = 0; w < input_image_width; w++) {
#pragma HLS PIPELINE
			for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
				for (int h = 0; h < rows_to_shift; h++) {
#pragma HLS UNROLL
					channels_buffer_0[d][h][w] = channels_buffer_0[d][h
							+ shift_starting_point][w];

				}
			}
		}
	} else {
//fill first time:
		for (int h = 0; h < rows_to_shift; h++) {
			const int h_offset = h * input_image_num_fms_groups_in_width;
			for (int d = 0; d < input_image_depth; d++) {
				const int d_offst = h_offset + d * input_image_num_fms_groups_in_a_channel;
				for (int i = 0; i < input_image_num_fms_groups_in_width; i++) {
					tmp_buffer[d][i] = channels[d_offst + i];
				}
			}
			fill_row(tmp_buffer, channels_buffer_0, input_image_num_fms_groups_in_width, h);
		}
	}

//fill

	for (int h = rows_to_shift; h < buffer_height; h++) {
		const int h_offset = filling_starting_offset
				+ (h - rows_to_shift) * input_image_num_fms_groups_in_width;
		if ((h - rows_to_shift) + starting_h < input_image_height) {
			for (int d = 0; d < input_image_depth; d++) {
				const int d_offst = h_offset + d * input_image_num_fms_groups_in_a_channel;
				for (int i = 0; i < input_image_num_fms_groups_in_width; i++) {
					tmp_buffer[d][i] = channels[d_offst + i];
				}
			}
			fill_row(tmp_buffer, channels_buffer_0, input_image_num_fms_groups_in_width, h);
		} else {	//padding bottom
			for (int d = 0; d < input_image_depth; d++) {
				for (int w = 0; w < input_image_width; w++) {
					channels_buffer_0[d][h][w] = current_layer_zero_point;
				}
			}
		}
	}
//	cout << "*********************\n";
//	for (int h = 0; h < buffer_height; h++) {
//		for (int w = 0; w < layer_0_ifm_width; w++) {
//			cout << channels_buffer_0[1][h][w]<<" ";
//		}
//		cout << "\n";
//	}
//	cout << "*********************\n";
}

void cnn_pipeline_7_mob_v2(
		fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
		fms_dt result[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size]) {
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim
			+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
	fms_dt _6_layer_0_3x3_conv_out_0[layer_2_dw_depth][_7_stages_layer_0_rows_at_once][layer_2_dw_ifm_width] =
			{ 0 };
//##############
	fms_dt _6_layer_2_dw_upper[layer_2_dw_depth][layer_2_dw_filter_size
			- layer_2_dw_strides][layer_2_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_upper complete dim = 3

	fms_dt _6_layer_2_dw_lower[layer_2_dw_depth][_7_stages_layer_2_rows_at_once][layer_2_dw_ifm_width];
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_lower complete dim = 3

	fms_dt _6_layer_2_dw_out_0[layer_3_pw_depth][_7_stages_layer_2_rows_at_once][layer_3_pw_ifm_width] =
			{ 0 };
#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_out_0 complete dim = 1
//##############

	fms_dt _6_layer_3_pw_out_0[layer_4_pw_depth][_7_stages_layer_3_rows_at_once][layer_5_dw_ifm_width] =
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

	fms_dt _6_layer_6_pw_out_0[layer_7_pw_depth][layer_7_pw_ifm_width] = { 0 };
#pragma HLS ARRAY_PARTITION variable = _6_layer_6_pw_out_0 complete dim = 1
//###########################################################

//#########################odd###############################
//	fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim
//			+ (_7_stages_layer_2_rows_at_once - 1) * layer_0_strides][input_image_width];
//#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
//#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2

	fms_dt _6_layer_0_3x3_conv_out_1[layer_2_dw_depth][_7_stages_layer_0_rows_at_once][layer_2_dw_ifm_width] =
			{ 0 };
//##############

	fms_dt _6_layer_2_dw_out_1[layer_3_pw_depth][_7_stages_layer_2_rows_at_once][layer_3_pw_ifm_width] =
			{ 0 };
//##############
	fms_dt _6_layer_3_pw_out_1[layer_4_pw_depth][_7_stages_layer_3_rows_at_once][layer_5_dw_ifm_width] =
			{ 0 };

	fms_dt _6_layer_4_5_pw_dw_out_1[layer_6_pw_depth][layer_6_pw_ifm_width] = {
			0 };

#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = _6_layer_0_3x3_conv_out_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = _6_layer_2_dw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_3_pw_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _6_layer_4_5_pw_dw_out_1 cyclic factor = layer_4_pw_parallelism_in/2 dim = 1

	fms_dt _6_layer_6_pw_out_1[layer_7_pw_depth][layer_7_pw_ifm_width] = { 0 };
#pragma HLS ARRAY_PARTITION variable = _6_layer_6_pw_out_1 complete dim = 1

//###########################################################
// pipeline filling##########################################
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
//****************************
//	const int _6_layer_2_dw_upper_height = layer_2_dw_filter_size
//			- layer_2_dw_strides;
//	const fms_dt layer_2_ifms_zero_point = conv_fms_zero_points[2];
//	for (int d = 0; d < layer_2_dw_depth; d++) {
//		for (int h = 0; h < layer_2_dw_padding_top; h++) {
//			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//				_6_layer_2_dw_upper[d][h][w] = layer_2_ifms_zero_point;
//			}
//		}
//		for (int h = layer_2_dw_padding_top; h < _6_layer_2_dw_upper_height;
//				h++) {
//			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//				_6_layer_2_dw_upper[d][h][w] = _6_layer_0_3x3_conv_out_0[d][h
//						- layer_2_dw_padding_top][w];
//			}
//		}
//	}
//	//****************************
//
//	cout << "\n*******1*********\n";
//	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
//		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//			cout << _6_layer_2_dw_upper[0][h][w] << " ";
//		}
//		cout << "\n";
//	}
//	cout << "\n*********1*******\n";

//##########
//	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 3);
//	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
//			_6_layer_0_3x3_conv_out_0);
//	cout << "\n*******2*********\n";
//	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
//		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//			cout << _6_layer_0_3x3_conv_out_0[0][h][w] << " ";
//		}
//		cout << "\n";
//	}
//	cout << "\n*********2*******\n";
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 5);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_1);

	const int _6_layer_2_dw_upper_height = layer_2_dw_filter_size
			- layer_2_dw_strides;
	const fms_dt layer_2_ifms_zero_point = conv_fms_zero_points[2];
//first time only second row of what _6_layer_2_dw is going to be valid, hence fill with zero points
	for (int d = 0; d < layer_2_dw_depth; d++) {
		for (int h = 0; h < _6_layer_2_dw_upper_height; h++) {
			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
				_6_layer_2_dw_upper[d][h][w] = layer_2_ifms_zero_point;
			}
		}
	}
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_out_0, 1);
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 9);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_out_1, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
//	cout << "\n*******3*********\n";
//	for (int h = 0; h < _6_stages_layer_3_rows_at_once; h++) {
//		for (int w = 0; w < layer_3_pw_ofm_width; w++) {
//			cout << _6_layer_3_pw_out_0[0][h][w] << " ";
//		}
//		cout << "\n";
//	}
//	cout << "\n*********3*******\n";
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 13);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_1);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_out_0, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			0);
//##########//_6_layer_4_pw_5_dw first run does not produce any valid output
//	cout << "\n*******3-2*********\n";
//	for (int h = 0; h < _6_stages_layer_3_rows_at_once; h++) {
//		for (int w = 0; w < layer_3_pw_ofm_width; w++) {
//			cout << _6_layer_3_pw_out_1[0][h][w] << " ";
//		}
//		cout << "\n";
//	}
//	cout << "\n*********3-2*******\n";
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 17);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_out_1, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			1);	//first call of _6_layer_4_pw_5_dw the pw part produces only 1 valid row, that is why the increment is 1 not 2
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 21);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_1);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_out_0, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			3);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, _6_layer_6_pw_out_1);
	write_to_tmp_channel(_6_layer_6_pw_out_1, tmp_channels, 0);

//	cout << "***6***\n";
//	for (int w = 0; w < 5; w++) {
//		for (int d = 0; d < layer_7_pw_depth; d++)
//			cout << _6_layer_6_pw_out_0[d][w] << " ";
//		cout << "\n";
//	}

	const int pipeline_filling_stages = 6;
	int odd_even = pipeline_filling_stages % 2;
	int h = pipeline_filling_stages;
	int _6_layer_4_pw_5_dw_starting_h = (h - 3) * _7_stages_layer_4_rows_at_once
			- 1;
	const int channels_buffer_0_height = layer_0_filter_dim
			+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides;
	const int channels_buffer_0_rows_filled_each_time =
			_7_stages_layer_0_rows_at_once * layer_0_strides;
	const int channels_buffer_0_rows_filled_first_time =
			channels_buffer_0_height;

	const int extra_rows_filled_first_time =
			channels_buffer_0_rows_filled_first_time
					- channels_buffer_0_rows_filled_each_time;

	main_pipeline_loop: for (; h < switch_point_fms_height; h++) {
		if (odd_even) {
			_7_stages_fill_channels_buffer(channels, channels_buffer_0,
					h * channels_buffer_0_rows_filled_each_time
							+ extra_rows_filled_first_time);
			_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
					_6_layer_0_3x3_conv_out_1);
			_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2,
					_6_layer_2_dw_upper, _6_layer_2_dw_out_0, 0);
			_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3,
					_6_layer_3_pw_out_1);
			_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
					_6_layer_5_dw_upper, _6_layer_5_dw_lower,
					_6_layer_4_5_pw_dw_out_0, _6_layer_4_pw_5_dw_starting_h);
			_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6,
					_6_layer_6_pw_out_1);
			write_to_tmp_channel(_6_layer_6_pw_out_1, tmp_channels,
					h - pipeline_filling_stages + 1);
			_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
					h - pipeline_filling_stages);
		} else {
			_7_stages_fill_channels_buffer(channels, channels_buffer_0,
					h * channels_buffer_0_rows_filled_each_time
							+ extra_rows_filled_first_time);
			_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
					_6_layer_0_3x3_conv_out_0);
			_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2,
					_6_layer_2_dw_upper, _6_layer_2_dw_out_1, 0);
			_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3,
					_6_layer_3_pw_out_0);
			_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
					_6_layer_5_dw_upper, _6_layer_5_dw_lower,
					_6_layer_4_5_pw_dw_out_1, _6_layer_4_pw_5_dw_starting_h);
			_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6,
					_6_layer_6_pw_out_0);
			write_to_tmp_channel(_6_layer_6_pw_out_0, tmp_channels,
					h - pipeline_filling_stages + 1);
			_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
					h - pipeline_filling_stages);
		}
		odd_even = 1 - odd_even;
		_6_layer_4_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
	}
//###########################################################
// pipeline flushing##########################################
	_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages);
//##########
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, _6_layer_6_pw_out_0);
	write_to_tmp_channel(_6_layer_6_pw_out_0, tmp_channels,
			h - pipeline_filling_stages + 1);
	_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages + 1);
//##########
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			_6_layer_4_pw_5_dw_starting_h);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, _6_layer_6_pw_out_1);
	write_to_tmp_channel(_6_layer_6_pw_out_1, tmp_channels,
			h - pipeline_filling_stages + 2);
	_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages + 2);
	_6_layer_4_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
//##########
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			_6_layer_4_pw_5_dw_starting_h);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, _6_layer_6_pw_out_0);
	write_to_tmp_channel(_6_layer_6_pw_out_0, tmp_channels,
			h - pipeline_filling_stages + 3);
	_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages + 3);
	_6_layer_4_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
//##########
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_out_1, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			_6_layer_4_pw_5_dw_starting_h);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, _6_layer_6_pw_out_1);
	write_to_tmp_channel(_6_layer_6_pw_out_1, tmp_channels,
			h - pipeline_filling_stages + 4);
	_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages + 4);
	_6_layer_4_pw_5_dw_starting_h += _7_stages_layer_4_rows_at_once;
//##########
//	cout << "\n_6_layer_4_pw_5_dw_starting_h " << _6_layer_4_pw_5_dw_starting_h
//			<< "\n";
//padding bottom
// by the end of the previous _6_layer_2_dw, _6_layer_2_dw_upper contains the last two valid rows of the first layer output.
//This time, _6_layer_2_dw will produce only one valid row
	for (int d = 0; d < layer_2_dw_depth; d++) {
		for (int h = 0; h < layer_2_dw_padding_bottom; h++) {
			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
				_6_layer_0_3x3_conv_out_0[d][h][w] = layer_2_ifms_zero_point;
			}
		}
	}
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_out_0, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			_6_layer_4_pw_5_dw_starting_h);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, _6_layer_6_pw_out_0);
	write_to_tmp_channel(_6_layer_6_pw_out_0, tmp_channels,
			h - pipeline_filling_stages + 5);
	_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages + 5);
//##########

//	_7_stages_fill_channels_buffer(channels, channels_buffer_0,
//			h * channels_buffer_0_rows_filled_each_time
//					+ extra_rows_filled_first_time);
//	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
//			_6_layer_0_3x3_conv_out_1);
//	//****************
//	for (int d = 0; d < layer_2_dw_depth; d++) {
//		for (int h = _7_stages_layer_0_rows_at_once - layer_2_dw_padding_bottom;
//				h < _7_stages_layer_0_rows_at_once; h++) {
//			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//				_6_layer_0_3x3_conv_out_1[d][h][w] = layer_2_ifms_zero_point;
//			}
//		}
//	}
//	//****************
//	//****************************
//
////	cout << "\n*******1*********\n";
////	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
////		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
////			cout << _6_layer_2_dw_upper[0][h][w] << " ";
////		}
////		cout << "\n";
////	}
////	cout << "\n*********1*******\n";
////	cout << "\n*******2*********\n";
////	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
////		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
////			cout << _6_layer_0_3x3_conv_out_0[0][h][w] << " ";
////		}
////		cout << "\n";
////	}
////	cout << "\n*********2*******\n";
//	//##########
//	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
//			_6_layer_2_dw_lower, _6_layer_2_dw_out_1, 0);
//	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
//	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
//			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
//			1);
//	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, _6_layer_6_pw_out_1);
//	_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
//			switch_point_fms_height - pipeline_filling_stages - 5);

//#########
//	_7_stages_fill_channels_buffer(channels, channels_buffer_0,
//			(switch_point_fms_height * _7_stages_layer_0_rows_at_once - 1)
//					* layer_0_strides);
//	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
//			_6_layer_0_3x3_conv_out_0);
//	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
//			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 0);
//	// padding bottom
//	for (int d = 0; d < layer_3_pw_depth; d++) {
//		for (int w = 0; w < layer_3_pw_ifm_width; w++) {
//			_6_layer_2_dw_out_0[d][_7_stages_layer_0_rows_at_once - 1][w] = 0;
//		}
//	}
//	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
//	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
//			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
//			1);
//	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, _6_layer_6_pw_out_0);
//	_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
//			switch_point_fms_height - 1);
}
