#include <iostream>
#include <math.h>
#include "../headers/pipeline.h"

using namespace std;

void _7_layer_6_pw(
		fms_dt channels_buffer[layer_6_pw_depth][layer_6_pw_ifm_width],
		const weights_dt weights[layer_6_pw_num_fils][layer_6_pw_depth],
		fms_dt result[layer_7_pw_depth][layer_7_pw_ifm_width]) {

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
				normalization.ofm_scale_rec = conv_fms_scales_rec[6 + 1];
				result[o_o_d_offset + o_d][w] = dw_relu_norm(tmp, normalization,
						6);
			}
		}
	}
}

void _7_layer_7_pw(
		fms_dt channels_buffer[layer_7_pw_depth][layer_7_pw_ifm_width],
		const weights_dt weights[layer_7_pw_num_fils][layer_7_pw_depth],
		fms_dt result[max_fms_size], int starting_h) {

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
			o_o_d < layer_6_pw_num_fils / layer_7_pw_parallelism_out; o_o_d++) {
		int o_o_d_offset = o_o_d * layer_7_pw_parallelism_out;
		// filters loop
		layer_7_pw_pipeline: for (int w = 0; w < layer_7_pw_ifm_width; w++) {
#pragma HLS PIPELINE
			// FMs width loop
			layer_7_pw_loops: for (int o_d = 0;
					o_d < layer_7_pw_parallelism_out; o_d++) {
#pragma HLS UNROLL
				// parallelized filters loop
				pss_dt tmp = 0;
				int offset_in_result =
						(o_o_d * layer_7_pw_parallelism_out + o_d)
								* switch_point_fms_height
								* switch_point_fms_width
								+ starting_h * switch_point_fms_width + w;
				for (int d = 0; d < layer_7_pw_parallelism_in; d++) {
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += ((fms_dt) channels_buffer[d][w]) * weights[o_d][d];
				}
				fms_quantization_scheme normalization = { 0, 0, 0, 0 };
				normalization.fused_scales =
						fused_scales[layers_fused_parameters_offsets[7] + o_d];
				normalization.fused_zero_point =
						fused_zero_points[layers_fused_parameters_offsets[7]
								+ o_d];
				normalization.ofm_zero_point = conv_fms_zero_points[7 + 1];
				normalization.ofm_scale_rec = conv_fms_scales_rec[7 + 1];
				result[offset_in_result] = pw_relu_norm(tmp, normalization,
						layer_7_relu);
			}
		}
	}
}

void _7_stages_fill_channels_buffer(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim
				+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides][input_image_width],
		int starting_h) {

	const fms_dt current_layer_zero_point = conv_fms_zero_points[0];

	const int buffer_height = layer_0_filter_dim
			+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides;
	const int rows_to_shift = layer_0_filter_dim - layer_0_strides;

	const int shift_starting_point =
			starting_h > buffer_height - rows_to_shift ?
					buffer_height - rows_to_shift : starting_h - 1;
	const int fill_starting_point = rows_to_shift;
	const int first_time_offset =
			starting_h == 0 ? (layer_0_filter_dim - layer_0_strides) : 0;
	//shift or fill first
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
	//fill
	for (int w = 0; w < input_image_width; w++) {
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++) {
#pragma HLS UNROLL
			for (int h = fill_starting_point; h < buffer_height; h++) {
#pragma HLS UNROLL
				if (starting_h + h - fill_starting_point < input_image_height) {
					channels_buffer_0[d][h][w] = channels[d][starting_h
							+ first_time_offset + h - fill_starting_point][w];
				} else {
					channels_buffer_0[d][h][w] = current_layer_zero_point;
				}
			}
		}
	}
}

void cnn_pipeline_7_mob_v2(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size]) {
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
	const int _6_layer_2_dw_upper_height = layer_2_dw_filter_size
			- layer_2_dw_strides;
	const fms_dt layer_2_ifms_zero_point = conv_fms_zero_points[2];
	for (int d = 0; d < layer_2_dw_depth; d++) {
		for (int h = 0; h < layer_2_dw_padding_top; h++) {
			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
				_6_layer_2_dw_upper[d][h][w] = layer_2_ifms_zero_point;
			}
		}
		for (int h = layer_2_dw_padding_top; h < _6_layer_2_dw_upper_height;
				h++) {
			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
				_6_layer_2_dw_upper[d][h][w] = _6_layer_0_3x3_conv_out_0[d][h
						- layer_2_dw_padding_top][w];
			}
		}
	}
	//****************************

	cout << "\n*******1*********\n";
	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
			cout << _6_layer_2_dw_upper[0][h][w] << " ";
		}
		cout << "\n";
	}
	cout << "\n*********1*******\n";

//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 3);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	cout << "\n*******2*********\n";
	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
			cout << _6_layer_0_3x3_conv_out_0[0][h][w] << " ";
		}
		cout << "\n";
	}
	cout << "\n*********2*******\n";
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 7);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_1);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 1);
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 11);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_1, 0);
	cout << "\n*******3*********\n";
		for (int h = 0; h < _6_stages_layer_2_rows_at_once; h++) {
			for (int w = 0; w < layer_3_pw_ifm_width; w++) {
				cout << _6_layer_2_dw_out_0[0][h][w] << " ";
			}
			cout << "\n";
		}
		cout << "\n*********3*******\n";
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 15);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_1);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			1);
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0, 19);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_0);
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_1, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			1);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, _6_layer_6_pw_out_0);
	int even_odd = 1;
	const int pipeline_filling_stages = 6;
	int h = pipeline_filling_stages;
	const int channels_buffer_0_height = layer_0_filter_dim
			+ (_7_stages_layer_0_rows_at_once - 1) * layer_0_strides;
	const int channels_buffer_0_rows_filled_each_time =
			_7_stages_layer_0_rows_at_once * layer_0_strides;
	const int channels_buffer_0_rows_filled_first_time = 3;

	const int less_rows_filled_first_time =
			channels_buffer_0_rows_filled_each_time
					- channels_buffer_0_rows_filled_first_time;

	main_pipeline_loop: for (; h < switch_point_fms_height; h++) {
		if (even_odd) {
			_7_stages_fill_channels_buffer(channels, channels_buffer_0,
					h * channels_buffer_0_rows_filled_each_time
							- less_rows_filled_first_time);
			_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
					_6_layer_0_3x3_conv_out_1);
			_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2,
					_6_layer_2_dw_upper, _6_layer_2_dw_lower,
					_6_layer_2_dw_out_0, 0);
			_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3,
					_6_layer_3_pw_out_1);
			_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
					_6_layer_5_dw_upper, _6_layer_5_dw_lower,
					_6_layer_4_5_pw_dw_out_0, 1);
			_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6,
					_6_layer_6_pw_out_1);
			_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
					h - pipeline_filling_stages);
		} else {
			_7_stages_fill_channels_buffer(channels, channels_buffer_0,
					h * channels_buffer_0_rows_filled_each_time
							- less_rows_filled_first_time);
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
			_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6,
					_6_layer_6_pw_out_0);
			_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
					h - pipeline_filling_stages);
		}
		even_odd = 1 - even_odd;
	}
//###########################################################
// pipeline flushing##########################################
	_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages);
//##########
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, _6_layer_6_pw_out_1);
	_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages - 1);
//##########
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			1);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, _6_layer_6_pw_out_0);
	_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages - 2);
//##########
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			1);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, _6_layer_6_pw_out_1);
	_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages - 3);
//##########
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_0, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_0, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_0, pw_weights_3, _6_layer_3_pw_out_0);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_0, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_0,
			1);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_0, pw_weights_6, _6_layer_6_pw_out_0);
	_7_layer_7_pw(_6_layer_6_pw_out_0, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages - 4);
//##########
	_7_stages_fill_channels_buffer(channels, channels_buffer_0,
			switch_point_fms_height * channels_buffer_0_rows_filled_each_time
					- less_rows_filled_first_time);
	_6_layer_0_3x3_conv(channels_buffer_0, weights_0,
			_6_layer_0_3x3_conv_out_1);
	//****************
	for (int d = 0; d < layer_2_dw_depth; d++) {
		for (int h = _7_stages_layer_0_rows_at_once - layer_2_dw_padding_bottom;
				h < _7_stages_layer_0_rows_at_once; h++) {
			for (int w = 0; w < layer_2_dw_ifm_width; w++) {
				_6_layer_0_3x3_conv_out_1[d][h][w] = layer_2_ifms_zero_point;
			}
		}
	}
	//****************
	//****************************

//	cout << "\n*******1*********\n";
//	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
//		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//			cout << _6_layer_2_dw_upper[0][h][w] << " ";
//		}
//		cout << "\n";
//	}
//	cout << "\n*********1*******\n";
//	cout << "\n*******2*********\n";
//	for (int h = 0; h < _6_stages_layer_0_rows_at_once; h++) {
//		for (int w = 0; w < layer_2_dw_ifm_width; w++) {
//			cout << _6_layer_0_3x3_conv_out_0[0][h][w] << " ";
//		}
//		cout << "\n";
//	}
//	cout << "\n*********2*******\n";
	//##########
	_6_layer_2_dw(_6_layer_0_3x3_conv_out_1, dw_weights_2, _6_layer_2_dw_upper,
			_6_layer_2_dw_lower, _6_layer_2_dw_out_1, 0);
	_6_layer_3_pw(_6_layer_2_dw_out_1, pw_weights_3, _6_layer_3_pw_out_1);
	_6_layer_4_pw_5_dw(_6_layer_3_pw_out_1, pw_weights_4, dw_weights_5,
			_6_layer_5_dw_upper, _6_layer_5_dw_lower, _6_layer_4_5_pw_dw_out_1,
			1);
	_7_layer_6_pw(_6_layer_4_5_pw_dw_out_1, pw_weights_6, _6_layer_6_pw_out_1);
	_7_layer_7_pw(_6_layer_6_pw_out_1, pw_weights_7, result,
			switch_point_fms_height - pipeline_filling_stages - 5);
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
