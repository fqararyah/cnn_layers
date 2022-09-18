#include "./headers/cnn_functions.h"
#include "./headers/utils.h"

void _4_layer_2_pw(
	fms_dt channels_buffer[layer_2_pw_depth][layer_2_pw_ifm_width],
	weights_dt weights[layer_2_pw_num_fils][layer_2_pw_depth],
	fms_dt result[max_fms_size], int starting_h)
{

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < layer_2_pw_num_fils / layer_2_pw_parallelism_out;
		 o_o_d++)
	{
		int o_o_d_offset = o_o_d * layer_2_pw_parallelism_out;
		// filters loop
	layer_2_pw_pipeline:
		for (int w = 0; w < layer_2_pw_ifm_width; w++)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_2_pw_loops:
			for (int o_d = 0;
				 o_d < layer_2_pw_parallelism_out; o_d++)
			{
#pragma HLS UNROLL
				// parallelized filters loop
				pss_dt tmp = 0;
				for (int d = 0; d < layer_2_pw_parallelism_in; d++)
				{
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += ((fms_dt)channels_buffer[d][w]) * weights[o_o_d_offset + o_d][d];
				}
				fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
				if (scaled_val > 0)
				{
					result[(o_o_d_offset + o_d) * switch_point_fms_height * switch_point_fms_width + starting_h * switch_point_fms_width + w] = scaled_val;
				}
			}
		}
	}
}

void mobilenet_v2_pipeline_4(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size])
{
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

	dw_weights_dt dw_weights_1[layer_1_dw_depth][3][3];

#pragma HLS ARRAY_PARTITION variable = dw_weights_1 type = complete dim = 1

	layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3];

	weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth];
	weights_dt pw_weights_2[layer_2_pw_num_fils][layer_2_pw_depth];
	weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth];
	_5_fill_layers_weights(weights_0, dw_weights_1, pw_weights_1, pw_weights_2,
						   pw_weights_3);

	//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width];

	fms_dt _5_layer_0_3x3_conv_out_0[layer_1_pw_depth][layer_1_pw_ifm_width] =
		{0};

	fms_dt _5_layer_1_pw_out_0[layer_1_dw_depth][layer_1_dw_ifm_width] = {0};

	fms_dt _5_layer_1_dw_upper[layer_1_dw_depth][2][layer_1_dw_ifm_width] = {0};
	fms_dt _5_layer_1_dw_lower[layer_1_dw_depth][layer_1_dw_strides][layer_1_dw_ifm_width] = {0};
	fms_dt _5_layer_1_dw_out_0[layer_2_pw_depth][layer_2_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_0_3x3_conv_out_0 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_upper cyclic factor = 3 dim = 2

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_out_0 cyclic factor = layer_2_pw_parallelism_in/2 dim = 1

	//###########################################################

	//#########################odd###############################
	fms_dt channels_buffer_1[input_image_depth][layer_0_filter_size + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width];

	fms_dt _5_layer_0_3x3_conv_out_1[layer_1_pw_depth][layer_1_pw_ifm_width] =
		{0};

	fms_dt _5_layer_1_pw_out_1[layer_1_dw_depth][layer_1_dw_ifm_width] = {0};

	fms_dt _5_layer_1_dw_out_1[layer_2_pw_depth][layer_2_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 cyclic factor = 3 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_0_3x3_conv_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_out_1 cyclic factor = layer_2_pw_parallelism_in/2 dim = 1

	//###########################################################
	// pipeline filling###########################################
	_5_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
	//##########
	_5_stages_fill_channels_buffer(channels, channels_buffer_1, 2);
	_5_layer_0_3x3_conv(channels_buffer_0, weights_0,
						_5_layer_0_3x3_conv_out_0);
	//##########
	_5_stages_fill_channels_buffer(channels, channels_buffer_0, 4);
	_5_layer_0_3x3_conv(channels_buffer_1, weights_0,
						_5_layer_0_3x3_conv_out_1);
	_5_layer_1_pw_dw(_5_layer_0_3x3_conv_out_0, pw_weights_1, dw_weights_1,
					 _5_layer_1_dw_upper, _5_layer_1_dw_lower, _5_layer_1_dw_out_0, 0);
	//this sequence does not produce a valid _5_layer_1_dw_out_0 yet, as it needs 2
	//rows but only 1 has been feed so far
	//##########
	_5_stages_fill_channels_buffer(channels, channels_buffer_1, 6);
	_5_layer_0_3x3_conv(channels_buffer_0, weights_0,
						_5_layer_0_3x3_conv_out_0);
	_5_layer_1_pw_dw(_5_layer_0_3x3_conv_out_1, pw_weights_1, dw_weights_1,
					 _5_layer_1_dw_upper, _5_layer_1_dw_lower, _5_layer_1_dw_out_1, 1);
	//##########

	int even_odd = 1;
	int h = 4;
main_pipeline_loop:
	for (; h < switch_point_fms_height; h++)
	{
		if (even_odd)
		{
			_5_stages_fill_channels_buffer(channels, channels_buffer_0, h * _5_stages_layer_1_rows_at_once * layer_0_strides);
			_5_layer_0_3x3_conv(channels_buffer_1, weights_0,
								_5_layer_0_3x3_conv_out_1);
			_5_layer_1_pw_dw(
				_5_layer_0_3x3_conv_out_0,
				pw_weights_1,
				dw_weights_1,
				_5_layer_1_dw_upper,
				_5_layer_1_dw_lower,
				_5_layer_1_dw_out_0,
				1);
			_4_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_2,
						  result, h - 4);
		}
		else
		{
			_5_stages_fill_channels_buffer(channels, channels_buffer_1, h);
			_5_layer_0_3x3_conv(channels_buffer_0, weights_0,
								_5_layer_0_3x3_conv_out_0);
			_5_layer_1_pw_dw(
				_5_layer_0_3x3_conv_out_1,
				pw_weights_1,
				dw_weights_1,
				_5_layer_1_dw_upper,
				_5_layer_1_dw_lower,
				_5_layer_1_dw_out_1,
				1);
			_4_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_2,
						  result, h - 4);
		}
		even_odd = 1 - even_odd;
	}
	//###########################################################
	// pipeline flushing##########################################
	_4_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_2,
				  result, h - 4);
	//##########
	_5_layer_1_pw_dw(
		_5_layer_0_3x3_conv_out_0,
		pw_weights_1,
		dw_weights_1,
		_5_layer_1_dw_upper,
		_5_layer_1_dw_lower,
		_5_layer_1_dw_out_0,
		1);
	_4_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_2,
				  result, h - 3);
	//##########
	_5_layer_0_3x3_conv(channels_buffer_1, weights_0,
						_5_layer_0_3x3_conv_out_1);
	_5_layer_1_pw_dw(
		_5_layer_0_3x3_conv_out_1,
		pw_weights_1,
		dw_weights_1,
		_5_layer_1_dw_upper,
		_5_layer_1_dw_lower,
		_5_layer_1_dw_out_1,
		1);
	_4_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_2,
				  result, h - 2);
	//##########
	fms_dt padding_bottom[layer_1_pw_depth][layer_1_pw_ifm_width] = {0};
	_5_layer_1_pw_dw(
		padding_bottom,
		pw_weights_1,
		dw_weights_1,
		_5_layer_1_dw_upper,
		_5_layer_1_dw_lower,
		_5_layer_1_dw_out_0,
		1);
	_4_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_2,
				  result, h - 2);
}
