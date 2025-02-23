#include "./headers/utils.h"

void _5_layer_0_3x3_conv(
	fms_dt channels_buffer[input_image_depth][first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
	layer_0_weights_dt weights[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim],
	fms_dt result[layer_1_dw_depth][layer_1_dw_ifm_width])
{
#pragma HLS INLINE off

	fms_dt intermediate_channels_buffer[input_image_depth][first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][first_conv_layer_filter_dim] = {0};
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

	// fill the intermediate_channels_buffer
	for (int d = 0; d < input_image_depth; d++)
	{
		for (int h = 0; h < first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides; h++)
		{
			for (int w = 0; w < first_conv_layer_filter_dim - layer_0_padding_left; w++)
			{
				intermediate_channels_buffer[d][h][w + layer_0_padding_left] = channels_buffer[d][h][w];
			}
		}
	}
	// end fill the intermediate_channels_buffer

layer_0_ofms:
	for (int o_o_d = 0;
		 o_o_d < first_conv_layer_num_fils / sesl_layer_0_parallelism_ofms; o_o_d++)
	{
		// outer filters loop
		int o_o_d_offset = o_o_d * sesl_layer_0_parallelism_ofms; // for indexing in depth

	layer_0_pipeline:
		for (int w = 0; w < input_image_width; w +=
													layer_0_strides)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_0_parallelized_ofms:
			for (int o_d = 0;
				 o_d < sesl_layer_0_parallelism_ofms; o_d++)
			{
				first_conv_pss_dt tmp = 0;
#pragma HLS UNROLL
			// parallelized filters loop
			layer_0_d_loops:
				for (int d = 0; d < input_image_depth; d++)
				{
#pragma HLS UNROLL
				// parallelized depth loop
				layer_0_ch:
					for (int h = 0; h < first_conv_layer_filter_dim; h++)
					{
#pragma HLS UNROLL
					// conv height loop
					layer_0_cw:
						for (int c_w = 0; c_w < first_conv_layer_filter_dim; c_w++)
						{
#pragma HLS UNROLL
							// conv width loop
							tmp += intermediate_channels_buffer[d][h][c_w] * weights[o_o_d_offset + o_d][d][h][c_w];
						}
					}
				}
				fms_dt scaled_val = (fms_dt)(((ap_fixed<17, 12>)tmp - zero_point_dw) * ratio_dw_pss_to_fms);
				if (scaled_val > 0)
				{
					result[o_o_d_offset + o_d][w / layer_0_strides] =
						scaled_val;
				}
			}
			if (w < input_image_width - layer_0_strides)
			{
				// shift and fill the intermediate_channels_buffer
				for (int d = 0; d < input_image_depth; d++)
				{
#pragma HLS UNROLL
					for (int c_h = 0; c_h < first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides; c_h++)
					{
#pragma HLS UNROLL
						for (int c_w = 0; c_w < first_conv_layer_filter_dim - layer_0_strides; c_w++)
						{
#pragma HLS UNROLL
							intermediate_channels_buffer[d][c_h][c_w] = intermediate_channels_buffer[d][c_h][c_w + layer_0_strides];
						}
						for (int c_w = first_conv_layer_filter_dim - layer_0_strides; c_w < first_conv_layer_filter_dim; c_w++)
						{
#pragma HLS UNROLL
							intermediate_channels_buffer[d][c_h][c_w] = channels_buffer[d][c_h][c_w - (first_conv_layer_filter_dim - layer_0_strides) +
																								(w + first_conv_layer_filter_dim - layer_0_padding_left)];
						}
					}
				}
				// end shift and fill the intermediate_channels_buffer
			}
		}
	}
}

void _5_layer_1_pw_dw(
	fms_dt channels_buffer[layer_1_pw_depth][layer_1_pw_ifm_width],
	weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
	dw_weights_dt dw_weights[layer_1_dw_depth][layer_2_dw_specs.filter_size][layer_2_dw_specs.filter_size],
	fms_dt upper[layer_1_dw_depth][layer_2_dw_specs.filter_size - layer_2_dw_specs.strides][layer_1_dw_ifm_width],
	fms_dt lower[layer_1_dw_depth][layer_2_dw_specs.strides][layer_1_dw_ifm_width],
	fms_dt result[layer_2_pw_depth][layer_2_pw_ifm_width], int active_row)
{

	fms_dt intermediate_pw_results[layer_1_pw_parallelism_out][layer_2_dw_specs.filter_size][layer_2_dw_specs.filter_size];
#pragma HLS ARRAY_PARTITION variable = intermediate_pw_results type = complete dim = 0

#pragma HLS INLINE off
layer_1_pw__dw_main_loop:
	for (int h = 0; h < _5_stages_layer_1_rows_at_once;
		 h++)
	{
		// rows for next DW
		for (int o_o_d = 0;
			 o_o_d < layer_1_pw_num_fils / layer_1_pw_parallelism_out;
			 o_o_d++)
		{
			int o_o_d_offset = o_o_d * layer_1_pw_parallelism_out;

			// fill upper and lower (except last) rows:
			for (int d = 0; d < layer_1_pw_parallelism_out; d++)
			{
#pragma HLS UNROLL
				for (int h = 0; h < layer_2_dw_specs.filter_size - layer_2_dw_specs.strides; h++)
				{ // fill first two rows as the third will be filled by the pw results
#pragma HLS UNROLL
				  //  padding
					intermediate_pw_results[d][h][0] = 0;
					for (int w = layer_2_dw_specs.padding_left; w < layer_2_dw_specs.filter_size; w++)
					{
#pragma HLS UNROLL
						intermediate_pw_results[d][h][w] = upper[o_o_d_offset + d][h][w - layer_2_dw_specs.padding_left];
					}
				}
				intermediate_pw_results[d][layer_2_dw_specs.filter_size - 1][0] = 0;
			}

		layer_1_pw_pipeline:
			for (int w = 0; w < layer_1_dw_ifm_width + v1_layer_3_dw_filter_size - (v1_layer_3_dw_padding_left + layer_2_dw_specs.padding_right);
				 w++)
			{
#pragma HLS PIPELINE
			//###################PW#######################
			layer_1_pw_loops:
				for (int o_d = 0;
					 o_d < layer_1_pw_parallelism_out; o_d++)
				{
#pragma HLS UNROLL
					// parallelized filters loop
					if (w < layer_1_dw_ifm_width)
					{
						// FMs width loop
						pss_dt tmp = 0;
						for (int d = 0; d < layer_1_pw_parallelism_in; d++)
						{
#pragma HLS UNROLL
							// parallelized depth loop
							tmp +=
								((fms_dt)channels_buffer[d][h][w]) * weights[o_o_d_offset + o_d][d];
						}
						fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
						if (scaled_val > 0)
						{
							if (w + layer_2_dw_specs.padding_left < layer_2_dw_specs.filter_size)
							{
								intermediate_pw_results[o_d][layer_2_dw_specs.filter_size - 1][w + layer_2_dw_specs.padding_left] = scaled_val;
							}
							lower[o_o_d_offset + o_d][0][w] = scaled_val;
						}
					}
					//###############end PW####################
					//###############DW########################
					if (w + 1 >= layer_2_dw_specs.filter_size - layer_2_dw_specs.padding_left)
					{
						if (active_row)
						{
							if (w < layer_1_dw_ifm_width)
							{
								intermediate_pw_results[o_d][layer_2_dw_specs.filter_size - 1][layer_2_dw_specs.filter_size - layer_2_dw_specs.padding_left] =
									lower[o_o_d_offset + o_d][0][w];
							}
							else
							{
								intermediate_pw_results[o_d][layer_2_dw_specs.filter_size - 1][layer_2_dw_specs.filter_size - layer_2_dw_specs.padding_left] = 0;
							}

							dw_pss_dt tmp = 0;
							// parallelized depth loop
							for (int c_h = 0; c_h < layer_2_dw_specs.filter_size; c_h++)
							{
#pragma HLS UNROLL
								for (int c_w = 0; c_w < layer_2_dw_specs.filter_size; c_w++)
								{
									// conv width loop
#pragma HLS UNROLL
									tmp += intermediate_pw_results[o_d][c_h][c_w] * dw_weights[o_o_d_offset + o_d][c_h][c_w];
								}
							}
							fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
							if (scaled_val > 0)
							{
								result[o_o_d_offset + o_d][(w + 1 - (layer_2_dw_specs.filter_size - layer_2_dw_specs.padding_left)) / layer_2_dw_specs.strides] =
									scaled_val;
							}
							//#####################end DW################
							//#####################shift and fill intermediate#################
							for (int c_h = 0; c_h < layer_2_dw_specs.filter_size; c_h++)
							{
#pragma HLS UNROLL
								for (int c_w = 0; c_w < layer_2_dw_specs.filter_size - layer_2_dw_specs.strides; c_w++)
								{
#pragma HLS UNROLL
									intermediate_pw_results[o_d][c_h][c_w] = intermediate_pw_results[o_d][c_h][c_w + layer_2_dw_specs.strides];
								}
								if (c_h < layer_2_dw_specs.filter_size - layer_2_dw_specs.strides)
								{
									for (int c_w = layer_2_dw_specs.filter_size - layer_2_dw_specs.strides; c_w < layer_2_dw_specs.filter_size; c_w++)
									{
#pragma HLS UNROLL
										if (w < layer_1_dw_ifm_width)
										{
											intermediate_pw_results[o_d][c_h][c_w] = upper[o_d][c_h][w + layer_2_dw_specs.padding_left];
										}
										else
										{
											intermediate_pw_results[o_d][c_h][c_w] = 0;
										}
									}
								}
							}
							//#####################end shift and fill intermediate#################
						}
						//#####################shift#################
						for (int c_h = 0; c_h < layer_2_dw_specs.filter_size - 2 * layer_2_dw_specs.strides; c_h++)
						{
#pragma HLS UNROLL
							for (int c_w = 0; c_w < layer_2_dw_specs.strides; c_w++)
							{
#pragma HLS UNROLL
								upper[o_o_d_offset + o_d][c_h][(w + 1 - (layer_2_dw_specs.filter_size - 1)) + c_w] =
									upper[o_o_d_offset + o_d][c_h + layer_2_dw_specs.strides][(w + 1 - (layer_2_dw_specs.filter_size - layer_2_dw_specs.padding_left)) + c_w];
							}
						}
						for (int c_h = 0; c_h < layer_2_dw_specs.strides; c_h++)
						{
#pragma HLS UNROLL
							for (int c_w = 0; c_w < layer_2_dw_specs.strides; c_w++)
							{
#pragma HLS UNROLL
								upper[o_o_d_offset + o_d][c_h + layer_2_dw_specs.filter_size - 2 * layer_2_dw_specs.strides]
								[(w + 1 - (layer_2_dw_specs.filter_size - layer_2_dw_specs.padding_left)) + c_w] =
									lower[o_o_d_offset + o_d][c_h][(w + 1 - (layer_2_dw_specs.filter_size - layer_2_dw_specs.padding_left)) + c_w];
							}
						}
						//#####################end shift#################
					}
				}
			}
		}
	}
}

void _5_layer_2_pw(
	fms_dt channels_buffer[layer_2_pw_depth][layer_2_pw_ifm_width],
	weights_dt weights[layer_3_pw_specs.num_fils][layer_2_pw_depth],
	fms_dt result[layer_3_pw_num_fils][layer_3_dw_ifm_width])
{

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < layer_3_pw_specs.num_fils / layer_2_pw_parallelism_out;
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
					result[o_o_d_offset + o_d][w] = scaled_val;
				}
			}
		}
	}
}

void _5_layer_3_pw(
	fms_dt channels_buffer[layer_3_pw_depth][layer_3_dw_ifm_width],
	weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
	fms_dt result[max_fms_size], int starting_h)
{

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < layer_3_pw_num_fils / layer_3_pw_parallelism_out;
		 o_o_d++)
	{
		int o_o_d_offset = o_o_d * layer_3_pw_parallelism_out;
		// filters loop
	layer_3_pw_pipeline:
		for (int w = 0; w < layer_3_pw_ifm_width; w++)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_3_pw_loops:
			for (int o_d = 0;
				 o_d < layer_3_pw_parallelism_out; o_d++)
			{
#pragma HLS UNROLL
				// parallelized filters loop
				pss_dt tmp = 0;
				for (int d = 0; d < layer_3_pw_parallelism_in; d++)
				{
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += ((fms_dt)channels_buffer[d][w]) * weights[o_o_d_offset + o_d][d];
				}
				fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
				if (scaled_val > 0)
				{
					result[(o_o_d_offset + o_d) * switch_point_fms_height * switch_point_fms_width + starting_h * switch_point_fms_width + w] =
						scaled_val; // must be fully unrolled in the IFMs depth dimension
				}
			}
		}
	}
}

void _5_stages_fill_channels_buffer(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt channels_buffer_0[input_image_depth][first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
	int starting_h)
{

#pragma HLS INLINE off

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides - layer_0_strides; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = channels_buffer_0[d][h + layer_0_strides][w];
			}
		}
	}

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides - layer_0_strides; h < first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = channels[d][starting_h + h - (first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides - layer_0_strides)][w];
			}
		}
	}
}

void mobilenet_v2_pipeline_5(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size])
{
#pragma HLS INLINE off

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

	dw_weights_dt dw_weights_2[layer_1_dw_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim];

#pragma HLS ARRAY_PARTITION variable = dw_weights_2 type = complete dim = 1

	layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][layer_2_dw_specs.filter_size][layer_2_dw_specs.filter_size];

	weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth];
	weights_dt pw_weights_3[layer_3_pw_specs.num_fils][layer_2_pw_depth];
	weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth];
	_5_fill_layers_weights(weights_1, dw_weights_2, pw_weights_1, pw_weights_3,
						   pw_weights_3);

	//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width] = {0};

	fms_dt _5_layer_0_3x3_conv_out_0[layer_1_pw_depth][layer_1_pw_ifm_width] =
		{0};

	fms_dt _5_layer_1_pw_out_0[layer_1_dw_depth][layer_1_dw_ifm_width] = {0};

	fms_dt _5_layer_1_dw_upper[layer_1_dw_depth][2][layer_1_dw_ifm_width] = {0};
	fms_dt _5_layer_1_dw_lower[layer_1_dw_depth][layer_2_dw_specs.strides][layer_1_dw_ifm_width] = {0};
	fms_dt _5_layer_1_dw_out_0[layer_2_pw_depth][layer_2_pw_ifm_width] = {0};

	fms_dt _5_layer_2_pw_out_0[layer_3_pw_depth][layer_3_dw_ifm_width] = {0};

	fms_dt _5_layer_3_pw_out_0[layer_3_pw_depth][layer_3_dw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_0_3x3_conv_out_0 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_upper cyclic factor = 3 dim = 2

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_out_0 cyclic factor = layer_2_pw_parallelism_in / 2 dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_2_pw_out_0 cyclic factor = layer_3_pw_parallelism_in / 2 dim = 1
	//###########################################################

	//#########################odd###############################
	fms_dt channels_buffer_1[input_image_depth][first_conv_layer_filter_dim + (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width] = {0};

	fms_dt _5_layer_0_3x3_conv_out_1[layer_1_pw_depth][layer_1_pw_ifm_width] =
		{0};

	fms_dt _5_layer_1_pw_out_1[layer_1_dw_depth][layer_1_dw_ifm_width] = {0};

	fms_dt _5_layer_1_dw_out_1[layer_2_pw_depth][layer_2_pw_ifm_width] = {0};

	fms_dt _5_layer_2_pw_out_1[layer_3_pw_depth][layer_3_dw_ifm_width] = {0};

	fms_dt _5_layer_3_pw_out_1[layer_3_pw_depth][layer_3_dw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 cyclic factor = 3 dim = 3

#pragma HLS ARRAY_PARTITION variable = _5_layer_0_3x3_conv_out_1 complete dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_1_dw_out_1 cyclic factor = layer_2_pw_parallelism_in / 2 dim = 1

#pragma HLS ARRAY_PARTITION variable = _5_layer_2_pw_out_1 cyclic factor = layer_3_pw_parallelism_in / 2 dim = 1

	//###########################################################
	// pipeline filling###########################################
	_5_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
	//##########
	_5_stages_fill_channels_buffer(channels, channels_buffer_1, 2);
	_5_layer_0_3x3_conv(channels_buffer_0, weights_1,
						_5_layer_0_3x3_conv_out_0);
	//##########
	_5_stages_fill_channels_buffer(channels, channels_buffer_0, 4);
	_5_layer_0_3x3_conv(channels_buffer_1, weights_1,
						_5_layer_0_3x3_conv_out_1);
	_5_layer_1_pw_dw(_5_layer_0_3x3_conv_out_0, pw_weights_1, dw_weights_2,
					 _5_layer_1_dw_upper, _5_layer_1_dw_lower, _5_layer_1_dw_out_0, 0);
	// this sequence does not produce a valid _5_layer_1_dw_out_0 yet, as it needs 2
	// rows but only 1 has been feed so far
	//##########
	_5_stages_fill_channels_buffer(channels, channels_buffer_1, 6);
	_5_layer_0_3x3_conv(channels_buffer_0, weights_1,
						_5_layer_0_3x3_conv_out_0);
	_5_layer_1_pw_dw(_5_layer_0_3x3_conv_out_1, pw_weights_1, dw_weights_2,
					 _5_layer_1_dw_upper, _5_layer_1_dw_lower, _5_layer_1_dw_out_1, 1);
	//##########
	_5_stages_fill_channels_buffer(channels, channels_buffer_0, 8);
	_5_layer_0_3x3_conv(channels_buffer_1, weights_1,
						_5_layer_0_3x3_conv_out_1);
	_5_layer_1_pw_dw(_5_layer_0_3x3_conv_out_0, pw_weights_1, dw_weights_2,
					 _5_layer_1_dw_upper, _5_layer_1_dw_lower, _5_layer_1_dw_out_0, 1);
	_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
				  _5_layer_2_pw_out_1);
	//##########
	int even_odd = 0;
	int h = 5;
main_pipeline_loop:
	for (; h < switch_point_fms_height; h++)
	{
		if (even_odd)
		{
			_5_stages_fill_channels_buffer(channels, channels_buffer_0, h * _5_stages_layer_1_rows_at_once * layer_0_strides);
			_5_layer_0_3x3_conv(channels_buffer_1, weights_1,
								_5_layer_0_3x3_conv_out_1);
			_5_layer_1_pw_dw(
				_5_layer_0_3x3_conv_out_0,
				pw_weights_1,
				dw_weights_2,
				_5_layer_1_dw_upper,
				_5_layer_1_dw_lower,
				_5_layer_1_dw_out_0,
				1);
			_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
						  _5_layer_2_pw_out_1);
			_5_layer_3_pw(_5_layer_2_pw_out_0, pw_weights_3,
						  result, h - 5);
		}
		else
		{
			_5_stages_fill_channels_buffer(channels, channels_buffer_1, h);
			_5_layer_0_3x3_conv(channels_buffer_0, weights_1,
								_5_layer_0_3x3_conv_out_0);
			_5_layer_1_pw_dw(
				_5_layer_0_3x3_conv_out_1,
				pw_weights_1,
				dw_weights_2,
				_5_layer_1_dw_upper,
				_5_layer_1_dw_lower,
				_5_layer_1_dw_out_1,
				1);
			_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
						  _5_layer_2_pw_out_0);
			_5_layer_3_pw(_5_layer_2_pw_out_1, pw_weights_3,
						  result, h - 5);
		}
		even_odd = 1 - even_odd;
	}
	//###########################################################
	// pipeline flushing##########################################
	_5_layer_3_pw(_5_layer_2_pw_out_0, pw_weights_3,
				  result, switch_point_fms_height - 5);
	//##########
	_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
				  _5_layer_2_pw_out_1);
	_5_layer_3_pw(_5_layer_2_pw_out_1, pw_weights_3,
				  result, switch_point_fms_height - 4);
	//##########
	_5_layer_1_pw_dw(
		_5_layer_0_3x3_conv_out_0,
		pw_weights_1,
		dw_weights_2,
		_5_layer_1_dw_upper,
		_5_layer_1_dw_lower,
		_5_layer_1_dw_out_0,
		1);
	_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
				  _5_layer_2_pw_out_0);
	_5_layer_3_pw(_5_layer_2_pw_out_0, pw_weights_3,
				  result, switch_point_fms_height - 3);
	//##########
	_5_layer_0_3x3_conv(channels_buffer_1, weights_1,
						_5_layer_0_3x3_conv_out_1);
	_5_layer_1_pw_dw(
		_5_layer_0_3x3_conv_out_1,
		pw_weights_1,
		dw_weights_2,
		_5_layer_1_dw_upper,
		_5_layer_1_dw_lower,
		_5_layer_1_dw_out_1,
		1);
	_5_layer_2_pw(_5_layer_1_dw_out_1, pw_weights_3,
				  _5_layer_2_pw_out_1);
	_5_layer_3_pw(_5_layer_2_pw_out_1, pw_weights_3,
				  result, switch_point_fms_height - 2);
	//##########
	fms_dt padding_bottom[layer_1_pw_depth][layer_1_pw_ifm_width] = {0};
	_5_layer_1_pw_dw(
		padding_bottom,
		pw_weights_1,
		dw_weights_2,
		_5_layer_1_dw_upper,
		_5_layer_1_dw_lower,
		_5_layer_1_dw_out_0,
		1);
	_5_layer_2_pw(_5_layer_1_dw_out_0, pw_weights_3,
				  _5_layer_2_pw_out_0);
	_5_layer_3_pw(_5_layer_2_pw_out_0, pw_weights_3,
				  result, switch_point_fms_height - 1);
}
