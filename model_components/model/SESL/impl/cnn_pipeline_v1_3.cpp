#include "./headers/utils.h"
#include "./headers/cnn_functions_v1.h"

void v1_3_layer_0_3x3_conv(
	fms_dt channels_buffer[input_image_depth][layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
	layer_0_weights_dt weights[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
	fms_dt result[v1_layer_1_dw_depth][v1_layer_1_dw_ifm_width])
{
#pragma HLS INLINE off

	fms_dt intermediate_channels_buffer[input_image_depth][layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides][layer_0_s_filter_dim] = {0};
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

	// fill the intermediate_channels_buffer
	for (int d = 0; d < input_image_depth; d++)
	{
		for (int h = 0; h < layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides; h++)
		{
			for (int w = 0; w < layer_0_s_filter_dim - layer_0_padding_left; w++)
			{
				intermediate_channels_buffer[d][h][w + layer_0_padding_left] = channels_buffer[d][h][w];
			}
		}
	}
	// end fill the intermediate_channels_buffer

layer_0_ofms:
	for (int o_o_d = 0;
		 o_o_d < layer_0_s_num_fils / v1_sesl_layer_0_parallelism_ofms; o_o_d++)
	{
		// outer filters loop
		int o_o_d_offset = o_o_d * v1_sesl_layer_0_parallelism_ofms; // for indexing in depth

	layer_0_pipeline:
		for (int w = 0; w < input_image_width; w +=
													layer_0_strides)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_0_parallelized_ofms:
			for (int o_d = 0;
				 o_d < v1_sesl_layer_0_parallelism_ofms; o_d++)
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
					for (int h = 0; h < layer_0_s_filter_dim; h++)
					{
#pragma HLS UNROLL
					// conv height loop
					layer_0_cw:
						for (int c_w = 0; c_w < layer_0_s_filter_dim; c_w++)
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
					for (int c_h = 0; c_h < layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides; c_h++)
					{
#pragma HLS UNROLL
						for (int c_w = 0; c_w < layer_0_s_filter_dim - layer_0_strides; c_w++)
						{
#pragma HLS UNROLL
							intermediate_channels_buffer[d][c_h][c_w] = intermediate_channels_buffer[d][c_h][c_w + layer_0_strides];
						}
						for (int c_w = layer_0_s_filter_dim - layer_0_strides; c_w < layer_0_s_filter_dim; c_w++)
						{
#pragma HLS UNROLL
							intermediate_channels_buffer[d][c_h][c_w] = channels_buffer[d][c_h][c_w - (layer_0_s_filter_dim - layer_0_strides) +
																								(w + layer_0_s_filter_dim - layer_0_padding_left)];
						}
					}
				}
				// end shift and fill the intermediate_channels_buffer
			}
		}
	}
}

void v1_3_layer_1_dw(fms_dt upper[v1_layer_1_dw_depth][v1_layer_1_dw_filter_size - v1_layer_1_dw_strides][v1_layer_1_dw_ifm_width],
					 fms_dt lower[v1_layer_1_dw_depth][v1_layer_1_dw_ifm_width],
					 dw_weights_dt dw_weights[v1_layer_1_dw_depth][v1_layer_1_dw_filter_size*v1_layer_1_dw_filter_size],
					 fms_dt result[v1_layer_1_dw_num_fils][v1_layer_1_dw_ofm_width])
{

#pragma HLS INLINE off

	fms_dt intermediate_pw_results[v1_layer_1_dw_parallelism][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size];
#pragma HLS ARRAY_PARTITION variable = intermediate_pw_results type = complete dim = 0

	for (int o_d = 0; o_d < v1_layer_1_dw_depth / v1_layer_1_dw_parallelism;
		 o_d++)
	{
		// depth loop
		int o_d_offset = o_d * v1_layer_1_dw_parallelism;
		// fill upper and lower (except last) rows:
		for (int d = 0; d < v1_layer_1_dw_parallelism; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < v1_layer_1_dw_filter_size - v1_layer_1_dw_strides; h++)
			{ // fill first two rows as the third will be filled by the pw results
#pragma HLS UNROLL
			  //  padding
				intermediate_pw_results[d][h][0] = 0;
				for (int w = v1_layer_1_dw_padding_left; w < v1_layer_1_dw_filter_size; w++)
				{
#pragma HLS UNROLL
					intermediate_pw_results[d][h][w] = upper[o_d_offset + d][h][w - v1_layer_1_dw_padding_left];
				}
			}
			for (int h = v1_layer_1_dw_filter_size - v1_layer_1_dw_strides; h < v1_layer_1_dw_filter_size; h++)
			{ // fill first two rows as the third will be filled by the pw results
#pragma HLS UNROLL
			  //  padding
				intermediate_pw_results[d][h][0] = 0;
				for (int w = v1_layer_1_dw_padding_left; w < v1_layer_1_dw_filter_size; w++)
				{
#pragma HLS UNROLL
					intermediate_pw_results[d][h][w] = lower[o_d_offset + d][w - v1_layer_1_dw_padding_left];
				}
			}
			intermediate_pw_results[d][v1_layer_1_dw_filter_size - 1][0] = 0;
		}

	layer_1_dw_pipeline:
		for (int w = 0; w < v1_layer_1_dw_ifm_width; w +=
														layer_1_dw_strides)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_1_dw_loops:
			for (int d = 0; d < v1_layer_1_dw_parallelism; d++)
			{
#pragma HLS UNROLL
				dw_pss_dt tmp = 0;
				// parallelized depth loop
				for (int c_h = 0; c_h < layer_1_dw_filter_size; c_h++)
				{
#pragma HLS UNROLL
					for (int c_w = 0; c_w < layer_1_dw_filter_size; c_w++)
					{
						// conv width loop
#pragma HLS UNROLL
						tmp += intermediate_pw_results[o_d][c_h][c_w] * dw_weights[o_d_offset + o_d][c_h* layer_2_dw_filter_size + c_w];
					}
				}
				fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
				if (scaled_val > 0)
				{
					result[o_d_offset + d][w / layer_1_dw_strides] =
						scaled_val;
				}
			}
		}
	}
	for (int w = 0; w < v1_layer_1_dw_ifm_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < v1_layer_1_dw_depth; d++)
		{
#pragma HLS UNROLL factor = 8
			upper[d][0][w] = upper[d][1][w];
			upper[d][1][w] = lower[d][w];
		}
	}
}

void v1_3_layer_2_pw(
	fms_dt channels_buffer[v1_layer_2_pw_depth][v1_layer_2_pw_ifm_width],
	weights_dt weights[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth],
	fms_dt result[max_fms_size], int starting_h)
{

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < v1_layer_2_pw_num_fils / v1_layer_2_pw_parallelism_out;
		 o_o_d++)
	{
		int o_o_d_offset = o_o_d * v1_layer_2_pw_parallelism_out;
		// filters loop
	layer_2_pw_pipeline:
		for (int w = 0; w < v1_layer_2_pw_ifm_width; w++)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_2_pw_loops:
			for (int o_d = 0;
				 o_d < v1_layer_2_pw_parallelism_out; o_d++)
			{
#pragma HLS UNROLL
				// parallelized filters loop
				pss_dt tmp = 0;
				for (int d = 0; d < v1_layer_2_pw_parallelism_in; d++)
				{
#pragma HLS UNROLL
					// parallelized depth loop
					tmp += ((fms_dt)channels_buffer[d][w]) * weights[o_o_d_offset + o_d][d];
				}
				fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
				if (scaled_val > 0)
				{
					result[(o_o_d_offset + o_d) * switch_point_fms_height * switch_point_fms_width + starting_h * switch_point_fms_width + w] =
						scaled_val;
				}
			}
		}
	}
}

void v1_3_stages_fill_channels_buffer(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt channels_buffer_0[input_image_depth][layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
	int starting_h)
{

#pragma HLS INLINE off

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides - layer_0_strides; h++)
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
			for (int h = layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides - layer_0_strides; h < layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = channels[d][starting_h + h - (layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides - layer_0_strides)][w];
			}
		}
	}
}

void mobilenet_v1_pipeline_3(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size])
{
#pragma HLS INLINE off

	layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim];

	dw_weights_dt dw_weights_1[v1_layer_1_dw_depth][v1_layer_1_dw_filter_size][v1_layer_1_dw_filter_size];
	weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth];

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

#pragma HLS ARRAY_PARTITION variable = dw_weights_1 type = complete dim = 1

	v1_3_fill_layers_weights(weights_0, dw_weights_1, pw_weights_2);

	//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width] = {0};

	fms_dt v1_3_layer_0_3x3_conv_out_0[v1_layer_2_pw_depth][v1_layer_2_pw_ifm_width] =
		{0};

	fms_dt v1_3_layer_1_dw_upper[v1_layer_1_dw_depth][v1_layer_1_dw_filter_size - v1_layer_1_dw_strides][v1_layer_1_dw_ifm_width] = {0};
	fms_dt v1_3_layer_1_dw_out_0[v1_layer_2_pw_depth][v1_layer_2_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = v1_3_layer_0_3x3_conv_out_0 cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = v1_3_layer_0_3x3_conv_out_0 cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = v1_3_layer_0_3x3_conv_out_0 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = v1_3_layer_1_dw_upper cyclic factor = 8 dim = 1

#pragma HLS ARRAY_PARTITION variable = v1_3_layer_1_dw_out_0 cyclic factor = 2 dim = 2

	//###########################################################

	//#########################odd###############################
	fms_dt channels_buffer_1[input_image_depth][layer_0_s_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width] = {0};

	fms_dt v1_3_layer_0_3x3_conv_out_1[v1_layer_2_pw_depth][v1_layer_2_pw_ifm_width] =
		{0};

	fms_dt v1_3_layer_1_dw_out_1[v1_layer_2_pw_depth][v1_layer_2_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = v1_3_layer_0_3x3_conv_out_1 cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = v1_3_layer_0_3x3_conv_out_1 cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = v1_3_layer_0_3x3_conv_out_1 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = v1_3_layer_1_dw_out_1 cyclic factor = 2 dim = 2

	//###########################################################
	// pipeline filling###########################################
	v1_3_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
	//##########
	v1_3_stages_fill_channels_buffer(channels, channels_buffer_1, 2);
	v1_3_layer_0_3x3_conv(channels_buffer_0, weights_0,
						  v1_3_layer_0_3x3_conv_out_0);
	// this sequence does not produce a valid _5_layer_1_dw_out_0 yet, as it needs 2
	// rows but only 1 has been feed so far
	//##########
	v1_3_stages_fill_channels_buffer(channels, channels_buffer_0, 4);
	v1_3_layer_0_3x3_conv(channels_buffer_1, weights_0,
						  v1_3_layer_0_3x3_conv_out_1);
	//##########
	v1_3_stages_fill_channels_buffer(channels, channels_buffer_1, 6);
	v1_3_layer_0_3x3_conv(channels_buffer_0, weights_0,
						  v1_3_layer_0_3x3_conv_out_0);
	v1_3_layer_1_dw(v1_3_layer_1_dw_upper, v1_3_layer_0_3x3_conv_out_1, dw_weights_1, v1_3_layer_1_dw_out_1);
	//##########
	int even_odd = 1;
	int h = 4;
main_pipeline_loop:
	for (; h < switch_point_fms_height; h++)
	{
		if (even_odd)
		{
			v1_3_stages_fill_channels_buffer(channels, channels_buffer_0, h * v1_3_stages_layer_1_rows_at_once * layer_0_strides);
			v1_3_layer_0_3x3_conv(channels_buffer_1, weights_0,
								  v1_3_layer_0_3x3_conv_out_1);
			v1_3_layer_1_dw(v1_3_layer_1_dw_upper, v1_3_layer_0_3x3_conv_out_0, dw_weights_1, v1_3_layer_1_dw_out_0);
			v1_3_layer_2_pw(v1_3_layer_1_dw_out_1, pw_weights_2, result, h - 4);
		}
		else
		{
			v1_3_stages_fill_channels_buffer(channels, channels_buffer_1, h * v1_3_stages_layer_1_rows_at_once * layer_0_strides);
			v1_3_layer_0_3x3_conv(channels_buffer_0, weights_0,
								  v1_3_layer_0_3x3_conv_out_0);
			v1_3_layer_1_dw(v1_3_layer_1_dw_upper, v1_3_layer_0_3x3_conv_out_1, dw_weights_1, v1_3_layer_1_dw_out_1);
			v1_3_layer_2_pw(v1_3_layer_1_dw_out_0, pw_weights_2, result, h - 4);
		}
		even_odd = 1 - even_odd;
	}
	//###########################################################
	// pipeline flushing##########################################
	v1_3_layer_2_pw(v1_3_layer_1_dw_out_1, pw_weights_2, result, switch_point_fms_height - 4);
	//##########
	v1_3_layer_1_dw(v1_3_layer_1_dw_upper, v1_3_layer_0_3x3_conv_out_0, dw_weights_1, v1_3_layer_1_dw_out_0);
	v1_3_layer_2_pw(v1_3_layer_1_dw_out_0, pw_weights_2, result, switch_point_fms_height - 3);
	//##########
	v1_3_layer_0_3x3_conv(channels_buffer_1, weights_0,
						  v1_3_layer_0_3x3_conv_out_1);
	v1_3_layer_1_dw(v1_3_layer_1_dw_upper, v1_3_layer_0_3x3_conv_out_1, dw_weights_1, v1_3_layer_1_dw_out_1);
	v1_3_layer_2_pw(v1_3_layer_1_dw_out_1, pw_weights_2, result, switch_point_fms_height - 2);
	//##########
	fms_dt padding_bottom[v1_layer_2_pw_depth][v1_layer_2_pw_ifm_width] = {0};
	v1_3_layer_1_dw(v1_3_layer_1_dw_upper, padding_bottom, dw_weights_1, v1_3_layer_1_dw_out_0);
	v1_3_layer_2_pw(v1_3_layer_1_dw_out_0, pw_weights_2, result, switch_point_fms_height - 1);
}
