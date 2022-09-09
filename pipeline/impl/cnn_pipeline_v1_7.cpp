#include "./headers/utils.h"
#include "./headers/cnn_functions_v1.h"

void v1_7_layer_2_pw_dw(
	fms_dt channels_buffer[v1_layer_2_pw_depth][v1_7_stages_layer_1_rows_at_once][v1_layer_2_dw_ifm_width],
	weights_dt weights[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth],
	dw_weights_dt dw_weights[v1_layer_2_dw_depth][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size],
	fms_dt upper[v1_layer_2_dw_depth][v1_layer_2_dw_ifm_width],
	fms_dt lower[v1_layer_2_dw_depth][v1_layer_2_dw_strides][v1_layer_2_dw_ifm_width],
	fms_dt result[v1_layer_3_pw_depth][v1_layer_3_pw_ifm_width], int active_row)
{

	fms_dt intermediate_pw_results[v1_layer_2_pw_parallelism_out][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size];
#pragma HLS ARRAY_PARTITION variable = intermediate_pw_results type = complete dim = 0

#pragma HLS INLINE off
layer_1_2_pw_dw_main_loop:
	for (int o_o_d = 0;
		 o_o_d < v1_layer_2_pw_num_fils / v1_layer_2_pw_parallelism_out;
		 o_o_d++)
	{
		int o_o_d_offset = o_o_d * v1_layer_2_pw_parallelism_out;

		// fill upper and lower (except last) rows:
		for (int d = 0; d < v1_layer_2_pw_parallelism_out; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < v1_layer_2_dw_filter_size - v1_layer_2_dw_strides; h++)
			{
#pragma HLS UNROLL
				// padding
				intermediate_pw_results[o_o_d_offset + d][h][0] = 0;
				for (int w = 1; w < v1_layer_2_dw_filter_size; w++)
				{
#pragma HLS UNROLL
					intermediate_pw_results[o_o_d_offset + d][h][w] = upper[o_o_d_offset + d][h][w];
				}
			}
		}

	layer_1_2_pw_dw_pipeline:
		for (int w = 0; w < v1_layer_2_dw_ifm_width + v1_layer_2_dw_padding_left + v1_layer_2_dw_padding_right;
			 w++)
		{
#pragma HLS PIPELINE
		//###################PW#######################
		layer_1_2_pw_dw_loops:
			for (int o_d = 0;
				 o_d < v1_layer_2_pw_parallelism_out; o_d++)
			{
#pragma HLS UNROLL
				// parallelized filters loop
				if (w < v1_layer_2_dw_ifm_width)
				{
					for (int row = 0; row < v1_4_stages_layer_1_rows_at_once;
						 row++)
					{
#pragma HLS UNROLL
						// FMs width loop
						pss_dt tmp = 0;
						for (int d = 0; d < v1_layer_2_pw_parallelism_in; d++)
						{
#pragma HLS UNROLL
							// parallelized depth loop
							tmp +=
								((fms_dt)channels_buffer[d][row][w]) * weights[o_o_d_offset + o_d][d];
						}
						fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
						if (scaled_val > 0)
						{
							if (w + 1 < layer_1_dw_filter_size)
							{
								intermediate_pw_results[o_d][row + v1_layer_2_dw_filter_size - v1_layer_2_dw_strides][w + 1] = scaled_val;
							}
							lower[o_o_d_offset + o_d][row][w] = scaled_val;
						}
					}
				}
				//###############end PW####################
				//###############DW########################
				if (w + 1 >= v1_layer_2_dw_filter_size - v1_layer_2_dw_padding_left && (w + 1 - (v1_layer_2_dw_filter_size - v1_layer_2_dw_padding_left)) % v1_layer_2_dw_strides == 0)
				{
					for (int row = 0; row < v1_layer_2_dw_strides; row++)
					{
						for (int c_w = v1_layer_2_dw_filter_size - v1_layer_2_dw_strides; c_w < v1_layer_2_dw_filter_size; c_w++)
						{
							// conv width loop
#pragma HLS UNROLL
							intermediate_pw_results[o_d][v1_layer_2_dw_filter_size - v1_layer_2_dw_strides + row][c_w] =
								lower[o_o_d_offset + o_d][row][w + 1 - (v1_layer_2_dw_filter_size - v1_layer_2_dw_padding_left) +
															   (c_w - (v1_layer_2_dw_filter_size - v1_layer_2_dw_strides))];
						}
					}
					dw_pss_dt tmp = 0;
					// parallelized depth loop
					for (int c_h = 0; c_h < v1_layer_2_dw_filter_size; c_h++)
					{
#pragma HLS UNROLL
						for (int c_w = 0; c_w < v1_layer_2_dw_filter_size; c_w++)
						{
							// conv width loop
#pragma HLS UNROLL
							tmp += intermediate_pw_results[o_d][c_h][w + c_w] * dw_weights[o_o_d_offset + o_d][c_h][c_w];
						}
					}
					fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
					if (scaled_val > 0)
					{
						result[o_o_d_offset + o_d][(w - layer_3_dw_padding_left) / layer_0_strides] = scaled_val;
					}
					//#####################end DW################
					//#####################shift and fill intermediate#################
					for (int c_h = 0; c_h < v1_layer_2_dw_filter_size; c_h++)
					{
#pragma HLS UNROLL
						for (int c_w = 0; c_w < v1_layer_2_dw_filter_size - v1_layer_2_dw_strides; c_w++)
						{
#pragma HLS UNROLL
							intermediate_pw_results[o_d][c_h][c_w] = intermediate_pw_results[o_d][c_h][c_w + v1_layer_2_dw_strides];
						}
						if (c_h < v1_layer_2_dw_filter_size - v1_layer_2_dw_strides)
						{
							for (int c_w = v1_layer_2_dw_filter_size - v1_layer_2_dw_strides; c_w < v1_layer_2_dw_filter_size; c_w++)
							{
#pragma HLS UNROLL
								intermediate_pw_results[o_d][c_h][c_w] = upper[o_o_d_offset + o_d][w + (c_w - (v1_layer_2_dw_filter_size - v1_layer_2_dw_strides)) +
																								   v1_layer_2_dw_padding_left]; // to do add offset
							}
						}
					}
					//#####################end shift and fill intermediate#################
				}
			}
		}
	}

layer_1_2_pw_dw_shift_loop:
	for (int w = 0; w < v1_layer_2_pw_ifm_width;
		 w++)
	{
#pragma HLS PIPELINE
	//###################PW#######################
	layer_1_shift_loops:
		for (int o_d = 0;
			 o_d < v1_layer_2_dw_depth; o_d++)
		{
#pragma HLS UNROLL factor = 8
			upper[o_d][w] =
				lower[o_d][1][w];
		}
	}
}

void v1_7_layer_3_pw_dw(
	fms_dt channels_buffer[v1_layer_3_pw_depth][v1_layer_3_pw_ifm_width],
	weights_dt weights[v1_layer_3_pw_num_fils][v1_layer_3_pw_depth],
	dw_weights_dt dw_weights[v1_layer_3_dw_depth][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size],
	fms_dt upper[v1_layer_3_dw_depth][v1_layer_3_dw_filter_size - v1_layer_3_dw_strides][v1_layer_3_dw_ifm_width],
	fms_dt lower[v1_layer_3_dw_depth][v1_layer_3_dw_strides][v1_layer_3_dw_ifm_width],
	fms_dt result[v1_layer_4_pw_depth][v1_layer_4_pw_ifm_width], int active_row)
{

	fms_dt intermediate_pw_results[v1_layer_3_pw_parallelism_out][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size];
#pragma HLS ARRAY_PARTITION variable = intermediate_pw_results type = complete dim = 0

#pragma HLS INLINE off
layer_1_pw__dw_main_loop:
	for (int h = 0; h < v1_7_stages_layer_3_rows_at_once;
		 h++)
	{
		// rows for next DW
		for (int o_o_d = 0;
			 o_o_d < v1_layer_3_pw_num_fils / v1_layer_3_pw_parallelism_out;
			 o_o_d++)
		{
			int o_o_d_offset = o_o_d * v1_layer_3_pw_parallelism_out;

			// fill upper and lower (except last) rows:
			for (int d = 0; d < v1_layer_3_pw_parallelism_out; d++)
			{
#pragma HLS UNROLL
				for (int h = 0; h < v1_layer_3_dw_filter_size - v1_layer_3_dw_strides; h++)
				{ // fill first two rows as the third will be filled by the pw results
#pragma HLS UNROLL
				  //  padding
					intermediate_pw_results[d][h][0] = 0;
					for (int w = v1_layer_3_dw_padding_left; w < v1_layer_3_dw_filter_size; w++)
					{
#pragma HLS UNROLL
						intermediate_pw_results[d][h][w] = upper[o_o_d_offset + d][h][w - v1_layer_3_dw_padding_left];
					}
				}
				intermediate_pw_results[d][v1_layer_3_dw_filter_size - 1][0] = 0;
			}

		layer_1_pw_pipeline:
			for (int w = 0; w < layer_1_dw_ifm_width + v1_layer_3_dw_filter_size - (v1_layer_3_dw_padding_left + layer_1_dw_padding_right);
				 w++)
			{
#pragma HLS PIPELINE
			//###################PW#######################
			layer_1_pw_loops:
				for (int o_d = 0;
					 o_d < v1_layer_3_pw_parallelism_out; o_d++)
				{
#pragma HLS UNROLL
					// parallelized filters loop
					if (w < layer_1_dw_ifm_width)
					{
						// FMs width loop
						pss_dt tmp = 0;
						for (int d = 0; d < v1_layer_3_pw_parallelism_in; d++)
						{
#pragma HLS UNROLL
							// parallelized depth loop
							tmp +=
								((fms_dt)channels_buffer[d][h][w]) * weights[o_o_d_offset + o_d][d];
						}
						fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
						if (scaled_val > 0)
						{
							if (w + v1_layer_3_dw_padding_left < v1_layer_3_dw_filter_size)
							{
								intermediate_pw_results[o_d][v1_layer_3_dw_filter_size - 1][w + v1_layer_3_dw_padding_left] = scaled_val;
							}
							lower[o_o_d_offset + o_d][0][w] = scaled_val;
						}
					}
					//###############end PW####################
					//###############DW########################
					if (w + 1 >= v1_layer_3_dw_filter_size - v1_layer_3_dw_padding_left)
					{
						if (active_row)
						{
							if (w < layer_1_dw_ifm_width)
							{
								intermediate_pw_results[o_d][v1_layer_3_dw_filter_size - 1][v1_layer_3_dw_filter_size - v1_layer_3_dw_padding_left] =
									lower[o_o_d_offset + o_d][0][w];
							}
							else
							{
								intermediate_pw_results[o_d][v1_layer_3_dw_filter_size - 1][v1_layer_3_dw_filter_size - v1_layer_3_dw_padding_left] = 0;
							}

							dw_pss_dt tmp = 0;
							// parallelized depth loop
							for (int c_h = 0; c_h < v1_layer_3_dw_filter_size; c_h++)
							{
#pragma HLS UNROLL
								for (int c_w = 0; c_w < v1_layer_3_dw_filter_size; c_w++)
								{
									// conv width loop
#pragma HLS UNROLL
									tmp += intermediate_pw_results[o_d][c_h][c_w] * dw_weights[o_o_d_offset + o_d][c_h][c_w];
								}
							}
							fms_dt scaled_val = (fms_dt)((((ap_fixed<17, 12>)tmp) - zero_point_dw) * ratio_dw_pss_to_fms);
							if (scaled_val > 0)
							{
								result[o_o_d_offset + o_d][(w + 1 - (v1_layer_3_dw_filter_size - v1_layer_3_dw_padding_left)) / v1_layer_1_dw_strides] =
									scaled_val;
							}
							//#####################end DW################
							//#####################shift and fill intermediate#################
							for (int c_h = 0; c_h < v1_layer_3_dw_filter_size; c_h++)
							{
#pragma HLS UNROLL
								for (int c_w = 0; c_w < v1_layer_3_dw_filter_size - v1_layer_3_dw_strides; c_w++)
								{
#pragma HLS UNROLL
									intermediate_pw_results[o_d][c_h][c_w] = intermediate_pw_results[o_d][c_h][c_w + v1_layer_3_dw_strides];
								}
								if (c_h < v1_layer_3_dw_filter_size - v1_layer_3_dw_strides)
								{
									for (int c_w = v1_layer_3_dw_filter_size - v1_layer_3_dw_strides; c_w < v1_layer_3_dw_filter_size; c_w++)
									{
#pragma HLS UNROLL
										if (w < layer_1_dw_ifm_width)
										{
											intermediate_pw_results[o_d][c_h][c_w] = upper[o_d][c_h][w + v1_layer_3_dw_padding_left];
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
						for (int c_h = 0; c_h < v1_layer_3_dw_filter_size - 2 * v1_layer_3_dw_strides; c_h++)
						{
#pragma HLS UNROLL
							for (int c_w = 0; c_w < v1_layer_3_dw_strides; c_w++)
							{
#pragma HLS UNROLL
								upper[o_o_d_offset + o_d][c_h][(w + 1 - (v1_layer_3_dw_filter_size - 1)) + c_w] =
									upper[o_o_d_offset + o_d][c_h + v1_layer_3_dw_strides][(w + 1 - (v1_layer_3_dw_filter_size - v1_layer_3_dw_padding_left)) + c_w];
							}
						}
						for (int c_h = 0; c_h < v1_layer_3_dw_strides; c_h++)
						{
#pragma HLS UNROLL
							for (int c_w = 0; c_w < v1_layer_3_dw_strides; c_w++)
							{
#pragma HLS UNROLL
								upper[o_o_d_offset + o_d][c_h + v1_layer_3_dw_filter_size - 2 * v1_layer_3_dw_strides]
									 [(w + 1 - (v1_layer_3_dw_filter_size - v1_layer_3_dw_padding_left)) + c_w] =
										 lower[o_o_d_offset + o_d][c_h][(w + 1 - (v1_layer_3_dw_filter_size - v1_layer_3_dw_padding_left)) + c_w];
							}
						}
						//#####################end shift#################
					}
				}
			}
		}
	}
}

void v1_7_layer_4_pw(
	fms_dt channels_buffer[v1_layer_4_pw_depth][v1_layer_4_pw_ifm_width],
	weights_dt weights[v1_layer_4_pw_num_fils][v1_layer_4_pw_depth],
	fms_dt result[max_fms_size], int starting_h)
{

#pragma HLS INLINE off

	// rows for next DW
	for (int o_o_d = 0;
		 o_o_d < v1_layer_4_pw_num_fils / v1_layer_4_pw_parallelism_out;
		 o_o_d++)
	{
		int o_o_d_offset = o_o_d * v1_layer_4_pw_parallelism_out;
		// filters loop
	layer_2_pw_pipeline:
		for (int w = 0; w < v1_layer_4_pw_ifm_width; w++)
		{
#pragma HLS PIPELINE
		// FMs width loop
		layer_2_pw_loops:
			for (int o_d = 0;
				 o_d < v1_layer_4_pw_parallelism_out; o_d++)
			{
#pragma HLS UNROLL
				// parallelized filters loop
				pss_dt tmp = 0;
				for (int d = 0; d < v1_layer_4_pw_parallelism_in; d++)
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

void mobilenet_v1_pipeline_7(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size])
{
#pragma HLS INLINE off

	layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size];
	dw_weights_dt dw_weights_1[v1_layer_1_dw_depth][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size];
	dw_weights_dt dw_weights_2[v1_layer_2_dw_depth][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size];
	dw_weights_dt dw_weights_3[v1_layer_3_dw_depth][v1_layer_3_dw_filter_size][v1_layer_3_dw_filter_size];
	weights_dt pw_weights_2[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth];
	weights_dt pw_weights_3[v1_layer_3_pw_num_fils][v1_layer_3_pw_depth];
	weights_dt pw_weights_4[v1_layer_4_pw_num_fils][v1_layer_4_pw_depth];

#pragma HLS ARRAY_PARTITION variable = channels type = complete dim = 1

#pragma HLS ARRAY_PARTITION variable = dw_weights_1 type = complete dim = 1

	v1_7_fill_layers_weights(weights_0, dw_weights_1, dw_weights_2, dw_weights_3, pw_weights_2, pw_weights_3, pw_weights_4);

	//#########################even###############################
	fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size + (v1_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width] = {0};

	fms_dt v1_7_layer_0_3x3_conv_out_0[v1_layer_2_pw_depth][v1_7_stages_layer_1_rows_at_once][v1_layer_2_pw_ifm_width] =
		{0};

	fms_dt v1_7_layer_1_dw_upper[v1_layer_1_dw_depth][v1_layer_3_dw_filter_size - v1_layer_1_dw_strides][v1_layer_1_dw_ifm_width] = {0};
	fms_dt v1_7_layer_2_dw_upper[v1_layer_2_dw_depth][v1_layer_2_dw_ifm_width] = {0};
	fms_dt v1_7_layer_2_dw_lower[v1_layer_2_dw_depth][v1_layer_2_dw_strides][v1_layer_2_dw_ifm_width] = {0};
	fms_dt v1_7_layer_1_dw_out_0[v1_layer_2_pw_depth][v1_7_stages_layer_1_rows_at_once][v1_layer_2_pw_ifm_width] = {0};

	fms_dt v1_7_layer_2_pw_dw_out_0[v1_layer_3_pw_depth][v1_layer_3_pw_ifm_width] = {0};

	fms_dt v1_7_layer_3_dw_upper[v1_layer_3_dw_depth][v1_layer_3_dw_filter_size - v1_layer_3_dw_strides][v1_layer_3_dw_ifm_width] = {0};
	fms_dt v1_7_layer_3_dw_lower[v1_layer_3_dw_depth][v1_layer_3_dw_strides][v1_layer_3_dw_ifm_width] = {0};
	fms_dt v1_7_layer_3_pw_dw_out_0[v1_layer_4_pw_depth][v1_layer_4_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_0_3x3_conv_out_0 cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = v1_7_layer_0_3x3_conv_out_0 cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = v1_7_layer_0_3x3_conv_out_0 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_2_pw_dw_out_0 cyclic factor = 32 dim = 1

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_1_dw_upper complete dim = 1

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_2_dw_upper complete dim = 1
#pragma HLS ARRAY_PARTITION variable = v1_7_layer_2_dw_lower cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = v1_7_layer_2_dw_lower cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = v1_7_layer_2_dw_lower cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_1_dw_out_0 cyclic factor = 2 dim = 3

	//###########################################################

	//#########################odd###############################
	fms_dt channels_buffer_1[input_image_depth][layer_0_filter_size + (v1_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width] = {0};

	fms_dt v1_7_layer_0_3x3_conv_out_1[v1_layer_2_pw_depth][v1_7_stages_layer_1_rows_at_once][v1_layer_2_pw_ifm_width] =
		{0};

	fms_dt v1_7_layer_1_dw_out_1[v1_layer_2_pw_depth][v1_7_stages_layer_1_rows_at_once][v1_layer_2_pw_ifm_width] = {0};

	fms_dt v1_7_layer_2_pw_dw_out_1[v1_layer_3_pw_depth][v1_layer_3_pw_ifm_width] = {0};

	fms_dt v1_7_layer_3_pw_dw_out_1[v1_layer_4_pw_depth][v1_layer_4_pw_ifm_width] = {0};

#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_0_3x3_conv_out_1 cyclic factor = 16 dim = 1
#pragma HLS ARRAY_PARTITION variable = v1_7_layer_0_3x3_conv_out_1 cyclic factor = 2 dim = 2
#pragma HLS ARRAY_PARTITION variable = v1_7_layer_0_3x3_conv_out_1 cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_2_pw_dw_out_1 cyclic factor = 32 dim = 1

#pragma HLS ARRAY_PARTITION variable = v1_7_layer_1_dw_out_1 cyclic factor = 2 dim = 3

	//###########################################################
	// pipeline filling###########################################
	v1_4_stages_fill_channels_buffer(channels, channels_buffer_0, 0);
	//##########
	v1_4_stages_fill_channels_buffer(channels, channels_buffer_1, 2);
	v1_4_layer_0_3x3_conv(channels_buffer_0, weights_0,
						  v1_7_layer_0_3x3_conv_out_0);
	// this sequence does not produce a valid _5_layer_1_dw_out_0 yet, as it needs 2
	// rows but only 1 has been feed so far
	//##########
	v1_4_stages_fill_channels_buffer(channels, channels_buffer_0, 6);
	v1_4_layer_0_3x3_conv(channels_buffer_1, weights_0,
						  v1_7_layer_0_3x3_conv_out_1);
	v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_0, dw_weights_1, v1_7_layer_1_dw_out_0, 0);
	//##########
	v1_4_stages_fill_channels_buffer(channels, channels_buffer_1, 10);
	v1_4_layer_0_3x3_conv(channels_buffer_0, weights_0,
						  v1_7_layer_0_3x3_conv_out_0);
	v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_1, dw_weights_1, v1_7_layer_1_dw_out_1, 1);
	//##########
	v1_4_stages_fill_channels_buffer(channels, channels_buffer_0, 14);
	v1_4_layer_0_3x3_conv(channels_buffer_1, weights_0,
						  v1_7_layer_0_3x3_conv_out_1);
	v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_0, dw_weights_1, v1_7_layer_1_dw_out_0, 1);
	v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_1, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
					   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_1, 1);
	//##########
	v1_4_stages_fill_channels_buffer(channels, channels_buffer_1, 18);
	v1_4_layer_0_3x3_conv(channels_buffer_0, weights_0,
						  v1_7_layer_0_3x3_conv_out_0);
	v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_1, dw_weights_1, v1_7_layer_1_dw_out_1, 1);
	v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_0, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
					   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_0, 1);
	v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_1, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
					   v1_7_layer_3_pw_dw_out_1, 1);
	//##########

	int even_odd = 1;
	int h = 6;
main_pipeline_loop:
	for (; h < switch_point_fms_height; h++)
	{
		if (even_odd)
		{
			v1_4_stages_fill_channels_buffer(channels, channels_buffer_0, (h * v1_7_stages_layer_1_rows_at_once - 1) * layer_0_strides);
			v1_4_layer_0_3x3_conv(channels_buffer_1, weights_0,
								  v1_7_layer_0_3x3_conv_out_1);
			v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_0, dw_weights_1, v1_7_layer_1_dw_out_0, 1);
			v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_1, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
							   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_1, 1);
			v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_0, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
							   v1_7_layer_3_pw_dw_out_0, 1);
			v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_1, pw_weights_4, result, h - 6);
		}
		else
		{
			v1_4_stages_fill_channels_buffer(channels, channels_buffer_1, (h * v1_7_stages_layer_1_rows_at_once - 1) * layer_0_strides);
			v1_4_layer_0_3x3_conv(channels_buffer_0, weights_0,
								  v1_7_layer_0_3x3_conv_out_0);
			v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_1, dw_weights_1, v1_7_layer_1_dw_out_1, 1);
			v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_0, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
							   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_0, 1);
			v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_1, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
							   v1_7_layer_3_pw_dw_out_1, 1);
			v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_0, pw_weights_4, result, h - 6);
		}
		even_odd = 1 - even_odd;
	}
	//###########################################################
	// pipeline flushing##########################################
	v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_1, pw_weights_4, result, switch_point_fms_height - 6);
	//##########
	v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_0, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
					   v1_7_layer_3_pw_dw_out_0, 1);
	v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_0, pw_weights_4, result, switch_point_fms_height - 5);
	//##########
	v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_1, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
					   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_1, 1);
	v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_1, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
					   v1_7_layer_3_pw_dw_out_1, 1);
	v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_1, pw_weights_4, result, switch_point_fms_height - 4);
	//##########
	v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_0, dw_weights_1, v1_7_layer_1_dw_out_0, 1);
	v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_0, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
					   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_0, 1);
	v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_0, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
					   v1_7_layer_3_pw_dw_out_0, 1);
	v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_0, pw_weights_4, result, switch_point_fms_height - 3);
	//##########
	v1_4_layer_0_3x3_conv(channels_buffer_1, weights_0, v1_7_layer_0_3x3_conv_out_1);
	v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_1, dw_weights_1, v1_7_layer_1_dw_out_1, 1);
	v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_1, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
					   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_1, 1);
	v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_1, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
					   v1_7_layer_3_pw_dw_out_1, 1);
	v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_1, pw_weights_4, result, h - 2);
	//##########
	v1_4_stages_fill_channels_buffer(channels, channels_buffer_0, (switch_point_fms_height * v1_7_stages_layer_1_rows_at_once - 1) * layer_0_strides);
	v1_4_layer_0_3x3_conv(channels_buffer_0, weights_0, v1_7_layer_0_3x3_conv_out_0);
	v1_4_layer_1_dw(v1_7_layer_1_dw_upper, v1_7_layer_0_3x3_conv_out_0, dw_weights_1, v1_7_layer_1_dw_out_0, 1);
	for (int d = 0; d < layer_2_pw_depth; d++)
	{
		for (int w = 0; w < layer_2_pw_ifm_width; w++)
		{
			v1_7_layer_1_dw_out_0[d][v1_4_stages_layer_1_rows_at_once - 1][w] = 0;
		}
	}
	v1_7_layer_2_pw_dw(v1_7_layer_1_dw_out_0, pw_weights_2, dw_weights_2, v1_7_layer_2_dw_upper,
					   v1_7_layer_2_dw_lower, v1_7_layer_2_pw_dw_out_0, 1);
	v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_0, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
					   v1_7_layer_3_pw_dw_out_0, 1);
	v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_0, pw_weights_4, result, h - 1);
	//##########
	for (int d = 0; d < layer_2_pw_depth; d++)
	{
		for (int w = 0; w < layer_2_pw_ifm_width; w++)
		{
			v1_7_layer_2_pw_dw_out_0[d][v1_4_stages_layer_1_rows_at_once - 1][w] = 0;
		}
	}
	v1_7_layer_3_pw_dw(v1_7_layer_2_pw_dw_out_0, pw_weights_3, dw_weights_3, v1_7_layer_3_dw_upper, v1_7_layer_3_dw_lower,
					   v1_7_layer_3_pw_dw_out_0, 1);
	v1_7_layer_4_pw(v1_7_layer_3_pw_dw_out_0, pw_weights_4, result, h - 1);
	//##########
}
