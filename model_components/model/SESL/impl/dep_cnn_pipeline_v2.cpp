#include "utils.h"
//
void fill_channels_buffer_v2(fms_dt channels[32][112][112],
							 fms_dt dw_channels_buffer_1[layer_1_dw_depth][max_conv_h + layer_2_dw_specs.strides][layer_1_dw_width], int starting_h)
{
	for (int w = 0; w < 112; w++)
	{
#pragma HLS PIPELINE
		for (int h = 0; h < layer_2_dw_strides; h++)
		{
			for (int d = 0; d < 32; d++)
			{
#pragma HLS UNROLL
				dw_channels_buffer_1[d][max_conv_h + h - 1][w] =
					channels[d][starting_h + h][w];
			}
		}
	}
}

void layer_1_dw_pw(
	fms_dt channels_buffer[layer_1_dw_depth][max_conv_h + layer_2_dw_specs.strides][layer_1_dw_width],
	dw_weights_dt dw_weights[layer_1_dw_depth][max_conv_h][max_conv_w],
	weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
	fms_dt result[layer_2_dw_depth][layer_2_dw_strides][layer_2_dw_width],
	int h_index)
{

	// dw_pss_dt tmp_result[2][layer_1_pw_depth];

layer_1_dw_depth_loop:
	for (int o_d = 0;
		 o_d < layer_1_dw_depth / dw_layer_1_parallelism; o_d++)
	{
		int o_d_indx = o_d * dw_layer_1_parallelism;
		for (int row = 0; row < layer_2_dw_strides; row++)
		{

		layer_1_dw_pipeline:
			for (int w = 0; w < layer_1_dw_width - 2; w +=
													  layer_2_dw_specs.strides)
			{
#pragma HLS PIPELINE

				pss_dt tmp_result[2][pw_layer_1_parallelism_out];
				fms_dt depth_buffer[layer_1_dw_depth][max_conv_h][max_conv_w];
#pragma HLS ARRAY_PARTITION variable = depth_buffer type = complete dim = 0
			layer_1_dw_loops:
				for (int d = 0; d < dw_layer_1_parallelism;
					 d++)
				{
#pragma HLS UNROLL
					dw_pss_dt tmp = 0;

					for (int h = 0; h < max_conv_h; h++)
					{
#pragma HLS UNROLL
						depth_buffer[o_d_indx + d][h][2] =
							channels_buffer[o_d_indx + d][h][w];
						for (int c_w = 0; c_w < max_conv_w - 1; c_w++)
						{
#pragma HLS UNROLL
							depth_buffer[o_d_indx + d][h][c_w] =
								depth_buffer[o_d_indx + d][h][c_w + 1];
						}
					}
				}
			layer_1_dw_loops_:
				for (int d = 0; d < dw_layer_1_parallelism;
					 d++)
				{
#pragma HLS UNROLL
					dw_pss_dt tmp = 0;
					for (int c_w = 0; c_w < max_conv_w; c_w++)
					{
#pragma HLS UNROLL
						for (int h = 0; h < max_conv_h; h++)
						{
#pragma HLS UNROLL
							tmp += depth_buffer[o_d_indx + d][h][c_w] * dw_weights[o_d_indx + d][h][c_w];
						}
					}
				layer_1_pw_loops:
					for (int pw_o_d = 0;
						 pw_o_d < pw_layer_1_parallelism_out; pw_o_d++)
					{
#pragma HLS UNROLL
						tmp_result[0][pw_o_d] += tmp * weights[o_d][o_d_indx + d];
					}
				}
			layer_1_pw_loops_write:
				for (int pw_o_d = 0;
					 pw_o_d < pw_layer_1_parallelism_out; pw_o_d++)
				{
#pragma HLS UNROLL
					fms_dt scaled_val = (fms_dt)(tmp_result[0][pw_o_d]);
					if (scaled_val > 0)
					{ // ReLU
						result[pw_o_d][row][w] = scaled_val;
					}
				}
			}
		}
	}
}

void layer_2_dw_pw(
	fms_dt upper_dw_channels_buffer_2[layer_2_dw_depth][layer_2_dw_width],
	fms_dt lower_dw_channels_buffer_2[layer_2_dw_depth][layer_2_dw_strides][layer_2_dw_width],
	dw_weights_dt dw_weights[layer_2_dw_depth][max_conv_h][max_conv_w],
	weights_dt weights[layer_3_pw_specs.num_fils][layer_2_pw_depth],
	fms_dt result[layer_3_pw_depth][layer_3_pw_width], int h_index)
{

	// dw_pss_dt tmp_result[2][layer_1_pw_depth];

layer_1_dw_depth_loop:
	for (int o_d = 0;
		 o_d < layer_2_dw_depth / dw_layer_2_parallelism; o_d++)
	{
		int o_d_indx = o_d * dw_layer_2_parallelism;

	layer_1_dw_pipeline:
		for (int w = 0; w < layer_2_dw_width - 2; w +=
												  layer_2_dw_strides)
		{
#pragma HLS PIPELINE

			pss_dt tmp_result[2][pw_layer_1_parallelism_out];
			fms_dt depth_buffer[layer_1_dw_depth][max_conv_h][max_conv_w];
#pragma HLS ARRAY_PARTITION variable = depth_buffer type = complete dim = 0
		layer_1_dw_loops:
			for (int d = 0; d < dw_layer_2_parallelism;
				 d++)
			{
#pragma HLS UNROLL
				dw_pss_dt tmp = 0;
			layer_1_dw_loops_:
				for (int d = 0; d < dw_layer_2_parallelism;
					 d++)
				{
#pragma HLS UNROLL
					dw_pss_dt tmp = 0;
					for (int c_w = 0; c_w < max_conv_w; c_w++)
					{
#pragma HLS UNROLL
						tmp += upper_dw_channels_buffer_2[o_d_indx + d][c_w + w] * dw_weights[o_d_indx + d][0][c_w];
					}
					for (int c_w = 0; c_w < max_conv_w; c_w++)
					{
#pragma HLS UNROLL
						for (int h = max_conv_h - layer_2_dw_strides; h < max_conv_h; h++)
						{
#pragma HLS UNROLL
							tmp += lower_dw_channels_buffer_2[o_d_indx + d][h - (max_conv_h - layer_2_dw_strides)][c_w + w] * dw_weights[o_d_indx + d][h][c_w];
						}
					}
				layer_1_pw_loops:
					for (int pw_o_d = 0;
						 pw_o_d < pw_layer_2_parallelism_out; pw_o_d++)
					{
#pragma HLS UNROLL
						tmp_result[0][pw_o_d] += tmp * weights[o_d][o_d_indx + d];
					}
				}
			layer_1_pw_loops_write:
				for (int pw_o_d = 0;
					 pw_o_d < pw_layer_2_parallelism_out; pw_o_d++)
				{
#pragma HLS UNROLL
					fms_dt scaled_val = (fms_dt)(tmp_result[0][pw_o_d]);
					if (scaled_val > 0)
					{ // ReLU
						result[pw_o_d][w / layer_2_dw_strides] = scaled_val;
					}
				}
			}
		}
	}
}

void conv_pipeline_v2(fms_dt channels[32][112][112],
					  fms_grp_dt result[max_fms_size])
{

#pragma HLS ARRAY_PARTITION variable = channels complete dim = 1

	dw_weights_dt dw_weights_2[layer_1_dw_depth][max_conv_h][max_conv_w];
	dw_weights_dt dw_weights_2[layer_2_dw_depth][max_conv_h][max_conv_w];
	dw_weights_dt dw_weights_3[layer_3_dw_depth][max_conv_h][max_conv_w];

	weights_dt pw_weights_1[layer_1_pw_num_fils][layer_1_pw_depth];
	weights_dt pw_weights_3[layer_3_pw_specs.num_fils][layer_2_pw_depth];
	weights_dt pw_weights_3[layer_3_pw_num_fils][layer_3_pw_depth];
	fill_layers_weights(dw_weights_2, dw_weights_2, dw_weights_3, pw_weights_1,
						pw_weights_3, pw_weights_3);

	fms_dt dw_channels_buffer_1[layer_1_dw_depth][max_conv_h + layer_2_dw_specs.strides][layer_1_dw_width];
	fms_dt upper_dw_channels_buffer_2[layer_2_dw_depth][layer_2_dw_width];
	fms_dt lower_dw_channels_buffer_2[layer_2_dw_depth][layer_2_dw_strides][layer_2_dw_width] =
		{0};
	fms_dt upper_dw_channels_buffer_3[layer_3_dw_depth][max_conv_h - layer_3_dw_strides][layer_3_dw_width];
	fms_dt lower_dw_channels_buffer_3[layer_3_dw_depth][layer_3_dw_width] =
		{0};

#pragma HLS ARRAY_PARTITION variable = dw_channels_buffer_1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = dw_channels_buffer_1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = upper_dw_channels_buffer_2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lower_dw_channels_buffer_2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lower_dw_channels_buffer_2 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = upper_dw_channels_buffer_3 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lower_dw_channels_buffer_3 complete dim = 1

	fms_dt dw_channels_buffer_b1[layer_1_dw_depth][max_conv_h + layer_2_dw_specs.strides][layer_1_dw_width];
	fms_dt upper_dw_channels_buffer_b2[layer_2_dw_depth][layer_2_dw_width];
	fms_dt lower_dw_channels_buffer_b2[layer_2_dw_depth][layer_2_dw_strides][layer_2_dw_width] =
		{0};
	fms_dt upper_dw_channels_buffer_b3[layer_3_dw_depth][max_conv_h - layer_3_dw_strides][layer_3_dw_width];
	fms_dt lower_dw_channels_buffer_b3[layer_3_dw_depth][layer_3_dw_width] =
		{0};

#pragma HLS ARRAY_PARTITION variable = dw_channels_buffer_b1 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = dw_channels_buffer_b1 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = upper_dw_channels_buffer_b2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lower_dw_channels_buffer_b2 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lower_dw_channels_buffer_b2 complete dim = 2

#pragma HLS ARRAY_PARTITION variable = upper_dw_channels_buffer_b3 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = lower_dw_channels_buffer_b3 complete dim = 1

	int odd_even = 0;
	//	fill_channels_buffer_v2(channels, dw_channels_buffer_1, 0);
	//	layer_1_dw_pw(dw_channels_buffer_1, dw_weights_2, pw_weights_1,
	//			lower_dw_channels_buffer_b2, 0);

	fill_channels_buffer_v2(channels, dw_channels_buffer_b1, 1);

main_pipeline_loop:
	for (int h = 0; h < 56; h++)
	{
		if (odd_even)
		{
			layer_1_dw_pw(dw_channels_buffer_1, dw_weights_2, pw_weights_1,
						  lower_dw_channels_buffer_b2, h > 2);
			layer_2_dw_pw(
				upper_dw_channels_buffer_2,
				lower_dw_channels_buffer_2,
				dw_weights_2,
				pw_weights_3,
				lower_dw_channels_buffer_b3, h > 2);
			fill_channels_buffer_v2(channels, dw_channels_buffer_b1, h);
			for (int i = 0; i < 7; i++)
			{
				result[i] = lower_dw_channels_buffer_b3[h][i];
			}
		}
		else
		{
			layer_1_dw_pw(dw_channels_buffer_b1, dw_weights_2, pw_weights_1,
						  lower_dw_channels_buffer_2, h > 2);
			layer_2_dw_pw(
				upper_dw_channels_buffer_b2,
				lower_dw_channels_buffer_b2,
				dw_weights_2,
				pw_weights_3,
				lower_dw_channels_buffer_3, h > 2);
			fill_channels_buffer_v2(channels, dw_channels_buffer_1, h);

			for (int i = 0; i < 7; i++)
			{
				result[i] = lower_dw_channels_buffer_3[h][i];
			}
		}
		odd_even = 1 - odd_even;
	}
}
