#include "utils.h"
#include <stdint.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

void fill_layer_weight_groups_tile_off_chip(weights_grp_dt *weights,
											weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
											int starting_filter, const int layer_depth,
											const int num_of_weight_groups, const int layer_weights_offset,
											const int layer_num_fils)
{
#pragma HLS INLINE off

	const int current_fill_offset = layer_weights_offset + (starting_filter * layer_depth) / weights_group_items;

	if (starting_filter < layer_num_fils)
	{
	fill_weights_loop:
		for (int weight_grp_index = 0;
			 weight_grp_index < num_of_weight_groups; weight_grp_index++)
		{
			if (current_fill_offset + weight_grp_index < all_off_chip_pw_s_weights)
			{
				weight_groups_buffer[weight_grp_index] = weights[current_fill_offset + weight_grp_index];
			}
		}
	}
}

void fill_on_chip_weights_cpu(weights_grp_dt *on_chip_weights_src,
							  weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS])
{
	for (int i = 0; i < all_on_chip_pw_s_weights / ON_CHIP_WEIGHTS_PORTS; i++)
	{
		for (int j = 0; j < ON_CHIP_WEIGHTS_PORTS; j++)
		{
			on_chip_weights[i][j] = on_chip_weights_src[i * ON_CHIP_WEIGHTS_PORTS + j];
		}
	}
}

#if HW == _FPGA
void fill_on_chip_weights_fpga(weights_grp_dt *on_chip_weights_src,
							   weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
							   const int starting_filter)
{
	const int num_ierations = (all_on_chip_pw_s_weights + num_of_weight_groups_in_the_largest_weight_tile - 1) /
							  num_of_weight_groups_in_the_largest_weight_tile;

	weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile];
	for (int iter = 0; iter < num_ierations; iter++)
	{
		fill_layer_weight_groups_tile_off_chip(on_chip_weights_src, weight_groups_buffer,
											   0,
											   0,
											   num_of_weight_groups_in_the_largest_weight_tile,
											   iter * num_of_weight_groups_in_the_largest_weight_tile,
											   1);
		int current_fill_offset =
			iter * num_of_weight_groups_in_the_largest_weight_tile * weights_group_items / ON_CHIP_WEIGHTS_PORTS;
		for (int weight_grp_index = 0;
			 weight_grp_index < num_of_weight_groups_in_the_largest_weight_tile; weight_grp_index++)
		{
			weights_grp_dt chunck = weight_groups_buffer[weight_grp_index];
			for (int within_filter_index = 0;
				 within_filter_index < num_of_weights_in_the_same_filter_and_group_on_chip;
				 within_filter_index++)
			{
#pragma HLS UNROLL
				for (int filter_index = 0; filter_index < ON_CHIP_WEIGHTS_PORTS;
					 filter_index++)
				{
#pragma HLS UNROLL
					on_chip_weights[filter_index][current_fill_offset +
												  weight_grp_index * num_of_weights_in_the_same_filter_and_group_on_chip +
												  within_filter_index] = (weights_dt)chunck((within_filter_index *
																								 ON_CHIP_WEIGHTS_PORTS +
																							 filter_index) *
																									weights_dt_width +
																								weights_dt_offset,
																							(within_filter_index *
																								 ON_CHIP_WEIGHTS_PORTS +
																							 filter_index) *
																								weights_dt_width);
				}
			}
		}
	}
}

void fill_weights_tile_from_weight_groups_tile(
	weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile],
	weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
	const int layer_depth,
	const int num_of_weight_groups)
{
#pragma HLS INLINE off

	// assumes pw_parallelism_out * filter depth is divisable by weight group number

fill_weights_loop:
	for (int weight_grp_index = 0;
		 weight_grp_index < num_of_weight_groups; weight_grp_index++)
	{
		weights_grp_dt chunck = weight_groups_buffer[weight_grp_index];
		for (int within_filter_index = 0;
			 within_filter_index < num_of_weights_in_the_same_filter_and_group;
			 within_filter_index++)
		{
#pragma HLS UNROLL
			int in_filter_index = weight_grp_index * num_of_weights_in_the_same_filter_and_group + within_filter_index;
			if (in_filter_index< layer_depth)
							{
			for (int filter_index = 0; filter_index < pw_conv_parallelism_out;
				 filter_index++)
			{
#pragma HLS UNROLL
					weights_tile[filter_index][in_filter_index] = (weights_dt)chunck(
						(within_filter_index * pw_conv_parallelism_out + filter_index) * weights_dt_width + weights_dt_offset,
						(within_filter_index * pw_conv_parallelism_out + filter_index) * weights_dt_width);
				}
			}
		}
	}
}

void fill_weights_tile_off_chip(weights_grp_dt *weights,
								weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
								int starting_filter, const int layer_depth,
								const int num_of_weight_groups, const int layer_weights_offset)
{
	// assumes pw_parallelism_out * filter depth is divisable by weight group number
	const int current_fill_offset = layer_weights_offset + starting_filter * layer_depth / weights_group_items;

fill_weights_loop:
	for (int weight_grp_index = 0;
		 weight_grp_index < num_of_weight_groups; weight_grp_index++)
	{
		weights_grp_dt chunck = weights[current_fill_offset + weight_grp_index];
		for (int within_filter_index = 0;
			 within_filter_index < num_of_weights_in_the_same_filter_and_group;
			 within_filter_index++)
		{
#pragma HLS UNROLL
			for (int filter_index = 0; filter_index < pw_conv_parallelism_out;
				 filter_index++)
			{
#pragma HLS UNROLL
				weights_tile[filter_index][weight_grp_index * num_of_weights_in_the_same_filter_and_group + within_filter_index] = (weights_dt)chunck(
					(within_filter_index * pw_conv_parallelism_out + filter_index) * weights_dt_width + weights_dt_offset,
					(within_filter_index * pw_conv_parallelism_out + filter_index) * weights_dt_width);
			}
		}
	}
}
#endif

void fill_layers_weights_cpu(weights_dt *weights,
							 weights_dt weights_buffer[][max_conv_d],
							 int starting_filter, const int layer_depth,
							 const int layer_weights_offset,
							 const int layer_num_fils)
{
	const int layer_Weights_offset_cpu = layer_weights_offset * weights_group_items;
	const int current_fill_offset = layer_Weights_offset_cpu + starting_filter * layer_depth;
	for (int filter_index = 0; filter_index < pw_conv_parallelism_out; filter_index++)
	{
		if (filter_index + starting_filter < layer_num_fils)
		{
			for (int d = 0; d < layer_depth; d++)
			{
				weights_buffer[filter_index][d] = weights[current_fill_offset + filter_index * layer_depth + d];
			}
		}
	}
}

void fill_layers_weights_cpu_pw_conv(weights_dt *weights,
									 weights_dt weights_buffer[][max_conv_d][max_filter_area],
									 int starting_filter, const int layer_depth,
									 const int layer_weights_offset,
									 const int layer_num_fils)
{
	const int layer_Weights_offset_cpu = layer_weights_offset * weights_group_items;
	const int current_fill_offset = layer_Weights_offset_cpu + starting_filter * max_filter_area * layer_depth; // todo max_filter_area

	for (int filter_index = 0; filter_index < pw_conv_parallelism_out; filter_index++)
	{
		if (filter_index + starting_filter < layer_num_fils)
		{
			for (int d = 0; d < layer_depth; d++)
			{
				for (int i = 0; i < max_filter_area; i++)
				{
					weights_buffer[filter_index][d][i] = weights[current_fill_offset +
																 (filter_index * layer_depth + d) * max_filter_area + i]; // todo
				}
			}
		}
	}
}

void fill_layer_0_weights(
	layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][3][3])
{
	for (int i = 0; i < first_conv_layer_num_fils; i++)
	{
		for (int j = 0; j < first_conv_layer_depth; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{
					weights_1[i][j][k][l] = i * j * k * l % 8;
				}
			}
		}
	}
}

// void fill_dw_layer_weights(
// 	const dw_weights_dt src[max_conv_d][max_conv_h * max_conv_w],
// 	dw_weights_dt dst[max_conv_d][max_conv_h * max_conv_w],
// 	const int conv_d, const int conv_h, const int conv_w)
// {
// #pragma HLS INLINE OFF
// 	for (int d = 0; d < conv_d; d++)
// 	{
// 		if (d < conv_d)
// 		{
// 			for (int h = 0; h < conv_h; h++)
// 			{
// 				if (h < conv_h)
// 				{
// 					for (int w = 0; w < conv_w; w++)
// 					{
// 						if (w < conv_w)
// 						{
// 							dst[d][h * conv_w + w] = src[d][h * conv_w + w];
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }

void fill_fused_zero_points_buffer(const biases_dt fused_zero_points[],
								   biases_dt fused_zero_points_buffer[], int starting_d, int layer,
								   const int current_layer_fused_parameters_offset)
{
	const int absolute_current_layer_fused_parameters_offset = current_layer_fused_parameters_offset + starting_d;
	for (int i = 0; i < pw_conv_parallelism_out; i++)
	{
		fused_zero_points_buffer[i] = fused_zero_points[absolute_current_layer_fused_parameters_offset + i];
	}
}

void fill_fused_scales_buffer(const fused_scales_dt fused_scales[],
							  fused_scales_dt fused_scales_buffer[],int starting_d,
							  int layer, const int current_layer_fused_parameters_offset)
{
	const int absolute_current_layer_fused_parameters_offset = current_layer_fused_parameters_offset + starting_d;
	for (int i = 0; i < pw_conv_parallelism_out; i++)
	{
		fused_scales_buffer[i] = fused_scales[absolute_current_layer_fused_parameters_offset + i];
	}
}

void fill_fused_scales_and_zero_points(
	const fused_scales_dt layer_fused_scales[],
	fused_scales_dt fused_scales[],
	const biases_dt layer_fused_zero_points[],
	biases_dt fused_zero_points[], const int layer_num_filters)
{
#pragma HLS INLINE off

	for (int i = 0; i < max_conv_d; i++)
	{
		if (i >= layer_num_filters)
		{
			break;
		}
		fused_scales[i] = layer_fused_scales[i];
		fused_zero_points[i] = layer_fused_zero_points[i];
	}
}

void copy_channels_to_tmp_channels(fms_dt channels[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size])
{
	for (int i = 0; i < max_tmp_fms_size / pw_tile_w; i++)
	{
#pragma HLS PIPELINE
		for (int par = 0; par < pw_tile_w; par++)
		{
#pragma HLS UNROLL
			tmp_channels[i * pw_tile_w + par] = channels[i * pw_tile_w + par];
		}
	}
}

void copy_channels_to_tmp_channels(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
								   fms_dt tmp_channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH])
{
	for (int d = 0; d < MAX_TMP_FMS_BUFFER_DEPTH; d++)
	{
		for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
		{
#pragma HLS PIPELINE
			for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
			{
#pragma HLS UNROLL
				tmp_channels[d][h][w] = channels[d][h][w];
			}
		}
	}
}

void fill_model_configs_list(const int model_configs_list_src[2 * max_conv_layers],
							 int model_configs_list[2 * max_conv_layers])
{
	for (int i = 0; i < 2 * max_conv_layers; i++)
	{
		model_configs_list[i] = model_configs_list_src[i];
	}
}
