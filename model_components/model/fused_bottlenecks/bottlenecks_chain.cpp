#include "bottlenecks_chain.h"

// padding left and right
// padding top: just do not fill

void _7_stages_fill_ifm_groups_buffer(
	fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_in_rows_at_once], int starting_h,
	const int elements_to_fill_from_an_ifm)
{ //_7_stages_layer_0_in_rows_at_once * input_image_num_fms_groups_in_width
#pragma HLS INLINE off

	const int start_filling_offset = starting_h * input_image_num_fms_groups_in_width;
	int elements_avaiable_in_input_image;
	if (starting_h + _7_stages_layer_0_in_rows_at_once >= input_image_height)
	{
		elements_avaiable_in_input_image = (input_image_height - starting_h) * input_image_num_fms_groups_in_width;
		if (elements_avaiable_in_input_image < 0)
		{
			elements_avaiable_in_input_image = 0;
		}
	}
	for (int d = 0; d < input_image_depth; d++)
	{
		const int d_offst = start_filling_offset + d * input_image_num_fms_groups_in_a_channel;
		for (int i = 0; i < elements_to_fill_from_an_ifm; i++)
		{
			if (i < elements_avaiable_in_input_image)
			{
				fms_groups_buffer[d][i] = channels[d_offst + i];
			}
		}
	}
}

void fill_row_from_groups_buffer(
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_in_rows_at_once],
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_in_buffer_height][input_image_width],
	int row, const int channels_buffer_start_filling_h)
{
#pragma HLS INLINE

	const int start_filling_offset = row * input_image_num_fms_groups_in_width;

	for (int o_w = 0; o_w < input_image_num_fms_groups_in_width; o_w++)
	{
#pragma HLS PIPELINE off
		const int o_w_offset = o_w * input_image_group_items;
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			fms_grp_dt chunck = fms_groups_buffer[d][start_filling_offset + o_w];
			for (int w = 0; w < input_image_group_items; w++)
			{
#pragma HLS PIPELINE
				if (o_w_offset + w < input_image_width)
				{
					channels_buffer_0[d][channels_buffer_start_filling_h + row][o_w_offset + w] = (fms_dt)chunck(
						w * fms_dt_width + fms_dt_offset, w * fms_dt_width);
				}
			}
		}
	}
}

void _7_stages_shift_channels_buffer_rows(
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_in_buffer_height][input_image_width],
	const int rows_to_shift)
{
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = 0; h < rows_to_shift; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = channels_buffer_0[d][h + _7_stages_layer_0_in_rows_at_once][w];
			}
		}
	}
}

void _7_stages_padd_bottom_channels_buffer_rows(
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_in_buffer_height][input_image_width],
	const fms_dt zero_point)
{
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = _7_stages_layer_0_in_buffer_height - layer_0_dw_padding_bottom;
				 h < _7_stages_layer_0_in_buffer_height; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = zero_point;
			}
		}
	}
}

void _7_stages_fill_channels_buffer_from_groups_buffer(
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_in_rows_at_once],
	fms_dt channels_buffer_0[input_image_depth][_7_stages_layer_0_in_buffer_height][input_image_width],
	int starting_h, bool shift, const fms_dt zero_point)
{
#pragma HLS INLINE off

	const int rows_to_shift = _7_stages_layer_0_in_buffer_height - _7_stages_layer_0_in_rows_at_once;
	if (shift)
	{
		_7_stages_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
	}
	const int channels_buffer_start_filling_h =
		starting_h == 0 ? layer_0_dw_padding_top : rows_to_shift;
	for (int h = 0; h < _7_stages_layer_0_in_rows_at_once; h++)
	{
		if (starting_h + h < input_image_height)
		{
			fill_row_from_groups_buffer(fms_groups_buffer, channels_buffer_0, h,
										channels_buffer_start_filling_h);
		}
		else
		{
			_7_stages_padd_bottom_channels_buffer_rows(channels_buffer_0,
													   zero_point);
		}
	}
}

void fill_first_bottleneck_input(fms_dt channels_buffer_0[input_image_depth][bottlenck_0_input_buffer_height][input_image_width],
								 fms_dt first_bottleneck_input[],
								 const int starting_w,
								 fms_dt zero_point, const int first_fill_dst_left_offset)
{

	const int channels_buffer_0_hw = bottlenck_0_input_buffer_height * input_image_width;
	const int start_filling_index_in_first_bottleneck_input =
		first_fill_dst_left_offset * bottleneck_1_ifms_depth;

	for (int d = 0; d < input_image_depth; d++)
	{
#pragma HLS UNROLL
		for (int h = 0; h < bottleneck_0_fill_each_time; h++)
		{
#pragma HLS UNROLL
			for (int w = 0; w < bottleneck_0_fill_each_time; w++)
			{
				first_bottleneck_input[start_filling_index_in_first_bottleneck_input + d * bottlenck_0_input_buffer_size +
									   h * bottlenck_0_input_buffer_width + w] =
					channels_buffer_0[starting_w + d * channels_buffer_0_hw + h * input_image_width + w];
			}
		}
	}
}

void save_chain_output(fms_dt chain_output[], fms_dt result[max_fms_size],
					   const bottlenecks_chain_specs chain_specs, int h, int w)
{
#pragma HLS INLINE off

	const int num_of_tiles_w = chain_specs.chain_output_num_tiles_w;
	const int num_of_tiles_hw = chain_specs.chain_output_num_tiles_h * chain_specs.chain_output_num_tiles_w;
	const int tile_in_h = h / pw_tile_h;
	const int in_tile_h = h % pw_tile_h;
	const int tile_in_w = w / pw_tile_w;
	const int in_tile_w = w % pw_tile_w;

	for (int d = 0; d < chain_specs.chain_ofms_depth; d++)
	{
		const int tile_in_d = d / pw_tile_d;
		const int in_tile_d = d % pw_tile_d;
		const int tile_index = tile_in_d * num_of_tiles_hw + tile_in_h * num_of_tiles_w + tile_in_w;

		const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

		const int index_in_result = tile_index * pw_tile_size + in_tile_index;

		result[index_in_result] = chain_output[d];
	}
}

void _1_bottlenecks_chain(fms_dt channels[max_fms_size], fms_dt chain_input[], // chain_input_height*chain_input_width*chain_input_depth
						  fms_dt result[max_fms_size], const bottlenecks_chain_specs chain_specs,
						  int starting_h, int filling_row)
{
#pragma HLS INLINE off

	fms_dt bottleneck_0_input[bottlenck_0_input_buffer_size];
	fms_dt bottleneck_0_output[bottleneck_0_ofms_depth];
	fms_dt bottleneck_0_previous_pass_dw_input_1[bottlenck_0_inter_pass_dw_input_size];
	fms_dt bottleneck_0_previous_pass_dw_input_2[bottlenck_0_inter_pass_dw_input_size];

#pragma HLS ARRAY_PARTITION variable = previous_pass_dw_input_1 type = cyclic factor = 2
#pragma HLS ARRAY_PARTITION variable = previous_pass_dw_input_2 type = cyclic factor = 2

	const fms_dt first_dw_layer_in_the_chain_zero_point =
		conv_fms_zero_points[chain_specs.first_dw_layer_in_the_chain];
	const int bottleneck_1_first_fill_offset = bottleneck_1_dw_filter_dim - bottleneck_1_dw_strides;
	const int first_fill_from_left_offset = chain_specs.first_filter_dim - chain_specs.first_strides;
	const fms_dt layer_0_fms_zero_point = conv_fms_zero_points[0];
	const int num_of_ifm_groups_read_each_time =
		input_image_num_fms_groups_in_width * bottleneck_0_rows_at_once;

	fms_dt channels_buffer_0[input_image_depth][bottlenck_0_input_buffer_height][input_image_width];
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = channels_buffer_0 complete dim = 2

	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * _7_stages_layer_0_in_rows_at_once];

	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 filling_row, input_image_num_fms_groups_in_width);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0, filling_row, false,
													  layer_0_fms_zero_point);

	_7_stages_fill_ifm_groups_buffer(channels, fms_groups_buffer,
									 filling_row, num_of_ifm_groups_read_each_time);
	_7_stages_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
													  channels_buffer_0, filling_row, false,
													  layer_0_fms_zero_point);

	fill_first_bottleneck_input(channels_buffer_0, bottleneck_0_input,
								0, first_dw_layer_in_the_chain_zero_point,
								first_fill_from_left_offset);

	fill_first_bottleneck_input(channels_buffer_0, bottleneck_0_input,
								1, first_dw_layer_in_the_chain_zero_point,
								first_fill_from_left_offset);

	mob_v2_bottleneck_0(bottleneck_0_input,
						bottleneck_0_output,
						bottleneck_0_previous_pass_dw_input_1,
						bottleneck_0_previous_pass_dw_input_2, starting_h, 0);

	// mob_v2_bottleneck(bottleneck_1_input, bottleneck_1_output,
	// 				  previous_pass_dw_input_1, previous_pass_dw_input_2, pw_weights_4,
	// 				  dw_weights_5, pw_weights_6, layer_4_fused_scales,
	// 				  layer_4_fused_scales_log_2_shifts, layer_4_relu_6_fused_scales,
	// 				  layer_4_fused_zero_points, layer_5_fused_scales,
	// 				  layer_5_fused_scales_log_2_shifts, layer_5_relu_6_fused_scales,
	// 				  layer_5_fused_zero_points, layer_6_fused_scales,
	// 				  layer_6_fused_scales_log_2_shifts, layer_6_relu_6_fused_scales,
	// 				  layer_6_fused_zero_points, bottleneck_1_ifms_depth,
	// 				  bottleneck_1_ifms_height, bottleneck_1_ifms_width,
	// 				  bottleneck_1_ofms_depth, bottleneck_1_ofms_width,
	// 				  bottleneck_1_expanded_ifms_depth, bottleneck_1_dw_filter_dim,
	// 				  bottleneck_1_dw_strides, starting_h, 0,
	// 				  bottleneck_1_expansion_parallelism_h,
	// 				  bottleneck_1_expansion_parallelism_w,
	// 				  bottleneck_1_expansion_layer_index, bottleneck_1_dw_layer_index,
	// 				  bottleneck_1_projection_layer_index,
	// 				  bottleneck_1_expansion_layer_relu, bottleneck_1_dw_layer_relu,
	// 				  bottleneck_1_projection_layer_relu, bottleneck_1_dw_padding_left,
	// 				  bottleneck_1_dw_padding_right, bottleneck_1_dw_padding_top,
	// 				  bottleneck_1_dw_padding_bottom, first_fill_from_left_offset);

	for (int w = 0; w < chain_specs.chain_ofms_width; w++)
	{
		const int fill_input_index = w * bottleneck_1_dw_strides * bottleneck_1_rows_at_once + bottleneck_1_first_fill_offset;
		fill_first_bottleneck_input(chain_input, bottleneck_0_input, fill_input_index,
									first_dw_layer_in_the_chain_zero_point, 0);
		if (w % 2 == 0)
		{
			mob_v2_bottleneck_0(bottleneck_0_input,
								bottleneck_0_output,
								bottleneck_0_previous_pass_dw_input_2,
								bottleneck_0_previous_pass_dw_input_1, starting_h, w);
			// 	mob_v2_bottleneck(bottleneck_1_input, bottleneck_1_output,
			// 					  previous_pass_dw_input_2, previous_pass_dw_input_1,
			// 					  pw_weights_4, dw_weights_5, pw_weights_6, layer_4_fused_scales,
			// 					  layer_4_fused_scales_log_2_shifts, layer_4_relu_6_fused_scales,
			// 					  layer_4_fused_zero_points, layer_5_fused_scales,
			// 					  layer_5_fused_scales_log_2_shifts, layer_5_relu_6_fused_scales,
			// 					  layer_5_fused_zero_points, layer_6_fused_scales,
			// 					  layer_6_fused_scales_log_2_shifts, layer_6_relu_6_fused_scales,
			// 					  layer_6_fused_zero_points, bottleneck_1_ifms_depth,
			// 					  bottleneck_1_ifms_height, bottleneck_1_ifms_width,
			// 					  bottleneck_1_ofms_depth, bottleneck_1_ofms_width,
			// 					  bottleneck_1_expanded_ifms_depth, bottleneck_1_dw_filter_dim,
			// 					  bottleneck_1_dw_strides, starting_h, w,
			// 					  bottleneck_1_expansion_parallelism_h,
			// 					  bottleneck_1_expansion_parallelism_w,
			// 					  bottleneck_1_expansion_layer_index, bottleneck_1_dw_layer_index,
			// 					  bottleneck_1_projection_layer_index,
			// 					  bottleneck_1_expansion_layer_relu, bottleneck_1_dw_layer_relu,
			// 					  bottleneck_1_projection_layer_relu,
			// 					  bottleneck_1_dw_padding_left, bottleneck_1_dw_padding_right,
			// 					  bottleneck_1_dw_padding_top, bottleneck_1_dw_padding_bottom, 0);
		}
		else
		{
			mob_v2_bottleneck_0(bottleneck_0_input,
								bottleneck_0_output,
								bottleneck_0_previous_pass_dw_input_1,
								bottleneck_0_previous_pass_dw_input_2, starting_h, w);
			// 	mob_v2_bottleneck(bottleneck_1_input, bottleneck_1_output,
			// 					  previous_pass_dw_input_1, previous_pass_dw_input_2,
			// 					  pw_weights_4, dw_weights_5, pw_weights_6, layer_4_fused_scales,
			// 					  layer_4_fused_scales_log_2_shifts, layer_4_relu_6_fused_scales,
			// 					  layer_4_fused_zero_points, layer_5_fused_scales,
			// 					  layer_5_fused_scales_log_2_shifts, layer_5_relu_6_fused_scales,
			// 					  layer_5_fused_zero_points, layer_6_fused_scales,
			// 					  layer_6_fused_scales_log_2_shifts, layer_6_relu_6_fused_scales,
			// 					  layer_6_fused_zero_points, bottleneck_1_ifms_depth,
			// 					  bottleneck_1_ifms_height, bottleneck_1_ifms_width,
			// 					  bottleneck_1_ofms_depth, bottleneck_1_ofms_width,
			// 					  bottleneck_1_expanded_ifms_depth, bottleneck_1_dw_filter_dim,
			// 					  bottleneck_1_dw_strides, starting_h, w,
			// 					  bottleneck_1_expansion_parallelism_h,
			// 					  bottleneck_1_expansion_parallelism_w,
			// 					  bottleneck_1_expansion_layer_index, bottleneck_1_dw_layer_index,
			// 					  bottleneck_1_projection_layer_index,
			// 					  bottleneck_1_expansion_layer_relu, bottleneck_1_dw_layer_relu,
			// 					  bottleneck_1_projection_layer_relu,
			// 					  bottleneck_1_dw_padding_left, bottleneck_1_dw_padding_right,
			// 					  bottleneck_1_dw_padding_top, bottleneck_1_dw_padding_bottom, 0);
		}

		save_chain_output(bottleneck_1_output, result, chain_specs, starting_h,
						  w);
	}
}
