#include "../../basic_defs/simulation_constants.h"

#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE && CHAIN_LENGTH == 6 && MODEL_ID != 1  && ! ONLY_SEML

#include "bottlenecks_chain.h"

// padding left and right
// padding top: just do not fill
void bottleneck_chain_0_1_fill_ifm_groups_buffer(
	fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * bottleneck_0_fill_each_time], int starting_h,
	const int elements_to_fill_from_an_ifm)
{ // chain_0_1_layer_0_s_in_rows_at_once * input_image_num_fms_groups_in_width
#pragma HLS INLINE off

	const int start_filling_offset = starting_h * input_image_num_fms_groups_in_width;
	int elements_avaiable_in_input_image;

	elements_avaiable_in_input_image = (input_image_height - starting_h) * input_image_num_fms_groups_in_width;

	for (int d = 0; d < input_image_depth; d++)
	{
#pragma HLS PIPELINE off
		const int d_offst = start_filling_offset + d * input_image_num_fms_groups_in_a_channel;
		for (int i = 0; i < elements_to_fill_from_an_ifm; i++)
		{
#pragma HLS PIPELINE off
			if (i < elements_avaiable_in_input_image)
			{
				fms_groups_buffer[d][i] = channels[d_offst + i];
			}
		}
	}
}

void chain_0_1_fill_row_from_groups_buffer(
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * bottleneck_0_fill_each_time],
	fms_dt channels_buffer_0[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
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
#if HW == _FPGA
					channels_buffer_0[d][channels_buffer_start_filling_h + row][o_w_offset + w] = (fms_dt)chunck(
						w * fms_dt_width + fms_dt_offset, w * fms_dt_width);
#endif
				}
			}
		}
	}
}

void chain_0_1_shift_channels_buffer_rows(
	fms_dt channels_buffer_0[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
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
				channels_buffer_0[d][h][w] = channels_buffer_0[d][h + chain_0_1_rows_filled_each_time][w];
			}
		}
	}
}

void chain_0_1_padd_bottom_channels_buffer_rows(
	fms_dt channels_buffer_0[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
	const fms_dt zero_point)
{
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = chain_0_1_in_buffer_height - first_conv_layer_specs.padding_bottom;
				 h < chain_0_1_in_buffer_height; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = zero_point;
			}
		}
	}
}

void bottleneck_chain_fill_channels_buffer_from_groups_buffer(
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * chain_0_1_rows_filled_each_time],
	fms_dt channels_buffer_0[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
	int starting_h, bool shift, const fms_dt zero_point)
{
#pragma HLS INLINE off

	const int rows_to_shift = chain_0_1_in_buffer_height - chain_0_1_rows_filled_each_time;
	if (shift)
	{
		chain_0_1_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
	}
	const int channels_buffer_start_filling_h =
		starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
	for (int h = 0; h < chain_0_1_rows_filled_each_time; h++)
	{
		if (starting_h + h < input_image_height)
		{
			chain_0_1_fill_row_from_groups_buffer(fms_groups_buffer,
												  channels_buffer_0, h, channels_buffer_start_filling_h);
		}
		else
		{
			chain_0_1_padd_bottom_channels_buffer_rows(channels_buffer_0,
													   zero_point);
		}
	}
}

void fill_first_cols_of_first_bottleneck_input(
	fms_dt channels_buffer_0[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
	fms_dt bottleneck_0_input[bottleneck_0_input_buffer_size],
	int starting_h)
{

	const int offset_in_bottleneck_0_input = chain_0_1_first_strides;
fill_first_cols_of_first_bottleneck_input:
	for (int d = 0;
		 d < input_image_depth; d++)
	{
		for (int h = 0; h < chain_0_1_in_buffer_height; h++)
		{
			for (int w = 0;
				 w < chain_0_1_first_filter_dim - chain_0_1_first_strides;
				 w++)
			{
				bottleneck_0_input[d * bottleneck_0_input_buffer_hw + h * bottlenck_0_input_buffer_width + w + chain_0_1_first_strides] =
					channels_buffer_0[d][starting_h + h][w];
			}
		}
	}
}

void shift_and_fill_bottleneck_0_input(
	fms_dt channels_buffer_0[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
	fms_dt bottleneck_0_input[bottleneck_0_input_buffer_size],
	const int starting_w, int starting_h, fms_dt zero_point)
{
#pragma HLS INLINE

	const int to_be_shifted = chain_0_1_first_filter_dim - chain_0_1_first_strides;
	const int start_filling_index_in_first_bottleneck_input =
		chain_0_1_first_filter_dim - chain_0_1_first_strides;

fill_bottleneck_0_input:
	for (int d = 0; d < chain_0_1_ifms_depth; d++)
	{
#pragma HLS UNROLL
		for (int h = 0; h < chain_0_1_first_filter_dim; h++)
		{
#pragma HLS UNROLL
			for (int w = 0; w < to_be_shifted; w++)
			{
#pragma HLS UNROLL
				const int to_fill_in_index = d * bottleneck_0_input_buffer_hw + h * bottlenck_0_input_buffer_width + w;
				bottleneck_0_input[to_fill_in_index] =
					bottleneck_0_input[to_fill_in_index + chain_0_1_first_strides];
			}
			for (int w = 0; w < chain_0_1_first_strides; w++)
			{
#pragma HLS UNROLL
				if (starting_w + w < chain_0_1_ifms_width + chain_0_1_first_padding_left)
				{
					bottleneck_0_input[d * bottleneck_0_input_buffer_hw + h * bottlenck_0_input_buffer_width + w + start_filling_index_in_first_bottleneck_input] =
						channels_buffer_0[d][h + starting_h][starting_w + w];
				}
				else
				{
					bottleneck_0_input[d * bottleneck_0_input_buffer_hw + h * bottlenck_0_input_buffer_width + w + start_filling_index_in_first_bottleneck_input] =
						zero_point;
				}
			}
		}
	}
}

void fill_bottleneck_1_input(
	fms_dt bottleneck_0_1_communication_buffer_prev[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width],
	fms_dt bottleneck_1_input_buffer[bottleneck_1_input_buffer_size],
	const int offset_w)
{
#pragma HLS INLINE
	for (int h = 0; h < chain_0_1_bottleneck_0_rows_at_once; h++)
	{
#pragma HLS UNROLL

		for (int w = 0; w < chain_0_1_bottleneck_0_rows_at_once; w++)
		{
#pragma HLS UNROLL

			for (int d = 0; d < bottleneck_1_ifms_depth; d++)
			{
#pragma HLS PIPELINE
				bottleneck_1_input_buffer[h * chain_0_1_bottleneck_0_rows_at_once * bottleneck_1_ifms_depth + w * bottleneck_1_ifms_depth + d] = bottleneck_0_1_communication_buffer_prev[d][h][w + offset_w];
			}
		}
	}
}

void save_chain_output(fms_dt chain_output[], fms_dt result[max_fms_size],
					   const int chain_output_num_tiles_w, const int chain_output_num_tiles_h,
					   const int chain_ofms_depth, int h, int w)
{
#pragma HLS INLINE off

	const int num_of_tiles_w = chain_output_num_tiles_w;
	const int num_of_tiles_hw = chain_output_num_tiles_h * num_of_tiles_w;
	const int tile_in_h = h / pw_tile_h;
	const int in_tile_h = h % pw_tile_h;
	const int tile_in_w = w / pw_tile_w;
	const int in_tile_w = w % pw_tile_w;

save_chain_output:
	for (int d = 0; d < chain_ofms_depth; d++)
	{
		const int tile_in_d = d / pw_tile_d;
		const int in_tile_d = d % pw_tile_d;
		const int tile_index = tile_in_d * num_of_tiles_hw + tile_in_h * num_of_tiles_w + tile_in_w;

		const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

		const int index_in_result = tile_index * pw_tile_size + in_tile_index;

		result[index_in_result] = chain_output[d];
	}
}

void chain_0_1_fill_channels_buffer_cpu(
	fms_dt channels[input_image_depth * input_image_hw],
	fms_dt channels_buffer_0[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
	int starting_h, bool shift, const fms_dt zero_point)
{
#pragma HLS INLINE off

	const int rows_to_shift = chain_0_1_in_buffer_height - chain_0_1_rows_filled_each_time;
	if (shift)
	{
		chain_0_1_shift_channels_buffer_rows(channels_buffer_0, rows_to_shift);
	}
	const int channels_buffer_start_filling_h =
		starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
	for (int h = 0; h < chain_0_1_rows_filled_each_time; h++)
	{
		if (starting_h + h < input_image_height)
		{
			for (int d = 0; d < input_image_depth; d++)
			{
				for (int w = 0; w < input_image_width; w++)
				{
					channels_buffer_0[d][h + channels_buffer_start_filling_h][w] =
						channels[d * input_image_hw + (h + starting_h) * input_image_width + w];
				}
			}
		}
		else
		{
			chain_0_1_padd_bottom_channels_buffer_rows(channels_buffer_0,
													   zero_point);
		}
	}
}

void bottleneck_0_pipeline_filling_stage(
	fms_dt chain_input[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
	fms_dt bottleneck_0_input[bottleneck_0_input_buffer_size],
	pss_dt bottleneck_0_projection_kernel_output[bottleneck_0_ofms_depth],
	pss_dt bottleneck_0_projection_kernel_output_prev[bottleneck_0_ofms_depth],
	fms_dt bottleneck_0_1_communication_buffer[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width],
	fms_dt bottleneck_0_dw_lower_buffer[bottleneck_0_expanded_ifms_depth][bottleneck_0_dw_filter_dim],
	fms_dt bottleneck_0_previous_pass_dw_input[bottleneck_0_expanded_ifms_depth][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
	const int bottleneck_0_extra_cols_filled_first_time,
	const int bottleneck_0_first_fill_offset,
	const fms_dt layer_0_s_ifms_zero_point,
	const fms_dt bottleneck_0_dw_ifms_zero_point)
{
#pragma HLS INLINE off

	relu_6_fused_scales_dt layer_3_pw_relu_6_fused_scale = layer_3_pw_relu_6_fused_scales[0];

	bottleneck_0_padding_top_right(bottleneck_0_previous_pass_dw_input,
								   bottleneck_0_dw_ifms_zero_point);

	for (int h = 0; h < chain_0_1_bottleneck_0_rows_at_once; h++)
	{
		fill_first_cols_of_first_bottleneck_input(chain_input,
												  bottleneck_0_input, h * first_conv_layer_specs.strides); // starting_h + 2

		shift_and_fill_bottleneck_0_input(chain_input, bottleneck_0_input,
										  bottleneck_0_extra_cols_filled_first_time,
										  h * first_conv_layer_specs.strides, layer_0_s_ifms_zero_point); // starting_h + 2

		bottleneck_0_do_padding_left(bottleneck_0_previous_pass_dw_input,
									 bottleneck_0_dw_lower_buffer, bottleneck_0_dw_ifms_zero_point);
		mob_v2_bottleneck_0(bottleneck_0_input,
							bottleneck_0_projection_kernel_output,
							bottleneck_0_projection_kernel_output_prev,
							bottleneck_0_1_communication_buffer,
							bottleneck_0_previous_pass_dw_input,
							bottleneck_0_dw_lower_buffer, h, h, 0); // strtaing_h+1

		for (int o_w = 0; o_w < chain_0_1_bottleneck_0_rows_at_once; o_w++)
		{
			for (int i_w = 0; i_w < chain_0_1_ofms_width; i_w++)
			{
				const int w = o_w * chain_0_1_ofms_width + i_w;
				const int fill_input_index = (w + 1) * chain_0_1_first_strides + bottleneck_0_first_fill_offset;
				shift_and_fill_bottleneck_0_input(chain_input,
												  bottleneck_0_input, fill_input_index,
												  h * first_conv_layer_specs.strides, // starting_h + 2
												  layer_0_s_ifms_zero_point);
				mob_v2_bottleneck_0(bottleneck_0_input,
									bottleneck_0_projection_kernel_output,
									bottleneck_0_projection_kernel_output_prev,
									bottleneck_0_1_communication_buffer,
									bottleneck_0_previous_pass_dw_input,
									bottleneck_0_dw_lower_buffer, h, h, w + 1); // starting_h + 1
			}
		}
		//##########################corner cases###################################
		// normalize last col in each row

		for (int d = 0; d < bottleneck_0_ofms_depth; d++)
		{
			bottleneck_0_1_communication_buffer[d][h][bottleneck_0_ofms_width - 1] = normalize_projection_kernel_output(
				bottleneck_0_projection_kernel_output_prev,
				layer_3_pw_fused_scales,
				layer_3_pw_fused_scales_log_2_shifts,
				layer_3_pw_relu_6_fused_scale,
				layer_3_pw_fused_zero_points, d, layer_3_pw_specs.layer_activation, layer_3_pw_specs);
		}
		// perform last update for the dw inter step buffer
		for (int d = 0; d < bottleneck_0_expanded_ifms_depth; d++)
		{
			if (h == 1)
			{
				bottleneck_0_previous_pass_dw_input[d][0][bottleneck_0_inter_pass_dw_input_width - bottleneck_0_dw_padding_right - 1] =
					bottleneck_0_previous_pass_dw_input[d][1][bottleneck_0_inter_pass_dw_input_width - bottleneck_0_dw_padding_right - 1];
			}
			bottleneck_0_previous_pass_dw_input[d][1][bottleneck_0_inter_pass_dw_input_width - bottleneck_0_dw_padding_right - 1] =
				bottleneck_0_dw_lower_buffer[d][0];
		}
		//############################corner cases#################################
	}
}

void bottleneck_0_within_pipeline_stage(
	fms_dt chain_input[input_image_depth][chain_0_1_in_buffer_height][input_image_width],
	fms_dt bottleneck_0_input[bottleneck_0_input_buffer_size],
	pss_dt bottleneck_0_projection_kernel_output[bottleneck_0_ofms_depth],
	pss_dt bottleneck_0_projection_kernel_output_prev[bottleneck_0_ofms_depth],
	fms_dt bottleneck_0_1_communication_buffer[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width],
	fms_dt bottleneck_0_dw_lower_buffer[bottleneck_0_expanded_ifms_depth][bottleneck_0_dw_filter_dim],
	fms_dt bottleneck_0_previous_pass_dw_input[bottleneck_0_expanded_ifms_depth][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
	const int starting_h,
	const int bottleneck_0_extra_cols_filled_first_time,
	const int bottleneck_0_first_fill_offset,
	const fms_dt layer_0_s_ifms_zero_point,
	const fms_dt bottleneck_0_dw_ifms_zero_point)
{
#pragma HLS INLINE off

	relu_6_fused_scales_dt layer_3_pw_relu_6_fused_scale = layer_3_pw_relu_6_fused_scales[0];
	relu_6_fused_scales_dt layer_7_pw_relu_6_fused_scale = layer_7_pw_relu_6_fused_scales[0];

	int bottleneck_0_h = starting_h * chain_0_1_bottleneck_0_rows_at_once;

	for (int h = 0; h < chain_0_1_bottleneck_0_rows_at_once; h++)
	{
		fill_first_cols_of_first_bottleneck_input(chain_input,
												  bottleneck_0_input, h * first_conv_layer_specs.strides); // starting_h + 2

		shift_and_fill_bottleneck_0_input(chain_input, bottleneck_0_input,
										  bottleneck_0_extra_cols_filled_first_time,
										  h * first_conv_layer_specs.strides, layer_0_s_ifms_zero_point); // starting_h + 2

		bottleneck_0_do_padding_left(bottleneck_0_previous_pass_dw_input,
									 bottleneck_0_dw_lower_buffer, bottleneck_0_dw_ifms_zero_point);
		mob_v2_bottleneck_0(bottleneck_0_input,
							bottleneck_0_projection_kernel_output,
							bottleneck_0_projection_kernel_output_prev,
							bottleneck_0_1_communication_buffer,
							bottleneck_0_previous_pass_dw_input,
							bottleneck_0_dw_lower_buffer, bottleneck_0_h + h, h, 0); // strtaing_h+1

		for (int w = 0; w < bottleneck_0_ofms_width; w++)
		{
			const int fill_input_index = (w + 1) * chain_0_1_first_strides + bottleneck_0_first_fill_offset;
			shift_and_fill_bottleneck_0_input(chain_input, bottleneck_0_input,
											  fill_input_index, h * first_conv_layer_specs.strides, // starting_h + 2
											  layer_0_s_ifms_zero_point);
			mob_v2_bottleneck_0(bottleneck_0_input,
								bottleneck_0_projection_kernel_output,
								bottleneck_0_projection_kernel_output_prev,
								bottleneck_0_1_communication_buffer,
								bottleneck_0_previous_pass_dw_input,
								bottleneck_0_dw_lower_buffer, bottleneck_0_h + h, h, w + 1); // starting_h + 1
		}
		//##########################corner cases###################################
		// normalize last col in each row
		for (int d = 0; d < bottleneck_0_ofms_depth; d++)
		{
			bottleneck_0_1_communication_buffer[d][h][bottleneck_0_ofms_width - 1] = normalize_projection_kernel_output(
				bottleneck_0_projection_kernel_output_prev,
				layer_3_pw_fused_scales,
				layer_3_pw_fused_scales_log_2_shifts,
				layer_3_pw_relu_6_fused_scale,
				layer_3_pw_fused_zero_points, d, layer_3_pw_specs.layer_activation, layer_3_pw_specs);
		}
		// perform last update for the dw inter step buffer
		for (int d = 0; d < bottleneck_0_expanded_ifms_depth; d++)
		{
			bottleneck_0_previous_pass_dw_input[d][0][bottleneck_0_inter_pass_dw_input_width - bottleneck_0_dw_padding_right - 1] =
				bottleneck_0_previous_pass_dw_input[d][1][bottleneck_0_inter_pass_dw_input_width - bottleneck_0_dw_padding_right - 1];
			bottleneck_0_previous_pass_dw_input[d][1][bottleneck_0_inter_pass_dw_input_width - bottleneck_0_dw_padding_right - 1] =
				bottleneck_0_dw_lower_buffer[d][0];
		}
		//############################corner cases#################################
	}
}

void bottleneck_1_pipeline_filling_stage(
	fms_dt bottleneck_1_previous_pass_dw_input_1[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width],
	fms_dt bottleneck_1_previous_pass_dw_input_2[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width],
	const fms_dt bottleneck_1_dw_ifms_zero_point)
{
	bottleneck_1_padding_right(bottleneck_1_previous_pass_dw_input_1,
							   bottleneck_1_dw_ifms_zero_point);
	bottleneck_1_padding_right(bottleneck_1_previous_pass_dw_input_2,
							   bottleneck_1_dw_ifms_zero_point);
}

void bottleneck_1_within_pipeline_stage(
	fms_dt bottleneck_1_input[bottleneck_1_input_buffer_size],
	pss_dt bottleneck_1_projection_kernel_output[bottleneck_1_ofms_depth],
	pss_dt bottleneck_1_projection_kernel_output_prev[bottleneck_1_ofms_depth],
	fms_dt bottleneck_0_1_communication_buffer[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width],
	fms_dt chain_seml_communication_buffer[bottleneck_1_ofms_depth][bottleneck_1_ofms_width],
	fms_dt bottleneck_1_dw_lower_buffer[bottleneck_1_expanded_ifms_depth][bottleneck_1_dw_filter_dim * bottleneck_1_dw_strides],
	fms_dt bottleneck_1_previous_pass_dw_input_1[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width],
	fms_dt bottleneck_1_previous_pass_dw_input_2[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width],
	const int starting_h, const fms_dt bottleneck_1_dw_ifms_zero_point)
{
#pragma HLS INLINE off

	relu_6_fused_scales_dt layer_7_pw_relu_6_fused_scale = layer_7_pw_relu_6_fused_scales[0];
	const int bottleneck_1_first_fill_offset = 1;
	const int bottleneck_1_warm_up_rows = bottleneck_1_dw_filter_dim - bottleneck_1_dw_strides;
	int bottleneck_1_h =
		starting_h == 0 ? 0 : starting_h * bottleneck_1_rows_at_once * bottleneck_1_dw_strides - bottleneck_1_warm_up_rows;

	fill_bottleneck_1_input(bottleneck_0_1_communication_buffer,
							bottleneck_1_input, 0);

	mob_v2_bottleneck_1(bottleneck_1_input,
						bottleneck_1_projection_kernel_output,
						bottleneck_1_projection_kernel_output_prev,
						chain_seml_communication_buffer,
						bottleneck_1_previous_pass_dw_input_1,
						bottleneck_1_previous_pass_dw_input_2, bottleneck_1_dw_lower_buffer,
						bottleneck_1_h, 0); // strtaing_h+1

	for (int o_w = 0; o_w < chain_0_1_bottleneck_1_rows_at_once; o_w++)
	{
		for (int i_w = 0; i_w < chain_0_1_ofms_width; i_w++)
		{
			const int w = o_w * chain_0_1_ofms_width + i_w;
			const int fill_input_index = w * bottleneck_1_dw_strides + bottleneck_1_first_fill_offset;
			fill_bottleneck_1_input(bottleneck_0_1_communication_buffer,
									bottleneck_1_input, fill_input_index);
			if (starting_h % 2 == 1)
			{
				mob_v2_bottleneck_1(bottleneck_1_input,
									bottleneck_1_projection_kernel_output,
									bottleneck_1_projection_kernel_output_prev,
									chain_seml_communication_buffer,
									bottleneck_1_previous_pass_dw_input_1,
									bottleneck_1_previous_pass_dw_input_2,
									bottleneck_1_dw_lower_buffer, bottleneck_1_h,
									fill_input_index);
			}
			else
			{
				mob_v2_bottleneck_1(bottleneck_1_input,
									bottleneck_1_projection_kernel_output,
									bottleneck_1_projection_kernel_output_prev,
									chain_seml_communication_buffer,
									bottleneck_1_previous_pass_dw_input_2,
									bottleneck_1_previous_pass_dw_input_1,
									bottleneck_1_dw_lower_buffer, bottleneck_1_h,
									fill_input_index);
			}
		}
	}

	for (int d = 0; d < bottleneck_1_ofms_depth; d++)
	{
#pragma HLS PIPELINE
		chain_seml_communication_buffer[d][bottleneck_1_ofms_width - 1] =
			normalize_projection_kernel_output(
				bottleneck_1_projection_kernel_output_prev,
				layer_7_pw_fused_scales,
				layer_7_pw_fused_scales_log_2_shifts,
				layer_7_pw_relu_6_fused_scale,
				layer_7_pw_fused_zero_points, d, layer_7_pw_specs.layer_activation, layer_7_pw_specs);
	}
}

void copy_bottleneck_0_1_communication_buffer(
	fms_dt bottleneck_0_1_communication_buffer[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width],
	fms_dt bottleneck_0_1_communication_buffer_prev[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width])
{
#pragma HLS INLINE off

	for (int d = 0;
		 d < bottleneck_0_ofms_depth / bottleneck_0_1_communication_buffer_partitioning_factor_in_d;
		 d++)
	{
		for (int w = 0; w < bottleneck_0_ofms_width; w++)
		{
#pragma HLS PIPELINE
			for (int h = 0; h < chain_0_1_bottleneck_0_rows_at_once; h++)
			{
#pragma HLS UNROLL
				for (int d_inner = 0;
					 d_inner < bottleneck_0_1_communication_buffer_partitioning_factor_in_d;
					 d_inner++)
				{
#pragma HLS UNROLL
					bottleneck_0_1_communication_buffer_prev[d * bottleneck_0_1_communication_buffer_partitioning_factor_in_d + d_inner][h][w] =
						bottleneck_0_1_communication_buffer[d * bottleneck_0_1_communication_buffer_partitioning_factor_in_d + d_inner][h][w];
				}
			}
		}
	}
}

void write_chain_seml_communication_buffer(
	fms_dt chain_seml_communication_buffer[bottleneck_1_ofms_depth][bottleneck_1_ofms_width],
	fms_dt result[max_fms_size], const int starting_h)
{
#pragma HLS INLINE off

	const int num_tiles_hw = layer_8_pw_specs.layer_num_of_ifm_tiles_h * layer_8_pw_specs.layer_num_of_ifm_tiles_w;
	const int tile_in_h = starting_h / pw_tile_h;
	const int in_tile_h = starting_h % pw_tile_h;

	for (int d = 0; d < chain_0_1_ofms_depth; d++)
	{
		const int tile_in_d = d / pw_tile_d;
		const int in_tile_d = d % pw_tile_d;
		for (int w = 0; w < chain_0_1_ofms_width; w++)
		{
#pragma HLS PIPELINE
			const int tile_in_w = w / pw_tile_w;
			const int tile_index = tile_in_d * num_tiles_hw + tile_in_h * layer_8_pw_specs.layer_num_of_ofm_tiles_w + tile_in_w;

			const int in_tile_w = w % pw_tile_w;
			const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

			const int index_in_result = tile_index * pw_tile_size + in_tile_index;

			result[index_in_result] = chain_seml_communication_buffer[d][w];
		}
	}
}

void write_chain_seml_communication_buffer(
	fms_dt chain_seml_communication_buffer[bottleneck_1_ofms_depth][bottleneck_1_ofms_width],
	fms_dt result[][MIN_FMS_HEIGHT][MIN_FMS_WIDTH], const int starting_h)
{
#pragma HLS INLINE off

	const int num_of_tiles_w = layer_8_pw_specs.layer_num_of_ofm_tiles_w;
	const int num_tiles_hw = layer_8_pw_specs.layer_num_of_ifm_tiles_h * num_of_tiles_w;
	const int tile_in_h = starting_h / pw_tile_h;
	const int in_tile_h = starting_h % pw_tile_h;

	for (int d = 0; d < chain_0_1_ofms_depth; d++)
	{
		const int tile_in_d = d / pw_tile_d;
		const int in_tile_d = d % pw_tile_d;
		for (int w = 0; w < chain_0_1_ofms_width; w++)
		{
#pragma HLS PIPELINE
			const int tile_in_w = w / pw_tile_w;
			const int tile_index = tile_in_d * num_tiles_hw + tile_in_h * num_of_tiles_w + tile_in_w;
			const int in_tile_w = w % pw_tile_w;

			result[tile_index][in_tile_h][in_tile_w] = chain_seml_communication_buffer[d][w];
		}
	}
}

//##########################################################################################################################

void _0_1_bottlenecks_chain(
	fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_dt result[][MIN_FMS_HEIGHT][MIN_FMS_WIDTH])
{
#pragma HLS INLINE off

	fms_dt bottleneck_0_1_communication_buffer[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width];
	fms_dt bottleneck_0_1_communication_buffer_prev[bottleneck_0_ofms_depth][chain_0_1_bottleneck_0_rows_at_once][bottleneck_0_ofms_width];
	fms_dt chain_seml_communication_buffer[bottleneck_1_ofms_depth][bottleneck_1_ofms_width];

#pragma HLS ARRAY_PARTITION variable = bottleneck_0_1_communication_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = bottleneck_0_1_communication_buffer_prev type = cyclic factor = bottleneck_0_1_communication_buffer_partitioning_factor_in_d dim = 1
#pragma HLS ARRAY_PARTITION variable = bottleneck_0_1_communication_buffer_prev type = complete dim = 2
	//******************************************************************************************************************
	fms_dt chain_input[input_image_depth][chain_0_1_in_buffer_height][input_image_width];
	fms_dt bottleneck_0_input[bottleneck_0_input_buffer_size];
	pss_dt bottleneck_0_projection_kernel_output[bottleneck_0_ofms_depth];
	pss_dt bottleneck_0_projection_kernel_output_prev[bottleneck_0_ofms_depth];

#pragma HLS ARRAY_PARTITION variable = bottleneck_0_projection_kernel_output complete

	fms_dt bottleneck_0_expansion_results_buffer[bottleneck_0_dw_filter_dim - bottleneck_0_dw_strides];
	fms_dt bottleneck_0_dw_lower_buffer[bottleneck_0_expanded_ifms_depth][bottleneck_0_dw_filter_dim];
	fms_dt bottleneck_0_previous_pass_dw_input[bottleneck_0_expanded_ifms_depth][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width];

#pragma HLS ARRAY_PARTITION variable = bottleneck_0_input type = complete dim = 0

#pragma HLS ARRAY_PARTITION variable = bottleneck_0_previous_pass_dw_input type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = bottleneck_0_previous_pass_dw_input type = cyclic factor = 2 dim = 3

#pragma HLS ARRAY_PARTITION variable = bottleneck_0_dw_lower_buffer complete

	const fms_dt bottleneck_0_dw_ifms_zero_point = layer_2_dw_specs.layer_ifms_zero_point;
	const int bottleneck_0_first_fill_offset = chain_0_1_first_filter_dim - chain_0_1_first_strides;
	const int bottleneck_0_extra_cols_filled_first_time =
		chain_0_1_first_filter_dim - chain_0_1_first_strides;
	const fms_dt layer_0_s_ifms_zero_point = first_conv_layer_specs.layer_ifms_zero_point;
	const int num_of_ifm_groups_read_each_time =
		input_image_num_fms_groups_in_width * chain_0_1_rows_filled_each_time;

#pragma HLS ARRAY_PARTITION variable = chain_input complete dim = 1
#pragma HLS ARRAY_PARTITION variable = chain_input complete dim = 2

	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * chain_0_1_rows_filled_each_time];
	//******************************************************************************************************************
	fms_dt bottleneck_1_input[bottleneck_1_input_buffer_size];
	pss_dt bottleneck_1_projection_kernel_output[bottleneck_1_ofms_depth];
	pss_dt bottleneck_1_projection_kernel_output_prev[bottleneck_1_ofms_depth];
#pragma HLS ARRAY_PARTITION variable = bottleneck_1_projection_kernel_output complete

	fms_dt bottleneck_1_dw_lower_buffer[bottleneck_1_expanded_ifms_depth][bottleneck_1_dw_filter_dim * bottleneck_1_dw_strides];
#pragma HLS ARRAY_PARTITION variable = bottleneck_1_dw_lower_buffer complete

	fms_dt bottleneck_1_previous_pass_dw_input_1[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width];
	fms_dt bottleneck_1_previous_pass_dw_input_2[bottleneck_1_expanded_ifms_depth][bottleneck_1_inter_pass_dw_input_width];

	const fms_dt bottleneck_1_dw_ifms_zero_point = layer_2_dw_specs.layer_ifms_zero_point;
		//conv_fms_zero_points[bottleneck_1_dw_layer_index];
	//******************************************************************************************************************

	int filling_h = 0;
	int bottleneck_0_h = 0;
	int bottleneck_1_h = 0;
	int writing_to_result_h = 0;

	//#######################################pipeline filling###########################################
	//-------------------------------------------------------------------------
#if HW == _FPGA
	bottleneck_chain_0_1_fill_ifm_groups_buffer(channels, fms_groups_buffer,
												0,
												input_image_num_fms_groups_in_width); // to do
	bottleneck_chain_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
															 chain_input, 0, false,
															 layer_0_s_ifms_zero_point);
	bottleneck_chain_0_1_fill_ifm_groups_buffer(channels, fms_groups_buffer,
												chain_0_1_extra_rows_filled_first_time,
												num_of_ifm_groups_read_each_time); // to do
	bottleneck_chain_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
															 chain_input, chain_0_1_extra_rows_filled_first_time, false,
															 layer_0_s_ifms_zero_point);
#elif HW == CPU
	chain_0_1_fill_channels_buffer_cpu(channels, chain_input, 0, false, layer_0_s_ifms_zero_point);
	chain_0_1_fill_channels_buffer_cpu(channels,
									   chain_input, chain_0_1_extra_rows_filled_first_time, false,
									   layer_0_s_ifms_zero_point);
#endif
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	bottleneck_0_pipeline_filling_stage(chain_input, bottleneck_0_input,
										bottleneck_0_projection_kernel_output,
										bottleneck_0_projection_kernel_output_prev,
										bottleneck_0_1_communication_buffer, bottleneck_0_dw_lower_buffer,
										bottleneck_0_previous_pass_dw_input,
										bottleneck_0_extra_cols_filled_first_time,
										bottleneck_0_first_fill_offset, layer_0_s_ifms_zero_point,
										bottleneck_0_dw_ifms_zero_point);
	copy_bottleneck_0_1_communication_buffer(
		bottleneck_0_1_communication_buffer,
		bottleneck_0_1_communication_buffer_prev);
	bottleneck_1_pipeline_filling_stage(bottleneck_1_previous_pass_dw_input_1,
										bottleneck_1_previous_pass_dw_input_2,
										bottleneck_1_dw_ifms_zero_point);
	filling_h++;
	bottleneck_0_h++;

	int filling_row = filling_h * chain_0_1_rows_filled_each_time + chain_0_1_extra_rows_filled_first_time;
	//-------------------------------------------------------------------------
#if HW == CPU
	chain_0_1_fill_channels_buffer_cpu(channels,
									   chain_input, filling_row, true,
									   layer_0_s_ifms_zero_point);
#elif HW == _FPGA
	bottleneck_chain_0_1_fill_ifm_groups_buffer(channels, fms_groups_buffer,
												filling_row, num_of_ifm_groups_read_each_time);
	bottleneck_chain_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
															 chain_input, filling_row, true, layer_0_s_ifms_zero_point);
#endif
	//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	filling_h++;
	//#######################################end pipeline filling###########################################

	for (filling_h; filling_h <= chain_0_1_ofms_height; filling_h++)
	{
		bottleneck_1_within_pipeline_stage(bottleneck_1_input,
										   bottleneck_1_projection_kernel_output,
										   bottleneck_1_projection_kernel_output_prev,
										   bottleneck_0_1_communication_buffer_prev,
										   chain_seml_communication_buffer, bottleneck_1_dw_lower_buffer,
										   bottleneck_1_previous_pass_dw_input_1,
										   bottleneck_1_previous_pass_dw_input_2, bottleneck_1_h,
										   bottleneck_1_dw_ifms_zero_point);
		write_chain_seml_communication_buffer(chain_seml_communication_buffer,
											  result, writing_to_result_h);

		bottleneck_0_within_pipeline_stage(chain_input, bottleneck_0_input,
										   bottleneck_0_projection_kernel_output,
										   bottleneck_0_projection_kernel_output_prev,
										   bottleneck_0_1_communication_buffer,
										   bottleneck_0_dw_lower_buffer,
										   bottleneck_0_previous_pass_dw_input, bottleneck_0_h,
										   bottleneck_0_extra_cols_filled_first_time,
										   bottleneck_0_first_fill_offset, layer_0_s_ifms_zero_point,
										   bottleneck_0_dw_ifms_zero_point);
		copy_bottleneck_0_1_communication_buffer(
			bottleneck_0_1_communication_buffer,
			bottleneck_0_1_communication_buffer_prev);

		filling_row = filling_h * chain_0_1_rows_filled_each_time + chain_0_1_extra_rows_filled_first_time;
		//-------------------------------------------------------------------------
#if HW == CPU
		chain_0_1_fill_channels_buffer_cpu(channels,
										   chain_input, filling_row, true,
										   layer_0_s_ifms_zero_point);
#elif HW == _FPGA
		bottleneck_chain_0_1_fill_ifm_groups_buffer(channels, fms_groups_buffer,
													filling_row, num_of_ifm_groups_read_each_time);
		bottleneck_chain_fill_channels_buffer_from_groups_buffer(
			fms_groups_buffer, chain_input, filling_row, true,
			layer_0_s_ifms_zero_point);
#endif
		//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

		bottleneck_0_h++;
		bottleneck_1_h++;
		writing_to_result_h = bottleneck_1_h - 1;
	}
	//#######################################pipeline draining###########################################
	bottleneck_1_within_pipeline_stage(bottleneck_1_input,
									   bottleneck_1_projection_kernel_output,
									   bottleneck_1_projection_kernel_output_prev,
									   bottleneck_0_1_communication_buffer_prev,
									   chain_seml_communication_buffer, bottleneck_1_dw_lower_buffer,
									   bottleneck_1_previous_pass_dw_input_1,
									   bottleneck_1_previous_pass_dw_input_2, bottleneck_1_h,
									   bottleneck_1_dw_ifms_zero_point);
	write_chain_seml_communication_buffer(chain_seml_communication_buffer,
										  result, writing_to_result_h);
	bottleneck_1_h++;
	writing_to_result_h = bottleneck_1_h - 1;
	//#########################
	bottleneck_0_within_pipeline_stage(chain_input, bottleneck_0_input,
									   bottleneck_0_projection_kernel_output,
									   bottleneck_0_projection_kernel_output_prev,
									   bottleneck_0_1_communication_buffer, bottleneck_0_dw_lower_buffer,
									   bottleneck_0_previous_pass_dw_input, bottleneck_0_h,
									   bottleneck_0_extra_cols_filled_first_time,
									   bottleneck_0_first_fill_offset, layer_0_s_ifms_zero_point,
									   bottleneck_0_dw_ifms_zero_point);
	bottleneck_1_within_pipeline_stage(bottleneck_1_input,
									   bottleneck_1_projection_kernel_output,
									   bottleneck_1_projection_kernel_output_prev,
									   bottleneck_0_1_communication_buffer,
									   chain_seml_communication_buffer, bottleneck_1_dw_lower_buffer,
									   bottleneck_1_previous_pass_dw_input_1,
									   bottleneck_1_previous_pass_dw_input_2, bottleneck_1_h,
									   bottleneck_1_dw_ifms_zero_point);
	write_chain_seml_communication_buffer(chain_seml_communication_buffer,
										  result, writing_to_result_h);
	//#######################################end pipeline draining###########################################
}

#endif
