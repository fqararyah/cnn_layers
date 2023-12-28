#include "../headers/layers_imp_common_includes.h"
#include "../headers/conv.h"
#include "../headers/pw_conv.h"

#if FIBHA_VERSION == 2

void shift_hannels_buffer_rows(
	fms_dt channels_tile[input_image_depth][first_conv_layer_filter_dim][input_image_width],
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
				channels_tile[d][h][w] = channels_tile[d][h + (first_conv_layer_filter_dim - rows_to_shift)][w];
			}
		}
	}
}

void chain_0_1_padd_bottom_channels_buffer_rows(
	fms_dt channels_buffer_0[input_image_depth][first_conv_layer_filter_dim][input_image_width],
	const fms_dt zero_point)
{
#pragma HLS INLINE

	for (int w = 0; w < input_image_width; w++)
	{
#pragma HLS PIPELINE
		for (int d = 0; d < input_image_depth; d++)
		{
#pragma HLS UNROLL
			for (int h = first_conv_layer_filter_dim - first_conv_layer_specs.padding_bottom;
				 h < first_conv_layer_filter_dim; h++)
			{
#pragma HLS UNROLL
				channels_buffer_0[d][h][w] = zero_point;
			}
		}
	}
}

void fill_channels_buffer_cpu(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_dt channels_tile[input_image_depth][first_conv_layer_filter_dim][input_image_width],
	int starting_h)
{
#pragma HLS INLINE off

	const int rows_to_shift = first_conv_layer_filter_dim - first_conv_layer_specs.strides;

	if (starting_h >= first_conv_layer_filter_dim - first_conv_layer_specs.padding_top)
	{
		shift_hannels_buffer_rows(channels_tile, rows_to_shift);
	}
	const int channels_buffer_start_filling_h =
		starting_h == 0 ? first_conv_layer_specs.padding_top : rows_to_shift;
	for (int h = 0; h < first_conv_layer_specs.strides; h++)
	{
		if (starting_h + h < input_image_height)
		{
			for (int d = 0; d < input_image_depth; d++)
			{
				for (int w = 0; w < input_image_width; w++)
				{
					channels_tile[d][h + channels_buffer_start_filling_h][w] =
						input_image[d * input_image_hw + (h + starting_h) * input_image_width + w];
				}
			}
		}
		else
		{
			chain_0_1_padd_bottom_channels_buffer_rows(channels_tile, first_conv_layer_specs.layer_ifms_zero_point);
		}
	}
}

// Note that this implementation of layer_0 is not not very optimized
void layer_0_s_conv_engine(
	const layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim],
	fms_dt channels_tile[input_image_depth][first_conv_layer_filter_dim][input_image_width],
	fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH], int starting_h,
	const fused_scales_dt fused_scales[],
	const relu_6_fused_scales_dt relu_6_fused_scales[],
	const biases_dt fused_zero_points[])
{

	const biases_dt current_layer_zero_point = first_conv_layer_specs.layer_ifms_zero_point;
	for (int f = 0; f < first_conv_layer_num_fils; f++)
	{
		fms_quantization_scheme normalization = {0, 0, 0, 0};
		normalization.ofm_zero_point = first_conv_layer_specs.layer_ofms_zero_point;
		normalization.ofm_scale = first_conv_layer_specs.layer_ofms_scale;
		normalization.fused_zero_point = fused_zero_points[f];
		normalization.fused_scales = fused_scales[f];
		normalization.fused_scales_log_2_shift = fused_scales_log_2_shifts[f];
		normalization.layer_0_relu_6_fused_scale = relu_6_fused_scales[f];
		for (int w = 0; w < first_conv_layer_specs.layer_ofm_width; w++)
		{
#pragma HLS PIPELINE
			pss_dt tmp = 0;
			for (int d = 0; d < first_conv_layer_depth; d++)
			{
#pragma HLS UNROLL
				for (int c_h = 0; c_h < first_conv_layer_filter_dim; c_h++)
				{
#pragma HLS UNROLL
					for (int c_w = 0; c_w < first_conv_layer_filter_dim; c_w++)
					{
#pragma HLS UNROLL
						if (w * first_conv_layer_specs.strides + c_w < first_conv_layer_ifm_width)
						{
							tmp += weights_1[f][d][c_h][c_w] * channels_tile[d][c_h][w * first_conv_layer_specs.strides + c_w];
							if (starting_h == 0 && w == 0 && d == 0 && f == 0)
							{
								printf("%d * %d \n", (int)weights_1[f][d][c_h][c_w],
									   (int)channels_tile[d][c_h][w * first_conv_layer_specs.strides + c_w]);
							}
						}
						else
						{
							tmp += weights_1[f][d][c_h][c_w] * current_layer_zero_point;
						}
					}
				}
			}
			const int tile_in_d = f / pw_tile_d;
			const int tile_in_h = starting_h / pw_tile_h;
			const int tile_in_w = w / pw_tile_w;
			const int tile_index = tile_in_d * (first_conv_layer_specs.layer_num_fils * first_conv_layer_specs.layer_num_of_ofm_tiles_w) + tile_in_h * first_conv_layer_specs.layer_num_of_ofm_tiles_w + tile_in_w;

			// const int in_tile_d = f % pw_tile_d;
			const int in_tile_h = starting_h % pw_tile_h;
			const int in_tile_w = w % pw_tile_w;
			// const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;

			result[tile_index][in_tile_h][in_tile_w] = conv_relu_norm(
				tmp, normalization, 6);
		}
	}
}

void layer_0_s_3x3(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH])
{
	const int rows_filled_first_time = first_conv_layer_filter_dim - first_conv_layer_strides;
	fms_dt channels_tile[first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_ifm_width];
	fms_grp_dt fms_groups_buffer[input_image_depth][input_image_num_fms_groups_in_width * first_conv_layer_strides];

#if HW == CPU
	fill_channels_buffer_cpu(input_image, channels_tile, 0);
	fill_channels_buffer_cpu(input_image, channels_tile, rows_filled_first_time);
#elif HW == _FPGA
	fill_input_image_groups_buffer(input_image, fms_groups_buffer, 0,
								   input_image_num_fms_groups_in_width); // to do
	input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
														channels_tile, 0, 0,
														first_conv_layer_specs.layer_ifms_zero_point);

	fill_input_image_groups_buffer(input_image, fms_groups_buffer,
								   rows_filled_first_time,
								   num_of_ifm_groups_read_each_time); // to do
	input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
														channels_tile,
														rows_filled_first_time,
														rows_filled_first_time,
														first_conv_layer_specs.layer_ifms_zero_point);
#endif
	for (int h = 0; h < first_conv_layer_specs.layer_ofm_height; h++)
	{
		const int start_reading_h = h * first_conv_layer_strides + rows_filled_first_time;
		layer_0_s_conv_engine(first_layer_weights, channels_tile, result, h,
							  first_conv_layer_fused_scales, first_conv_layer_relu_6_fused_scales,
							  first_conv_layer_fused_zero_points);
#if HW == CPU
		fill_channels_buffer_cpu(input_image, channels_tile, start_reading_h);
#elif HW == _FPGA
		fill_input_image_groups_buffer(input_image, fms_groups_buffer,
									   start_reading_h, num_of_ifm_groups_read_each_time); // to do
		input_image_fill_channels_buffer_from_groups_buffer(fms_groups_buffer,
															channels_tile, start_reading_h,
															start_reading_h % first_conv_layer_filter_dim,
															first_conv_layer_specs.layer_ifms_zero_point);
#endif
	}
}

#endif