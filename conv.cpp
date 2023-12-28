#include "../headers/layers_imp_common_includes.h"
#include "../headers/conv.h"
#include "../headers/pw_conv.h"

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

	shift_hannels_buffer_rows(channels_tile, rows_to_shift);
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
			chain_0_1_padd_bottom_channels_buffer_rows(channels_tile, conv_fms_zero_points[0]);
		}
	}
}

// Note that this implementation of layer_0 is not not very optimized
void layer_0_s_conv_engine(
	const layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim],
	fms_dt channels_tile[input_image_depth][first_conv_layer_filter_dim][input_image_width],
	fms_dt results[max_fms_size], int starting_h, fused_scales_dt fused_scales[],
	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[], relu_6_fused_scales_dt relu_6_fused_scales[], biases_dt fused_zero_points[])
{

	const biases_dt current_layer_zero_point = conv_fms_zero_points[0];
	for (int f = 0; f < first_conv_layer_num_fils; f++)
	{
		fms_quantization_scheme normalization = {0, 0, 0, 0};
		normalization.ofm_zero_point = conv_fms_zero_points[2];
		normalization.ofm_scale_rec = conv_fms_scales_rec[2];
		normalization.ofm_scale = conv_fms_scales[2];
		normalization.fused_zero_point = fused_zero_points[f];
		normalization.fused_scales = fused_scales[f];
		normalization.fused_scales_log_2_shift = fused_scales_log_2_shifts[f];
		normalization.layer_0_relu_6_fused_scale = relu_6_fused_scales[f];
		for (int w = 0; w < first_conv_layer_specs.layer_ofm_width w++)
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

			const int in_tile_d = f % pw_tile_d;
			const int in_tile_h = starting_h % pw_tile_h;
			const int in_tile_w = w % pw_tile_w;
			const int in_tile_index = in_tile_d * pw_tile_hw + in_tile_h * pw_tile_w + in_tile_w;
			
			results[tile_index * pw_tile_size + in_tile_index] = conv_relu_norm(
				tmp, normalization, 6);
		}
	}
}

// void layer_0_s_3x3(
// 	const layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim],
// 	fms_grp_dt channels[input_image_depth * input_image_height * input_image_width / input_image_group_items],
// 	fms_dt result[max_fms_size], fused_scales_dt fused_scales[],
// 	fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[], relu_6_fused_scales_dt relu_6_fused_scales[], biases_dt fused_zero_points[])
// {
// 	fms_dt channels_tile[first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_ifm_width];
// 	for (int h = 0; h < first_conv_layer_specs.layer_ofm_height h++)
// 	{
// 		fill_channels_buffer_cpu(channels, channels_tile, h);
// 		layer_0_s_conv_engine(weights_1, channels_tile, result, h, fused_scales, fused_scales_log_2_shifts, relu_6_fused_scales, fused_zero_points);
// 	}
// }
