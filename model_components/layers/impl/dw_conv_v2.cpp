#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/conv_utils.h"

void dw_conv_engine(
    dw_weights_dt weights[CHANNELS_PIPELINE_DEPTH][max_filter_hw_dim * max_filter_hw_dim],
    fms_dt ifms_buffer[CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
    dw_pss_dt result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
    const int filter_dim, const int strides,
    const int layer_d,
    const int starting_d)
{
#pragma HLS INLINE off

dw_conv_engine:
    for (int c_h = 0; c_h < max_filter_hw_dim; c_h++)
    {
        for (int c_w = 0; c_w < max_filter_hw_dim; c_w++)
        {
            for (int d_in_pipeline = 0; d_in_pipeline < CHANNELS_PIPELINE_DEPTH;
                 d_in_pipeline++)
            {
#pragma HLS PIPELINE
                if (c_h == 0 && c_w == 0)
                {
                    // TDO fill
                }
                if (starting_d + d_in_pipeline >= layer_d)
                {
                    break;
                }
                for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                    {
#pragma HLS UNROLL
                        if (c_w >= filter_dim || c_h >= filter_dim || h >= dw_tile_h / strides || w >= dw_tile_w / strides)
                        {
                            break;
                        }
                        if (c_h == 0 && c_w == 0)
                        {
                            result_tile[d_in_pipeline][h][w] =
                                ifms_buffer[d_in_pipeline][h * strides + c_h][w * strides + c_w] * weights[d_in_pipeline][c_h * filter_dim + c_w];
                        }
                        else
                        {
                            result_tile[d_in_pipeline][h][w] +=
                                ifms_buffer[d_in_pipeline][h * strides + c_h][w * strides + c_w] * weights[d_in_pipeline][c_h * filter_dim + c_w];
                        }
                    }
                }
            }
        }
    }
}

void fill_dw_weights_and_scales_tiles(const dw_weights_dt weights[][3 * 3],
                                      dw_weights_dt weights_tile[][3 * 3],
                                      const fused_scales_dt fused_scales[],
                                      fused_scales_dt fused_scales_tile[],
                                      const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                                      fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
                                      const relu_6_fused_scales_dt relu_6_fused_scales[],
                                      relu_6_fused_scales_dt relu_6_fused_scales_tile[],
                                      const biases_dt fused_zero_points[], biases_dt fused_zero_points_tile[],
                                      int starting_d, const int current_dw_layer_weights_offset,
                                      const int current_layer_fused_parameters_offset)
{
#pragma HLS INLINE

    const int absolute_current_layer_fused_parameters_offset =
        current_layer_fused_parameters_offset + starting_d;
    const int absolute_current_layer_weights_offset =
        current_dw_layer_weights_offset + starting_d;
    for (int d = 0; d < dw_pipeline_depth; d++)
    {
#pragma HLS PIPELINE
        for (int i = 0; i < 3 * 3; i++)
        {
#pragma HLS UNROLL
            weights_tile[d][i] = weights[absolute_current_layer_weights_offset + d][i];
        }
        fused_scales_tile[d] =
            fused_scales[absolute_current_layer_fused_parameters_offset + d];
        fused_scales_log_2_shifts_tile[d] =
            fused_scales_log_2_shifts[absolute_current_layer_fused_parameters_offset + d];
        relu_6_fused_scales_tile[d] =
            relu_6_fused_scales[absolute_current_layer_fused_parameters_offset + d];
        fused_zero_points_tile[d] =
            fused_zero_points[absolute_current_layer_fused_parameters_offset + d];
    }
}

void dw_conv_copy_engine_result_tile(
	dw_pss_dt engine_result_tile[dw_pipeline_depth][switch_point_fms_width],
	dw_pss_dt engine_result_tile_copy[dw_pipeline_depth][switch_point_fms_width], const int ofms_width)
{
#pragma HLS INLINE off

	for (int d = 0; d < dw_pipeline_depth; d++)
	{
#pragma HLS PIPELINE
		for (int w = 0; w < switch_point_fms_width; w++)
		{
#pragma HLS UNROLL
			if (w >= ofms_width)
			{
				break;
			}
			engine_result_tile_copy[d][w] = engine_result_tile[d][w];
		}
	}
}

void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
                 fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                 fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                 const int layer, const int layer_conv_d, const int layer_width, const int layer_height,
                 const int num_of_tiles_d,
                 const int num_of_ifms_tiles_h, const int num_of_ifms_tiles_w,
                 const int num_of_ofms_tiles_h, const int num_of_ofms_tiles_w,
                 const int strides, const int padding_left, const int padding_right, const int padding_top,
                 const fused_scales_dt fused_scales[],
                 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                 const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[],
                 const fused_scales_dt fused_scales_part2[],
                 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
                 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
                 const biases_dt fused_zero_points_part2[])
{
#pragma HLS INLINE off

    const int padding_bottom = padding_right;

    const int num_of_ifms_tiles_hw = num_of_ifms_tiles_h * num_of_ifms_tiles_w;
    const int num_of_ofms_tiles_hw = num_of_ofms_tiles_h * num_of_ofms_tiles_w;

    fms_quantization_scheme normalization = {0, 0, 0, 0};

    const int current_dw_layer_weights_offset = dw_layers_weights_offsets[layer];
    const int current_layer_fused_parameters_offset =
        layers_fused_parameters_offsets[layer];

    dw_weights_dt weights_tile[dw_pipeline_depth][3 * 3];
#pragma HLS ARRAY_PARTITION variable = weights_tile type = complete dim = 1

    fused_scales_dt fused_scales_tile[dw_pipeline_depth];
    fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[dw_pipeline_depth];
    relu_6_fused_scales_dt relu_6_fused_scales_tile[dw_pipeline_depth];
    biases_dt fused_zero_points_tile[dw_pipeline_depth];

    normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
    normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];
    normalization.ofm_scale = conv_fms_scales[layer + 1];

    const fms_dt current_layer_fms_zero_point = conv_fms_zero_points[layer];

    fms_dt ifms_buffer[dw_pipeline_depth][dw_max_v2_buffer_height][dw_max_v2_buffer_width];
    dw_pss_dt engine_result_tile[dw_pipeline_depth][dw_tile_h][dw_tile_w];
    dw_pss_dt engine_result_tile_copy[dw_pipeline_depth][dw_tile_h][dw_tile_w];

    const int skip_padding_left = padding_left == 0 ? padding_right : 0;

#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = engine_result_tile_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile_copy type = complete dim = 3

    for ()
    {
        for ()
        {
            for (int dw_pipeline_in_d = 0;
                 dw_pipeline_in_d < layer_conv_d / dw_pipeline_depth;
                 dw_pipeline_in_d++)
            {
                const int tile_in_d = dw_pipeline_in_d * (dw_pipeline_depth / dw_tile_d);
                if (current_layer_fused_parameters_offset < first_quantization_arrays_num_elements)
                {
                    fill_dw_weights_and_scales_tiles(weights, weights_tile,
                                                     fused_scales, fused_scales_tile, fused_scales_log_2_shifts,
                                                     fused_scales_log_2_shifts_tile, relu_6_fused_scales,
                                                     relu_6_fused_scales_tile, fused_zero_points,
                                                     fused_zero_points_tile, tile_in_d * dw_tile_d,
                                                     current_dw_layer_weights_offset,
                                                     current_layer_fused_parameters_offset);
                }
                else
                {
                    fill_dw_weights_and_scales_tiles(weights, weights_tile,
                                                     fused_scales_part2, fused_scales_tile,
                                                     fused_scales_log_2_shifts_part2,
                                                     fused_scales_log_2_shifts_tile, relu_6_fused_scales_part2,
                                                     relu_6_fused_scales_tile, fused_zero_points_part2,
                                                     fused_zero_points_tile, tile_in_d * dw_tile_d,
                                                     current_dw_layer_weights_offset,
                                                     current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
                }
                normalize_and_write_back_result_tile(result,
                                                     engine_result_tile_copy, normalization, 6,
                                                     fused_scales_tile, fused_scales_log_2_shifts_tile,
                                                     relu_6_fused_scales_tile, fused_zero_points_tile,
                                                     absolute_offset_in_results, num_of_ofms_tiles_w,
                                                     num_of_ofms_tiles_hw, layer_ifm_width / strides);

                dw_conv_engine(weights_tile, ifms_buffer, engine_result_tile,
                               layer_ifm_width / strides, 3, strides, skip_padding_left);

                dw_conv_copy_engine_result_tile(engine_result_tile,
                                                engine_result_tile_copy, layer_ifm_width / strides);

                dw_conv_fill_from_channels(channels, ifms_buffer_lower_part, 3,
                                           layer_ifm_height, layer_ifm_width,
                                           h * strides + rows_filled_first_time,
                                           absolute_offset_in_ifms, absolute_offset_in_ifms_2,
                                           num_of_ifms_tiles_w, num_of_ifms_tiles_hw, strides,
                                           padding_right, current_layer_fms_zero_point);
                dw_conv_copy_to_ifm_buffer(ifms_buffer_lower_part, ifms_buffer,
                                           strides, 3, layer_ifm_width, padding_right);
            }
        }
    }
}