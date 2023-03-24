#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/pw_conv.h"

void fill_first_rows(fms_dt channels[max_fms_size],
                     fms_dt ifms_buffer[dw_pipeline_depth][max_filter_hw_dim][switch_point_fms_width + max_padding_lr], const int filter_dim,
                     const int num_of_tiles_w, const int num_of_tiles_hw,
                     const int padding_left_right, const int strides, const int padding_top,
                     const int absolute_offset_in_ifms, const int ifms_width,
                     fms_dt fms_zero_point)
{
    for (int h = 0; h < max_padding; h++)
    {
        if (h >= padding_top)
        {
            break;
        }
        for (int d = 0; d < dw_pipeline_depth; d++)
        {
            for (int w = 0; w < switch_point_fms_width + max_padding_lr; w++)
            {
                if (w >= ifms_width + 2 * padding_left_right)
                {
                    break;
                }
                ifms_buffer[d][h + strides][w] = fms_zero_point;
            }
        }
    }

    for (int h = 0; h < filter_dim - min_strides; h++)
    {
        if (h >= filter_dim - strides - padding_top)
        {
            break;
        }
        for (int o_d = 0; o_d < dw_pipeline_depth / dw_tile_d; o_d++)
        {
            for (int o_w = 0; o_w < num_of_tiles_w; o_w++)
            {
                const int current_absolute_offset = absolute_offset_in_ifms + o_d * num_of_tiles_hw * dw_tile_size + o_w * dw_tile_size;
                for (int d = 0; d < dw_tile_d; d++)
                {
                    for (int w = 0; w < dw_tile_w; w++)
                    {
                        if (o_w * dw_tile_w + w >= ifms_width)
                        {
                            break;
                        }
                        ifms_buffer[o_d * dw_tile_d + d][h + padding_top + strides][o_w * dw_tile_w + w + padding_left_right] =
                            channels[current_absolute_offset + d * dw_tile_hw + h * dw_tile_w + w];
                    }
                }
            }
        }
    }
}

void padd_left_right(
    fms_dt ifms_buffer[dw_pipeline_depth][max_filter_hw_dim][switch_point_fms_width + max_padding_lr], const int padding_left_right,
    const int ifms_width, fms_dt fms_zero_point)
{
    for (int d = 0; d < dw_pipeline_depth; d++)
    {
#pragma HLS PIPELINE
        for (int h = 0; h < max_filter_hw_dim; h++)
        {
            for (int w = 0; w < padding_left_right; w++)
            {
                ifms_buffer[d][h][w] = fms_zero_point;
                ifms_buffer[d][h][ifms_width + 2 * padding_left_right - w - 1] =
                    fms_zero_point;
            }
        }
    }
}

void dw_conv_fill_from_channels(fms_dt channels[max_fms_size],
                                fms_dt ifms_buffer_lower_part[dw_pipeline_depth][max_strides][switch_point_fms_width + max_padding_lr], const int ifms_buffer_height,
                                const int ifms_height, const int ifms_width,
                                const int ifms_buffer_offset_h,
                                const int global_absolute_tile_offset_in_ifms,
                                const int global_absolute_tile_offset_in_ifms_2,
                                const int num_of_tiles_w, const int num_of_tiles_hw, const int strides,
                                const int padding_left_or_right, fms_dt fms_zero_point)
{
#pragma HLS INLINE off

dw_fill_channels_tile:
    for (int o_d = 0;
         o_d < dw_pipeline_depth / dw_tile_d; o_d++)
    {
        const int current_absolute_offset_after_o_d =
            global_absolute_tile_offset_in_ifms + o_d * num_of_tiles_hw * dw_tile_size;
        const int current_absolute_offset_after_o_d_2 =
            global_absolute_tile_offset_in_ifms_2 + o_d * num_of_tiles_hw * dw_tile_size;

        for (int o_w = 0; o_w < num_of_tiles_w; o_w++)
        {
            const int current_absolute_offset_after_o_w =
                current_absolute_offset_after_o_d + o_w * dw_tile_size;
            const int current_absolute_offset_after_o_w_2 =
                current_absolute_offset_after_o_d_2 + o_w * dw_tile_size;
            const int current_buffer_offset_after_o_w = o_w * dw_tile_w + padding_left_or_right;

            for (int d = 0; d < dw_tile_d; d++)
            {
#pragma HLS PIPELINE
                for (int w = 0; w < dw_tile_w; w++)
                {
#pragma HLS UNROLL
                    if (ifms_buffer_offset_h < ifms_height && o_w * dw_tile_w + w < ifms_width)
                    {
                        ifms_buffer_lower_part[o_d * dw_tile_d + d][0][current_buffer_offset_after_o_w + w] =
                            channels[current_absolute_offset_after_o_w + d * dw_tile_hw + w];
                    }
                    else
                    {
                        ifms_buffer_lower_part[o_d * dw_tile_d + d][0][current_buffer_offset_after_o_w + w] = fms_zero_point;
                    }
                    if (strides == 2)
                    {
                        if (ifms_buffer_offset_h + 1 < ifms_height && o_w * dw_tile_w + w < ifms_width)
                        {
                            ifms_buffer_lower_part[o_d * dw_tile_d + d][1][current_buffer_offset_after_o_w + w] =
                                channels[current_absolute_offset_after_o_w_2 + d * dw_tile_hw + w];
                        }
                        else
                        {
                            ifms_buffer_lower_part[o_d * dw_tile_d + d][1][current_buffer_offset_after_o_w + w] = fms_zero_point;
                        }
                    }
                }
            }
        }
    }
}

void dw_conv_copy_to_ifm_buffer(
    fms_dt ifms_buffer_lower_part[max_strides][max_strides][switch_point_fms_width + max_padding_lr],
    fms_dt ifms_buffer[dw_pipeline_depth][max_filter_hw_dim][switch_point_fms_width + max_padding_lr], const int strides, const int filter_dim,
    const int ifms_width, const int padding_left_or_right)
{
#pragma HLS INLINE off

dw_shift_d:
    for (int d = 0; d < dw_pipeline_depth; d++)
    {
#pragma HLS PIPELINE
    dw_shift_w:
        for (int w = 0; w < switch_point_fms_width + max_padding_lr;
             w++)
        {
#pragma HLS UNROLL
            if (w >= ifms_width)
            {
                break;
            }
            if (strides == 1)
            {
                ifms_buffer[d][0][w + padding_left_or_right] =
                    ifms_buffer[d][1][w + padding_left_or_right];
                ifms_buffer[d][1][w + padding_left_or_right] =
                    ifms_buffer[d][2][w + padding_left_or_right];
            }
            else if (strides == 2)
            {
                ifms_buffer[d][0][w + padding_left_or_right] =
                    ifms_buffer[d][2][w + padding_left_or_right];
            }
        }
    }
    for (int d = 0; d < dw_pipeline_depth; d++)
    {
#pragma HLS PIPELINE
        for (int w = 0; w < switch_point_fms_width; w++)
        {
#pragma HLS UNROLL
            if (w >= ifms_width)
            {
                break;
            }
            ifms_buffer[d][filter_dim - strides][w + padding_left_or_right] =
                ifms_buffer_lower_part[d][0][w + padding_left_or_right];
            if (strides == 2)
            {
                ifms_buffer[d][filter_dim - strides + 1][w + padding_left_or_right] =
                    ifms_buffer_lower_part[d][1][w + padding_left_or_right];
            }
        }
    }
}

void dw_conv_engine(
    dw_weights_dt weights[][max_filter_hw_dim * max_filter_hw_dim],
    fms_dt ifms_buffer[dw_pipeline_depth][max_filter_hw_dim][switch_point_fms_width + max_padding_lr],
    dw_pss_dt result_tile[dw_pipeline_depth][switch_point_fms_width],
    const int ofms_w, const int filter_dim, const int strides,
    const int skip_padding_left)
{
#pragma HLS INLINE off
dw_conv_engine:
    for (int c_h = 0; c_h < max_filter_hw_dim; c_h++)
    {
        for (int d = 0; d < dw_pipeline_depth; d++)
        {
#pragma HLS PIPELINE
            for (int w = 0; w < switch_point_fms_width; w++)
            {
#pragma HLS UNROLL
                if (c_h >= filter_dim || w >= ofms_w)
                {
                    break;
                }
                dw_pss_dt tmp = 0;
                for (int c_w = 0; c_w < max_filter_hw_dim; c_w++)
                {
#pragma HLS UNROLL
                    if (c_w >= filter_dim)
                    {
                        break;
                    }
                    tmp += ifms_buffer[d][c_h][w * strides + c_w + skip_padding_left] * weights[d][c_h * filter_dim + c_w];
                }
                if (c_h == 0)
                {
                    result_tile[d][w] = tmp;
                }
                else
                {
                    result_tile[d][w] += tmp;
                }
            }
        }
    }
}

void normalize_and_write_back_result_tile(fms_dt result[max_fms_size],
                                          dw_pss_dt result_tile[dw_pipeline_depth][switch_point_fms_width],
                                          fms_quantization_scheme normalization, const int layer_relu,
                                          const fused_scales_dt fused_scales_tile[],
                                          const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
                                          const relu_6_fused_scales_dt relu_6_fused_scales_tile[],
                                          const biases_dt fused_zero_points_tile[],
                                          const int absolute_offset_in_results, const int num_of_tiles_w,
                                          const int num_of_tiles_hw, const int ofms_width)
{
#pragma HLS INLINE off

    for (int d_in_pipeline = 0; d_in_pipeline < dw_pipeline_depth / dw_tile_d;
         d_in_pipeline++)
    {
        const int current_absolute_offset_after_o_d = absolute_offset_in_results + d_in_pipeline * num_of_tiles_hw * dw_tile_size;
        for (int o_w = 0; o_w < num_of_tiles_w; o_w++)
        {
            const int current_absolute_offset_after_o_w =
                current_absolute_offset_after_o_d + o_w * dw_tile_size;

            for (int d = 0; d < dw_tile_d; d++)
            {
#pragma HLS PIPELINE
                for (int w = 0; w < dw_tile_w; w++)
                {
#pragma HLS UNROLL
                    if (o_w * dw_tile_w + w >= ofms_width)
                    {
                        break;
                    }
                    normalization.fused_scales = fused_scales_tile[d_in_pipeline * dw_tile_d + d];
                    normalization.fused_scales_log_2_shift =
                        fused_scales_log_2_shifts_tile[d_in_pipeline * dw_tile_d + d];
                    normalization.relu_6_fused_scale =
                        relu_6_fused_scales_tile[d_in_pipeline * dw_tile_d + d];
                    normalization.fused_zero_point =
                        fused_zero_points_tile[d_in_pipeline * dw_tile_d + d];

                    result[current_absolute_offset_after_o_w + d * dw_tile_hw + w] = dw_relu_norm(
                        result_tile[d_in_pipeline * dw_tile_d + d][o_w * dw_tile_w + w], normalization,
                        layer_relu);
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
                 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
                 const int layer, const int layer_conv_d, const int layer_ifm_width,
                 const int layer_ifm_height, const int num_of_tiles_d,
                 const int num_of_ifms_tiles_h, const int num_of_ifms_tiles_w,
                 const int num_of_ofms_tiles_h, const int num_of_ofms_tiles_w,
                 const int strides, const int padding_left, const int padding_right,
                 const int padding_top, const int direction,
                 const fused_scales_dt fused_scales[],
                 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                 const relu_6_fused_scales_dt relu_6_fused_scales[],
                 const biases_dt fused_zero_points[],
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

    fms_dt ifms_buffer[dw_pipeline_depth][max_filter_hw_dim][switch_point_fms_width + max_padding_lr];
    fms_dt ifms_buffer_lower_part[dw_pipeline_depth][max_strides][switch_point_fms_width + max_padding_lr];
    dw_pss_dt engine_result_tile[dw_pipeline_depth][switch_point_fms_width];
    dw_pss_dt engine_result_tile_copy[dw_pipeline_depth][switch_point_fms_width];
    const int skip_padding_left = padding_left == 0 ? padding_right : 0;

#pragma HLS ARRAY_PARTITION variable = ifms_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = ifms_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = ifms_buffer_lower_part type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile_copy type = complete dim = 2

    padd_left_right(ifms_buffer, padding_right, layer_ifm_width,
                    current_layer_fms_zero_point);

    const int rows_filled_first_time = 3 - strides - padding_top;
    const int dw_num_of_runs_in_pipe = 2;
    for (int dw_pipeline_in_d = 0;
         dw_pipeline_in_d < layer_conv_d / dw_pipeline_depth;
         dw_pipeline_in_d++)
    {
        const int tile_in_d = dw_pipeline_in_d * (dw_pipeline_depth / dw_tile_d);
        fill_first_rows(channels, ifms_buffer, 3, num_of_ifms_tiles_w,
                        num_of_ifms_tiles_hw, padding_right, strides, padding_top,
                        tile_in_d * num_of_ifms_tiles_hw * dw_tile_size,
                        layer_ifm_width, current_layer_fms_zero_point);
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

        for (int h = 0; h < dw_num_of_runs_in_pipe + layer_ifm_height / strides;
             h++)
        {
            int read_results_h =
                h < layer_ifm_height / strides ? h : (layer_ifm_height / strides) - 1;
            const int absolute_offset_in_ifms = (tile_in_d * num_of_ifms_tiles_hw + ((read_results_h * strides + rows_filled_first_time) / dw_tile_h) * num_of_ifms_tiles_w) * dw_tile_size + ((read_results_h * strides + rows_filled_first_time) % dw_tile_h) * dw_tile_w;

            const int absolute_offset_in_ifms_2 = (tile_in_d * num_of_ifms_tiles_hw + ((read_results_h * strides + rows_filled_first_time + 1) / dw_tile_h) * num_of_ifms_tiles_w) * dw_tile_size + ((read_results_h * strides + rows_filled_first_time + 1) % dw_tile_h) * dw_tile_w;

            int write_to_results_h =
                (h - dw_num_of_runs_in_pipe) < 0 ? 0 : (h - dw_num_of_runs_in_pipe);
            const int absolute_offset_in_results = (tile_in_d * num_of_ofms_tiles_hw + (write_to_results_h / dw_tile_h) * num_of_ofms_tiles_w) * dw_tile_size + (write_to_results_h % dw_tile_h) * dw_tile_w;

            //*************************

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
            //*************************
        }
    }
}
