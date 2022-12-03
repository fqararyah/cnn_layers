#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"


void fill_ifms_tile_depth_col_segment(fms_dt channels[max_fms_size], const int tile_index_in_h, fms_dt col_segment[dw_tile_d][dw_tile_h],
                                      int absolute_offset, const bool is_padding_col,
                                      const int ifms_height, const fms_dt zero_point)
{
#pragma HLS INLINE

    const int starting_fill_h_offset = tile_index_in_h * dw_tile_size;
    for (int h = 0; h < dw_tile_h; h++)
    {
#pragma HLS UNROLL
        for (int d = 0; d < dw_tile_d, d++)
        {
#pragma HLS UNROLL
            if (!is_padding_col && h + starting_fill_h_offset < ifms_height)
            {
                col_segment[d][h] = channels[absolute_offset + d * dw_tile_hw + h * dw_tile_w];
            }
            else
            {
                dw_ifms_buffer[d][h] = zero_point;
            }
        }
    }
}

void shift_col_segments(fms_dt left_col_segment[dw_tile_d][dw_tile_h], fms_dt right_col_segment[dw_tile_d][dw_tile_h])
{
#pragma HLS INLINE

    for (int h = 0; h < dw_tile_h; h++)
    {
#pragma HLS UNROLL
        for (int d = 0; d < dw_tile_d, d++)
        {
#pragma HLS UNROLL
            left_col_segment[d][h] = right_col_segment[d][h];
        }
    }
}

void fill_ifms_tile_depth_row_segment(fms_dt channels[max_fms_size], fms_dt row[dw_tile_d][], const int tile_index_in_w,
                                      int absolute_offset, const int padding_left,
                                      const bool is_padding_row, const int ifms_width, const fms_dt zero_point)
{
#pragma HLS INLINE

    const int starting_fill_w_offset = tile_index_in_w * dw_tile_size;
    for (int w = 0; w < dw_tile_w; w++)
    {
#pragma HLS UNROLL
        for (int d = 0; d < dw_tile_d; d++)
        {
#pragma HLS UNROLL
            if (!is_padding_row && w + starting_fill_w_offset < ifms_width)
            {
                row[d][w + starting_fill_w_offset + padding_left] = channels[absolute_offset + d * dw_tile_hw + w];
            }
            else
            {
                dw_ifms_buffer[d][w + starting_fill_w_offset + padding_left] = zero_point;
            }
        }
    }
}

void shift_segment_between_rows(fms_dt upper_row[dw_tile_d][], fms_dt lower_row[dw_tile_d][], const int tile_index_in_w, const int padding_left)
{
#pragma HLS INLINE

    const int starting_shift_index = tile_index_in_w * dw_tile_size + padding_left;
    for (int w = 0; w < dw_tile_w; w++)
    {
#pragma HLS UNROLL
        for (int d = 0; d < dw_tile_d; d++)
        {
#pragma HLS UNROLL
            upper_row[d][w + starting_shift_index] = lower_row[d][w + starting_shift_index];
        }
    }
}

void fill_ifms_tile_depth_row(fms_dt channels[max_fms_size], fms_dt ifms_row[dw_tile_d][], const int ifms_width, const int num_of_tiles_w,
                              const int num_of_tiles_hw, const int ifms_height,
                              int starting_tile_in_d, int row_index,
                              const int padding_left, const int padding_right,
                              const bool is_padding_row,
                              const fms_dt zero_point)
{
    const int h_offset = num_of_tiles_w * (row_index / dw_tile_h) * dw_tile_size + (row_index % dw_tile_h) * dw_tile_w;
    int absolute_offset = starting_tile_in_d * num_of_tiles_hw * dw_tile_size + h_offset;

    for (int tile_in_w = 0; tile_in_w < num_of_tiles_w; tile_in_w++)
    {
        absolute_offset += dw_tile_size;

        fill_ifms_tile_depth_segment(channels, ifms_row, tile_in_w, absolute_offset, padding_left, is_padding_row,
                                     ifms_width, zero_point)
    }
    // padding left
    for (int w = 0; w < padding_left; w++)
    {
#pragma HLS UNROLL
        for (int d = 0; d < dw_tile_d; d++)
        {
#pragma HLS UNROLL
            dw_ifms_buffer[d][w] = zero_point;
        }
    }
    // padding right
    for (int w = 0; w < padding_left; w++)
    {
#pragma HLS UNROLL
        for (int d = 0; d < dw_tile_d; d++)
        {
#pragma HLS UNROLL
            dw_ifms_buffer[d][w + ifms_width] = zero_point;
        }
    }
}

void dw_conv_engine(const dw_weights_dt weights[max_conv_d][], fms_dt channels_buffer[dw_tile_d][dw_max_v2_buffer_height][],
                    fms_dt results_buffer[dw_tile_d][dw_tile_h], const int filter_dim, const int in_tile_d, const int conv_d, const int strides,
                    fms_quantization_scheme normalization, const int layer_relu)
{
#pragma HLS INLINE
    for (int h = 0; h < dw_tile_h / strides; h++)
    {
#pragma HLS UNROLL
        dw_pss_dt tmp_pss;
        for (int c_h = 0; c_h < filter_dim; c_h++)
        {
#pragma HLS UNROLL
            for (int c_w = 0; c_w < filter_dim; c_w++)
            {
#pragma HLS UNROLL
                tmp_pss += weights[conv_d][c_h * filter_dim + c_w] * channels_buffer[in_tile_d][h * strides + c_h][c_w];
            }
        }
        results_buffer[in_tile_d][h] = dw_relu_norm(tmp_pss, normalization, layer_relu);
    }
}

void dw_conv_3x3(dw_weights_dt weights[max_conv_d][max_conv_h][max_conv_w],
                 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
                 const int layer, const int layer_conv_d, const int layer_ifm_width,
                 const int layer_ifm_height, const int num_of_tiles_d,
                 const int num_of_ofms_tiles_h, const int num_of_ofms_tiles_w, const int strides,
                 const int padding_left, const int padding_top, const int direction,
                 fused_scales_dt fused_scales[], relu_6_fused_scales_dt relu_6_fused_scales[],
                 biases_dt fused_zero_points[])
{
#pragma HLS INLINE off

    fms_dt upper_row[dw_tile_d][layer_ifm_width];
    fms_dt lower_row[dw_tile_d][layer_ifm_width];

    fms_dt col_segment_left[dw_tile_d][dw_tile_h];
    fms_dt col_segment_right[dw_tile_d][dw_tile_h];

    const int num_of_ifms_tiles_h = num_of_ofms_tiles_h * strides;
    const int num_of_ifms_tiles_w = num_of_ofms_tiles_w * strides;

    const int num_of_ifm_tiles_hw = num_of_ifms_tiles_h * num_of_ifms_tiles_w;

    fms_quantization_scheme normalization = {0, 0, 0, 0};
    const int current_layer_fused_parameters_offsets =
        layers_fused_parameters_offsets[layer];

    const int current_layer_fms_zero_point = conv_fms_zero_points[layer];

    for (int tile_in_d = 0; tile_in_d < num_of_tiles_d; tile_in_d++)
    {
        fill_ifms_tile_depth_row(channels, upper_row, layer_ifm_width, ,
                                 num_of_ifm_tiles_hw, layer_ifm_height,
                                 tile_in_d, 0,
                                 padding_left, 1,
                                 true,
                                 current_layer_fms_zero_point);
        for (int tile_in_h = 0; tile_in_h < num_of_ifms_tiles_h; tile_in_h++)
        {
            fill_ifms_tile_depth_col_segment(channels, 0, col_segment_left,
                                             absolute_offset, true,
                                             layer_ifm_height, current_layer_fms_zero_point);
            for (int tile_in_w = 0; tile_in_w < num_of_ifms_tiles_w; tile_in_w++)
            {
                fill_ifms_tile_depth_col_segment(channels, tile_in_w, col_segment_right,
                                                 absolute_offset,tile_in_w == num_of_ifms_tiles_w,
                                                 layer_ifm_height, current_layer_fms_zero_point);
                dw_conv_engine(weights, fms_dt channels_buffer[dw_tile_d][dw_max_v2_buffer_height][],
                               fms_dt results_buffer[dw_tile_d][dw_tile_h], const int filter_dim, const int in_tile_d, const int conv_d, const int strides,
                               fms_quantization_scheme normalization, const int layer_relu);
                shift_col_segments(col_segment_left, col_segment_right);
            }
        }
    }
}