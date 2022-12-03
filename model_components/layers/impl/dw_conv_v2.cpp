#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"

void fill_ifms_tile_depth_row_segment(fms_dt channels[max_fms_size], fms_dt row[dw_tile_d][], const int tile_index_in_row,
                                      int absolute_offset, const int padding_left,
                                      const bool is_padding_row, const int ifms_width, const fms_dt zero_point)
{
#pragma HLS INLINE

    const int starting_fill_w_offset = tile_index_in_row * dw_tile_size;
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

void fill_ifms_tile_depth_col_segment(fms_dt channels[max_fms_size], const int tile_index_in_col, fms_dt col_segment[dw_tile_d][dw_tile_h],
                                      int absolute_offset, const bool is_padding_col,
                                      const int ifms_height, const fms_dt zero_point)
{
    const int starting_fill_h_offset = tile_index_in_col * dw_tile_size;
    for (int h = 0; h < dw_tile_h; h++)
    {
        for (int d = 0; d < dw_tile_d, d++)
        {
            if (!is_padding_col && h + starting_fill_h_offset < ifms_height)
            {
                col_segment[d][h] = channels[absolute_offset + d * dw_tile_hw + h * dw_tile_w];
            } else{
                dw_ifms_buffer[d][h] = zero_point;
            }
        }
    }
}

void shift_segment_between_rows(fms_dt upper_row[dw_tile_d][], fms_dt lower_row[dw_tile_d][], const int tile_index, const int padding_left)
{
#pragma HLS INLINE

    const int starting_shift_index = tile_index * dw_tile_size + padding_left;
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
                              const int num_of_tiles_hw, const int fms_height,
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
                    fms_dt results_buffer[dw_tile_d][dw_tile_h], const int filter_dim, const int in_tile_d, const int conv_d, const int strides)
{
#pragma HLS INLINE
    for (int h = 0; h < dw_tile_h / strides; h++)
    {
#pragma HLS UNROLL
        results_buffer[in_tile_d][h] = 0;
        for (int c_h = 0; c_h < filter_dim; c_h++)
        {
#pragma HLS UNROLL
            for (int c_w = 0; c_w < filter_dim; c_w++)
            {
#pragma HLS UNROLL
                results_buffer[in_tile_d][h] += weights[conv_d][c_h * filter_dim + c_w] * channels_buffer[in_tile_d][h * strides + c_h][c_w];
            }
        }
        results_buffer[in_tile_d][h] = //HERE
    }
}

void dw_conv()
{
    fms_dt dw_ifms_buffer[dw_tile_d][112][112]; // partition dw_tile_d,_, complete
}