#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"

void padding_top(fms_dt channels_tile[dw_tile_d][dw_tile_h][dw_tile_w], const int padding_top, fms_dt zero_point)
{
    for (int d = 0; d < dw_tile_d; d++)
    {
#pragma HLS UNROLL
        for (int h = 0; h < padding_top; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < dw_tile_w; w++)
            {
#pragma HLS UNROLL
                channels_tile[d][h][w] = zero_point;
            }
        }
    }
}

void padding_bottom(fms_dt channels_tile[dw_tile_d][dw_tile_h][dw_tile_w], const int padding_bottom_starting_row,
                    fms_dt zero_point)
{
    for (int d = 0; d < dw_tile_d; d++)
    {
#pragma HLS UNROLL
        for (int h = padding_bottom_starting_row; h < dw_tile_h; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < dw_tile_w; w++)
            {
#pragma HLS UNROLL
                channels_tile[d][h][w] = zero_point;
            }
        }
    }
}

void padding_left(fms_dt channels_tile[dw_tile_d][dw_tile_h][dw_tile_w], const int padding_left,
                  fms_dt zero_point)
{
    for (int d = 0; d < dw_tile_d; d++)
    {
#pragma HLS UNROLL
        for (int h = 0; h < dw_tile_h; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < padding_left; w++)
            {
#pragma HLS UNROLL
                channels_tile[d][h][w] = zero_point;
            }
        }
    }
}

void padding_right(fms_dt channels_tile[dw_tile_d][dw_tile_h][dw_tile_w], const int padding_right_starting_col,
                   fms_dt zero_point)
{
    for (int d = 0; d < dw_tile_d; d++)
    {
#pragma HLS UNROLL
        for (int h = 0; h < dw_tile_h; h++)
        {
#pragma HLS UNROLL
            for (int w = padding_right_starting_col; w < dw_tile_w; w++)
            {
#pragma HLS UNROLL
                channels_tile[d][h][w] = zero_point;
            }
        }
    }
}

void dw_fill_channels_buffer_from_single_tile(fms_dt channels[max_fms_size],
                                                 fms_dt channels_tile[dw_tile_d][dw_tile_h][dw_tile_w],
                                                 const int absolute_tile_offset,
                                                 const int starting_row_in_src_tile,
                                                 const int starting_row_in_dst_tile,
                                                 const int rows_to_fill)
{
#pragma HLS INLINE

    const int top_offset = starting_row_in_src_tile * dw_tile_w;
    for (int d = 0; d < dw_tile_d; d++)
    {
        for (int h = 0; h < rows_to_fill; h++)
        {
            for (int w = 0; w < dw_tile_w; w++)
            {
                channels_tile[d][h + starting_row_in_dst_tile][w] =
                    channels[absolute_tile_offset + top_offset + w]
            }
        }
    }
}

void dw_fill_

void dw_fill_channels_buffer(fms_dt channels[max_fms_size],
                             fms_dt channels_tile[dw_tile_d][dw_tile_h][dw_tile_w],
                             const int starting_h, const int starting_w,
                             const int layer, const int ifms_w, const int ifms_h,
                             const int starting_tile_index,
                             const int rows_cols_to_fill // must be <= dw_tile_w and dw_tile_h
)
{
#pragma HLS INLINE off

    const int num_of_tiles_to_fill_from_horizontally = 1 + (starting_w % dw_tile_w);
    const int num_of_tiles_to_fill_from_vertically = 1 + (starting_h % dw_tile_h);
}
