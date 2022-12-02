#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"

void fill_ifms_buffer(fms_dt channels[max_fms_size], fms_dt dw_ifms_buffer[dw_tile_d][112][112], const int num_of_tiles_h, const int num_of_tiles_w,
                      int starting_tile_in_d)
{
    const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;
    int absolute_offset = starting_tile_in_d * num_of_tiles_hw * dw_tile_size;
    const int tile_h_step_offset = num_of_tiles_w * dw_tile_size;
    for (int tile_in_h = 0; tile_in_h < num_of_tiles_h)
    {
        absolute_offset += tile_h_step_offset;
        for (int tile_in_w = 0; tile_in_w < num_of_tiles_w; tile_in_w++)
        {
            absolute_offset += dw_tile_size;
            for (int h = 0; h < dw_tile_h; h++)
            {
#pragma HLS PIPELINE
                for (int d = 0; d < dw_tile_d; d++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < dw_tile_w; w++)
                    {
#pragma HLS UNROLL
                        dw_ifms_buffer[d][tile_in_h * dw_tile_h + h][tile_in_w * dw_tile_w + w] =
                            channels[absolute_offset + d * dw_tile_hw + h * dw_tile_w + w];
                    }
                }
            }
        }
    }
}

void dw_conv_engine(const dw_weights_dt weights[max_conv_d][], fms_dt channels_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
    dw_pss_dt results_buffer[dw_tile_d][dw_tile_w], const int filter_dim, const int in_tile_d, const int conv_d, const int strides)
{
#pragma HLS INLINE
    for (int w = 0; w < dw_tile_w; w++)
    {
#pragma HLS UNROLL
        results_buffer[in_tile_d][w] = 0;
        for (int c_h = 0; c_h < filter_dim; c_h++)
        {
#pragma HLS UNROLL
            for (int c_w = 0; c_w < filter_dim; c_w++)
            {
#pragma HLS UNROLL
                results_buffer[in_tile_d][w] += weights[conv_d][c_h * filter_dim + c_w] * channels_buffer[in_tile_d][c_h][w * strides + c_w];
            }
        }
        results_buffer[in_tile_d][w] = HERE
    }
}

void dw_conv()
{
    fms_dt dw_ifms_buffer[dw_tile_d][112][112]; // partition dw_tile_d,_, complete
}