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

void dw_fill_ifms_buffer(ms_dt channels[max_fms_size],
                         fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
                         fms_dt upper_row[dw_tile_d][], fms_dt lower_row[dw_tile_d][],
                         fms_dt left_col_segment[dw_tile_d][dw_tile_h], fms_dt right_col_segment[dw_tile_d][dw_tile_h],
                         int absolute_offset, const int ifms_height, const int ifms_width, int ifms_tile_in_w, int ifms_tile_in_h,
                         const int use_left_col, const int use_upper_row)
{
#pragma HLS INLINE

    const int starting_fill_w_from_row_offset = ifms_tile_in_w * dw_tile_size;
    // fill upper row
    if (use_upper_row)
    {
        for (int d = 0; d < dw_tile_d; d++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < dw_tile_w; w++)
            {
#pragma HLS UNROLL
                ifms_buffer[d][0][w] = upper_row[d][starting_fill_w_from_row_offset + w];
            }
        }
    }

    // fill lower row
    for (int d = 0; d < dw_tile_d; d++)
    {
#pragma HLS UNROLL
        for (int w = 0; w < dw_tile_w; w++)
        {
#pragma HLS UNROLL
            ifms_buffer[d][dw_max_v2_buffer_height - 1][w] = lower_row[d][starting_fill_w_from_row_offset + w];
        }
    }

    // fill left col
    if (use_left_col)
    {
        for (int d = 0; d < dw_tile_d; d++)
        {
#pragma HLS UNROLL
            for (int h = 0; h < dw_tile_h; h++)
            {
#pragma HLS UNROLL
                ifms_buffer[d][h + use_upper_row][0] = left_col_segment[d][h];
            }
        }
    }

    // fill right col
    for (int d = 0; d < dw_tile_d; d++)
    {
#pragma HLS UNROLL
        for (int h = 0; h < dw_tile_h; h++)
        {
#pragma HLS UNROLL
            ifms_buffer[d][h + use_upper_row][dw_max_v2_buffer_width - use_left_col - 1] = right_col_segment[d][h];
        }
    }

    // fill body
    for (int d = 0; d < dw_tile_d; d++)
    {
#pragma HLS UNROLL
        for (int h = 0; h < dw_tile_h; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < dw_tile_w; w++)
            {
#pragma HLS UNROLL
                ifms_buffer[d][h + use_upper_row][w + use_left_col] = channels[absolute_offset + d * dw_tile_hw + h * dw_tile_w + w];
            }
        }
    }
}

void dw_conv_engine(const dw_weights_dt weights[max_conv_d][], fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
                    fms_dt results_buffer[dw_tile_d], const int filter_dim, const int in_tile_d, const int tile_d_offset, const int strides,
                    fms_quantization_scheme normalization, const int layer_relu)
{
#pragma HLS INLINE

    dw_pss_dt tmp_pss;
    for (int c_h = 0; c_h < filter_dim; c_h++)
    {
#pragma HLS UNROLL
        for (int c_w = 0; c_w < filter_dim; c_w++)
        {
#pragma HLS UNROLL
            tmp_pss += weights[tile_d_offset][c_h * filter_dim + c_w] * ifms_buffer[in_tile_d][h * strides + c_h][c_w];
        }
    }
}

void dw_conv_pipeline(ms_dt channels[max_fms_size], const dw_weights_dt weights[max_conv_d][],
                      fms_dt upper_row[dw_tile_d][max_dw_input_width],
                      fms_dt lower_row[dw_tile_d][max_dw_input_width],
                      fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width],
                      fms_dt col_segment_left[dw_tile_d][dw_tile_h],
                      fms_dt col_segment_right[dw_tile_d][dw_tile_h],
                      const int filter_dim,
                      const int strides,
                      const int padding_left,
                      const int padding_top,
                      const int ifm_width,
                      const int ifm_height,
                      int ifm_tile_in_h, int ifm_tile_in_w,
                      const int tile_d_offset,
                      const int num_of_ifms_tiles_w,
                      const int num_of_ifms_tiles_h,
                      int absolute_offset_in_ifms,
                      const fms_dt fms_zero_point,
                      fms_quantization_scheme normalization,
                      const int layer_relu)
{
#pragma HLS INLINE off

    fill_ifms_tile_depth_col_segment(channels, ifm_tile_in_w, col_segment_right,
                                     absolute_offset_in_ifms, ifm_tile_in_w == num_of_ifms_tiles_w,
                                     ifm_height, fms_zero_point);

    dw_fill_ifms_buffer(channels,
                        ifms_buffer, upper_row, lower_row,
                        col_segment_left, col_segment_right,
                        absolute_offset_in_ifms, ifm_height, ifm_width, num_of_ifms_tiles_w, num_of_ifms_tiles_h,
                        strides == 1, strides == 1);

    dw_conv_engine(weights, ifms_buffer, filter_dim, tile_d_offset, strides,
                   normalization, layer_relu);

    bool is_padding_row = (padding_top && ifm_tile_in_h == 0) || ifm_tile_in_h == num_of_ifms_tiles_h - 1;
    fill_ifms_tile_depth_row_segment(channels, lower_row, ifm_tile_in_w,
                                     absolute_offset_in_ifms, padding_left,
                                     is_padding_row, ifm_width, fms_zero_point);

    shift_col_segments(col_segment_left, col_segment_right);
    shift_segment_between_rows(upper_row, lower_row, ifm_tile_in_w, padding_left);
}

void dw_conv_3x3(dw_weights_dt weights[max_conv_d][max_conv_h][max_conv_w],
                 fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
                 const int layer, const int layer_conv_d, const int layer_ifm_width,
                 const int layer_ifm_height, const int num_of_tiles_d,
                 const int num_of_ofms_tiles_h, const int num_of_ofms_tiles_w, const int strides,
                 const int padding_left, const int padding_right, const int padding_top, const int direction,
                 fused_scales_dt fused_scales[], relu_6_fused_scales_dt relu_6_fused_scales[],
                 biases_dt fused_zero_points[])
{
#pragma HLS INLINE off

    fms_dt upper_row[dw_tile_d][max_dw_input_width];
    fms_dt lower_row[dw_tile_d][max_dw_input_width];

    fms_dt col_segment_left[dw_tile_d][dw_tile_h];
    fms_dt col_segment_right[dw_tile_d][dw_tile_h];

    const int num_of_ifms_tiles_h = num_of_ofms_tiles_h * strides;
    const int num_of_ifms_tiles_w = num_of_ofms_tiles_w * strides;

    const int num_of_ifms_tiles_hw = num_of_ifms_tiles_h * num_of_ifms_tiles_w;
    const int num_of_ofms_tiles_hw = num_of_ofms_tiles_h * num_of_ofms_tiles_w;

    fms_quantization_scheme normalization = {0, 0, 0, 0};
    const int current_layer_fused_parameters_offsets =
        layers_fused_parameters_offsets[layer];

    const fms_dt current_layer_fms_zero_point = conv_fms_zero_points[layer];
    int absolute_index_in_ifms;
    int absolute_index_in_ofms;

    for (int tile_in_d = 0; tile_in_d < num_of_tiles_d; tile_in_d++)
    {
        absolute_index_in_ifms = tile_in_d * num_of_ifms_tiles_hw * dw_tile_size;
        absolute_index_in_ofms = tile_in_d * num_of_ofms_tiles_hw * dw_tile_size;
        fill_ifms_tile_depth_row(channels, upper_row, layer_ifm_width, ,
                                 num_of_ifms_tiles_hw, layer_ifm_height,
                                 tile_in_d, 0,
                                 padding_left, 1,
                                 true,
                                 current_layer_fms_zero_point);
        for (int ofm_tile_in_h = 0; ofm_tile_in_h < num_of_ofms_tiles_h; ofm_tile_in_h++)
        {
            absolute_index_in_ofms += num_of_ofms_tiles_w * dw_tile_size;
            for (int ifm_tile_in_ofm_tile_h = 0; ifm_tile_in_ofm_tile_h < strides; ifm_tile_in_ofm_tile_h++)
            {
                absolute_index_in_ifms += num_of_ifms_tiles_w * dw_tile_size;
                fill_ifms_tile_depth_col_segment(channels, 0, col_segment_left,
                                                 absolute_index_in_ifms, true,
                                                 layer_ifm_height, current_layer_fms_zero_point);
                for (int ofm_tile_in_w = 0; ofm_tile_in_w < num_of_ofms_tiles_w; ofm_tile_in_w++)
                {
                    absolute_index_in_ofms += dw_tile_size;
                    for (int ifm_tile_in_ofm_tile_w = 0; ifm_tile_in_ofm_tile_w < strides; ifm_tile_in_ofm_tile_w++)
                    {
                        int ifm_tile_in_w = ofm_tile_in_w * strides;
                        absolute_index_in_ifms += dw_tile_size;
                        fms_dt ifms_buffer[dw_tile_d][dw_max_v2_buffer_height][dw_max_v2_buffer_width];
                    }
                }
            }
        }
    }
}