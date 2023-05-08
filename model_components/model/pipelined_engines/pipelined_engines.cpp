#include "pipelined_engines.h"

#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE && !ONLY_SEML

using namespace pipelined_engines;
#ifndef PIPELINED_PW_CONV
#define PIPELINED_PW_CONV

void pipelined_engines::padd_left_dw_channels_tile(fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                                   fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                                   layer_specs layer_specs_struct)
{
    const fms_dt current_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
    const int padding_left = layer_specs_struct.padding_left;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < DW_BUFFER_HEIGHT; h++)
        {
            for (int w = 0; w < MAX_DW_PADDING_IN_PIPE; w++)
            {
                if (w < padding_left)
                {
                    dw_channels_tile[d][h][w] = current_layer_ifms_zero_point;
                    dw_channels_tile_copy[d][h][w] = current_layer_ifms_zero_point;
                }
            }
        }
    }
}

void pipelined_engines::fill_fused_scales_and_zps_buffer(const fused_scales_dt fused_scales[],
                                                         const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                                                         const relu_6_fused_scales_dt relu_6_fused_scales[],
                                                         const biases_dt fused_zero_points[],
                                                         fms_quantization_scheme normalization_buffer[],
                                                         int starting_d,
                                                         const int current_layer_fused_parameters_offset,
                                                         const int buffer_size,
                                                         const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int absolute_current_layer_fused_parameters_offset = current_layer_fused_parameters_offset + starting_d;
    for (int i = 0; i < buffer_size; i++)
    {
        normalization_buffer[i].ofm_scale = layer_specs_struct.layer_ofms_scale;
        normalization_buffer[i].ofm_zero_point = layer_specs_struct.layer_ofms_zero_point;
        normalization_buffer[i].fused_scales = fused_scales[absolute_current_layer_fused_parameters_offset + i];
        normalization_buffer[i].fused_scales_log_2_shift =
            fused_scales_log_2_shifts[absolute_current_layer_fused_parameters_offset + i];
        normalization_buffer[i].relu_6_fused_scale = relu_6_fused_scales[absolute_current_layer_fused_parameters_offset + i];
        normalization_buffer[i].fused_zero_point = fused_zero_points[absolute_current_layer_fused_parameters_offset + i];
    }
}

void pipelined_engines::load_pw_weights(const weights_dt pw_weights[],
                                        weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
                                        const int starting_filter,
                                        layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_depth = layer_specs_struct.layer_depth;
    const int layer_num_filters = layer_specs_struct.layer_num_fils;
    const int filling_weights_offset = layer_specs_struct.layer_weights_offset_on_chip +
                                       starting_filter * layer_depth;

    for (int filter_index = 0; filter_index < PARALLELISM_PW_OFMS; filter_index++)
    {
        //#pragma HLS UNROLL
        if (filter_index + starting_filter >= layer_num_filters)
        {
            break;
        }
        const int current_filling_weights_offset = filling_weights_offset + filter_index * layer_depth;
        for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
        {
            if (d >= layer_depth)
            {
                break;
            }
            weights_tile[filter_index][d] = pw_weights[current_filling_weights_offset + d];
        }
    }
}

void pipelined_engines::pw_conv_engine(weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
                                       fms_dt channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                       pss_dt engine_result[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                       const int starting_filter,
                                       const int starting_w,
                                       const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int layer_depth = layer_specs_struct.layer_depth;

pw_engine_d:
    for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
    {
#pragma HLS PIPELINE
        if (d >= layer_depth)
        {
            break;
        }
        for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
        {
#pragma HLS UNROLL
            for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < PARALLELISM_PW_W; w++)
                {
#pragma HLS UNROLL
                    if (w + starting_w >= layer_ifms_width)
                    {
                        break;
                    }
                    if (d == 0)
                    {
                        engine_result[f][h][w] = weights_tile[f][d] * channels[d][h][w + starting_w];
                    }
                    else
                    {
                        engine_result[f][h][w] += weights_tile[f][d] * channels[d][h][w + starting_w];
                    }
                }
            }
        }
    }
}

void first_pass_pw_normalize_engine_result(fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                           pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
                                           fms_dt normalized_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                           fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                           fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
                                           const fms_quantization_scheme normalization_buffer[],
                                           const int starting_d,
                                           const int starting_h,
                                           const bool fused_pw_dw,
                                           const layer_specs layer_specs_struct,
                                           const layer_specs dw_layer_specs_struct)
{
#pragma HLs INLINE off

    const int layer_relu = layer_specs_struct.layer_activation;
    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int layer_ifms_height = layer_specs_struct.layer_ifm_height;

    const int filter_dim = dw_layer_specs_struct.filter_size;
    const int padding_left = dw_layer_specs_struct.padding_left;
    const int padding_top = dw_layer_specs_struct.padding_top;
    const int strides = dw_layer_specs_struct.strides;
    const int filter_dim_minus_strides = filter_dim - strides;
    const int write_offset_h_in_normalized_tile = filter_dim_minus_strides;

    const int useful_rows_in_channels_tile = DW_BUFFER_HEIGHT - (DW_BUFFER_HEIGHT - filter_dim) % strides;
    const int write_offset_h_in_channels_tile = useful_rows_in_channels_tile - filter_dim_minus_strides;

    const int num_of_tiles_in_w = (layer_ifms_width / DW_PIPE_OVERLAP_BUFFER_WIDTH);
    const int d_offset_in_overlap_buffer = starting_d * filter_dim_minus_strides * num_of_tiles_in_w +
                                           (dw_layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);

    const fms_dt dw_layer_ifms_zero_point = dw_layer_specs_struct.layer_ifms_zero_point;

    for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
    {
        for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
        {
            int h_in_overlap_buffer = h - (PW_BUFFER_HEIGHT - filter_dim_minus_strides);
            int current_read_offset_in_overlap_buffer = d_offset_in_overlap_buffer +
                                                        f * filter_dim_minus_strides * num_of_tiles_in_w +
                                                        (h_in_overlap_buffer * layer_ifms_width) / DW_PIPE_OVERLAP_BUFFER_WIDTH;
            int current_write_offset_in_overlap_buffer = d_offset_in_overlap_buffer +
                                                         f * filter_dim_minus_strides * num_of_tiles_in_w +
                                                         (h_in_overlap_buffer * layer_ifms_width) / DW_PIPE_OVERLAP_BUFFER_WIDTH;
            for (int w = 0; w < MAX_FILTER_MINUS_STRIDES; w++)
            {
                if (w >= filter_dim_minus_strides)
                {
                    break;
                }
                if (w < padding_left)
                {
                    continue;
                }
                fms_dt scaled_val = pw_relu_norm_6(
                    engine_result_tile[f][h][w - padding_left], normalization_buffer[f],
                    layer_relu);
                if (h_in_overlap_buffer >= 0)
                {
                    normalized_tile[f][h_in_overlap_buffer][w] = dw_pipe_overlap_buffer[current_read_offset_in_overlap_buffer]
                                                                                       [w - padding_left];
                    dw_pipe_overlap_buffer[current_write_offset_in_overlap_buffer][w - padding_left] = scaled_val;
                }
                if (starting_h + h < layer_ifms_height && (DW_BUFFER_HEIGHT - filter_dim != h + write_offset_h_in_normalized_tile ||
                                                           padding_top == 0 || starting_h != 0))
                { // first time, we start from the last filter_dim rows, hence the row DW_BUFFER_HEIGHT - filter_dim should pe a padding
                    normalized_tile[f][h + filter_dim_minus_strides][w] = scaled_val;
                }
                else
                {
                    normalized_tile[f][h + filter_dim_minus_strides][w] = dw_layer_ifms_zero_point;
                }
            }
        }
    }
}

void pipelined_engines::pw_normalize_engine_result(pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
                                                   fms_dt normalized_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                                   fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
                                                   fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
                                                   const fms_quantization_scheme normalization_buffer[],
                                                   const int starting_d,
                                                   const int starting_h,
                                                   const int starting_w,
                                                   const bool fused_pw_dw,
                                                   const layer_specs layer_specs_struct,
                                                   const layer_specs dw_layer_specs_struct)
{
#pragma HLs INLINE off

    if (starting_w >= 0)
    {
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int layer_ifms_height = layer_specs_struct.layer_ifm_height;
        const int layer_ifms_depth = layer_specs_struct.layer_depth;
        const int layer_relu = layer_specs_struct.layer_activation;

        const int strides = dw_layer_specs_struct.strides;
        const int filter_dim = dw_layer_specs_struct.filter_size;
        const int padding_left = dw_layer_specs_struct.padding_left;
        const fms_dt dw_layer_ifms_zero_point = dw_layer_specs_struct.layer_ifms_zero_point;
        const int padding_top = dw_layer_specs_struct.padding_top;

        const int useful_cols_in_channels_tile = DW_BUFFER_WIDTH - (DW_BUFFER_HEIGHT - filter_dim) % strides;

        const int filter_minus_strides = filter_dim - strides;

        const int write_offset_h_in_normalized_tile = filter_minus_strides;

        scales_dt skip_connection_other_layer_scale = layer_specs_struct.skip_connection_other_layer_scale;
        biases_dt skip_connection_other_layer_zero_point = layer_specs_struct.skip_connection_other_layer_zero_point;

        rec_scales_dt add_layer_scale_reciprocal = layer_specs_struct.add_layer_scale_reciprocal;
        biases_dt add_layer_zero_point = layer_specs_struct.add_layer_zero_point;

        for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
        {
            for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
            {
                for (int w = 0; w < PARALLELISM_PW_W; w++)
                {
#pragma HLS UNROLL
                    pss_dt tmp_pss = engine_result_tile[f][h][w];
                    if (fused_pw_dw)
                    {
                        if (starting_h + h < layer_ifms_height &&
                            (DW_BUFFER_HEIGHT - filter_dim != h + write_offset_h_in_normalized_tile ||
                             padding_top == 0 || starting_h != 0) &&
                            starting_w + filter_minus_strides + w < layer_ifms_width + padding_left)
                        {
                            normalized_tile[f][h + write_offset_h_in_normalized_tile][filter_minus_strides + w] = pw_relu_norm_6(
                                tmp_pss, normalization_buffer[f],
                                layer_relu);
                        }
                        else
                        {
                            normalized_tile[f][h + write_offset_h_in_normalized_tile][filter_minus_strides + w] =
                                dw_layer_ifms_zero_point;
                        }
                    }
                    else
                    {
                        fms_dt normalized_val;
                        if (layer_specs_struct.fused_with_add == 0)
                        {
                            normalized_val = pw_relu_norm_6(tmp_pss, normalization_buffer[f], layer_relu);
                        }
                        else
                        {
                            pss_f_dt tmp_channels_scaled_val =
                                skip_connection_other_layer_scale *
                                (tmp_channels[starting_d + f][h][starting_w + w] - skip_connection_other_layer_zero_point);
                            pss_f_dt scaled_tmp =
                                pw_relu_norm_no_q_no_relu(
                                    tmp_pss,
                                    normalization_buffer[f], layer_relu);

                            pss_f_dt addition_result = (scaled_tmp + tmp_channels_scaled_val) *
                                                           add_layer_scale_reciprocal +
                                                       add_layer_zero_point;
                            addition_result = addition_result + quant_half - (addition_result < 0);
                            normalized_val = clamp(addition_result);
                        }
                        // result[starting_d + f][h][starting_w + w] = normalized_val;
                        normalized_tile[f][h][w] = normalized_val;
                        if (layer_specs_struct.write_to_tmp)
                        {
                            if (h == 0)
                            {
                                tmp_channels[starting_d + f][0][starting_w + w] = tmp_channels[starting_d + f][PW_BUFFER_HEIGHT][starting_w + w];
                            }
                            tmp_channels[starting_d + f][h + 1][starting_w + w] = normalized_val;
                        }
                    }
                }
            }
            if (fused_pw_dw)
            {
                for (int h = 0; h < DW_BUFFER_HEIGHT; h++)
                {
                    for (int w = 0; w < MAX_FILTER_MINUS_STRIDES; w++)
                    {
                        if (w >= filter_minus_strides)
                        {
                            break;
                        }
                        if (starting_w > 0)
                        {
                            normalized_tile[f][h][w] = dw_vertical_overlap_buffer[f][h][w];
                        }
                        dw_vertical_overlap_buffer[f][h][w] =
                            normalized_tile[f][h][useful_cols_in_channels_tile - filter_minus_strides + w];
                    }
                }
            }
        }
    }
}

void pipelined_engines::write_next_overlap_and_read_current(fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                                            fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH]
                                                                                             [DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
                                                            fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                                            const int starting_d,
                                                            const int starting_h,
                                                            const int starting_w,
                                                            layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    if (starting_w >= 0)
    {
        const fms_dt dw_layer_ifms_zero_point = layer_specs_struct.layer_ifms_zero_point;
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int strides = layer_specs_struct.strides;
        const int filter_dim = layer_specs_struct.filter_size;
        const int filter_minus_strides = filter_dim - strides;
        const int padding_left = layer_specs_struct.padding_left;

        const int num_of_tiles_in_w = (layer_ifms_width / DW_PIPE_OVERLAP_BUFFER_WIDTH);
        const int d_offset_in_overlap_buffer = starting_d * filter_minus_strides * num_of_tiles_in_w +
                                               (layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);

        const int useful_rows_in_channels_tile = DW_BUFFER_HEIGHT - (DW_BUFFER_HEIGHT - filter_dim) % strides;
        const int write_offset_h_in_channels_tile = useful_rows_in_channels_tile - filter_minus_strides;

        const int inner_offset_w = starting_w < DW_PIPE_OVERLAP_BUFFER_WIDTH
                                       ? starting_w
                                       : starting_w - DW_PIPE_OVERLAP_BUFFER_WIDTH; //%

        const int columns_filled_by_pw_in_the_firts_pass = filter_minus_strides - padding_left;

        for (int d = 0; d < PARALLELISM_PW_OFMS; d++)
        {
            for (int h = 0; h < MAX_FILTER_MINUS_STRIDES; h++)
            {
                if (h >= filter_minus_strides)
                {
                    break;
                }

                int current_write_offset_in_overlap_buffer = d_offset_in_overlap_buffer +
                                                             d * filter_minus_strides * num_of_tiles_in_w +
                                                             (h * layer_ifms_width + starting_w) / DW_PIPE_OVERLAP_BUFFER_WIDTH;
                int current_read_offset_in_overlap_buffer = d_offset_in_overlap_buffer +
                                                            d * filter_minus_strides * num_of_tiles_in_w +
                                                            (h * layer_ifms_width + starting_w) / DW_PIPE_OVERLAP_BUFFER_WIDTH;
                for (int w = 0; w < PARALLELISM_PW_W; w++)
                {
                    fms_dt read_val;
                    //#pragma HLS UNROLL
                    if (w + starting_w + columns_filled_by_pw_in_the_firts_pass < layer_ifms_width)
                    {
                        if (w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass >= DW_PIPE_OVERLAP_BUFFER_WIDTH)
                        {
                            if (starting_h != 0)
                            {
                                read_val =
                                    dw_pipe_overlap_buffer[current_read_offset_in_overlap_buffer + 1]
                                                          [w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass -
                                                           DW_PIPE_OVERLAP_BUFFER_WIDTH];
                            }
                            dw_pipe_overlap_buffer[current_write_offset_in_overlap_buffer + 1]
                                                  [w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass -
                                                   DW_PIPE_OVERLAP_BUFFER_WIDTH] =
                                                      dw_channels_tile[d][write_offset_h_in_channels_tile + h][w + filter_minus_strides];
                        }
                        else
                        {
                            if (starting_h != 0)
                            {
                                read_val =
                                    dw_pipe_overlap_buffer[current_read_offset_in_overlap_buffer]
                                                          [w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass];
                            }
                            dw_pipe_overlap_buffer[current_write_offset_in_overlap_buffer]
                                                  [w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass] =
                                                      dw_channels_tile[d][write_offset_h_in_channels_tile + h][w + filter_minus_strides];
                        }
                    }
                    else
                    {
                        read_val = dw_layer_ifms_zero_point;
                    }
                    dw_channels_tile[d][h][w + filter_minus_strides] = read_val;
                    if (w + filter_minus_strides >= PARALLELISM_PW_W)
                    {
                        dw_vertical_overlap_buffer[d][h][w + filter_minus_strides - PARALLELISM_PW_W] = read_val;
                    }
                }
            }
        }
    }
}

void pipelined_engines::fill_dw_weights_tile(const dw_weights_dt weights[][MAX_DW_FILTER_AREA_IN_PIPE],
                                             dw_weights_dt weights_tile[][MAX_DW_FILTER_AREA_IN_PIPE],
                                             int starting_d, const int current_dw_layer_weights_offset)
{
#pragma HLS INLINE off

    const int absolute_current_layer_weights_offset =
        current_dw_layer_weights_offset + starting_d;
    for (int d = 0; d < DW_BUFFER_DEPTH; d++)
    {
        for (int i = 0; i < MAX_DW_FILTER_AREA_IN_PIPE; i++)
        {
            weights_tile[d][i] = weights[absolute_current_layer_weights_offset + d][i];
        }
    }
}

void pipelined_engines::dw_conv_engine(
    dw_weights_dt weights[DW_TILE_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE],
    fms_dt channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    dw_pss_dt result_tile[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
    layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_ofms_width = layer_specs_struct.layer_ofm_width;
    const int layer_d = layer_specs_struct.layer_depth;
    const int filter_dim = layer_specs_struct.filter_size;
    const int strides = layer_specs_struct.strides;

dw_conv_engine:

    for (int c_h = 0; c_h < MAX_DW_FILTER_DIM_IN_PIPE; c_h++)
    {
        for (int c_w = 0; c_w < MAX_DW_FILTER_DIM_IN_PIPE; c_w++)
        {
            for (int d = 0; d < DW_TILE_DEPTH;
                 d++)
            {
#pragma HLS PIPELINE
                for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < PARALLELISM_DW_W; w++)
                    {
#pragma HLS UNROLL
                        if (c_w >= filter_dim || c_h >= filter_dim || h >= PW_BUFFER_HEIGHT / strides)
                        {
                            break;
                        }
                        if (c_h == 0 && c_w == 0)
                        {
                            result_tile[d][h][w] =
                                channels_tile[d][h * strides + c_h]
                                             [w * strides + c_w] *
                                weights[d][c_h * filter_dim + c_w];
                        }
                        else
                        {
                            result_tile[d][h][w] +=
                                channels_tile[d][h * strides + c_h]
                                             [w * strides + c_w] *
                                weights[d][c_h * filter_dim + c_w];
                        }
                    }
                }
            }
        }
    }
}

void pipelined_engines::dw_normalize_and_write_back_result_tile(dw_pss_dt result_tile[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PARALLELISM_PW_W],
                                                                fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                                const fms_quantization_scheme normalization_buffer[],
                                                                const int starting_d,
                                                                const int h_offset_in_result,
                                                                const int starting_w,
                                                                layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    if (starting_w >= 0)
    {
        const int layer_ofms_width = layer_specs_struct.layer_ofm_width;
        const int strides = layer_specs_struct.strides;
        const int ifms_d = layer_specs_struct.layer_depth;
        const int layer_relu = layer_specs_struct.layer_activation;
        const int offset_w = starting_w / strides;

        for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
        {
            if (h + h_offset_in_result >= PW_BUFFER_HEIGHT)
            {
                break;
            }
            for (int d = 0; d < DW_TILE_DEPTH; d++)
            {
                //#pragma HLS PIPELINE
                for (int w = 0; w < PARALLELISM_DW_W; w++)
                {
                    //#pragma HLS UNROLL
                    if (w >= PARALLELISM_DW_W / strides)
                    {
                        break;
                    }
                    result[starting_d + d][h + h_offset_in_result][offset_w + w] = dw_relu_norm(
                        result_tile[d][h][w], normalization_buffer[d],
                        layer_relu);
                }
            }
        }
    }
}

void pipelined_engines::pw_write_back_result_tile(fms_dt result_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                                  fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                  const int starting_d,
                                                  const int starting_w)
{
    if (starting_w >= 0)
    {
        for (int d = 0; d < DW_TILE_DEPTH; d++)
        {
            for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
            {
                //#pragma HLS PIPELINE
                for (int w = 0; w < PARALLELISM_DW_W; w++)
                {
                    result[starting_d + d][h][starting_w + w] = result_tile[d][h][w];
                }
            }
        }
    }
}

void pipelined_engines::pw_dw_conv(const weights_dt pw_weights[],
                                   const dw_weights_dt weights[][3 * 3],
                                   fms_dt channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                   fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                   fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
                                   fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                   fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                   fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
                                   const int starting_h,
                                   const int h_offset_in_result,
                                   bool fused_pw_dw,
                                   const layer_specs pw_layer_specs_struct,
                                   const layer_specs dw_layer_specs_struct,
                                   const fused_scales_dt fused_scales[],
                                   const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                                   const relu_6_fused_scales_dt relu_6_fused_scales[],
                                   const biases_dt fused_zero_points[])
{
#pragma HLS INLINE off

    const int pw_layer = pw_layer_specs_struct.layer_index;
    const int dw_layer = dw_layer_specs_struct.layer_index;
    const int ifms_width = pw_layer_specs_struct.layer_ifm_width;
    const int num_of_filters = pw_layer_specs_struct.layer_num_fils;

    fms_quantization_scheme dw_normalization_buffer[DW_BUFFER_DEPTH];
    fms_quantization_scheme dw_normalization_buffer_copy[DW_BUFFER_DEPTH];

    fms_quantization_scheme pw_normalization_buffer[PARALLELISM_PW_OFMS];
    fms_quantization_scheme pw_normalization_buffer_copy[PARALLELISM_PW_OFMS];

    weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH];
    weights_dt weights_tile_copy[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH];

    dw_weights_dt dw_weights_tile[DW_BUFFER_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE];
    dw_weights_dt dw_weights_tile_copy[DW_BUFFER_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE];

    pss_dt pw_engine_result_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W];
    pss_dt pw_engine_result_tile_copy[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W];

    // fms_dt normalized_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W];
    // fms_dt normalized_tile_copy[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W];

    dw_pss_dt dw_result_tile[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PARALLELISM_PW_W];
    dw_pss_dt dw_result_tile_copy[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PARALLELISM_PW_W];

    fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES];

    int prev_w = -1;
    int prev_prev_w = -1;
    int prev_prev_prev_w = -1;

    if (fused_pw_dw)
    {
        for (int d = 0; d < num_of_filters; d += PARALLELISM_PW_OFMS)
        {
            //###############################

            padd_left_dw_channels_tile(dw_channels_tile, dw_channels_tile_copy,
                                       dw_layer_specs_struct);

            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts,
                                             relu_6_fused_scales,
                                             fused_zero_points,
                                             dw_normalization_buffer,
                                             d, // starting_d
                                             pipe_layers_fused_parameters_offsets[dw_layer],
                                             DW_TILE_DEPTH,
                                             dw_layer_specs_struct);
            pipelined_engines::fill_dw_weights_tile(weights,
                                                    dw_weights_tile,
                                                    d, dw_layers_weights_offsets[dw_layer]);
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts,
                                             relu_6_fused_scales,
                                             fused_zero_points,
                                             pw_normalization_buffer,
                                             d, // starting_d
                                             pipe_layers_fused_parameters_offsets[pw_layer],
                                             PARALLELISM_PW_OFMS,
                                             pw_layer_specs_struct);

            load_pw_weights(pw_weights,
                            weights_tile,
                            d,
                            pw_layer_specs_struct);

            pw_conv_engine(weights_tile,
                           channels,
                           pw_engine_result_tile_copy,
                           d,
                           0,
                           pw_layer_specs_struct);

            first_pass_pw_normalize_engine_result(dw_pipe_overlap_buffer,
                                                  pw_engine_result_tile_copy,
                                                  dw_channels_tile,
                                                  result,
                                                  tmp_channels,
                                                  pw_normalization_buffer,
                                                  d,
                                                  starting_h,
                                                  fused_pw_dw,
                                                  pw_layer_specs_struct,
                                                  dw_layer_specs_struct);

            for (int w = 0; w < ifms_width; w += PARALLELISM_PW_W)
            {
                prev_w = w - PARALLELISM_PW_W;
                prev_prev_w = prev_w - PARALLELISM_PW_W;
                prev_prev_prev_w = prev_prev_w - PARALLELISM_PW_W;
                if ((w / PARALLELISM_PW_W) % 2 == 0)
                {
                    dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                            result,
                                                            dw_normalization_buffer,
                                                            d,
                                                            h_offset_in_result,
                                                            prev_prev_prev_w,
                                                            dw_layer_specs_struct);
                    //###############################
                    dw_conv_engine(
                        dw_weights_tile,
                        dw_channels_tile,
                        dw_result_tile,
                        dw_layer_specs_struct);
                    //###############################
                    pw_normalize_engine_result(pw_engine_result_tile_copy,
                                               dw_channels_tile_copy,
                                               dw_vertical_overlap_buffer,
                                               tmp_channels,
                                               pw_normalization_buffer,
                                               d,
                                               starting_h,
                                               prev_w,
                                               fused_pw_dw,
                                               pw_layer_specs_struct,
                                               dw_layer_specs_struct);
                    write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                        dw_vertical_overlap_buffer,
                                                        dw_channels_tile_copy,
                                                        d,
                                                        starting_h,
                                                        prev_w,
                                                        dw_layer_specs_struct);
                    //###############################
                    pw_conv_engine(weights_tile,
                                   channels,
                                   pw_engine_result_tile,
                                   d,
                                   w + 1,
                                   pw_layer_specs_struct);
                }
                else
                {
                    dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                            result,
                                                            dw_normalization_buffer,
                                                            d,
                                                            h_offset_in_result,
                                                            prev_prev_prev_w,
                                                            dw_layer_specs_struct);
                    //###############################
                    dw_conv_engine(
                        dw_weights_tile,
                        dw_channels_tile_copy,
                        dw_result_tile_copy,
                        dw_layer_specs_struct);
                    //###############################
                    pw_normalize_engine_result(pw_engine_result_tile,
                                               dw_channels_tile,
                                               dw_vertical_overlap_buffer,
                                               tmp_channels,
                                               pw_normalization_buffer,
                                               d,
                                               starting_h,
                                               prev_w,
                                               fused_pw_dw,
                                               pw_layer_specs_struct,
                                               dw_layer_specs_struct);
                    write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                        dw_vertical_overlap_buffer,
                                                        dw_channels_tile,
                                                        d,
                                                        starting_h,
                                                        prev_w,
                                                        dw_layer_specs_struct);
                    //###############################
                    pw_conv_engine(weights_tile,
                                   channels,
                                   pw_engine_result_tile_copy,
                                   d,
                                   w + 1,
                                   pw_layer_specs_struct);
                }
            }

            if (((ifms_width / PARALLELISM_PW_W) - 1) % 2)
            {
                dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                        result,
                                                        dw_normalization_buffer,
                                                        d,
                                                        h_offset_in_result,
                                                        ifms_width - 3 * PARALLELISM_PW_W,
                                                        dw_layer_specs_struct);
                //###############################
                dw_conv_engine(
                    dw_weights_tile,
                    dw_channels_tile,
                    dw_result_tile,
                    dw_layer_specs_struct);
                //###############################
                pw_normalize_engine_result(pw_engine_result_tile_copy,
                                           dw_channels_tile_copy,
                                           dw_vertical_overlap_buffer,
                                           tmp_channels,
                                           pw_normalization_buffer,
                                           d,
                                           starting_h,
                                           ifms_width - PARALLELISM_PW_W,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                    dw_vertical_overlap_buffer,
                                                    dw_channels_tile_copy,
                                                    d,
                                                    starting_h,
                                                    ifms_width - PARALLELISM_PW_W,
                                                    dw_layer_specs_struct);
                //#######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                        result,
                                                        dw_normalization_buffer,
                                                        d,
                                                        h_offset_in_result,
                                                        ifms_width - 2 * PARALLELISM_PW_W,
                                                        dw_layer_specs_struct);
                //###############################
                dw_conv_engine(
                    dw_weights_tile,
                    dw_channels_tile_copy,
                    dw_result_tile,
                    dw_layer_specs_struct);
                //#######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                        result,
                                                        dw_normalization_buffer,
                                                        d,
                                                        h_offset_in_result,
                                                        ifms_width - PARALLELISM_PW_W,
                                                        dw_layer_specs_struct);
            }
            else
            {
                dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                        result,
                                                        dw_normalization_buffer,
                                                        d,
                                                        h_offset_in_result,
                                                        ifms_width - 3 * PARALLELISM_PW_W,
                                                        dw_layer_specs_struct);
                //###############################
                dw_conv_engine(
                    dw_weights_tile,
                    dw_channels_tile_copy,
                    dw_result_tile_copy,
                    dw_layer_specs_struct);
                pw_normalize_engine_result(pw_engine_result_tile,
                                           dw_channels_tile,
                                           dw_vertical_overlap_buffer,
                                           tmp_channels,
                                           pw_normalization_buffer,
                                           d,
                                           starting_h,
                                           ifms_width - PARALLELISM_PW_W,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                    dw_vertical_overlap_buffer,
                                                    dw_channels_tile,
                                                    d,
                                                    starting_h,
                                                    ifms_width - PARALLELISM_PW_W,
                                                    dw_layer_specs_struct);
                //#######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                        result,
                                                        dw_normalization_buffer_copy,
                                                        d,
                                                        h_offset_in_result,
                                                        ifms_width - 2 * PARALLELISM_PW_W,
                                                        dw_layer_specs_struct);
                //###############################
                dw_conv_engine(
                    dw_weights_tile,
                    dw_channels_tile,
                    dw_result_tile,
                    dw_layer_specs_struct);
                //#######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                        result,
                                                        dw_normalization_buffer,
                                                        d,
                                                        h_offset_in_result,
                                                        ifms_width - PARALLELISM_PW_W,
                                                        dw_layer_specs_struct);
            }
        }
    }
    else
    {
        for (int d = 0; d < num_of_filters; d += PARALLELISM_PW_OFMS)
        {
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts,
                                             relu_6_fused_scales,
                                             fused_zero_points,
                                             pw_normalization_buffer,
                                             d, // starting_d
                                             pipe_layers_fused_parameters_offsets[pw_layer],
                                             PARALLELISM_PW_OFMS,
                                             pw_layer_specs_struct);

            load_pw_weights(pw_weights,
                            weights_tile,
                            d,
                            pw_layer_specs_struct);

            for (int w = 0; w < ifms_width; w += PARALLELISM_PW_W)
            {
                prev_w = w - PARALLELISM_PW_W;
                if ((w / PARALLELISM_PW_W) % 2 == 0)
                {
                    pw_normalize_engine_result(pw_engine_result_tile_copy,
                                               dw_channels_tile_copy,
                                               dw_vertical_overlap_buffer,
                                               tmp_channels,
                                               pw_normalization_buffer,
                                               d,
                                               starting_h,
                                               prev_w,
                                               fused_pw_dw,
                                               pw_layer_specs_struct,
                                               dw_layer_specs_struct);
                    pw_write_back_result_tile(dw_channels_tile_copy,
                                              result,
                                              d,
                                              prev_w);
                    //###############################
                    pw_conv_engine(weights_tile,
                                   channels,
                                   pw_engine_result_tile,
                                   d,
                                   w,
                                   pw_layer_specs_struct);
                }
                else
                {
                    pw_normalize_engine_result(pw_engine_result_tile,
                                               dw_channels_tile,
                                               dw_vertical_overlap_buffer,
                                               tmp_channels,
                                               pw_normalization_buffer,
                                               d,
                                               starting_h,
                                               prev_w,
                                               fused_pw_dw,
                                               pw_layer_specs_struct,
                                               dw_layer_specs_struct);
                    pw_write_back_result_tile(dw_channels_tile,
                                              result,
                                              d,
                                              prev_w);
                    //###############################
                    pw_conv_engine(weights_tile,
                                   channels,
                                   pw_engine_result_tile_copy,
                                   d,
                                   w,
                                   pw_layer_specs_struct);
                }
            }

            if (((ifms_width / PARALLELISM_PW_W) - 1) % 2)
            {
                pw_normalize_engine_result(pw_engine_result_tile_copy,
                                           dw_channels_tile_copy,
                                           dw_vertical_overlap_buffer,
                                           tmp_channels,
                                           pw_normalization_buffer,
                                           d,
                                           starting_h,
                                           ifms_width - PARALLELISM_PW_W,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                pw_write_back_result_tile(dw_channels_tile_copy,
                                          result,
                                          d,
                                          ifms_width - PARALLELISM_PW_W);
            }
            else
            {
                pw_normalize_engine_result(pw_engine_result_tile,
                                           dw_channels_tile,
                                           dw_vertical_overlap_buffer,
                                           tmp_channels,
                                           pw_normalization_buffer,
                                           d,
                                           starting_h,
                                           ifms_width - PARALLELISM_PW_W,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                pw_write_back_result_tile(dw_channels_tile,
                                          result,
                                          d,
                                          ifms_width - PARALLELISM_PW_W);
            }
        }
    }
}

#endif

#endif
