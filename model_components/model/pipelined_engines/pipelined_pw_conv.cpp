#include "pipelined_engines.h"

#ifndef PIPELINED_PW_CONV
#define PIPELINED_PW_CONV

void pipelined_engines::fill_fused_scales_and_zps_buffer(const fused_scales_dt fused_scales[],
                                                         const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                                                         const relu_6_fused_scales_dt relu_6_fused_scales[],
                                                         const biases_dt fused_zero_points[],
                                                         fms_quantization_scheme normalization_buffer[],
                                                         int starting_d,
                                                         const int current_layer_fused_parameters_offset,
                                                         const layer_specs layer_specs_struct)
{
    const int absolute_current_layer_fused_parameters_offset = current_layer_fused_parameters_offset + starting_d;
    for (int i = 0; i < PARALLELISM_DW_OFMS; i++)
    {
        if (starting_d == 0)
        {
            normalization_buffer[i].ofm_zero_point = layer_specs_struct.layer_ofms_scale;
            normalization_buffer[i].ofm_zero_point = layer_specs_struct.layer_ofms_zero_point;
        }
        normalization_buffer[i].fused_scales = fused_scales[absolute_current_layer_fused_parameters_offset + i];
        normalization_buffer[i].fused_scales_log_2_shift =
            fused_scales_log_2_shifts[absolute_current_layer_fused_parameters_offset + i];
        normalization_buffer[i].relu_6_fused_scale = relu_6_fused_scales[absolute_current_layer_fused_parameters_offset + i];
        normalization_buffer[i].fused_zero_point = fused_zero_points[absolute_current_layer_fused_parameters_offset + i];
    }
}

void pipelined_engines::load_pw_weights(const weights_dt pw_weights[],
                                        weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_DW_BUFFER_DEPTH],
                                        layer_specs layer_specs_struct,
                                        const int starting_filter)
{
#pragma HLS INLINE off

    const int layer_depth = layer_specs_struct.layer_depth;
    const int layer_num_filters = layer_specs_struct.layer_num_fils;
    const int filling_weights_offset = layer_specs_struct.layer_weights_offset +
                                       starting_filter * layer_depth;

    for (int filter_index = 0; filter_index < PARALLELISM_PW_OFMS; filter_index++)
    {
#pragma HLS UNROLL
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

void pipelined_engines::pw_conv_engine(weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_DW_BUFFER_DEPTH],
                                       fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                       pss_dt engine_result[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                       const int starting_filter,
                                       const int starting_w,
                                       const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int layer_depth = layer_specs_struct.layer_ifm_width;

    for (int d = 0; d < MAX_DW_BUFFER_DEPTH; d++)
    {
#pragma HLS PIPELINE
        if (d >= layer_depth)
        {
            break;
        }
        for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
        {
#pragma HLS UNROLL
            for (int h = 0; h < MAX_PW_BUFFER_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < PARALLELISM_PW_W; w++)
                {
#pragma HLS UNROLL
                    if (d == 0)
                    {
                        engine_result[f][h][w] += weights_tile[starting_filter + f][d] * channels[d][h][starting_w + w];
                    }
                    else
                    {
                        engine_result[f][h][w] += weights_tile[starting_filter + f][d] * channels[d][h][starting_w + w];
                    }
                }
            }
        }
    }
}

void pipelined_engines::pw_normalize_engine_result(pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                                   fms_dt normalized_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                                   const fms_quantization_scheme normalization_buffer[],
                                                   const layer_specs layer_specs_struct)
{
#pragma HLs INLINE off

    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int layer_ifms_depth = layer_specs_struct.layer_depth;
    const int layer_relu = layer_specs_struct.layer_activation;

    for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
    {
        for (int h = 0; h < MAX_PW_BUFFER_HEIGHT; h++)
        {
#pragma HLS PIPELINE
            for (int w = 0; w < PARALLELISM_PW_W; w++)
            {
#pragma HLS UNROLL
                pss_dt tmp_pss = engine_result_tile[f][h][w];
                normalized_tile[f][h][w] = pw_relu_norm(
                    tmp_pss, normalization_buffer[f],
                    layer_relu);
            }
        }
    }
}

void pipelined_engines::write_next_overlap_and_read_current(fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                                            fms_dt normalized_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W],
                                                            fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED],
                                                            const int starting_d_read,
                                                            const int starting_d_write,
                                                            const int starting_w,
                                                            layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int read_offset_in_overlap_buffer = starting_d_read +
                                              (layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);
    const int write_offset_in_overlap_buffer = starting_d_write +
                                               (layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);
    const int filter_minus_strides = layer_specs_struct.filter_size - layer_specs_struct.strides;
    for (int d = 0; d < PARALLELISM_PW_OFMS; d++)
    {
        for (int h = 0; h < MAX_DW_STRIDES_IN_PIPE; h++)
        {
#pragma HLS PIPELINE
            if (h >= filter_minus_strides)
            {
                break;
            }
            for (int w = 0; w < PARALLELISM_PW_W; w++)
            {
#pragma HLS UNROLL
                dw_channels_tile[d][h][w] =
                    dw_pipe_overlap_buffer[read_offset_in_overlap_buffer + d * filter_minus_strides + h][w];
                dw_pipe_overlap_buffer[write_offset_in_overlap_buffer + d * filter_minus_strides + h][w] =
                    normalized_tile[d][PARALLELISM_PW_H - filter_minus_strides + h][w];
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
    for (int d = 0; d < dw_pipeline_depth; d++)
    {
#pragma HLS PIPELINE
        for (int i = 0; i < MAX_DW_FILTER_AREA_IN_PIPE; i++)
        {
#pragma HLS UNROLL
            weights_tile[d][i] = weights[absolute_current_layer_weights_offset + d][i];
        }
    }
}

void pipelined_engines::dw_conv_engine(
    dw_weights_dt weights[DW_TILE_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE],
    fms_dt channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][DW_TILE_WIDTH_PADDED],
    dw_pss_dt result_tile[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][DW_TILE_WIDTH],
    const int starting_d,
    layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

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
                for (int h = 0; h < MAX_PW_BUFFER_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < DW_TILE_WIDTH; w++)
                    {
#pragma HLS UNROLL
                        if (c_w >= filter_dim || c_h >= filter_dim || h >= dw_tile_h / strides || w >= dw_tile_w / strides)
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

void pipelined_engines::dw_normalize_and_write_back_result_tile(dw_pss_dt result_tile[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][DW_TILE_WIDTH],
                                                                fms_dt result[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                                const fms_quantization_scheme normalization_buffer[],
                                                                const int starting_d,
                                                                const int starting_w,
                                                                layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int strides = layer_specs_struct.strides;
    const int ifms_d = layer_specs_struct.layer_depth;
    const int layer_relu = layer_specs_struct.layer_activation;

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
        for (int h = 0; h < MAX_PW_BUFFER_HEIGHT; h++)
        {
#pragma HLS PIPELINE
            for (int w = 0; w < DW_TILE_WIDTH; w++)
            {
#pragma HLS UNROLL
                result[starting_d + d][h][starting_w + w] = dw_relu_norm(
                    result_tile[d][h][w], normalization_buffer[d],
                    layer_relu);
            }
        }
    }
}

void pipelined_engines::pw_dw_conv(const weights_dt pw_weights[],
                                   fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                   fms_dt result[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH],
                                   fms_dt tmp_channels[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH],
                                   int pw_layer,
                                   int dw_layer,
                                   const layer_specs pw_layer_specs_struct,
                                   const layer_specs dw_layer_specs_struct,
                                   const fused_scales_dt fused_scales[],
                                   const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                                   const relu_6_fused_scales_dt relu_6_fused_scales[],
                                   const biases_dt fused_zero_points[])
{

    const int ifms_width = pw_layer_specs_struct.layer_ifm_width;
    const int num_of_filters = pw_layer_specs_struct.layer_num_fils;
    int prev_d;
    int prev_prev_d;

    for (int w = 0; w < ifms_width; w += PARALLELISM_PW_W)
    {
        for (int d = 0; d < num_of_filters; d += PARALLELISM_PW_OFMS)
        {
            prev_d = d >= PARALLELISM_PW_OFMS ? d - PARALLELISM_PW_OFMS : 0;
            prev_prev_d = prev_d >= PARALLELISM_PW_OFMS ? prev_d - PARALLELISM_PW_OFMS : 0;
            if (d % 2 == 0)
            {
            }
            else
            {
            }
        }
    }
}

#endif