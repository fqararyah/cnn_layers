#include "pipelined_engines.h"

#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE && ! ONLY_SEML

using namespace pipelined_engines;
#ifndef PIPELINED_PW_CONV
#define PIPELINED_PW_CONV

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

    if (starting_d >= 0)
    {
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

void pipelined_engines::pw_conv_engine(weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
                                       fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                       pss_dt engine_result[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH],
                                       const int starting_filter,
                                       const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
    const int layer_depth = layer_specs_struct.layer_depth;

pw_engine_o_w:
    for (int o_w = 0; o_w < layer_ifms_width; o_w += PARALLELISM_PW_W)
    {
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
                for (int h = 0; h < MAX_PW_BUFFER_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < PARALLELISM_PW_W; w++)
                    {
#pragma HLS UNROLL
                        if (d == 0)
                        {
                            engine_result[f][h][w + o_w] = weights_tile[f][d] * channels[d][h][o_w + w];
                        }
                        else
                        {
                            engine_result[f][h][w + o_w] += weights_tile[f][d] * channels[d][h][o_w + w];
                        }
                    }
                }
            }
        }
    }
}

void pipelined_engines::pw_normalize_engine_result(pss_dt engine_result_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                   fms_dt normalized_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
                                                   fms_dt result[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                   fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
                                                   const fms_quantization_scheme normalization_buffer[],
                                                   const int starting_d,
                                                   const int starting_h,
                                                   const bool fused_pw_dw,
                                                   const layer_specs layer_specs_struct,
                                                   const layer_specs dw_layer_specs_struct)
{
#pragma HLs INLINE off

    if (starting_d >= 0)
    {
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int layer_ifms_height = layer_specs_struct.layer_ifm_height;
        const int layer_ifms_depth = layer_specs_struct.layer_depth;
        const int layer_relu = layer_specs_struct.layer_activation;

        const int strides = dw_layer_specs_struct.strides;
        const int filter_dim = dw_layer_specs_struct.filter_size;
        const int writing_w_offset = dw_layer_specs_struct.padding_left;
        const fms_dt dw_layer_ifms_zero_point = dw_layer_specs_struct.layer_ifms_zero_point;
        const int padding_top = dw_layer_specs_struct.padding_top;

        const int filter_minus_strides = filter_dim - strides;

        const int write_offset_h_in_normalized_tile = filter_minus_strides;

        scales_dt skip_connection_other_layer_scale = layer_specs_struct.skip_connection_other_layer_scale;
        biases_dt skip_connection_other_layer_zero_point = layer_specs_struct.skip_connection_other_layer_zero_point;

        rec_scales_dt add_layer_scale_reciprocal = layer_specs_struct.add_layer_scale_reciprocal;
        biases_dt add_layer_zero_point = layer_specs_struct.add_layer_zero_point;

        for (int o_w = 0; o_w < layer_ifms_width; o_w += PARALLELISM_PW_W)
        {
            for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
            {
                for (int h = 0; h < MAX_PW_BUFFER_HEIGHT; h++)
                {
                    for (int w = 0; w < PARALLELISM_PW_W; w++)
                    {
                        //#pragma HLS UNROLL
                        pss_dt tmp_pss = engine_result_tile[f][h][w + o_w];
                        if (fused_pw_dw)
                        {
                            if (starting_h + h < layer_ifms_height &&
                                (MAX_DW_BUFFER_HEIGHT - filter_dim != h + write_offset_h_in_normalized_tile ||
                                 padding_top == 0 || starting_h != 0))
                            {
                                normalized_tile[f][h + write_offset_h_in_normalized_tile][writing_w_offset + o_w + w] = pw_relu_norm(
                                    tmp_pss, normalization_buffer[f],
                                    layer_relu);
                            }
                            else
                            {
                                normalized_tile[f][h + write_offset_h_in_normalized_tile][writing_w_offset + o_w + w] = dw_layer_ifms_zero_point;
                            }
                        }
                        else
                        {
                            fms_dt normalized_val;
                            if (layer_specs_struct.fused_with_add == 0)
                            {
                                normalized_val = pw_relu_norm(tmp_pss, normalization_buffer[f], layer_relu);
                            }
                            else
                            {
                                pss_f_dt tmp_channels_scaled_val =
                                    skip_connection_other_layer_scale *
                                    (tmp_channels[starting_d + f][h][w + o_w] - skip_connection_other_layer_zero_point);
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
                            result[starting_d + f][h][o_w + w] = normalized_val;
                            if (layer_specs_struct.write_to_tmp)
                            {
                                if (h == 0)
                                {
                                    tmp_channels[starting_d + f][0][o_w + w] = tmp_channels[starting_d + f][MAX_PW_BUFFER_HEIGHT][o_w + w];
                                }
                                tmp_channels[starting_d + f][h + 1][o_w + w] = normalized_val;
                            }
                        }
                    }
                }
            }
        }
    }
}

void pipelined_engines::write_next_overlap_and_read_current(fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                                            fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
                                                            const int starting_d_read,
                                                            const int starting_d_write,
                                                            const int starting_h,
                                                            layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    if (starting_d_write >= 0)
    {
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int strides = layer_specs_struct.strides;
        const int filter_dim = layer_specs_struct.filter_size;
        const int filter_minus_strides = filter_dim - strides;
        const int padding_left = layer_specs_struct.padding_left;

        const int num_of_tiles_in_w = (layer_ifms_width / PARALLELISM_PW_W);
        const int read_offset_in_overlap_buffer = starting_d_read * filter_minus_strides * num_of_tiles_in_w +
                                                  (layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);
        const int write_offset_in_overlap_buffer = starting_d_write * filter_minus_strides * num_of_tiles_in_w +
                                                   (layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);

        const int useful_rows_in_channels_tile = MAX_DW_BUFFER_HEIGHT - (MAX_DW_BUFFER_HEIGHT - filter_dim) % strides;
        const int write_offset_h_in_channels_tile = useful_rows_in_channels_tile - filter_minus_strides;

        for (int d = 0; d < PARALLELISM_PW_OFMS; d++)
        {
            for (int h = 0; h < MAX_FILTER_MINUS_STRIDES; h++)
            {
                if (h >= filter_minus_strides)
                {
                    break;
                }
                for (int o_w = 0; o_w < layer_ifms_width; o_w += PARALLELISM_PW_W)
                {
                    int current_write_offset_in_overlap_buffer = write_offset_in_overlap_buffer +
                                                                 d * filter_minus_strides * num_of_tiles_in_w +
                                                                 (h * layer_ifms_width + o_w) / PARALLELISM_PW_W;
                    int current_read_offset_in_overlap_buffer = read_offset_in_overlap_buffer +
                                                                d * filter_minus_strides * num_of_tiles_in_w +
                                                                (h * layer_ifms_width + o_w) / PARALLELISM_PW_W;
                    for (int w = 0; w < PARALLELISM_PW_W; w++)
                    {
                        //#pragma HLS UNROLL
                        if (starting_h != 0)
                        {
                            dw_channels_tile[d][h][w + o_w + padding_left] =
                                dw_pipe_overlap_buffer[current_read_offset_in_overlap_buffer]
                                                      [w];
                        }
                        dw_pipe_overlap_buffer[current_write_offset_in_overlap_buffer][w] =
                            dw_channels_tile[d][write_offset_h_in_channels_tile + h][o_w + w + padding_left];
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

    if (starting_d >= 0)
    {
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
}

void pipelined_engines::dw_conv_engine(
    dw_weights_dt weights[DW_TILE_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE],
    fms_dt channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
    dw_pss_dt result_tile[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_ofms_width = layer_specs_struct.layer_ofm_width;
    const int layer_d = layer_specs_struct.layer_depth;
    const int filter_dim = layer_specs_struct.filter_size;
    const int strides = layer_specs_struct.strides;

dw_conv_engine:
    for (int o_w = 0; o_w < layer_ofms_width; o_w += PARALLELISM_DW_W)
    {
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
                        for (int w = 0; w < PARALLELISM_DW_W; w++)
                        {
#pragma HLS UNROLL
                            if (c_w >= filter_dim || c_h >= filter_dim || h >= MAX_PW_BUFFER_HEIGHT / strides)
                            {
                                break;
                            }
                            if (c_h == 0 && c_w == 0)
                            {
                                result_tile[d][h][w + o_w] =
                                    channels_tile[d][h * strides + c_h]
                                                 [(o_w + w) * strides + c_w] *
                                    weights[d][c_h * filter_dim + c_w];
                            }
                            else
                            {
                                result_tile[d][h][w + o_w] +=
                                    channels_tile[d][h * strides + c_h]
                                                 [(o_w + w) * strides + c_w] *
                                    weights[d][c_h * filter_dim + c_w];
                            }
                        }
                    }
                }
            }
        }
    }
}

void pipelined_engines::dw_normalize_and_write_back_result_tile(dw_pss_dt result_tile[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                                fms_dt result[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                                const fms_quantization_scheme normalization_buffer[],
                                                                const int starting_d,
                                                                const int h_offset_in_result,
                                                                layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    if (starting_d >= 0)
    {
        const int layer_ofms_width = layer_specs_struct.layer_ofm_width;
        const int strides = layer_specs_struct.strides;
        const int ifms_d = layer_specs_struct.layer_depth;
        const int layer_relu = layer_specs_struct.layer_activation;

        for (int h = 0; h < MAX_PW_BUFFER_HEIGHT; h++)
        {
            if (h + h_offset_in_result >= MAX_PW_BUFFER_HEIGHT)
            {
                break;
            }
            for (int o_w = 0; o_w < layer_ofms_width; o_w += PARALLELISM_DW_W)
            {
                for (int d = 0; d < DW_TILE_DEPTH; d++)
                {
                    //#pragma HLS PIPELINE
                    for (int w = 0; w < PARALLELISM_DW_W; w++)
                    {
                        //#pragma HLS UNROLL
                        result[starting_d + d][h + h_offset_in_result][o_w + w] = dw_relu_norm(
                            result_tile[d][h][o_w + w], normalization_buffer[d],
                            layer_relu);
                    }
                }
            }
        }
    }
}

void pipelined_engines::pw_dw_conv(const weights_dt pw_weights[],
                                   const dw_weights_dt weights[][3 * 3],
                                   fms_dt channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                   fms_dt result[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                   fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][MAX_PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
                                   fms_dt dw_pipe_overlap_buffer[][DW_PIPE_OVERLAP_BUFFER_WIDTH],
                                   fms_dt dw_channels_tile[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
                                   fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][MAX_DW_BUFFER_HEIGHT][MAX_DW_BUFFER_WIDTH],
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

    pss_dt pw_engine_result_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH];
    pss_dt pw_engine_result_tile_copy[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][MAX_PW_BUFFER_WIDTH];

    // fms_dt normalized_tile[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W];
    // fms_dt normalized_tile_copy[PARALLELISM_PW_OFMS][PARALLELISM_PW_H][PARALLELISM_PW_W];

    dw_pss_dt dw_result_tile[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];
    dw_pss_dt dw_result_tile_copy[DW_TILE_DEPTH][MAX_PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH];

    load_pw_weights(pw_weights,
                    weights_tile,
                    0,
                    pw_layer_specs_struct);

    int prev_d = 0;
    int prev_prev_d = 0;
    int prev_prev_prev_d = 0;
    int next_d = PARALLELISM_PW_OFMS;

    if (fused_pw_dw)
    {
        for (int d = 0; d < num_of_filters; d += PARALLELISM_PW_OFMS)
        {
            prev_d = d - PARALLELISM_PW_OFMS;
            prev_prev_d = prev_d - PARALLELISM_PW_OFMS;
            prev_prev_prev_d = prev_prev_d - PARALLELISM_PW_OFMS;
            next_d = d + PARALLELISM_PW_OFMS;
            if ((d / PARALLELISM_PW_OFMS) % 2 == 0)
            {
                dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                        result,
                                                        dw_normalization_buffer_copy,
                                                        prev_prev_prev_d,
                                                        h_offset_in_result,
                                                        dw_layer_specs_struct);
                //###############################
                fill_fused_scales_and_zps_buffer(fused_scales,
                                                 fused_scales_log_2_shifts,
                                                 relu_6_fused_scales,
                                                 fused_zero_points,
                                                 dw_normalization_buffer,
                                                 prev_prev_d, // starting_d
                                                 pipe_layers_fused_parameters_offsets[dw_layer],
                                                 DW_TILE_DEPTH,
                                                 dw_layer_specs_struct);
                //###############################
                dw_conv_engine(
                    dw_weights_tile,
                    dw_channels_tile,
                    dw_result_tile,
                    dw_layer_specs_struct);
                //###############################
                pipelined_engines::fill_dw_weights_tile(weights,
                                                        dw_weights_tile_copy,
                                                        prev_d, dw_layers_weights_offsets[dw_layer]);
                //###############################
                pw_normalize_engine_result(pw_engine_result_tile_copy,
                                           dw_channels_tile_copy,
                                           result,
                                           tmp_channels,
                                           pw_normalization_buffer_copy,
                                           prev_d,
                                           starting_h,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                    dw_channels_tile_copy,
                                                    prev_d,
                                                    prev_d,
                                                    starting_h,
                                                    dw_layer_specs_struct);
                //###############################
                pw_conv_engine(weights_tile,
                               channels,
                               pw_engine_result_tile,
                               d,
                               pw_layer_specs_struct);
                //###############################
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
                                weights_tile_copy,
                                next_d,
                                pw_layer_specs_struct);
            }
            else
            {
                dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                        result,
                                                        dw_normalization_buffer,
                                                        prev_prev_prev_d,
                                                        h_offset_in_result,
                                                        dw_layer_specs_struct);
                //###############################
                fill_fused_scales_and_zps_buffer(fused_scales,
                                                 fused_scales_log_2_shifts,
                                                 relu_6_fused_scales,
                                                 fused_zero_points,
                                                 dw_normalization_buffer_copy,
                                                 prev_prev_d, // starting_d
                                                 pipe_layers_fused_parameters_offsets[dw_layer],
                                                 DW_TILE_DEPTH,
                                                 dw_layer_specs_struct);
                //###############################
                dw_conv_engine(
                    dw_weights_tile_copy,
                    dw_channels_tile_copy,
                    dw_result_tile_copy,
                    dw_layer_specs_struct);
                //###############################
                pipelined_engines::fill_dw_weights_tile(weights,
                                                        dw_weights_tile,
                                                        prev_d, dw_layers_weights_offsets[dw_layer]);
                //###############################
                pw_normalize_engine_result(pw_engine_result_tile,
                                           dw_channels_tile,
                                           result,
                                           tmp_channels,
                                           pw_normalization_buffer,
                                           prev_d,
                                           starting_h,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                    dw_channels_tile,
                                                    prev_d,
                                                    prev_d,
                                                    starting_h,
                                                    dw_layer_specs_struct);
                //###############################
                pw_conv_engine(weights_tile_copy,
                               channels,
                               pw_engine_result_tile_copy,
                               d,
                               pw_layer_specs_struct);
                //###############################
                fill_fused_scales_and_zps_buffer(fused_scales,
                                                 fused_scales_log_2_shifts,
                                                 relu_6_fused_scales,
                                                 fused_zero_points,
                                                 pw_normalization_buffer_copy,
                                                 d, // starting_d
                                                 pipe_layers_fused_parameters_offsets[pw_layer],
                                                 PARALLELISM_PW_OFMS,
                                                 pw_layer_specs_struct);

                load_pw_weights(pw_weights,
                                weights_tile,
                                next_d,
                                pw_layer_specs_struct);
            }
        }

        if (((num_of_filters / PARALLELISM_PW_OFMS) - 1) % 2)
        {
            dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                    result,
                                                    dw_normalization_buffer_copy,
                                                    num_of_filters - 3 * PARALLELISM_PW_OFMS,
                                                    h_offset_in_result,
                                                    dw_layer_specs_struct);
            //###############################
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts,
                                             relu_6_fused_scales,
                                             fused_zero_points,
                                             dw_normalization_buffer,
                                             num_of_filters - 2 * PARALLELISM_PW_OFMS, // starting_d
                                             pipe_layers_fused_parameters_offsets[dw_layer],
                                             DW_TILE_DEPTH,
                                             dw_layer_specs_struct);
            //###############################
            dw_conv_engine(
                dw_weights_tile,
                dw_channels_tile,
                dw_result_tile,
                dw_layer_specs_struct);
            //###############################
            pipelined_engines::fill_dw_weights_tile(weights,
                                                    dw_weights_tile_copy,
                                                    num_of_filters - PARALLELISM_PW_OFMS, dw_layers_weights_offsets[dw_layer]);
            //###############################
            pw_normalize_engine_result(pw_engine_result_tile_copy,
                                       dw_channels_tile_copy,
                                       result,
                                       tmp_channels,
                                       pw_normalization_buffer_copy,
                                       num_of_filters - PARALLELISM_PW_OFMS,
                                       starting_h,
                                       fused_pw_dw,
                                       pw_layer_specs_struct,
                                       dw_layer_specs_struct);
            write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                dw_channels_tile_copy,
                                                num_of_filters - PARALLELISM_PW_OFMS,
                                                num_of_filters - PARALLELISM_PW_OFMS,
                                                starting_h,
                                                dw_layer_specs_struct);
            //#######################################################################################
            dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                    result,
                                                    dw_normalization_buffer,
                                                    num_of_filters - 2 * PARALLELISM_PW_OFMS,
                                                    h_offset_in_result,
                                                    dw_layer_specs_struct);
            //###############################
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts,
                                             relu_6_fused_scales,
                                             fused_zero_points,
                                             dw_normalization_buffer,
                                             num_of_filters - PARALLELISM_PW_OFMS, // starting_d
                                             pipe_layers_fused_parameters_offsets[dw_layer],
                                             DW_TILE_DEPTH,
                                             dw_layer_specs_struct);
            //###############################
            dw_conv_engine(
                dw_weights_tile_copy,
                dw_channels_tile_copy,
                dw_result_tile,
                dw_layer_specs_struct);
            //#######################################################################################
            dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                    result,
                                                    dw_normalization_buffer,
                                                    num_of_filters - PARALLELISM_PW_OFMS,
                                                    h_offset_in_result,
                                                    dw_layer_specs_struct);
        }
        else
        {
            dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                    result,
                                                    dw_normalization_buffer,
                                                    num_of_filters - 3 * PARALLELISM_PW_OFMS,
                                                    h_offset_in_result,
                                                    dw_layer_specs_struct);
            //###############################
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts,
                                             relu_6_fused_scales,
                                             fused_zero_points,
                                             dw_normalization_buffer_copy,
                                             num_of_filters - 2 * PARALLELISM_PW_OFMS, // starting_d
                                             pipe_layers_fused_parameters_offsets[dw_layer],
                                             DW_TILE_DEPTH,
                                             dw_layer_specs_struct);
            //###############################
            dw_conv_engine(
                dw_weights_tile_copy,
                dw_channels_tile_copy,
                dw_result_tile_copy,
                dw_layer_specs_struct);
            //###############################
            pipelined_engines::fill_dw_weights_tile(weights,
                                                    dw_weights_tile,
                                                    num_of_filters - PARALLELISM_PW_OFMS, dw_layers_weights_offsets[dw_layer]);
            //###############################
            pw_normalize_engine_result(pw_engine_result_tile,
                                       dw_channels_tile,
                                       result,
                                       tmp_channels,
                                       pw_normalization_buffer,
                                       num_of_filters - PARALLELISM_PW_OFMS,
                                       starting_h,
                                       fused_pw_dw,
                                       pw_layer_specs_struct,
                                       dw_layer_specs_struct);
            write_next_overlap_and_read_current(dw_pipe_overlap_buffer,
                                                dw_channels_tile,
                                                num_of_filters - PARALLELISM_PW_OFMS,
                                                num_of_filters - PARALLELISM_PW_OFMS,
                                                starting_h,
                                                dw_layer_specs_struct);
            //#######################################################################################
            dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                    result,
                                                    dw_normalization_buffer_copy,
                                                    num_of_filters - 2 * PARALLELISM_PW_OFMS,
                                                    h_offset_in_result,
                                                    dw_layer_specs_struct);
            //###############################
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts,
                                             relu_6_fused_scales,
                                             fused_zero_points,
                                             dw_normalization_buffer,
                                             num_of_filters - PARALLELISM_PW_OFMS, // starting_d
                                             pipe_layers_fused_parameters_offsets[dw_layer],
                                             DW_TILE_DEPTH,
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
                                                    num_of_filters - PARALLELISM_PW_OFMS,
                                                    h_offset_in_result,
                                                    dw_layer_specs_struct);
        }
    }
    else
    {
        for (int d = 0; d < num_of_filters; d += PARALLELISM_PW_OFMS)
        {
            prev_d = d - PARALLELISM_PW_OFMS;
            prev_prev_d = prev_d - PARALLELISM_PW_OFMS;
            prev_prev_prev_d = prev_prev_d - PARALLELISM_PW_OFMS;
            next_d = d + PARALLELISM_PW_OFMS;
            if ((d / PARALLELISM_PW_OFMS) % 2 == 0)
            {
                pw_normalize_engine_result(pw_engine_result_tile_copy,
                                           dw_channels_tile_copy,
                                           result,
                                           tmp_channels,
                                           pw_normalization_buffer_copy,
                                           prev_d,
                                           starting_h,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                //###############################
                pw_conv_engine(weights_tile,
                               channels,
                               pw_engine_result_tile,
                               d,
                               pw_layer_specs_struct);
                //###############################
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
                                weights_tile_copy,
                                next_d,
                                pw_layer_specs_struct);
            }
            else
            {
                //###############################
                pw_normalize_engine_result(pw_engine_result_tile,
                                           dw_channels_tile,
                                           result,
                                           tmp_channels,
                                           pw_normalization_buffer,
                                           prev_d,
                                           starting_h,
                                           fused_pw_dw,
                                           pw_layer_specs_struct,
                                           dw_layer_specs_struct);
                //###############################
                pw_conv_engine(weights_tile_copy,
                               channels,
                               pw_engine_result_tile_copy,
                               d,
                               pw_layer_specs_struct);
                //###############################
                fill_fused_scales_and_zps_buffer(fused_scales,
                                                 fused_scales_log_2_shifts,
                                                 relu_6_fused_scales,
                                                 fused_zero_points,
                                                 pw_normalization_buffer_copy,
                                                 d, // starting_d
                                                 pipe_layers_fused_parameters_offsets[pw_layer],
                                                 PARALLELISM_PW_OFMS,
                                                 pw_layer_specs_struct);

                load_pw_weights(pw_weights,
                                weights_tile,
                                next_d,
                                pw_layer_specs_struct);
            }
        }
        if (((num_of_filters / PARALLELISM_PW_OFMS) - 1) % 2)
        {
            pw_normalize_engine_result(pw_engine_result_tile_copy,
                                       dw_channels_tile_copy,
                                       result,
                                       tmp_channels,
                                       pw_normalization_buffer_copy,
                                       num_of_filters - PARALLELISM_PW_OFMS,
                                       starting_h,
                                       fused_pw_dw,
                                       pw_layer_specs_struct,
                                       dw_layer_specs_struct);
        }
        else
        {
            pw_normalize_engine_result(pw_engine_result_tile,
                                       dw_channels_tile,
                                       result,
                                       tmp_channels,
                                       pw_normalization_buffer,
                                       num_of_filters - PARALLELISM_PW_OFMS,
                                       starting_h,
                                       fused_pw_dw,
                                       pw_layer_specs_struct,
                                       dw_layer_specs_struct);
        }
    }
}

#endif

#endif
