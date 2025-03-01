#include "pipelined_engines.h"

#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE && !ONLY_SEML

using namespace pipelined_engines;
#ifndef PIPELINED_PW_CONV
#define PIPELINED_PW_CONV

fms_dt pipelined_engines::first_dw_layer_kernel(fms_dt slice_of_communication_buffer_intra[layer_2_dw_filter_dim * layer_2_dw_filter_dim],
                                                const dw_weights_dt weights[][layer_2_dw_filter_dim * layer_2_dw_filter_dim],
                                                const int filter_dim,
                                                const int current_d,
                                                const fms_quantization_scheme normalization,
                                                const int layer_activation)
{
#pragma HLS INLINE

    dw_pss_dt pss = 0;
    fms_dt result;
dw_kernel:
    for (int c_h = 0; c_h < filter_dim; c_h++)
    {
#pragma HLS UNROLL
        for (int c_w = 0; c_w < filter_dim; c_w++)
        {
#pragma HLS UNROLL
            pss += slice_of_communication_buffer_intra[c_h * filter_dim + c_w] * weights[current_d][c_h * filter_dim + c_w];
        }
    }

    result = dw_relu_norm_v2(pss, normalization.fused_zero_point, normalization.ofm_zero_point,
                             normalization.fused_scales, normalization.relu_6_fused_scale, RELU6);

    return result;
}

fms_dt pipelined_engines::first_layer_conv_kernel(fms_dt ifms_buffer[FIRST_CONV_LAYER_BUFFER_SIZE],
                                                  const layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth]
                                                                                    [first_conv_layer_filter_dim]
                                                                                    [first_conv_layer_filter_dim],
                                                  const int starting_filter,
                                                  const int starting_h,
                                                  const int starting_w,
                                                  const fms_quantization_scheme normalization)
{
#pragma HLS INLINE

    pss_dt pss = 0;
    fms_dt result = 0;

    const int ifms_buffer_w = first_conv_layer_specs.filter_size;
    const int ifms_buffer_hw = ifms_buffer_w * first_conv_layer_specs.filter_size;

conv_kernel:
    for (int d = 0; d < input_image_depth; d++)
    {
#pragma HLS UNROLL
        for (int c_h = 0; c_h < first_conv_layer_filter_dim; c_h++)
        {
#pragma HLS UNROLL
            for (int c_w = 0; c_w < first_conv_layer_filter_dim; c_w++)
            {
#pragma HLS UNROLL
                // if (starting_h == 0 && starting_w == 0 && d == 0 && starting_filter == 0)
                // {
                //     printf("%d * %d \n", (int)ifms_buffer[d * ifms_buffer_hw + c_h * ifms_buffer_w + c_w], (int)weights_1[starting_filter][d][c_h][c_w]);
                // }
                pss += ifms_buffer[d * ifms_buffer_hw + c_h * ifms_buffer_w + c_w] *
                       weights_1[starting_filter][d][c_h][c_w];
            }
        }
    }

    result = conv_relu_norm_v2(pss, normalization.fused_zero_point, normalization.ofm_zero_point, normalization.fused_scales,
                               normalization.relu_6_fused_scale, first_conv_layer_specs.layer_activation);
    // if (starting_h == 0 && starting_w == 0 && starting_filter == 0)
    // {
    // 							pss += normalization.ofm_zero_point;
    // 							printf("%d >> ", (int) pss);

    // 							pss_f_dt scaled_pss = pss * normalization.fused_scales;
    // 							printf("%f >> ", (float) scaled_pss);
    // 							printf("%f >> ", (float) normalization.fused_scales);
    // 							if (normalization.relu_6_fused_scale == 6
    // 									&& scaled_pss > normalization.relu_6_fused_scale) {
    // 								scaled_pss = normalization.relu_6_fused_scale;
    // 							}

    // 							scaled_pss += normalization.ofm_zero_point;
    // 							printf("%f >> ", (float) scaled_pss);
    // 							scaled_pss += quant_half - (scaled_pss < 0);
    // 							printf("%f \n ", (float) scaled_pss);
    //     printf("%d\n", (int)result);
    // }

    return result;
}

void pipelined_engines::padd_left_dw_channels_tile(
    fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const fms_dt current_layer_ifms_zero_point =
        layer_specs_struct.layer_ifms_zero_point;
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
                    dw_channels_tile_copy[d][h][w] =
                        current_layer_ifms_zero_point;
                }
            }
        }
    }
}

void pipelined_engines::fill_fused_scales_and_zps_buffer(
    const fused_scales_dt fused_scales[],
    const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
    const relu_6_fused_scales_dt relu_6_fused_scales[],
    const biases_dt fused_zero_points[],
    fms_quantization_scheme normalization_buffer[], int starting_d,
    const int current_layer_fused_parameters_offset, const int buffer_size,
    const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int absolute_current_layer_fused_parameters_offset =
        current_layer_fused_parameters_offset + starting_d;
    const int layer_index = layer_specs_struct.layer_index;
    for (int i = 0; i < buffer_size; i++)
    {
        normalization_buffer[i].ofm_scale = layer_specs_struct.layer_ofms_scale;
        normalization_buffer[i].ofm_zero_point =
            layer_specs_struct.layer_ofms_zero_point;
        normalization_buffer[i].fused_scales =
            fused_scales[absolute_current_layer_fused_parameters_offset + i];
        // normalization_buffer[i].fused_scales_log_2_shift =
        //     fused_scales_log_2_shifts[absolute_current_layer_fused_parameters_offset + i];
        normalization_buffer[i].relu_6_fused_scale =
            relu_6_fused_scales[layer_index];
        normalization_buffer[i].fused_zero_point =
            fused_zero_points[absolute_current_layer_fused_parameters_offset + i];
    }
}

void pipelined_engines::load_pw_weights(
    weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
    weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
    const int starting_filter, layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_depth = layer_specs_struct.layer_depth;
    const int layer_num_filters = layer_specs_struct.layer_num_fils;
    const int filling_weights_offset =
        (layer_specs_struct.layer_weights_offset_on_chip + starting_filter * layer_depth) / ON_CHIP_WEIGHTS_PORTS;

    for (int filter_g_index = 0;
         filter_g_index < PARALLELISM_PW_OFMS / ON_CHIP_WEIGHTS_PORTS;
         filter_g_index++)
    {
        int current_filling_weights_offset = filling_weights_offset + filter_g_index * layer_depth;

        for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
        {
#pragma HLS PIPELINE
            if (d >= layer_depth)
            {
                break;
            }
            for (int filter_index = 0; filter_index < ON_CHIP_WEIGHTS_PORTS;
                 filter_index++)
            {
#pragma HLS UNROLL
                if (filter_g_index * ON_CHIP_WEIGHTS_PORTS + filter_index + starting_filter < layer_num_filters)
                {
                    weights_tile[filter_g_index * ON_CHIP_WEIGHTS_PORTS + filter_index][d] =
                        on_chip_weights[current_filling_weights_offset + d][filter_index]; // TODO
                }
            }
        }
    }
}

void pipelined_engines::pw_conv_engine(
    weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH],
    fms_dt pw_engine_input_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
    pss_dt engine_result[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
    const int starting_filter,
    const layer_specs layer_specs_struct,
    const int model_configs_list[])
{
#pragma HLS INLINE off

    const int layer_depth = layer_specs_struct.layer_depth;

    const int model_configs_list_limit_i = model_configs_list[2 * layer_specs_struct.layer_index];

pw_engine_ls:
    for (int o_w = 0; o_w < PW_BUFFER_WIDTH; o_w +=
                                             PARALLELISM_PW_W)
    {

        for (int d = 0; d < MAX_PW_BUFFER_DEPTH; d++)
        {
#pragma HLS PIPELINE
            if (d >= layer_depth ||
                (model_configs_list_limit_i != 0 && d >= model_configs_list_limit_i))
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
                        if (d == 0)
                        {
                            engine_result[f][h][o_w + w] = weights_tile[f][d] * pw_engine_input_buffer[d][h][o_w + w];
                        }
                        else
                        {
                            engine_result[f][h][o_w + w] += weights_tile[f][d] * pw_engine_input_buffer[d][h][o_w + w];
                        }
                    }
                }
            }
        }
    }
}

void first_pass_pw_normalize_engine_result(
    fms_dt dw_pipe_overlap_buffer[][2][2][DW_PIPE_OVERLAP_BUFFER_WIDTH],
    pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
    fms_dt normalized_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
    const fms_quantization_scheme normalization_buffer[],
    const int starting_d, const int starting_h, const bool fused_pw_dw,
    const layer_specs layer_specs_struct,
    const layer_specs dw_layer_specs_struct,
    const int read_end)
{
#pragma HLS INLINE off

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
    const int d_offset_in_overlap_buffer = starting_d * filter_dim_minus_strides * num_of_tiles_in_w + (dw_layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);

    const fms_dt dw_layer_ifms_zero_point =
        dw_layer_specs_struct.layer_ifms_zero_point;

    for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
    {
        for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
        {
#pragma HLS PIPELINE

            fms_quantization_scheme current_normaliztion_scheme =
                normalization_buffer[f];
            int h_in_overlap_buffer = h - (PW_BUFFER_HEIGHT - filter_dim_minus_strides);
            int current_read_offset_in_overlap_buffer =
                d_offset_in_overlap_buffer + f * filter_dim_minus_strides * num_of_tiles_in_w;
            int current_write_offset_in_overlap_buffer =
                d_offset_in_overlap_buffer + f * filter_dim_minus_strides * num_of_tiles_in_w;
            for (int w = 0; w < MAX_FILTER_MINUS_STRIDES; w++)
            {
                if (w < padding_left)
                {
                    continue;
                }
                fms_dt scaled_val = pw_relu_norm_6_v2(
                    engine_result_tile[f][h][w - padding_left],
                    current_normaliztion_scheme.fused_zero_point, current_normaliztion_scheme.ofm_zero_point,
                    current_normaliztion_scheme.fused_scales, current_normaliztion_scheme.relu_6_fused_scale,
                    layer_relu);

                if (h < filter_dim_minus_strides)
                {
                    normalized_tile[f][h][w] =
                        dw_pipe_overlap_buffer[current_read_offset_in_overlap_buffer][read_end][h][w - padding_left];
                }
                else if (h_in_overlap_buffer >= 0)
                {
                    dw_pipe_overlap_buffer[current_write_offset_in_overlap_buffer][1 - read_end][h_in_overlap_buffer][w - padding_left] = scaled_val;
                }
                if (starting_h + h < layer_ifms_height && (DW_BUFFER_HEIGHT - filter_dim != h + write_offset_h_in_normalized_tile || padding_top == 0 || starting_h != 0))
                { // first time, we start from the last filter_dim rows, hence the row DW_BUFFER_HEIGHT - filter_dim should pe a padding
                    normalized_tile[f][h + filter_dim_minus_strides][w] =
                        scaled_val;
                }
                else
                {
                    normalized_tile[f][h + filter_dim_minus_strides][w] =
                        dw_layer_ifms_zero_point;
                }
            }
        }
    }
}

void pipelined_engines::pw_only_normalize_engine_result(
    pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
    fms_dt normalized_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
    const fms_quantization_scheme normalization_buffer[],
    const int starting_d, const int starting_h, const int starting_w,
    const int model_configs_list_limit,
    const layer_specs layer_specs_struct,
    const layer_specs dw_layer_specs_struct)
{
#pragma HLS INLINE off

    scales_dt skip_connection_other_layer_scale =
        layer_specs_struct.skip_connection_other_layer_scale;
    biases_dt skip_connection_other_layer_zero_point =
        layer_specs_struct.skip_connection_other_layer_zero_point;

    rec_scales_dt add_layer_scale_reciprocal =
        layer_specs_struct.add_layer_scale_reciprocal;
    biases_dt add_layer_zero_point = layer_specs_struct.add_layer_zero_point;
    const int layer_relu = layer_specs_struct.layer_activation;

    const int layer_num_filters = layer_specs_struct.layer_num_fils;

    for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
    {
#pragma HLS PIPELINE
        if (starting_d + f >= layer_num_filters ||
            (model_configs_list_limit != 0 && starting_d + f >= model_configs_list_limit))
        {
            break;
        }
        fms_quantization_scheme current_normaliztion_scheme =
            normalization_buffer[f];
        for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < PW_BUFFER_WIDTH; w++)
            {
#pragma HLS UNROLL
                pss_dt tmp_pss = engine_result_tile[f][h][w];
                fms_dt normalized_val;
                if (layer_specs_struct.fused_with_add == 0)
                {
                    normalized_val = pw_relu_norm_6_v2(
                        tmp_pss,
                        current_normaliztion_scheme.fused_zero_point, current_normaliztion_scheme.ofm_zero_point,
                        current_normaliztion_scheme.fused_scales, current_normaliztion_scheme.relu_6_fused_scale,
                        layer_relu);
                }
                else
                {
                    pss_f_dt tmp_channels_scaled_val =
                        skip_connection_other_layer_scale * (tmp_channels[starting_d + f][h][starting_w + w] - skip_connection_other_layer_zero_point);
                    pss_f_dt scaled_tmp = pw_relu_norm_no_q_no_relu_v2(
                        tmp_pss, current_normaliztion_scheme.fused_zero_point, current_normaliztion_scheme.fused_scales,
                        current_normaliztion_scheme.ofm_scale);

                    pss_f_dt addition_result = (scaled_tmp + tmp_channels_scaled_val) * add_layer_scale_reciprocal + add_layer_zero_point;
                    addition_result = addition_result + quant_half - (addition_result < 0);
                    normalized_val = clamp(addition_result);
                }
                normalized_tile[f][h][w] = normalized_val;
            }
        }
    }
}

void pipelined_engines::pw_normalize_engine_result(
    pss_dt engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
    fms_dt normalized_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
    fms_dt dw_horizontal_overlap_buffer[DW_TILE_DEPTH][MAX_FILTER_MINUS_STRIDES][PW_BUFFER_WIDTH],
    const fms_quantization_scheme normalization_buffer[],
    const int starting_d, const int starting_h, const int starting_w,
    const int model_configs_list_limit,
    const layer_specs layer_specs_struct,
    const layer_specs dw_layer_specs_struct)
{
#pragma HLS INLINE off

    if (starting_w >= 0)
    {
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int layer_ifms_height = layer_specs_struct.layer_ifm_height;
        const int layer_num_filters = layer_specs_struct.layer_num_fils;
        const int layer_relu = layer_specs_struct.layer_activation;

        const int strides = dw_layer_specs_struct.strides;
        const int filter_dim = dw_layer_specs_struct.filter_size;
        const int padding_left = dw_layer_specs_struct.padding_left;
        const fms_dt dw_layer_ifms_zero_point =
            dw_layer_specs_struct.layer_ifms_zero_point;
        const int padding_top = dw_layer_specs_struct.padding_top;

        const int useful_cols_in_channels_tile = DW_BUFFER_WIDTH - (DW_BUFFER_HEIGHT - filter_dim) % strides;

        const int filter_minus_strides = filter_dim - strides;

        for (int f = 0; f < PARALLELISM_PW_OFMS; f++)
        {
#pragma HLS PIPELINE
            if (starting_d + f >= layer_num_filters ||
                (model_configs_list_limit != 0 && starting_d + f >= model_configs_list_limit))
            {
                break;
            }
            fms_quantization_scheme current_normaliztion_scheme =
                normalization_buffer[f];
            for (int h = 0; h < DW_BUFFER_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < MAX_FILTER_MINUS_STRIDES; w++)
                {
#pragma HLS UNROLL
                    if (w < filter_minus_strides)
                    {
                        if (starting_w > 0)
                        {
                            normalized_tile[f][h][w] =
                                dw_vertical_overlap_buffer[f][h][w];
                        }
                    }
                }
                if (h >= filter_minus_strides)
                {
                    for (int w = MAX_FILTER_MINUS_STRIDES; w < DW_BUFFER_WIDTH; w++)
                    {
#pragma HLS UNROLL
                        int current_w = w - MAX_FILTER_MINUS_STRIDES + filter_minus_strides;

                        fms_dt normalized_val;
                        pss_dt tmp_pss = engine_result_tile[f][h - filter_minus_strides][current_w - filter_minus_strides];
                        if (starting_h + (h - filter_minus_strides) < layer_ifms_height &&
                            (DW_BUFFER_HEIGHT - filter_dim != h || padding_top == 0 || starting_h != 0) &&
                            starting_w + current_w < layer_ifms_width + padding_left)
                        {
                            normalized_val = pw_relu_norm_6_v2(
                                tmp_pss,
                                current_normaliztion_scheme.fused_zero_point, current_normaliztion_scheme.ofm_zero_point,
                                current_normaliztion_scheme.fused_scales, current_normaliztion_scheme.relu_6_fused_scale,
                                layer_relu);
                        }
                        else
                        {
                            normalized_val = dw_layer_ifms_zero_point;
                        }
                        if (current_w >= PW_BUFFER_WIDTH)
                        {
                            dw_vertical_overlap_buffer[f][h][current_w - PW_BUFFER_WIDTH] = normalized_val;
                        }
                        if (h >= DW_BUFFER_HEIGHT - MAX_FILTER_MINUS_STRIDES && current_w - filter_minus_strides < PW_BUFFER_WIDTH)
                        {
                            dw_horizontal_overlap_buffer[f]
                                                        [h - (DW_BUFFER_HEIGHT - MAX_FILTER_MINUS_STRIDES)][current_w - filter_minus_strides] = normalized_val;
                        }
                        normalized_tile[f][h][current_w] = normalized_val;
                    }
                }
            }
        }
    }
}

void pipelined_engines::write_next_overlap_and_read_current_only_p2(
    fms_dt dw_pipe_overlap_buffer[][2][2][DW_PIPE_OVERLAP_BUFFER_WIDTH],
    fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
    fms_dt dw_horizontal_overlap_buffer[DW_TILE_DEPTH][MAX_FILTER_MINUS_STRIDES][PW_BUFFER_WIDTH],
    fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    const int starting_d, const int starting_h, const int starting_w,
    const int model_configs_list_limit,
    layer_specs layer_specs_struct,
    const int read_end)
{
#pragma HLS INLINE off

    if (starting_w >= 0)
    {
        const fms_dt dw_layer_ifms_zero_point =
            layer_specs_struct.layer_ifms_zero_point;
        const int layer_depth = layer_specs_struct.layer_depth;
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int strides = layer_specs_struct.strides;
        const int filter_dim = layer_specs_struct.filter_size;
        const int filter_minus_strides = filter_dim - strides;
        const int padding_left = layer_specs_struct.padding_left;

        const int num_of_tiles_in_w = (layer_ifms_width / DW_PIPE_OVERLAP_BUFFER_WIDTH);
        const int d_offset_in_overlap_buffer = starting_d * filter_minus_strides * num_of_tiles_in_w + (layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);

        const int useful_rows_in_channels_tile = DW_BUFFER_HEIGHT - (DW_BUFFER_HEIGHT - filter_dim) % strides;
        const int write_offset_h_in_channels_tile = useful_rows_in_channels_tile - filter_minus_strides;

        const int inner_offset_w =
            starting_w < DW_PIPE_OVERLAP_BUFFER_WIDTH ? starting_w : starting_w - DW_PIPE_OVERLAP_BUFFER_WIDTH; //%

        const int columns_filled_by_pw_in_the_firts_pass = filter_minus_strides - padding_left;

        for (int d = 0; d < PARALLELISM_PW_OFMS; d++)
        {
#pragma HLS PIPELINE
            fms_dt first_read_val;
            fms_dt second_read_val;
            if (d + starting_d >= layer_depth ||
                (model_configs_list_limit != 0 && starting_d + d >= model_configs_list_limit))
            {
                break;
            }
            for (int h = 0; h < MAX_FILTER_MINUS_STRIDES; h++)
            {
#pragma HLS UNROLL
                if (h >= filter_minus_strides)
                {
                    break;
                }

                int current_offset_in_overlap_buffer =
                    d_offset_in_overlap_buffer + d * filter_minus_strides * num_of_tiles_in_w + starting_w / DW_PIPE_OVERLAP_BUFFER_WIDTH;

                if (read_end == 0) // this is because the stupid HLS synthesis is unable to recognaize that there is no dependence even with the dependence pragma
                {
                    if (starting_h != 0)
                    {
                        first_read_val =
                            dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][0][h][inner_offset_w + columns_filled_by_pw_in_the_firts_pass];
                    }
                    dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][1][h][inner_offset_w + columns_filled_by_pw_in_the_firts_pass] =
                        dw_horizontal_overlap_buffer[d][h][0];

                    //***********************************************

                    if (1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass == DW_PIPE_OVERLAP_BUFFER_WIDTH && 1 + starting_w + columns_filled_by_pw_in_the_firts_pass < layer_ifms_width)
                    {
                        if (starting_h != 0)
                        {
                            second_read_val =
                                dw_pipe_overlap_buffer[current_offset_in_overlap_buffer + 1][0][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass - DW_PIPE_OVERLAP_BUFFER_WIDTH];
                        }
                        dw_pipe_overlap_buffer[current_offset_in_overlap_buffer + 1][1][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass - DW_PIPE_OVERLAP_BUFFER_WIDTH] =
                            dw_horizontal_overlap_buffer[d][h][1];
                    }
                    else if (1 + starting_w + columns_filled_by_pw_in_the_firts_pass < layer_ifms_width)
                    {
                        if (starting_h != 0)
                        {
                            second_read_val =
                                dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][0][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass];
                        }
                        dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][1][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass] =
                            dw_horizontal_overlap_buffer[d][h][1];
                    }
                    else
                    {
                        second_read_val = dw_layer_ifms_zero_point;
                    }
                }
                else
                {
                    if (starting_h != 0)
                    {
                        first_read_val =
                            dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][1][h][inner_offset_w + columns_filled_by_pw_in_the_firts_pass];
                    }
                    dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][0][h][inner_offset_w + columns_filled_by_pw_in_the_firts_pass] =
                        dw_horizontal_overlap_buffer[d][h][0];

                    //***********************************************

                    if (1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass == DW_PIPE_OVERLAP_BUFFER_WIDTH && 1 + starting_w + columns_filled_by_pw_in_the_firts_pass < layer_ifms_width)
                    {
                        if (starting_h != 0)
                        {
                            second_read_val =
                                dw_pipe_overlap_buffer[current_offset_in_overlap_buffer + 1][1][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass - DW_PIPE_OVERLAP_BUFFER_WIDTH];
                        }
                        dw_pipe_overlap_buffer[current_offset_in_overlap_buffer + 1][0][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass - DW_PIPE_OVERLAP_BUFFER_WIDTH] =
                            dw_horizontal_overlap_buffer[d][h][1];
                    }
                    else if (1 + starting_w + columns_filled_by_pw_in_the_firts_pass < layer_ifms_width)
                    {
                        if (starting_h != 0)
                        {
                            second_read_val =
                                dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][1][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass];
                        }
                        dw_pipe_overlap_buffer[current_offset_in_overlap_buffer][0][h][1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass] =
                            dw_horizontal_overlap_buffer[d][h][1];
                    }
                    else
                    {
                        second_read_val = dw_layer_ifms_zero_point;
                    }
                }

                dw_channels_tile[d][h][filter_minus_strides] = first_read_val;

                if (filter_minus_strides == PW_BUFFER_WIDTH)
                {
                    dw_vertical_overlap_buffer[d][h][0] = first_read_val;
                }

                dw_channels_tile[d][h][1 + filter_minus_strides] =
                    second_read_val;

                dw_vertical_overlap_buffer[d][h][1 + filter_minus_strides - PW_BUFFER_WIDTH] = second_read_val;
            }
        }
    }
}

void pipelined_engines::write_next_overlap_and_read_current(
    fms_dt dw_pipe_overlap_buffer[][2][2][DW_PIPE_OVERLAP_BUFFER_WIDTH],
    fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES],
    fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    const int starting_d, const int starting_h, const int starting_w,
    layer_specs layer_specs_struct,
    const int read_end)
{
#pragma HLS INLINE off

    if (starting_w >= 0)
    {
        const fms_dt dw_layer_ifms_zero_point =
            layer_specs_struct.layer_ifms_zero_point;
        const int layer_ifms_width = layer_specs_struct.layer_ifm_width;
        const int strides = layer_specs_struct.strides;
        const int filter_dim = layer_specs_struct.filter_size;
        const int filter_minus_strides = filter_dim - strides;
        const int padding_left = layer_specs_struct.padding_left;

        const int num_of_tiles_in_w = (layer_ifms_width / DW_PIPE_OVERLAP_BUFFER_WIDTH);
        const int d_offset_in_overlap_buffer = starting_d * filter_minus_strides * num_of_tiles_in_w + (layer_specs_struct.dw_ifms_cumulative_width_offset / DW_PIPE_OVERLAP_BUFFER_WIDTH);

        const int useful_rows_in_channels_tile = DW_BUFFER_HEIGHT - (DW_BUFFER_HEIGHT - filter_dim) % strides;
        const int write_offset_h_in_channels_tile = useful_rows_in_channels_tile - filter_minus_strides;

        const int inner_offset_w =
            starting_w < DW_PIPE_OVERLAP_BUFFER_WIDTH ? starting_w : starting_w - DW_PIPE_OVERLAP_BUFFER_WIDTH; //%

        const int columns_filled_by_pw_in_the_firts_pass = filter_minus_strides - padding_left;

        for (int h = 0; h < MAX_FILTER_MINUS_STRIDES; h++)
        {
            if (h >= filter_minus_strides)
            {
                break;
            }
            for (int d = 0; d < PARALLELISM_PW_OFMS; d++)
            {
#pragma HLS PIPELINE

                int current_write_offset_in_overlap_buffer =
                    d_offset_in_overlap_buffer + d * filter_minus_strides * num_of_tiles_in_w + starting_w / DW_PIPE_OVERLAP_BUFFER_WIDTH;
                int current_read_offset_in_overlap_buffer =
                    d_offset_in_overlap_buffer + d * filter_minus_strides * num_of_tiles_in_w + starting_w / DW_PIPE_OVERLAP_BUFFER_WIDTH;
                for (int w = 0; w < PW_BUFFER_WIDTH; w++)
                {
#pragma HLS UNROLL

                    fms_dt read_val;
                    if (w + starting_w + columns_filled_by_pw_in_the_firts_pass < layer_ifms_width)
                    {
                        if (w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass != DW_PIPE_OVERLAP_BUFFER_WIDTH)
                        {
                            if (starting_h != 0)
                            {
                                read_val =
                                    dw_pipe_overlap_buffer[current_read_offset_in_overlap_buffer][read_end][h][w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass];
                            }
                            dw_pipe_overlap_buffer[current_write_offset_in_overlap_buffer][1 - read_end][h][w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass] =
                                dw_channels_tile[d][write_offset_h_in_channels_tile + h][w + filter_minus_strides];
                        }
                    }
                    else
                    {
                        read_val = dw_layer_ifms_zero_point;
                    }
                    dw_channels_tile[d][h][w + filter_minus_strides] = read_val;
                    if (w + filter_minus_strides >= PW_BUFFER_WIDTH)
                    {
                        dw_vertical_overlap_buffer[d][h][w + filter_minus_strides - PW_BUFFER_WIDTH] =
                            read_val;
                    }
                }
            }
        }
        //***********************************************
        if (PW_BUFFER_WIDTH - 1 + inner_offset_w + columns_filled_by_pw_in_the_firts_pass == DW_PIPE_OVERLAP_BUFFER_WIDTH && PW_BUFFER_WIDTH - 1 + starting_w + columns_filled_by_pw_in_the_firts_pass < layer_ifms_width)
        {
            const int current_w = PW_BUFFER_WIDTH - 1;
            for (int h = 0; h < MAX_FILTER_MINUS_STRIDES; h++)
            {
                if (h >= filter_minus_strides)
                {
                    break;
                }
                for (int d = 0; d < PARALLELISM_PW_OFMS; d++)
                {
#pragma HLS PIPELINE
                    int current_write_offset_in_overlap_buffer =
                        d_offset_in_overlap_buffer + d * filter_minus_strides * num_of_tiles_in_w + (h * layer_ifms_width + starting_w) / DW_PIPE_OVERLAP_BUFFER_WIDTH;
                    int current_read_offset_in_overlap_buffer =
                        d_offset_in_overlap_buffer + d * filter_minus_strides * num_of_tiles_in_w + (h * layer_ifms_width + starting_w) / DW_PIPE_OVERLAP_BUFFER_WIDTH;

                    fms_dt read_val;

                    if (starting_h != 0)
                    {
                        read_val =
                            dw_pipe_overlap_buffer[current_read_offset_in_overlap_buffer + 1][read_end][h][current_w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass - DW_PIPE_OVERLAP_BUFFER_WIDTH];
                    }
                    dw_pipe_overlap_buffer[current_write_offset_in_overlap_buffer + 1][1 - read_end][h][current_w + inner_offset_w + columns_filled_by_pw_in_the_firts_pass - DW_PIPE_OVERLAP_BUFFER_WIDTH] =
                        dw_channels_tile[d][write_offset_h_in_channels_tile + h][current_w + filter_minus_strides];

                    dw_channels_tile[d][h][current_w + filter_minus_strides] =
                        read_val;

                    dw_vertical_overlap_buffer[d][h][current_w + filter_minus_strides - PW_BUFFER_WIDTH] =
                        read_val;
                }
            }
        }
        //***********************************************
    }
}

void pipelined_engines::fill_dw_weights_tile(
    const dw_weights_dt weights[][MAX_DW_FILTER_AREA_IN_PIPE],
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
    const int starting_d,
    const int model_configs_list_limit,
    layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_ofms_width = layer_specs_struct.layer_ofm_width;
    const int layer_d = layer_specs_struct.layer_depth;
    const int filter_dim = layer_specs_struct.filter_size;
    const int strides = layer_specs_struct.strides;

dw_conv_engine:

    for (int d = 0; d < DW_TILE_DEPTH; d++)
    {
#pragma HLS PIPELINE
        if (starting_d + d >= layer_d ||
            (model_configs_list_limit != 0 && starting_d + d >= model_configs_list_limit))
        {
            break;
        }
        for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
        {
#pragma HLS UNROLL
            for (int w = 0; w < PARALLELISM_DW_W; w++)
            {
#pragma HLS UNROLL
                dw_pss_dt tmp = 0;
                for (int c_h = 0; c_h < MAX_DW_FILTER_DIM_IN_PIPE; c_h++)
                {
#pragma HLS UNROLL
                    for (int c_w = 0; c_w < MAX_DW_FILTER_DIM_IN_PIPE; c_w++)
                    {
#pragma HLS UNROLL
                        if (c_w >= filter_dim || c_h >= filter_dim || h >= PW_BUFFER_HEIGHT / strides)
                        {
                            break;
                        }
                        tmp += channels_tile[d][h * strides + c_h][w * strides + c_w] * weights[d][c_h * filter_dim + c_w];
                    }
                }
                result_tile[d][h][w] = tmp;
            }
        }
    }
}

void pipelined_engines::dw_normalize_and_write_back_result_tile(
    dw_pss_dt result_tile[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
    fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    const fms_quantization_scheme normalization_buffer[],
    const int starting_d, const int h_offset_in_result,
    const int starting_w, layer_specs layer_specs_struct,
    const int model_configs_list_limit)
{
#pragma HLS INLINE off

    if (starting_w >= 0)
    {
        const int layer_ofms_width = layer_specs_struct.layer_ofm_width;
        const int strides = layer_specs_struct.strides;
        const int ifms_d = layer_specs_struct.layer_depth;
        const int layer_relu = layer_specs_struct.layer_activation;
        const int offset_w = starting_w / strides;

        for (int d = 0; d < DW_TILE_DEPTH; d++)
        {
#pragma HLS PIPELINE
            fms_quantization_scheme normalization = normalization_buffer[d];
            const biases_dt fused_zero_point = normalization.fused_zero_point;
            const fms_dt ofm_zero_point = normalization.ofm_zero_point;
            const scales_dt fused_scales = normalization.fused_scales;
            const relu_6_fused_scales_dt relu_6_fused_scale = normalization.relu_6_fused_scale;
            if (starting_d + d >= ifms_d ||
                (model_configs_list_limit != 0 && starting_d + d >= model_configs_list_limit))
            {
                break;
            }
            for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < PW_BUFFER_WIDTH; w++)
                {
#pragma HLS UNROLL
                    fms_dt scaled_val = dw_relu_norm_v2(result_tile[d][h][w], fused_zero_point, ofm_zero_point, fused_scales, relu_6_fused_scale, layer_relu);
                    if (h_offset_in_result)
                    {
                        if (h + h_offset_in_result < PW_BUFFER_HEIGHT && h < (PW_BUFFER_HEIGHT >> (strides - 1)) && w < (PW_BUFFER_WIDTH >> (strides - 1)))
                        {
                            result[starting_d + d][h + OFFSET_H_IN_PIPELINE_RESULTS][offset_w + w] = scaled_val;
                        }
                        // if (layer_specs_struct.layer_index == 6 && starting_d + d == 0 && offset_w + w == 0 && h == 0)
                        // {
                        //     dw_pss_dt pss = result_tile[d][h][w];
                        //     pss += fused_zero_point;
                        //     printf("%d >> ", pss);
                        //     pss_f_dt scaled_pss = pss * fused_scales;
                        //     printf("%f >> ", fused_scales);
                        //     printf("%f >> ", scaled_pss);
                        //     if (scaled_pss > relu_6_fused_scale)
                        //     {
                        //         scaled_pss = relu_6_fused_scale;
                        //     }

                        //     scaled_pss += ofm_zero_point;
                        //     printf("%f >> ", scaled_pss);
                        //     scaled_pss += quant_half - (scaled_pss < 0);
                        //     printf("%f >> ", scaled_pss);
                        //     printf("%d \n", clamp(scaled_pss));
                        //     printf("\n**********\n");
                        // }
                    }
                    else
                    {
                        result[starting_d + d][h][offset_w + w] = scaled_val;
                        // if (layer_specs_struct.layer_index == 6 && starting_d + d == 0 && offset_w + w == 0 && h == 0)
                        // {
                        //     dw_pss_dt pss = result_tile[d][h][w];
                        //     pss += fused_zero_point;
                        //     printf("%d >> ", pss);
                        //     pss_f_dt scaled_pss = pss * fused_scales;
                        //     printf("%f >> ", fused_scales);
                        //     printf("%f >> ", scaled_pss);
                        //     if (scaled_pss > relu_6_fused_scale)
                        //     {
                        //         scaled_pss = relu_6_fused_scale;
                        //     }

                        //     scaled_pss += ofm_zero_point;
                        //     printf("%f >> ", scaled_pss);
                        //     scaled_pss += quant_half - (scaled_pss < 0);
                        //     printf("%f >> ", scaled_pss);
                        //     printf("%d \n", clamp(scaled_pss));
                        //     printf("\n**********\n");
                        // }
                    }
                }
            }
        }
    }
}

void pipelined_engines::pw_write_back_result_tile(
    fms_dt result_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
    const int starting_d, const int starting_w,
    const int model_configs_list_limit,
    const layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_num_fils = layer_specs_struct.layer_num_fils;

    if (starting_w >= 0)
    {
        for (int w = 0; w < PW_BUFFER_WIDTH; w++)
        {
            for (int d = 0; d < DW_TILE_DEPTH; d++)
            {
#pragma HLS PIPELINE
                if (d + starting_d >= layer_num_fils ||
                    (model_configs_list_limit != 0 && starting_d + d >= model_configs_list_limit))
                {
                    break;
                }
                for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    fms_dt normalized_val = result_tile[d][h][w];
                    result[starting_d + d][h][starting_w + w] = normalized_val;
                    if (layer_specs_struct.write_to_tmp)
                    {
                        if (h == 0)
                        {
                            tmp_channels[starting_d + d][0][starting_w + w] =
                                tmp_channels[starting_d + d][PW_BUFFER_HEIGHT][starting_w + w];
                        }
                        tmp_channels[starting_d + d][h + 1][starting_w + w] =
                            normalized_val;
                    }
                }
            }
        }
    }
}

void copy_from_pre_first_pipeline_layers_output(fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                                                                       [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                                                                       [PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
                                                fms_dt channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
                                                fms_dt pw_engine_input_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH],
                                                const int h_offset_in_pre_first_channels,
                                                const int h_offset_in_pre_first_pipeline_layers_output,
                                                const int starting_w,
                                                const bool first_layer_in_the_pipeline,
                                                layer_specs layer_specs_struct,
                                                const int model_configs_list[])
{
#pragma HLS INLINE off

    const int layer_ifm_width = layer_specs_struct.layer_ifm_width;
    const int layer_depth = model_configs_list[2 * layer_specs_struct.layer_index] != 0
                                ? model_configs_list[2 * layer_specs_struct.layer_index]
                                : layer_specs_struct.layer_depth;

    if (first_layer_in_the_pipeline)
    {

        for (int d = 0; d < layer_depth; d++)
        {
#pragma HLS PIPELINE
            for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
            {
#pragma HLS UNROLL
                if (h + h_offset_in_pre_first_channels < PW_BUFFER_HEIGHT)
                {
                    for (int w = 0; w < PW_BUFFER_WIDTH; w++)
                    {
#pragma HLS UNROLL
                        if (starting_w + w < layer_ifm_width)
                        {
                            pw_engine_input_buffer[d][h + h_offset_in_pre_first_channels][w] =
                                pre_first_pipeline_layers_output[d][h + h_offset_in_pre_first_pipeline_layers_output][starting_w + w];
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (int d = 0; d < layer_depth; d++)
        {
#pragma HLS PIPELINE
            for (int h = 0; h < PW_BUFFER_HEIGHT; h++)
            {
#pragma HLS UNROLL
                for (int w = 0; w < PW_BUFFER_WIDTH; w++)
                {
#pragma HLS UNROLL
                    if (starting_w + w < layer_ifm_width)
                    {
                        pw_engine_input_buffer[d][h][w] = channels[d][h][starting_w + w];
                    }
                }
            }
        }
    }
}

void pipelined_engines::pw_dw_conv(
    weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
    const dw_weights_dt dw_weights[][3 * 3],
    fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                           [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                           [PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
    fms_dt channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    fms_dt result[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][MAX_PW_BUFFER_WIDTH],
    fms_dt tmp_channels[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT + 1][MAX_PW_BUFFER_WIDTH],
    fms_dt dw_pipe_overlap_buffer[][2][2][DW_PIPE_OVERLAP_BUFFER_WIDTH],
    fms_dt dw_channels_tile[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    fms_dt dw_channels_tile_copy[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][DW_BUFFER_WIDTH],
    const int starting_h, const int h_offset_in_result,
    const int h_offset_in_pre_first_channels,
    const int h_offset_in_pre_first_pipeline_layers_output,
    bool fused_pw_dw,
    const layer_specs pw_layer_specs_struct,
    const layer_specs dw_layer_specs_struct,
    const fused_scales_dt fused_scales[],
    const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
    const relu_6_fused_scales_dt relu_6_fused_scales[],
    const biases_dt fused_zero_points[],
    const int odd_even,
    const bool first_layer_in_the_pipeline,
    const int model_configs_list[])
{
#pragma HLS INLINE off

    const int pw_layer = pw_layer_specs_struct.layer_index;
    const int dw_layer = dw_layer_specs_struct.layer_index;
    const int ifms_width = pw_layer_specs_struct.layer_ifm_width;
    const int num_of_filters = pw_layer_specs_struct.layer_num_fils;
    const int filter_dim_minus_strides = dw_layer_specs_struct.filter_size - dw_layer_specs_struct.strides;

    fms_quantization_scheme dw_normalization_buffer[DW_BUFFER_DEPTH];
    fms_quantization_scheme dw_normalization_buffer_copy[DW_BUFFER_DEPTH];

    fms_quantization_scheme pw_normalization_buffer[PARALLELISM_PW_OFMS];
    fms_quantization_scheme pw_normalization_buffer_copy[PARALLELISM_PW_OFMS];

    weights_dt weights_tile[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH];
    weights_dt weights_tile_copy[PARALLELISM_PW_OFMS][MAX_PW_BUFFER_DEPTH];

#pragma HLS ARRAY_PARTITION variable = weights_tile type = complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile_copy type = complete dim = 1

    dw_weights_dt dw_weights_tile[DW_BUFFER_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE];
    dw_weights_dt dw_weights_tile_copy[DW_BUFFER_DEPTH][MAX_DW_FILTER_AREA_IN_PIPE];

#pragma HLS ARRAY_PARTITION variable = dw_weights_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = dw_weights_tile_copy type = complete dim = 2

    pss_dt pw_engine_result_tile[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH];
    pss_dt pw_engine_result_tile_copy[PARALLELISM_PW_OFMS][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = pw_engine_result_tile type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = pw_engine_result_tile_copy type = complete dim = 0

    dw_pss_dt dw_result_tile[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH];
    dw_pss_dt dw_result_tile_copy[DW_TILE_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH];

    fms_dt dw_vertical_overlap_buffer[DW_TILE_DEPTH][DW_BUFFER_HEIGHT][MAX_FILTER_MINUS_STRIDES];
    fms_dt dw_horizontal_overlap_buffer[DW_TILE_DEPTH][MAX_FILTER_MINUS_STRIDES][PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = dw_vertical_overlap_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = dw_vertical_overlap_buffer type = complete dim = 3

#pragma HLS ARRAY_PARTITION variable = dw_horizontal_overlap_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = dw_horizontal_overlap_buffer type = complete dim = 3

    fms_dt pw_engine_input_buffer[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH];
    fms_dt pw_engine_input_buffer_copy[MAX_PW_BUFFER_DEPTH][PW_BUFFER_HEIGHT][PW_BUFFER_WIDTH];

#pragma HLS ARRAY_PARTITION variable = pw_engine_input_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = pw_engine_input_buffer type = complete dim = 3

#pragma HLS ARRAY_PARTITION variable = pw_engine_input_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = pw_engine_input_buffer_copy type = complete dim = 3

    int next_w = 0;
    int prev_w = -1;
    int prev_prev_w = -1;
    int prev_prev_prev_w = -1;

    const int model_configs_list_limit_o = model_configs_list[2 * pw_layer_specs_struct.layer_index + 1];

    if (fused_pw_dw)
    {
        for (int d = 0; d < num_of_filters; d += PARALLELISM_PW_OFMS)
        {
            if (model_configs_list_limit_o != 0 && d >= model_configs_list_limit_o)
            {
                break;
            }
            // ###############################
            load_pw_weights(on_chip_weights, weights_tile, d,
                            pw_layer_specs_struct);
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts, relu_6_fused_scales,
                                             fused_zero_points, pw_normalization_buffer,
                                             d, // starting_d
                                             pipe_layers_fused_parameters_offsets[pw_layer],
                                             PARALLELISM_PW_OFMS, pw_layer_specs_struct);

            padd_left_dw_channels_tile(dw_channels_tile, dw_channels_tile_copy,
                                       dw_layer_specs_struct);

            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts, relu_6_fused_scales,
                                             fused_zero_points, dw_normalization_buffer,
                                             d, // starting_d
                                             pipe_layers_fused_parameters_offsets[dw_layer],
                                             DW_TILE_DEPTH, dw_layer_specs_struct);

            pipelined_engines::fill_dw_weights_tile(dw_weights, dw_weights_tile, d,
                                                    pipe_dw_layers_weights_offsets[dw_layer]);

            copy_from_pre_first_pipeline_layers_output(pre_first_pipeline_layers_output, channels, pw_engine_input_buffer, h_offset_in_pre_first_channels,
                                                       h_offset_in_pre_first_pipeline_layers_output, 0, first_layer_in_the_pipeline,
                                                       pw_layer_specs_struct, model_configs_list);

            pw_conv_engine(weights_tile, pw_engine_input_buffer, pw_engine_result_tile_copy,
                           d, pw_layer_specs_struct, model_configs_list);

            first_pass_pw_normalize_engine_result(dw_pipe_overlap_buffer,
                                                  pw_engine_result_tile_copy, dw_channels_tile, result,
                                                  tmp_channels, pw_normalization_buffer, d, starting_h,
                                                  fused_pw_dw, pw_layer_specs_struct, dw_layer_specs_struct, odd_even);

            // if (dw_layer_specs_struct.layer_index == 6 && d == 0)
            //         {
            //             for (int hh = 0; hh < 3; hh++)
            //             {
            //                 for (int ww = 0; ww < 6; ww++)
            //                 {
            //                     printf("%d ", (int)dw_channels_tile[0][hh][ww]);
            //                 }
            //                 printf("\n");
            //             }
            //             printf("################");
            //         }

            copy_from_pre_first_pipeline_layers_output(pre_first_pipeline_layers_output, channels, pw_engine_input_buffer,
                                                       h_offset_in_pre_first_channels,
                                                       h_offset_in_pre_first_pipeline_layers_output, filter_dim_minus_strides,
                                                       first_layer_in_the_pipeline, pw_layer_specs_struct, model_configs_list);

            for (int w = 0; w < ifms_width; w += PW_BUFFER_WIDTH)
            {
                next_w = w + PW_BUFFER_WIDTH;
                prev_w = w - PW_BUFFER_WIDTH;
                prev_prev_w = prev_w - PW_BUFFER_WIDTH;
                prev_prev_prev_w = prev_prev_w - PW_BUFFER_WIDTH;
                if ((w / PW_BUFFER_WIDTH) % 2 == 0)
                {
                    // dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                    //                                         result, dw_normalization_buffer, d,
                    //                                         h_offset_in_result, prev_prev_prev_w,
                    //                                         dw_layer_specs_struct, model_configs_list_limit_o);
                    // ###############################
                    dw_conv_engine(dw_weights_tile, dw_channels_tile,
                                   dw_result_tile, d, model_configs_list_limit_o, dw_layer_specs_struct);

                    if (prev_prev_w == 0 && dw_layer_specs_struct.layer_index == 6 && d == 0)
                    {
                        for (int hh = 0; hh < 3; hh++)
                        {
                            for (int ww = 0; ww < 6; ww++)
                            {
                                printf("%d ", (int)dw_channels_tile[0][hh][ww]);
                            }
                            printf("\n");
                        }
                        printf("%d\n", (int)dw_result_tile[0][0][0]);
                    }
                    // ###############################
                    pw_normalize_engine_result(pw_engine_result_tile_copy,
                                               dw_channels_tile_copy, dw_vertical_overlap_buffer,
                                               dw_horizontal_overlap_buffer,
                                               pw_normalization_buffer, d,
                                               starting_h, prev_w,
                                               model_configs_list_limit_o,
                                               pw_layer_specs_struct, dw_layer_specs_struct);
                    write_next_overlap_and_read_current_only_p2(
                        dw_pipe_overlap_buffer, dw_vertical_overlap_buffer,
                        dw_horizontal_overlap_buffer,
                        dw_channels_tile_copy, d, starting_h, prev_w,
                        model_configs_list_limit_o,
                        dw_layer_specs_struct, odd_even);
                    // ###############################
                    pw_conv_engine(weights_tile, pw_engine_input_buffer,
                                   pw_engine_result_tile, d,
                                   pw_layer_specs_struct, model_configs_list);
                    // ###############################
                    copy_from_pre_first_pipeline_layers_output(pre_first_pipeline_layers_output, channels,
                                                               pw_engine_input_buffer_copy, h_offset_in_pre_first_channels,
                                                               h_offset_in_pre_first_pipeline_layers_output, 
                                                               next_w + filter_dim_minus_strides,
                                                               first_layer_in_the_pipeline, pw_layer_specs_struct, model_configs_list);
                }
                else
                {
                    // if (prev_prev_prev_w == 0 && dw_layer_specs_struct.layer_index == 6 && d == 0)
                    // {
                    // printf("*%d\n", (int)dw_result_tile[0][0][1]);
                    // }
                    dw_normalize_and_write_back_result_tile(dw_result_tile,
                                                            result, dw_normalization_buffer, d,
                                                            h_offset_in_result, prev_prev_prev_w,
                                                            dw_layer_specs_struct,
                                                            model_configs_list_limit_o);
                    // ###############################
                    dw_conv_engine(dw_weights_tile, dw_channels_tile_copy,
                                   dw_result_tile_copy, d, model_configs_list_limit_o,
                                   dw_layer_specs_struct);
                    // ###############################
                    pw_normalize_engine_result(pw_engine_result_tile,
                                               dw_channels_tile, dw_vertical_overlap_buffer,
                                               dw_horizontal_overlap_buffer,
                                               pw_normalization_buffer, d,
                                               starting_h, prev_w,
                                               model_configs_list_limit_o,
                                               pw_layer_specs_struct, dw_layer_specs_struct);
                    if (prev_w == 0 && dw_layer_specs_struct.layer_index == 6 && d == 0)
                    {
                        for (int hh = 0; hh < 3; hh++)
                        {
                            for (int ww = 0; ww < 6; ww++)
                            {
                                printf("%d >> %d ", (int)pw_engine_result_tile[0][hh][ww] , (int)dw_channels_tile[0][hh][ww]);
                            }
                            printf("\n");
                        }
                    }
                    write_next_overlap_and_read_current_only_p2(
                        dw_pipe_overlap_buffer, dw_vertical_overlap_buffer,
                        dw_horizontal_overlap_buffer,
                        dw_channels_tile, d, starting_h, prev_w,
                        model_configs_list_limit_o,
                        dw_layer_specs_struct, odd_even);
                    // ###############################
                    pw_conv_engine(weights_tile, pw_engine_input_buffer_copy,
                                   pw_engine_result_tile_copy, d,
                                   pw_layer_specs_struct, model_configs_list);
                    // ###############################
                    copy_from_pre_first_pipeline_layers_output(pre_first_pipeline_layers_output, channels,
                                                               pw_engine_input_buffer, h_offset_in_pre_first_channels,
                                                               h_offset_in_pre_first_pipeline_layers_output,
                                                                next_w + filter_dim_minus_strides,
                                                               first_layer_in_the_pipeline, pw_layer_specs_struct, model_configs_list);
                }
            }

            if (((ifms_width / PW_BUFFER_WIDTH) - 1) % 2)
            {
                dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                        result, dw_normalization_buffer, d, h_offset_in_result,
                                                        ifms_width - 3 * PW_BUFFER_WIDTH,
                                                        dw_layer_specs_struct,
                                                        model_configs_list_limit_o);
                // ###############################
                dw_conv_engine(dw_weights_tile, dw_channels_tile,
                               dw_result_tile, d, model_configs_list_limit_o, dw_layer_specs_struct);
                // ###############################
                pw_normalize_engine_result(pw_engine_result_tile_copy,
                                           dw_channels_tile_copy, dw_vertical_overlap_buffer,
                                           dw_horizontal_overlap_buffer,
                                           pw_normalization_buffer, d, starting_h,
                                           ifms_width - PW_BUFFER_WIDTH,
                                           model_configs_list_limit_o,
                                           pw_layer_specs_struct, dw_layer_specs_struct);
                write_next_overlap_and_read_current_only_p2(
                    dw_pipe_overlap_buffer, dw_vertical_overlap_buffer,
                    dw_horizontal_overlap_buffer,
                    dw_channels_tile_copy, d, starting_h,
                    ifms_width - PW_BUFFER_WIDTH,
                    model_configs_list_limit_o,
                    dw_layer_specs_struct, odd_even);
                // #######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile, result,
                                                        dw_normalization_buffer, d, h_offset_in_result,
                                                        ifms_width - 2 * PW_BUFFER_WIDTH,
                                                        dw_layer_specs_struct, model_configs_list_limit_o);
                // ###############################
                dw_conv_engine(dw_weights_tile, dw_channels_tile_copy,
                               dw_result_tile, d, model_configs_list_limit_o, dw_layer_specs_struct);
                // #######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile, result,
                                                        dw_normalization_buffer, d, h_offset_in_result,
                                                        ifms_width - PW_BUFFER_WIDTH, dw_layer_specs_struct,
                                                        model_configs_list_limit_o);
            }
            else
            {
                dw_normalize_and_write_back_result_tile(dw_result_tile, result,
                                                        dw_normalization_buffer, d, h_offset_in_result,
                                                        ifms_width - 3 * PW_BUFFER_WIDTH,
                                                        dw_layer_specs_struct,
                                                        model_configs_list_limit_o);
                // ###############################
                dw_conv_engine(dw_weights_tile, dw_channels_tile_copy,
                               dw_result_tile_copy, d, model_configs_list_limit_o, dw_layer_specs_struct);
                pw_normalize_engine_result(pw_engine_result_tile,
                                           dw_channels_tile, dw_vertical_overlap_buffer,
                                           dw_horizontal_overlap_buffer,
                                           pw_normalization_buffer, d, starting_h,
                                           ifms_width - PW_BUFFER_WIDTH,
                                           model_configs_list_limit_o,
                                           pw_layer_specs_struct, dw_layer_specs_struct);
                write_next_overlap_and_read_current_only_p2(
                    dw_pipe_overlap_buffer, dw_vertical_overlap_buffer,
                    dw_horizontal_overlap_buffer,
                    dw_channels_tile, d, starting_h,
                    ifms_width - PW_BUFFER_WIDTH,
                    model_configs_list_limit_o,
                    dw_layer_specs_struct, odd_even);
                // #######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile_copy,
                                                        result, dw_normalization_buffer_copy, d,
                                                        h_offset_in_result, ifms_width - 2 * PW_BUFFER_WIDTH,
                                                        dw_layer_specs_struct,
                                                        model_configs_list_limit_o);
                // ###############################
                dw_conv_engine(dw_weights_tile, dw_channels_tile,
                               dw_result_tile, d, model_configs_list_limit_o, dw_layer_specs_struct);
                // #######################################################################################
                dw_normalize_and_write_back_result_tile(dw_result_tile, result,
                                                        dw_normalization_buffer, d, h_offset_in_result,
                                                        ifms_width - PW_BUFFER_WIDTH, dw_layer_specs_struct,
                                                        model_configs_list_limit_o);
            }
        }
    }
    else
    {
        for (int d = 0; d < num_of_filters; d += PARALLELISM_PW_OFMS)
        {
            if (model_configs_list_limit_o != 0 && d >= model_configs_list_limit_o)
            {
                break;
            }
            fill_fused_scales_and_zps_buffer(fused_scales,
                                             fused_scales_log_2_shifts, relu_6_fused_scales,
                                             fused_zero_points, pw_normalization_buffer,
                                             d, // starting_d
                                             pipe_layers_fused_parameters_offsets[pw_layer],
                                             PARALLELISM_PW_OFMS, pw_layer_specs_struct);

            load_pw_weights(on_chip_weights, weights_tile, d,
                            pw_layer_specs_struct);

            copy_from_pre_first_pipeline_layers_output(pre_first_pipeline_layers_output, channels, pw_engine_input_buffer,
                                                       h_offset_in_pre_first_channels,
                                                       h_offset_in_pre_first_pipeline_layers_output, 0, first_layer_in_the_pipeline,
                                                       pw_layer_specs_struct, model_configs_list);

            for (int w = 0; w < ifms_width; w += PW_BUFFER_WIDTH)
            {
                next_w = w + PW_BUFFER_WIDTH;
                prev_w = w - PW_BUFFER_WIDTH;
                prev_prev_w = w - 2 * PW_BUFFER_WIDTH;
                if ((w / PW_BUFFER_WIDTH) % 2 == 0)
                {
                    pw_write_back_result_tile(dw_channels_tile, result,
                                              tmp_channels, d, prev_prev_w, model_configs_list_limit_o, pw_layer_specs_struct);
                    // ###############################
                    pw_only_normalize_engine_result(pw_engine_result_tile_copy,
                                                    dw_channels_tile_copy,
                                                    tmp_channels, pw_normalization_buffer, d,
                                                    starting_h, prev_w,
                                                    model_configs_list_limit_o,
                                                    pw_layer_specs_struct, dw_layer_specs_struct);
                    // ###############################
                    pw_conv_engine(weights_tile, pw_engine_input_buffer,
                                   pw_engine_result_tile, d, pw_layer_specs_struct, model_configs_list);
                    // ###############################
                    copy_from_pre_first_pipeline_layers_output(pre_first_pipeline_layers_output, channels, pw_engine_input_buffer_copy,
                                                               h_offset_in_pre_first_channels,
                                                               h_offset_in_pre_first_pipeline_layers_output, next_w,
                                                               first_layer_in_the_pipeline,
                                                               pw_layer_specs_struct, model_configs_list);
                }
                else
                {
                    pw_write_back_result_tile(dw_channels_tile_copy, result,
                                              tmp_channels, d, prev_prev_w,
                                              model_configs_list_limit_o, pw_layer_specs_struct);
                    // ###############################
                    pw_only_normalize_engine_result(pw_engine_result_tile,
                                                    dw_channels_tile,
                                                    tmp_channels, pw_normalization_buffer, d,
                                                    starting_h, prev_w,
                                                    model_configs_list_limit_o,
                                                    pw_layer_specs_struct, dw_layer_specs_struct);
                    // ###############################
                    pw_conv_engine(weights_tile, pw_engine_input_buffer_copy,
                                   pw_engine_result_tile_copy, d,
                                   pw_layer_specs_struct, model_configs_list);
                    // ###############################
                    copy_from_pre_first_pipeline_layers_output(pre_first_pipeline_layers_output, channels, pw_engine_input_buffer,
                                                               h_offset_in_pre_first_channels,
                                                               h_offset_in_pre_first_pipeline_layers_output,
                                                               next_w, first_layer_in_the_pipeline,
                                                               pw_layer_specs_struct, model_configs_list);
                }
            }

            if (((ifms_width / PW_BUFFER_WIDTH) - 1) % 2)
            {
                pw_write_back_result_tile(dw_channels_tile, result,
                                          tmp_channels, d, ifms_width - 2 * PW_BUFFER_WIDTH,
                                          model_configs_list_limit_o, pw_layer_specs_struct);
                // ###############################
                pw_only_normalize_engine_result(pw_engine_result_tile_copy,
                                                dw_channels_tile_copy,
                                                tmp_channels, pw_normalization_buffer, d, starting_h,
                                                ifms_width - PW_BUFFER_WIDTH,
                                                model_configs_list_limit_o,
                                                pw_layer_specs_struct, dw_layer_specs_struct);
                pw_write_back_result_tile(dw_channels_tile_copy, result,
                                          tmp_channels, d, ifms_width - PW_BUFFER_WIDTH,
                                          model_configs_list_limit_o, pw_layer_specs_struct);
            }
            else
            {
                pw_write_back_result_tile(dw_channels_tile_copy, result,
                                          tmp_channels, d, ifms_width - 2 * PW_BUFFER_WIDTH,
                                          model_configs_list_limit_o, pw_layer_specs_struct);
                // ###############################
                pw_only_normalize_engine_result(pw_engine_result_tile,
                                                dw_channels_tile,
                                                tmp_channels, pw_normalization_buffer, d, starting_h,
                                                ifms_width - PW_BUFFER_WIDTH,
                                                model_configs_list_limit_o,
                                                pw_layer_specs_struct, dw_layer_specs_struct);
                pw_write_back_result_tile(dw_channels_tile, result,
                                          tmp_channels, d, ifms_width - PW_BUFFER_WIDTH,
                                          model_configs_list_limit_o, pw_layer_specs_struct);
            }
        }
    }
}

#endif

#endif
