#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/conv_utils.h"

using namespace seml_engines;

#if FIBHA_VERSION == 2

void dw_conv_engine(
    dw_weights_dt weights[CHANNELS_PIPELINE_DEPTH][max_filter_hw_dim * max_filter_hw_dim],
    fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
    dw_pss_dt result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
    const int starting_d,
    const int tile_in_h,
    const int tile_in_w,
    const fms_dt current_layer_zero_point,
    layer_specs layer_specs_struct)
{
#pragma HLS INLINE off

    const int layer_d = layer_specs_struct.layer_depth;
    const int filter_dim = layer_specs_struct.filter_size;
    const int strides = layer_specs_struct.strides;

dw_conv_engine:
    for (int c_h = 0; c_h < max_filter_hw_dim; c_h++)
    {
        for (int c_w = 0; c_w < max_filter_hw_dim; c_w++)
        {
            for (int d_in_pipeline = 0; d_in_pipeline < CHANNELS_PIPELINE_DEPTH;
                 d_in_pipeline++)
            {
#pragma HLS PIPELINE
                if (starting_d + d_in_pipeline >= layer_d)
                {
                    break;
                }
                for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                    {
#pragma HLS UNROLL
                        if (c_w >= filter_dim || c_h >= filter_dim || h >= dw_tile_h / strides || w >= dw_tile_w / strides)
                        {
                            break;
                        }
                        if (c_h == 0 && c_w == 0)
                        {
                            result_tile[d_in_pipeline][h][w] =
                                channels_tile[d_in_pipeline][h * strides + c_h]
                                             [w * strides + c_w] *
                                weights[d_in_pipeline][c_h * max_filter_hw_dim + c_w];
                        }
                        else
                        {
                            result_tile[d_in_pipeline][h][w] +=
                                channels_tile[d_in_pipeline][h * strides + c_h]
                                             [w * strides + c_w] *
                                weights[d_in_pipeline][c_h * max_filter_hw_dim + c_w];
                        }
                    }
                }
            }
        }
    }
}

void fill_scales_tiles(const fused_scales_dt fused_scales[],
                       fused_scales_dt fused_scales_tile[],
                       const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                       fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
                       const relu_6_fused_scales_dt relu_6_fused_scales[],
                       relu_6_fused_scales_dt relu_6_fused_scales_tile[],
                       const biases_dt fused_zero_points[], biases_dt fused_zero_points_tile[],
                       const int layer_d,
                       const int current_layer_fused_parameters_offset)
{
#pragma HLS INLINE off

    for (int d = 0; d < MAX_DW_LAYER_D; d++)
    {
#pragma HLS PIPELINE
        if (d >= layer_d)
        {
            break;
        }
        fused_scales_tile[d] =
            fused_scales[current_layer_fused_parameters_offset + d];
        fused_scales_log_2_shifts_tile[d] =
            fused_scales_log_2_shifts[current_layer_fused_parameters_offset + d];
        relu_6_fused_scales_tile[d] =
            relu_6_fused_scales[current_layer_fused_parameters_offset + d];
        fused_zero_points_tile[d] =
            fused_zero_points[current_layer_fused_parameters_offset + d];
    }
}

void seml_engines::fill_dw_weights_tile(const dw_weights_dt weights[][3 * 3],
                                        dw_weights_dt weights_tile[][3 * 3],
                                        int starting_d, const int current_dw_layer_weights_offset)
{
#pragma HLS INLINE off

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
    }
}

void dw_conv_copy_engine_result_tile(
    dw_pss_dt engine_result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
    dw_pss_dt engine_result_tile_copy[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH], const int strides)
{
#pragma HLS INLINE off

    for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d++)
    {
#pragma HLS PIPELINE
        for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
        {
#pragma HLS UNROLL
            if (h >= (CHANNELS_TILE_HEIGHT + 1) / strides)
            {
                break;
            }
            for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
            {
#pragma HLS UNROLL
                if (w >= (CHANNELS_TILE_WIDTH + 1) / strides)
                {
                    break;
                }
                engine_result_tile_copy[d][h][w] = engine_result_tile[d][h][w];
            }
        }
    }
}

void dw_normalize_and_write_back_result_tile(fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                                             dw_pss_dt result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                                             fms_quantization_scheme normalization,
                                             const fused_scales_dt fused_scales_tile[],
                                             const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
                                             const relu_6_fused_scales_dt relu_6_fused_scales_tile[],
                                             const biases_dt fused_zero_points_tile[],
                                             const int tile_in_h,
                                             const int tile_in_w,
                                             const int starting_d,
                                             const int in_tile_h,
                                             const int in_tile_w,
                                             layer_specs layer_specs_struct)
{
#pragma HLS INLINE off
    const int num_of_ofm_tiles_h = layer_specs_struct.layer_num_of_ofm_tiles_h;
    const int num_of_ofm_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
    const int strides = layer_specs_struct.strides;
    const int ifms_d = layer_specs_struct.layer_depth;
    const int layer_relu = layer_specs_struct.layer_activation;

    const int num_of_tiles_hw = num_of_ofm_tiles_h * num_of_ofm_tiles_w;
    const int initial_tile_index = starting_d * num_of_tiles_hw + tile_in_h * num_of_ofm_tiles_w + tile_in_w;

    for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
    {
        if (h * strides >= CHANNELS_TILE_HEIGHT)
        {
            break;
        }
        for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d += CHANNELS_TILE_DEPTH)
        {
#pragma HLS PIPELINE
            const int main_tile_index = initial_tile_index + d * num_of_tiles_hw;
            if (starting_d + d >= ifms_d)
            {
                break;
            }
            normalization.fused_scales = fused_scales_tile[starting_d + d];
            normalization.fused_scales_log_2_shift =
                fused_scales_log_2_shifts_tile[starting_d + d];
            normalization.relu_6_fused_scale =
                relu_6_fused_scales_tile[starting_d + d];
            normalization.fused_zero_point =
                fused_zero_points_tile[starting_d + d];

            if (strides == 1)
            {
                for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                {
#pragma HLS UNROLL
                    result[main_tile_index][h][w] = dw_relu_norm(
                        result_tile[d][h][w], normalization,
                        layer_relu);
                }
            }
            else
            {
                for (int w = 0; w < CHANNELS_TILE_WIDTH / 2; w++)
                {
#pragma HLS UNROLL
                    result[main_tile_index][in_tile_h + h][in_tile_w + w] = dw_relu_norm(
                        result_tile[d][h][w], normalization,
                        layer_relu);
                }
            }
        }
    }
}

void seml_engines::dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
                               fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                               fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                               const int layer,
                               const layer_specs layer_specs_struct,
                               const fused_scales_dt fused_scales[],
                               const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                               const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[],
                               const fused_scales_dt fused_scales_part2[],
                               const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
                               const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
                               const biases_dt fused_zero_points_part2[],
                               const int model_configs_list[2 * max_conv_layers])
{
#pragma HLS INLINE off

    fms_quantization_scheme normalization;

    const int current_dw_layer_weights_offset = dw_layers_weights_offsets[layer];
    const int current_layer_fused_parameters_offset =
        layers_fused_parameters_offsets[layer];

    dw_weights_dt weights_tile[CHANNELS_PIPELINE_DEPTH][3 * 3];
#pragma HLS ARRAY_PARTITION variable = weights_tile type = complete dim = 2

    fused_scales_dt fused_scales_tile[MAX_DW_LAYER_D];
    fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[MAX_DW_LAYER_D];
    relu_6_fused_scales_dt relu_6_fused_scales_tile[MAX_DW_LAYER_D];
    biases_dt fused_zero_points_tile[MAX_DW_LAYER_D];

    const fms_dt current_layer_fms_zero_point = layer_specs_struct.layer_ifms_zero_point;

    normalization.ofm_scale = layer_specs_struct.layer_ofms_scale;
    normalization.ofm_zero_point = layer_specs_struct.layer_ofms_zero_point;

    fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED];
    fms_dt channels_tile_copy[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED];
    dw_pss_dt engine_result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];
    dw_pss_dt engine_result_tile_copy[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];

#pragma HLS ARRAY_PARTITION variable = channels_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_tile type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = channels_tile_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_tile_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = engine_result_tile_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile_copy type = complete dim = 3

    const int ifms_d = layer_specs_struct.layer_depth;
    const int strides = layer_specs_struct.strides;

    const int num_of_iterations_d = model_configs_list[2 * layer] == 0
                                        ? (ifms_d + CHANNELS_PIPELINE_DEPTH - 1) / CHANNELS_PIPELINE_DEPTH
                                        : (model_configs_list[2 * layer] + CHANNELS_PIPELINE_DEPTH - 1) /
                                              CHANNELS_PIPELINE_DEPTH;
    // cout << layer << " "<< (ifms_d + CHANNELS_PIPELINE_DEPTH - 1) / CHANNELS_PIPELINE_DEPTH << " "
    //      << (model_configs_list[2 * layer] + CHANNELS_PIPELINE_DEPTH - 1) / CHANNELS_PIPELINE_DEPTH << "\n";
    if (current_layer_fused_parameters_offset < first_quantization_arrays_num_elements)
    {
        fill_scales_tiles(fused_scales, fused_scales_tile, fused_scales_log_2_shifts,
                          fused_scales_log_2_shifts_tile, relu_6_fused_scales,
                          relu_6_fused_scales_tile, fused_zero_points,
                          fused_zero_points_tile,
                          ifms_d,
                          current_layer_fused_parameters_offset);
    }
    else
    {
        fill_scales_tiles(fused_scales_part2, fused_scales_tile, fused_scales_log_2_shifts_part2,
                          fused_scales_log_2_shifts_tile, relu_6_fused_scales_part2,
                          relu_6_fused_scales_tile, fused_zero_points_part2,
                          fused_zero_points_tile,
                          ifms_d,
                          current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
    }

    for (int tile_in_h = 0; tile_in_h < layer_specs_struct.layer_num_of_ifm_tiles_h; tile_in_h++)
    {
        for (int tile_in_w = 0; tile_in_w < layer_specs_struct.layer_num_of_ifm_tiles_w; tile_in_w++)
        {
            int in_tile_h = (tile_in_h % strides) * (CHANNELS_TILE_HEIGHT / strides);
            int in_tile_w = (tile_in_w % strides) * (CHANNELS_TILE_WIDTH / strides);

            fill_fms_tile(channels,
                          channels_tile,
                          0,
                          tile_in_h,
                          tile_in_w,
                          current_layer_fms_zero_point,
                          layer_specs_struct);

            for (int dw_pipeline_in_d = 0;
                 dw_pipeline_in_d < num_of_iterations_d;
                 dw_pipeline_in_d++)
            {
                const int tile_in_d = dw_pipeline_in_d * (dw_pipeline_depth / dw_tile_d);
                const int prev_tile_in_d = (dw_pipeline_in_d - 1) >= 0
                                               ? (dw_pipeline_in_d - 1) * (dw_pipeline_depth / dw_tile_d)
                                               : 0;
                const int next_tile_in_d = (dw_pipeline_in_d + 1) * (dw_pipeline_depth / dw_tile_d);

                if (dw_pipeline_in_d % 2 == 0)
                {
                    seml_engines::fill_dw_weights_tile(weights, weights_tile,
                                                       tile_in_d,
                                                       current_dw_layer_weights_offset);

                    dw_normalize_and_write_back_result_tile(result,
                                                            engine_result_tile_copy, normalization,
                                                            fused_scales_tile, fused_scales_log_2_shifts_tile,
                                                            relu_6_fused_scales_tile, fused_zero_points_tile,
                                                            tile_in_h / strides, tile_in_w / strides,
                                                            prev_tile_in_d,
                                                            in_tile_h, in_tile_w,
                                                            layer_specs_struct);

                    dw_conv_engine(weights_tile,
                                   channels_tile,
                                   engine_result_tile,
                                   tile_in_d,
                                   tile_in_h,
                                   tile_in_w,
                                   current_layer_fms_zero_point,
                                   layer_specs_struct);

                    fill_fms_tile(channels,
                                  channels_tile_copy,
                                  next_tile_in_d,
                                  tile_in_h,
                                  tile_in_w,
                                  current_layer_fms_zero_point,
                                  layer_specs_struct);
                }
                else
                {
                    seml_engines::fill_dw_weights_tile(weights, weights_tile,
                                                       tile_in_d,
                                                       current_dw_layer_weights_offset);

                    dw_normalize_and_write_back_result_tile(result,
                                                            engine_result_tile, normalization,
                                                            fused_scales_tile, fused_scales_log_2_shifts_tile,
                                                            relu_6_fused_scales_tile, fused_zero_points_tile,
                                                            tile_in_h / strides, tile_in_w / strides,
                                                            prev_tile_in_d,
                                                            in_tile_h, in_tile_w,
                                                            layer_specs_struct);

                    dw_conv_engine(weights_tile,
                                   channels_tile_copy,
                                   engine_result_tile_copy,
                                   tile_in_d,
                                   tile_in_h,
                                   tile_in_w,
                                   current_layer_fms_zero_point,
                                   layer_specs_struct);

                    fill_fms_tile(channels,
                                  channels_tile,
                                  next_tile_in_d,
                                  tile_in_h,
                                  tile_in_w,
                                  current_layer_fms_zero_point,
                                  layer_specs_struct);
                }
            }
            if ((num_of_iterations_d - 1) % 2 == 0)
            {
                dw_normalize_and_write_back_result_tile(result,
                                                        engine_result_tile, normalization,
                                                        fused_scales_tile, fused_scales_log_2_shifts_tile,
                                                        relu_6_fused_scales_tile, fused_zero_points_tile,
                                                        tile_in_h / strides, tile_in_w / strides,
                                                        ifms_d % CHANNELS_PIPELINE_DEPTH == 0
                                                            ? ifms_d - CHANNELS_PIPELINE_DEPTH
                                                            : ifms_d - (ifms_d % CHANNELS_PIPELINE_DEPTH),
                                                        in_tile_h, in_tile_w,
                                                        layer_specs_struct);
            }
            else
            {
                dw_normalize_and_write_back_result_tile(result,
                                                        engine_result_tile_copy, normalization,
                                                        fused_scales_tile, fused_scales_log_2_shifts_tile,
                                                        relu_6_fused_scales_tile, fused_zero_points_tile,
                                                        tile_in_h / strides, tile_in_w / strides,
                                                        ifms_d % CHANNELS_PIPELINE_DEPTH == 0
                                                            ? ifms_d - CHANNELS_PIPELINE_DEPTH
                                                            : ifms_d - (ifms_d % CHANNELS_PIPELINE_DEPTH),
                                                        in_tile_h, in_tile_w,
                                                        layer_specs_struct);
            }
        }
    }
}

#endif
