#include "../headers/layers_imp_common_includes.h"
#include "../headers/dw_conv.h"
#include "../headers/conv_utils.h"

#if FIBHA_VERSION == 2

void dw_conv_engine(
    dw_weights_dt weights[CHANNELS_PIPELINE_DEPTH][max_filter_hw_dim * max_filter_hw_dim],
    fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
    dw_pss_dt result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
    fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH],
    fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_TOP_LEFT],
    fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
    fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
    fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT],
    fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH],
    fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
    fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT],
    const int filter_dim, const int strides,
    const int layer_d,
    const int starting_d,
    const int padding_top_left,
    const int tile_in_h,
    const int tile_in_w,
    const int num_of_ifm_tiles_h,
    const int num_of_ifm_tiles_w,
    const int layer_ifm_height,
    const int layer_ifm_width,
    const fms_dt current_layer_zero_point)
{
#pragma HLS INLINE off

    fms_dt ifms_buffer[CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED];
#pragma HLS ARRAY_PARTITION variable = ifms_buffer type = complete dim = 0

dw_conv_engine:
    for (int d_in_pipeline = 0; d_in_pipeline < CHANNELS_PIPELINE_DEPTH;
         d_in_pipeline++)
    {
#pragma HLS PIPELINE
        fill_fms_tile(channels,
                      padding_top_buffer,
                      padding_left_buffer,
                      padding_top_left_buffer,
                      padding_top_right_buffer,
                      padding_bottom_left_buffer,
                      padding_bottom_buffer,
                      padding_right_buffer,
                      padding_bottom_right_buffer,
                      ifms_buffer,
                      starting_d + d_in_pipeline,
                      tile_in_h,
                      tile_in_w,
                      num_of_ifm_tiles_h,
                      num_of_ifm_tiles_w,
                      layer_ifm_height,
                      layer_ifm_width,
                      padding_top_left,
                      current_layer_zero_point);

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
                pss_dt tmp = 0;
                for (int c_h = 0; c_h < max_filter_hw_dim; c_h++)
                {
#pragma HLS UNROLL

                    for (int c_w = 0; c_w < max_filter_hw_dim; c_w++)
                    {
#pragma HLS UNROLL
                        if (c_w >= filter_dim || c_h >= filter_dim || h >= dw_tile_h / strides || w >= dw_tile_w / strides ||
                            tile_in_w * CHANNELS_TILE_WIDTH + w >= layer_ifm_width || tile_in_h * CHANNELS_TILE_HEIGHT + h >= layer_ifm_height)
                        {
                            break;
                        }
                        tmp +=
                            ifms_buffer[h * strides + c_h]
                                       [w * strides + c_w] *
                            weights[d_in_pipeline][c_h * filter_dim + c_w];
                    }
                }
                result_tile[d_in_pipeline][h][w] = tmp;
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

    void fill_dw_weights_tile(const dw_weights_dt weights[][3 * 3],
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
                                                 fms_quantization_scheme normalization, const int layer_relu,
                                                 const fused_scales_dt fused_scales_tile[],
                                                 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[],
                                                 const relu_6_fused_scales_dt relu_6_fused_scales_tile[],
                                                 const biases_dt fused_zero_points_tile[],
                                                 const int ifms_d,
                                                 const int tile_in_h,
                                                 const int tile_in_w,
                                                 const int starting_d,
                                                 const int in_tile_h,
                                                 const int in_tile_w,
                                                 const int strides,
                                                 const int num_of_ofm_tiles_h,
                                                 const int num_of_ofm_tiles_w)
    {
#pragma HLS INLINE off

        const int num_of_tiles_hw = num_of_ofm_tiles_h * num_of_ofm_tiles_w;

        for (int o_d = 0; o_d < CHANNELS_PIPELINE_DEPTH; o_d += CHANNELS_TILE_DEPTH)
        {
#pragma HLS PIPELINE
            for (int i_d = 0; i_d < CHANNELS_TILE_DEPTH; i_d++)
            {
#pragma HLS UNROLL
                const int d = o_d * CHANNELS_TILE_DEPTH + i_d;
                const int main_tile_index = (starting_d + d) * num_of_tiles_hw + tile_in_h * num_of_ofm_tiles_w + tile_in_w;
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

                for (int h = 0; h < CHANNELS_TILE_HEIGHT; h++)
                {
#pragma HLS UNROLL
                    for (int w = 0; w < CHANNELS_TILE_WIDTH; w++)
                    {
#pragma HLS UNROLL
                        if (in_tile_h + h >= CHANNELS_TILE_HEIGHT || in_tile_w + w >= CHANNELS_TILE_WIDTH)
                        {
                            break;
                        }
                        result[main_tile_index][in_tile_h + h][in_tile_w + w] = dw_relu_norm(
                            result_tile[d][h][w], normalization,
                            layer_relu);
                    }
                }
            }
        }
    }

    void dw_conv_3x3(const dw_weights_dt weights[][3 * 3],
                     fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                     fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                     const int layer, const int layer_conv_d, const int layer_ifm_width, const int layer_ifm_height,
                     const int num_of_tiles_d,
                     const int num_of_ifms_tiles_h, const int num_of_ifms_tiles_w,
                     const int num_of_ofms_tiles_h, const int num_of_ofms_tiles_w,
                     const int strides, const int padding_left, const int padding_right, const int padding_top,
                     const fused_scales_dt fused_scales[],
                     const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                     const relu_6_fused_scales_dt relu_6_fused_scales[], const biases_dt fused_zero_points[],
                     const fused_scales_dt fused_scales_part2[],
                     const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
                     const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
                     const biases_dt fused_zero_points_part2[])
    {
#pragma HLS INLINE off

        const int padding_bottom = padding_right;

        const int num_of_ifms_tiles_hw = num_of_ifms_tiles_h * num_of_ifms_tiles_w;
        const int num_of_ofms_tiles_hw = num_of_ofms_tiles_h * num_of_ofms_tiles_w;

        fms_quantization_scheme normalization = {0, 0, 0, 0};

        const int current_dw_layer_weights_offset = dw_layers_weights_offsets[layer];
        const int current_layer_fused_parameters_offset =
            layers_fused_parameters_offsets[layer];

        dw_weights_dt weights_tile[CHANNELS_PIPELINE_DEPTH][3 * 3];
#pragma HLS ARRAY_PARTITION variable = weights_tile type = complete dim = 2

        fused_scales_dt fused_scales_tile[MAX_DW_LAYER_D];
        fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_tile[MAX_DW_LAYER_D];
        relu_6_fused_scales_dt relu_6_fused_scales_tile[MAX_DW_LAYER_D];
        biases_dt fused_zero_points_tile[MAX_DW_LAYER_D];

        normalization.ofm_zero_point = conv_fms_zero_points[layer + 1];
        normalization.ofm_scale_rec = conv_fms_scales_rec[layer + 1];
        normalization.ofm_scale = conv_fms_scales[layer + 1];

        const fms_dt current_layer_fms_zero_point = conv_fms_zero_points[layer];

        dw_pss_dt engine_result_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];
        dw_pss_dt engine_result_tile_copy[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH];

        const int skip_padding_left = padding_left == 0 ? padding_right : 0;

#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = engine_result_tile_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = engine_result_tile_copy type = complete dim = 3

        fms_dt padding_top_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH];
        fms_dt padding_left_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_top_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_top_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_bottom_left_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_bottom_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH];
        fms_dt padding_right_buffer[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT];
        fms_dt padding_bottom_right_buffer[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT];

#pragma HLS ARRAY_PARTITION variable = padding_top_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_left_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_top_left_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_top_right_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_bottom_left_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_bottom_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_right_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_bottom_right_buffer type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_top_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_left_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_top_left_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_top_right_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_bottom_left_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_bottom_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_right_buffer type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_bottom_right_buffer type = complete dim = 3

        fms_dt padding_top_buffer_copy[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][CHANNELS_TILE_WIDTH];
        fms_dt padding_left_buffer_copy[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_top_left_buffer_copy[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_top_right_buffer_copy[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_bottom_left_buffer_copy[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_TOP_LEFT][MAX_TILE_PADDING_TOP_LEFT];
        fms_dt padding_bottom_buffer_copy[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][CHANNELS_TILE_WIDTH];
        fms_dt padding_right_buffer_copy[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT];
        fms_dt padding_bottom_right_buffer_copy[CHANNELS_PIPELINE_DEPTH][MAX_TILE_PADDING_BOTTOM_RIGHT][MAX_TILE_PADDING_BOTTOM_RIGHT];

#pragma HLS ARRAY_PARTITION variable = padding_top_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_left_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_top_left_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_top_right_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_bottom_left_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_bottom_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_right_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_bottom_right_buffer_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = padding_top_buffer_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_left_buffer_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_top_left_buffer_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_top_right_buffer_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_bottom_left_buffer_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_bottom_buffer_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_right_buffer_copy type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = padding_bottom_right_buffer_copy type = complete dim = 3

        if (current_layer_fused_parameters_offset < first_quantization_arrays_num_elements)
        {
            fill_scales_tiles(fused_scales, fused_scales_tile, fused_scales_log_2_shifts,
                              fused_scales_log_2_shifts_tile, relu_6_fused_scales,
                              relu_6_fused_scales_tile, fused_zero_points,
                              fused_zero_points_tile,
                              layer_conv_d,
                              current_layer_fused_parameters_offset);
        }
        else
        {
            fill_scales_tiles(fused_scales_part2, fused_scales_tile, fused_scales_log_2_shifts_part2,
                              fused_scales_log_2_shifts_tile, relu_6_fused_scales_part2,
                              relu_6_fused_scales_tile, fused_zero_points_part2,
                              fused_zero_points_tile,
                              layer_conv_d,
                              current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
        }

        for (int tile_in_h = 0; tile_in_h < num_of_ifms_tiles_h; tile_in_h++)
        {
            const int in_tile_h = (tile_in_h % strides) * CHANNELS_TILE_HEIGHT / strides;
            for (int tile_in_w = 0; tile_in_w < num_of_ifms_tiles_w; tile_in_w++)
            {
                const int in_tile_w = (tile_in_w % strides) * CHANNELS_TILE_WIDTH / strides;

                padd_fms_tile_top_left(channels,
                                       padding_top_buffer,
                                       padding_left_buffer,
                                       padding_top_left_buffer,
                                       padding_top_right_buffer,
                                       padding_bottom_left_buffer,
                                       0,
                                       tile_in_h,
                                       tile_in_w,
                                       padding_top,
                                       layer_conv_d,
                                       num_of_ifms_tiles_h,
                                       num_of_ifms_tiles_w,
                                       layer_ifm_height,
                                       layer_ifm_width,
                                       current_layer_fms_zero_point);
                padd_fms_tile_bottom_right(channels,
                                           padding_bottom_buffer,
                                           padding_right_buffer,
                                           padding_bottom_right_buffer,
                                           0,
                                           tile_in_h,
                                           tile_in_w,
                                           3 - padding_top,
                                           layer_conv_d,
                                           num_of_ifms_tiles_h,
                                           num_of_ifms_tiles_w,
                                           layer_ifm_height,
                                           layer_ifm_width,
                                           current_layer_fms_zero_point);

                for (int dw_pipeline_in_d = 0;
                     dw_pipeline_in_d < layer_conv_d / dw_pipeline_depth;
                     dw_pipeline_in_d++)
                {
                    const int tile_in_d = dw_pipeline_in_d * (dw_pipeline_depth / dw_tile_d);
                    const int prev_tile_in_d = (dw_pipeline_in_d - 1) >= 0
                                                   ? (dw_pipeline_in_d - 1) * (dw_pipeline_depth / dw_tile_d)
                                                   : 0;
                    const int next_tile_in_d = (dw_pipeline_in_d + 1) * (dw_pipeline_depth / dw_tile_d);

                    fill_dw_weights_tile(weights, weights_tile,
                                         tile_in_d,
                                         current_dw_layer_weights_offset);

                    copy_fms_tile_corners(padding_top_buffer,
                                          padding_left_buffer,
                                          padding_top_left_buffer,
                                          padding_top_right_buffer,
                                          padding_bottom_left_buffer,
                                          padding_bottom_buffer,
                                          padding_right_buffer,
                                          padding_bottom_right_buffer,
                                          padding_top_buffer_copy,
                                          padding_left_buffer_copy,
                                          padding_top_left_buffer_copy,
                                          padding_top_right_buffer_copy,
                                          padding_bottom_left_buffer_copy,
                                          padding_bottom_buffer_copy,
                                          padding_right_buffer_copy,
                                          padding_bottom_right_buffer_copy,
                                          tile_in_d,
                                          layer_conv_d,
                                          padding_top,
                                          3 - padding_top);

                    dw_normalize_and_write_back_result_tile(result,
                                                            engine_result_tile_copy, normalization, 6,
                                                            fused_scales_tile, fused_scales_log_2_shifts_tile,
                                                            relu_6_fused_scales_tile, fused_zero_points_tile,
                                                            layer_conv_d,
                                                            tile_in_h / strides, tile_in_w / strides,
                                                            prev_tile_in_d,
                                                            in_tile_h, in_tile_w, strides,
                                                            num_of_ofms_tiles_h, num_of_ofms_tiles_w);
                    dw_conv_engine(weights_tile,
                                   channels,
                                   engine_result_tile,
                                   padding_top_buffer_copy,
                                   padding_left_buffer_copy,
                                   padding_top_left_buffer_copy,
                                   padding_top_right_buffer_copy,
                                   padding_bottom_left_buffer_copy,
                                   padding_bottom_buffer_copy,
                                   padding_right_buffer_copy,
                                   padding_bottom_right_buffer_copy,
                                   3, strides,
                                   layer_conv_d,
                                   tile_in_d,
                                   padding_top,
                                   tile_in_h,
                                   tile_in_w, num_of_ifms_tiles_h, num_of_ifms_tiles_w, layer_ifm_height, layer_ifm_width,
                                   current_layer_fms_zero_point);

                    dw_conv_copy_engine_result_tile(engine_result_tile,
                                                    engine_result_tile_copy, strides);

                    padd_fms_tile_top_left(channels,
                                           padding_top_buffer,
                                           padding_left_buffer,
                                           padding_top_left_buffer,
                                           padding_top_right_buffer,
                                           padding_bottom_left_buffer,
                                           next_tile_in_d,
                                           tile_in_h,
                                           tile_in_w,
                                           padding_top,
                                           layer_conv_d,
                                           num_of_ifms_tiles_h,
                                           num_of_ifms_tiles_w,
                                           layer_ifm_height,
                                           layer_ifm_width,
                                           current_layer_fms_zero_point);
                    padd_fms_tile_bottom_right(channels,
                                               padding_bottom_buffer,
                                               padding_right_buffer,
                                               padding_bottom_right_buffer,
                                               next_tile_in_d,
                                               tile_in_h,
                                               tile_in_w,
                                               3 - padding_top,
                                               layer_conv_d,
                                               num_of_ifms_tiles_h,
                                               num_of_ifms_tiles_w,
                                               layer_ifm_height,
                                               layer_ifm_width,
                                               current_layer_fms_zero_point);
                }
                dw_normalize_and_write_back_result_tile(result,
                                                        engine_result_tile_copy, normalization, 6,
                                                        fused_scales_tile, fused_scales_log_2_shifts_tile,
                                                        relu_6_fused_scales_tile, fused_zero_points_tile,
                                                        layer_conv_d,
                                                        tile_in_h / strides, tile_in_w / strides,
                                                        layer_conv_d % CHANNELS_PIPELINE_DEPTH == 0
                                                            ? layer_conv_d - CHANNELS_PIPELINE_DEPTH
                                                            : layer_conv_d - (layer_conv_d % CHANNELS_PIPELINE_DEPTH),
                                                        in_tile_h, in_tile_w, strides,
                                                        num_of_ofms_tiles_h, num_of_ofms_tiles_w);
            }
        }
    }

#endif
