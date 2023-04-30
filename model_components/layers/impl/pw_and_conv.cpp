#include "../headers/layers_imp_common_includes.h"
#include "../headers/pw_conv.h"
#include "../headers/conv_utils.h"

#if FIBHA_VERSION == 2

void pw_and_conv_eng(weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d][max_filter_area],
                     fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED],
                     pss_dt results_tile[pw_tile_d][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                     const int starting_conv_d, const int starting_filter, const layer_specs layer_specs_struct)
{

    const int layer_depth = layer_specs_struct.layer_depth;
    const int layer_filters_hw = layer_specs_struct.filter_size;
    const int layer_num_of_filters = layer_specs_struct.layer_num_fils;
    const int strides = layer_specs_struct.strides;

#pragma HLS INLINE
pw_conv_eng_loops:
    for (int c_h = 0; c_h < max_filter_hw_dim; c_h++)
    {
        for (int c_w = 0; c_w < max_filter_hw_dim; c_w++)
        {
            for (int d = 0; d < CHANNELS_PIPELINE_DEPTH; d++)
            {
#pragma HLS PIPELINE
                for (int f_d = 0; f_d < pw_conv_parallelism_out; f_d++)
                {
#pragma HLS UNROLL
                    for (int t_h = 0; t_h < pw_tile_h; t_h++)
                    {
#pragma HLS UNROLL
                        for (int t_w = 0; t_w < pw_tile_w; t_w++)
                        {
#pragma HLS UNROLL
                            if (t_h >= dw_tile_h / strides || t_w >= dw_tile_w / strides ||
                                d + starting_conv_d >= layer_depth || c_h >= layer_filters_hw || c_w >= layer_filters_hw)
                            {
                                break;
                            }
                            results_tile[f_d][t_h][t_w] += channels_tile[d][t_h * strides + c_h][t_w * strides + c_w] *
                                                           weights_tile[f_d][starting_conv_d + d]
                                                                       [c_h * max_conv_filter_hw_dim + c_w];
                        }
                    }
                }
            }
        }
    }
}

void pw_conv_pipeline(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                      weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d][max_filter_area],
                      pss_dt results_tile[pw_tile_d][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
                      layer_specs layer_specs_struct,
                      const int td_o,
                      const int t_in_h, const int t_in_w)
{
#pragma HLS INLINE OFF

    const int num_of_tiles_d_in = layer_specs_struct.layer_num_of_tiles_in_d;
    const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
    const int num_of_tiles_hw = layer_specs_struct.layer_num_of_ifm_tiles_h * num_of_tiles_w;
    const int layer_ifms_depth = layer_specs_struct.layer_depth;
    const int layer_num_fils = layer_specs_struct.layer_num_fils;

    const int iterations_in_d = (layer_ifms_depth + CHANNELS_PIPELINE_DEPTH - 1) / CHANNELS_PIPELINE_DEPTH;

    fms_dt channels_tile[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED];
    fms_dt channels_tile_copy[CHANNELS_PIPELINE_DEPTH][CHANNELS_TILE_HEIGHT_PADDED][CHANNELS_TILE_WIDTH_PADDED];

    const fms_dt current_layer_fms_zero_point = layer_specs_struct.layer_ifms_zero_point;

#pragma HLS ARRAY_PARTITION variable = channels_tile type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_tile type = complete dim = 3
#pragma HLS ARRAY_PARTITION variable = channels_tile_copy type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = channels_tile_copy type = complete dim = 3

    fill_fms_tile(channels,
                  channels_tile,
                  0,
                  t_in_h,
                  t_in_w,
                  current_layer_fms_zero_point,
                  layer_specs_struct);

conv2_itd_loop:
    for (int td_i = 0; td_i < iterations_in_d; td_i++)
    {
        const int tile_in_d = td_i * CHANNELS_PIPELINE_DEPTH;
        const int next_tile_in_d = (td_i + 1) * CHANNELS_PIPELINE_DEPTH;
#pragma HLS ARRAY_PARTITION variable = channels_buffer complete dim = 0
        if (td_i % 2 == 0)
        {
            pw_and_conv_eng(weights_tile, channels_tile, results_tile,
                            tile_in_d, td_o * pw_conv_parallelism_out, layer_specs_struct);
            fill_fms_tile(channels,
                          channels_tile_copy,
                          next_tile_in_d,
                          t_in_h,
                          t_in_w,
                          current_layer_fms_zero_point,
                          layer_specs_struct);
        }
        else
        {
            pw_and_conv_eng(weights_tile, channels_tile_copy, results_tile,
                            tile_in_d, td_o * pw_conv_parallelism_out, layer_specs_struct);
            fill_fms_tile(channels,
                          channels_tile,
                          next_tile_in_d,
                          t_in_h,
                          t_in_w,
                          current_layer_fms_zero_point,
                          layer_specs_struct);
        }
    }
}

void do_conv(weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d][max_filter_area],
             fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
             fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
             fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
             const int layer, const layer_specs layer_specs_struct,
             fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out],
             fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out],
             relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out],
             biases_dt fused_zero_points_buffer[pw_conv_parallelism_out], int td_o)
{

#pragma HLS INLINE off

    const int num_of_tiles_h = layer_specs_struct.layer_num_of_ifm_tiles_h;
    const int num_of_tiles_w = layer_specs_struct.layer_num_of_ifm_tiles_w;
    const int num_of_ofm_tiles_h = layer_specs_struct.layer_num_of_ofm_tiles_h;
    const int num_of_ofm_tiles_w = layer_specs_struct.layer_num_of_ofm_tiles_w;
    const int num_of_tiles_hw = num_of_tiles_h * num_of_tiles_w;
    const int num_of_ofm_tiles_hw = num_of_ofm_tiles_h * num_of_ofm_tiles_w;
    const int strides = layer_specs_struct.strides;

    fms_quantization_scheme normalization;

    pss_f_dt tmp_channels_scaled_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = tmp_channels_scaled_tile complete dim = 3

    pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w] = {0};
    pss_dt prev_results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w] = {0};
    fms_dt scaled_result_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w];
#pragma HLS ARRAY_PARTITION variable = results_tile complete dim = 0
#pragma HLS ARRAY_PARTITION variable = prev_results_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = prev_results_tile complete dim = 2
#pragma HLS ARRAY_PARTITION variable = scaled_result_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = scaled_result_tile complete dim = 2

    int prev_tile_index = -1;
    int in_tile_h = -1;
    int in_tile_w = -1;

conv2_ith_loop:
    for (int t_in_h = 0; t_in_h < num_of_tiles_h; t_in_h++)
    {
        //############width loop##############
    conv2_itw_loop:
        for (int t_in_w = 0; t_in_w < num_of_tiles_w;
             t_in_w++)
        {
            //############depth loop##############
            int tile_index = td_o * (pw_conv_parallelism_out / pw_tile_d) * num_of_ofm_tiles_hw + (t_in_h / strides) * num_of_ofm_tiles_w +
                             (t_in_w / strides);

            scale_pss_tile(tmp_channels, prev_results_tile, scaled_result_tile,
                           layer_specs_struct, fused_scales_buffer,
                           fused_scales_log_2_shifts_buffer,
                           relu_6_fused_scales_buffer, fused_zero_points_buffer,
                           prev_tile_index);

            pw_conv_pipeline(channels, weights_tile, results_tile,
                             layer_specs_struct,
                             td_o, t_in_h, t_in_w);
            copy_pss_tile(results_tile, prev_results_tile);
            pw_write_results_tile(scaled_result_tile, result,
                                  prev_tile_index, tmp_channels, tmp_channels_scaled_tile,
                                  td_o * pw_conv_parallelism_out,
                                  in_tile_h, in_tile_w,
                                  layer_specs_struct);

            in_tile_w = (t_in_w % strides) * (CHANNELS_TILE_WIDTH / strides);
            in_tile_h = (t_in_h % strides) * (CHANNELS_TILE_HEIGHT / strides);
            prev_tile_index = tile_index;
        }
    }
    scale_pss_tile(tmp_channels, prev_results_tile, scaled_result_tile,
                   layer_specs_struct, fused_scales_buffer,
                   fused_scales_log_2_shifts_buffer,
                   relu_6_fused_scales_buffer, fused_zero_points_buffer,
                   prev_tile_index);

    pw_write_results_tile(scaled_result_tile, result,
                          prev_tile_index, tmp_channels, tmp_channels_scaled_tile,
                          td_o * pw_conv_parallelism_out,
                          strides == 1 ? 0 : (CHANNELS_TILE_HEIGHT / strides),
                          strides == 1 ? 0 : (CHANNELS_TILE_WIDTH / strides),
                          layer_specs_struct);
}

void pw_and_conv(weights_grp_dt *weights,
                 fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                 fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                 fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
                 int layer, const layer_specs layer_specs_struct,
                 const fused_scales_dt fused_scales[],
                 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[],
                 const relu_6_fused_scales_dt relu_6_fused_scales[],
                 const biases_dt fused_zero_points[],
                 const fused_scales_dt fused_scales_part2[],
                 const fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_part2[],
                 const relu_6_fused_scales_dt relu_6_fused_scales_part2[],
                 const biases_dt fused_zero_points_part2[])
{
#pragma HLS INLINE off

    weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d][max_filter_area];
#pragma HLS ARRAY_PARTITION variable = weights_tile complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights_tile cyclic dim = 2 factor = num_of_weights_in_the_same_filter_and_group

#if HW == _FPGA
    weights_grp_dt weight_groups_buffer[num_of_weight_groups_in_the_largest_weight_tile];
    fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer, 0,
                                           layer_specs_struct.layer_depth, layer_specs_struct.layer_num_of_weight_groups_for_one_pass,
                                           layer_specs_struct.layer_weights_offset,
                                           layer_specs_struct.layer_num_fils);
#elif HW == CPU
    fill_layers_weights_cpu_pw_conv(weights,
                                    weights_tile,
                                    0 * pw_conv_parallelism_out, layer_specs_struct.layer_depth,
                                    layer_specs_struct.layer_weights_offset, layer_specs_struct.layer_num_fils);
#endif

    biases_dt fused_zero_points_buffer[pw_conv_parallelism_out];
    fused_scales_dt fused_scales_buffer[pw_conv_parallelism_out];
    fused_scales_log_2_shifts_dt fused_scales_log_2_shifts_buffer[pw_conv_parallelism_out];
    relu_6_fused_scales_dt relu_6_fused_scales_buffer[pw_conv_parallelism_out];

    const int current_layer_fused_parameters_offset = layers_fused_parameters_offsets[layer];

conv2_ots_loop:
    for (int td_o = 0; td_o < layer_specs_struct.layer_num_of_tiles_out_d; td_o++)
    {
        if (current_layer_fused_parameters_offset < first_quantization_arrays_num_elements)
        {
            fill_fused_zero_points_buffer(fused_zero_points,
                                          fused_zero_points_buffer, td_o * pw_conv_parallelism_out,
                                          layer, current_layer_fused_parameters_offset);
            fill_fused_scales_buffer(fused_scales, fused_scales_buffer,
                                     fused_scales_log_2_shifts, fused_scales_log_2_shifts_buffer,
                                     relu_6_fused_scales, relu_6_fused_scales_buffer,
                                     td_o * pw_conv_parallelism_out, layer, current_layer_fused_parameters_offset);
        }
        else
        {
            fill_fused_zero_points_buffer(fused_zero_points_part2,
                                          fused_zero_points_buffer, td_o * pw_conv_parallelism_out,
                                          layer, current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
            fill_fused_scales_buffer(fused_scales_part2, fused_scales_buffer,
                                     fused_scales_log_2_shifts_part2, fused_scales_log_2_shifts_buffer,
                                     relu_6_fused_scales_part2, relu_6_fused_scales_buffer,
                                     td_o * pw_conv_parallelism_out, layer,
                                     current_layer_fused_parameters_offset - first_quantization_arrays_num_elements);
        }

#if HW == _FPGA
        fill_weights_tile_from_weight_groups_tile(weight_groups_buffer,
                                                  weights_tile, td_o * pw_conv_parallelism_out,
                                                  layer_specs_struct.layer_depth,
                                                  layer_specs_struct.layer_num_of_weight_groups_for_one_pass,
                                                  layer_specs_struct.layer_weights_offset);
#endif
        do_conv(weights_tile, channels, result, tmp_channels, layer, layer_specs_struct, fused_scales_buffer,
                fused_scales_log_2_shifts_buffer, relu_6_fused_scales_buffer,
                fused_zero_points_buffer, td_o);
#if HW == _FPGA
        fill_layer_weight_groups_tile_off_chip(weights, weight_groups_buffer,
                                               (td_o + 1) * pw_conv_parallelism_out,
                                               layer_specs_struct.layer_depth,
                                               layer_specs_struct.layer_num_of_weight_groups_for_one_pass,
                                               layer_specs_struct.layer_weights_offset,
                                               layer_specs_struct.layer_num_fils);
#elif HW == CPU
        fill_layers_weights_cpu_pw_conv(weights,
                                        weights_tile,
                                        (td_o + 1) * pw_conv_parallelism_out, layer_specs_struct.layer_depth,
                                        layer_specs_struct.layer_weights_offset, layer_specs_struct.layer_num_fils);
#endif
    }
}

#endif