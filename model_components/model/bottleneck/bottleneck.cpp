#include "bottleneck.h"

void mob_v2_bottleneck(fms_dt bottleneck_input[],
                       const int bottleneck_ifms_depth,
                       const int bottleneck_input_buffer_height,
                       const int bottleneck_ifms_width,
                       const int bottleneck_ofms_depth,
                       const int bottleneck_ofms_width,
                       const int bottleneck_expanded_ifms_depth,
                       fms_dt bottleneck_output[],    // bottleneck_1_ofms_depth*bottleneck_1_ofms_width
                       fms_dt previous_pass_output[]; // bottleneck_1_expanded_ifms_depth*bottleneck_1_ifms_width, height=1
                       const weights_dt expansion_layer_weights[],
                       const dw_weights_dt layer_5_dw_weights[max_of_bottlenecks_layers_depths][],
                       const weights_dt projection_layer_weights[max_of_bottlenecks_projection_filters][],
                       const int current_h)
{

#pragma HLS INLINE off
    fms_dt intermediate_channels_buffer[bottleneck_1_dw_filter_dim][bottleneck_1_dw_filter_dim]; // depth = 1
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

    const int expansion_layer_fused_parameters_offsets =
        layers_fused_parameters_offsets[4];
    const fms_dt expansion_layer_ofms_zero_point = conv_fms_zero_points[bottleneck_1_expansion_layer_index + 1];
    const rec_scales_dt expansion_layer_ofms_scale_rec = conv_fms_scales_rec[bottleneck_1_expansion_layer_index + 1];
    const rec_scales_dt expansion_layer_ofms_scale = conv_fms_scales[bottleneck_1_expansion_layer_index + 1];

    const int dw_layer_fused_parameters_offsets =
        layers_fused_parameters_offsets[bottleneck_1_dw_layer_index];
    const fms_dt dw_layer_ofms_zero_point = conv_fms_zero_points[bottleneck_1_dw_layer_index + 1];
    const rec_scales_dt dw_layer_ofms_scale_rec = conv_fms_scales_rec[bottleneck_1_dw_layer_index + 1];
    const rec_scales_dt dw_layer_ofms_scale = conv_fms_scales[bottleneck_1_dw_layer_index + 1];
    const fms_dt current_dw_ifms_zero_point = conv_fms_zero_points[bottleneck_1_dw_layer_index];

    const int projection_layer_fused_parameters_offsets =
        layers_fused_parameters_offsets[bottleneck_1_projection_layer_index];
    const fms_dt projection_layer_ofms_zero_point = conv_fms_zero_points[bottleneck_1_projection_layer_index + 1];
    const rec_scales_dt projection_layer_ofms_scale_rec = conv_fms_scales_rec[bottleneck_1_projection_layer_index + 1];
    const rec_scales_dt projection_layer_ofms_scale = conv_fms_scales[bottleneck_1_projection_layer_index + 1];

    for (int h = 0; h < bottleneck_1_input_buffer_height / bottleneck_1_expansion_parallelism_h; h++)
    {
        for (int w = 0; w < bottleneck_1_ofms_width; w++)
        {
            fms_dt expansion_input_buffer[bottleneck_1_ifms_depth * bottleneck_1_expansion_parallelism_h *
                                          bottleneck_1_expansion_parallelism_w];
            pss_dt expansion_results_buffer[bottleneck_1_expansion_parallelism_h * bottleneck_1_expansion_parallelism_w];
            fill_expansion_kernel_buffer(bottleneck_input, expansion_input_buffer, bottleneck_ifms_depth,
                                         bottleneck_ifms_width, p_h, int w + p_w);

            for (int p_h = 0; p_h < bottleneck_1_expansion_parallelism_h; p_h++)
            {
#pragma HLS UNROLL
                for (int p_w = 0; p_w < bottleneck_1_expansion_parallelism_w; p_w++)
                {
#pragma HLS UNROLL
                    expansion_results_buffer[h * bottleneck_1_expansion_parallelism_w + w] =
                        expansion_kernel(fms_dt ifms_buffer[], const int ifms_depth, weights_dt filter[]);
                }
            }
        }
    }

    const int filter_shift_rows = layer_5_dw_filter_size - layer_5_dw_strides;
    const int filter_shift_offset = filter_shift_rows;
    const int extra_cols_filled_first_time = layer_5_dw_filter_size - (layer_5_dw_padding_left + layer_5_dw_strides);
    const int cols_filled_first_time = layer_5_dw_filter_size - layer_5_dw_padding_left;

    const int pw_iterations_before_first_dw = filter_shift_offset;

layer_4_pw__dw_main_loop:
    for (int o_o_d = 0;
         o_o_d < layer_4_pw_num_fils / layer_4_pw_parallelism_out; o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_4_pw_parallelism_out;

    layer_4_pw_pipeline:
        for (int w = 0;
             w < layer_5_dw_ofm_width + pw_iterations_before_first_dw; w++)
        {
#pragma HLS PIPELINE
            //###################PW#######################
            const int pw_starting_point = w * layer_5_dw_strides;
            if (w < layer_5_dw_ofm_width)
            {
                for (int pw_w = 0;
                     pw_w < layer_5_dw_strides; pw_w++)
                {
                    const int pw_w_index = pw_starting_point + pw_w;
                layer_4_pw_loops:
                    for (int o_d = 0;
                         o_d < layer_4_pw_parallelism_out; o_d++)
                    {
#pragma HLS UNROLL
                        // parallelized filters loop
                        for (int row = 0; row < _6_stages_layer_4_rows_at_once;
                             row++)
                        {
#pragma HLS UNROLL
                            if (starting_h + row < layer_4_pw_ifm_height)
                            {
                                // FMs width loop
                                pss_dt tmp = 0;
                                for (int d = 0; d < layer_4_pw_parallelism_in;
                                     d++)
                                {
#pragma HLS UNROLL
                                    // parallelized depth loop
                                    tmp +=
                                        ((fms_dt)channels_buffer[d][row][pw_w_index]) * weights[o_o_d_offset + o_d][d];
                                }

                                fms_quantization_scheme normalization = {0, 0,
                                                                         0, 0};
                                normalization.fused_scales =
                                    layer_4_fused_scales[o_o_d_offset + o_d];
                                normalization.relu_6_fused_scale =
                                    layer_4_relu_6_fused_scales[o_o_d_offset + o_d];
                                normalization.fused_zero_point =
                                    layer_4_fused_zero_points[o_o_d_offset + o_d];
                                normalization.ofm_zero_point =
                                    current_pw_ofms_zero_point;
                                normalization.ofm_scale_rec =
                                    current_pw_ofms_scale_rec;
                                normalization.ofm_scale =
                                    current_pw_ofms_scale;
                                fms_dt scaled_val = pw_relu_norm(tmp,
                                                                 normalization, layer_4_relu);

                                lower[o_o_d_offset + o_d][row][pw_w_index] =
                                    scaled_val;
                                // fill first col if it is the beginning of a row
                                if (layer_5_dw_padding_left == 0 && w == 0 && pw_w_index < extra_cols_filled_first_time)
                                {
                                    intermediate_channels_buffer[o_d][row + filter_shift_rows][pw_w_index] =
                                        scaled_val;
                                }
                            }
                            else
                            {
                                lower[o_o_d_offset + o_d][row][pw_w_index] =
                                    current_dw_ifms_zero_point;
                                // fill first col if it is the beginning of a row
                                if (layer_5_dw_padding_left == 0 && w == 0 && pw_w_index < extra_cols_filled_first_time)
                                {
                                    intermediate_channels_buffer[o_d][row + filter_shift_rows][pw_w_index] =
                                        current_dw_ifms_zero_point;
                                }
                            }
                        }
                    }
                }
            }

            if (w == 0)
            { // not enough columns are ready
                continue;
            }

            //###############end PW####################
            //###############DW########################
            const int dw_starting_point = (w - pw_iterations_before_first_dw) * layer_5_dw_strides;
            const int dw_starting_next_iter_point = (w - pw_iterations_before_first_dw + 1) * layer_5_dw_strides;
        layer_5_fill_loops:
            for (int o_d = 0;
                 o_d < layer_4_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                for (int row = filter_shift_offset;
                     row < layer_5_dw_filter_size; row++)
                {
                    for (int c_w = filter_shift_offset;
                         c_w < layer_5_dw_filter_size; c_w++)
                    {
                        // conv width loop
#pragma HLS UNROLL
                        if (dw_starting_point + (c_w - filter_shift_offset) + extra_cols_filled_first_time < layer_5_dw_ifm_width)
                        {
                            intermediate_channels_buffer[o_d][row][c_w] =
                                lower[o_o_d_offset + o_d][row - filter_shift_offset][dw_starting_point + (c_w - filter_shift_offset) + extra_cols_filled_first_time];
                        }
                        else
                        { // padding right
                            intermediate_channels_buffer[o_d][row][c_w] =
                                current_dw_ifms_zero_point;
                        }
                    }
                }

                dw_pss_dt tmp = 0;
                // parallelized depth loop
                for (int c_h = 0; c_h < layer_5_dw_filter_size; c_h++)
                {
#pragma HLS UNROLL
                    for (int c_w = 0; c_w < layer_5_dw_filter_size; c_w++)
                    {
                        // conv width loop
#pragma HLS UNROLL
                        tmp += intermediate_channels_buffer[o_d][c_h][c_w] * dw_weights[o_o_d_offset + o_d][c_h][c_w];
                    }
                }

                fms_quantization_scheme normalization = {0, 0, 0, 0};
                normalization.fused_scales =
                    layer_5_fused_scales[o_o_d_offset + o_d];
                normalization.relu_6_fused_scale =
                    layer_5_relu_6_fused_scales[o_o_d_offset + o_d];
                normalization.fused_zero_point =
                    layer_5_fused_zero_points[o_o_d_offset + o_d];
                normalization.ofm_zero_point = current_dw_ofms_zero_point;
                normalization.ofm_scale_rec = current_dw_ofms_scale_rec;
                normalization.ofm_scale = current_dw_ofms_scale;
                result[o_o_d_offset + o_d][w - pw_iterations_before_first_dw] =
                    dw_relu_norm(tmp, normalization, 6);

                //#####################end DW################
                //#####################shift and fill intermediate#################
                for (int c_h = 0; c_h < layer_5_dw_filter_size; c_h++)
                {
#pragma HLS UNROLL
                    for (int c_w = 0; c_w < filter_shift_offset; c_w++)
                    {
#pragma HLS UNROLL
                        intermediate_channels_buffer[o_d][c_h][c_w] =
                            intermediate_channels_buffer[o_d][c_h][c_w + layer_5_dw_strides];
                    }
                }
                for (int c_h = 0; c_h < filter_shift_rows; c_h++)
                {
#pragma HLS UNROLL
                    for (int c_w = filter_shift_offset;
                         c_w < layer_5_dw_filter_size; c_w++)
                    {
#pragma HLS UNROLL
                        if (dw_starting_next_iter_point + (c_w - filter_shift_offset) + extra_cols_filled_first_time < layer_5_dw_ifm_width)
                        {
                            intermediate_channels_buffer[o_d][c_h][c_w] =
                                upper[o_o_d_offset + o_d][dw_starting_next_iter_point + (c_w - filter_shift_offset) + extra_cols_filled_first_time];
                        }
                        else
                        { // padding right
                            intermediate_channels_buffer[o_d][c_h][c_w] =
                                current_dw_ifms_zero_point;
                        }
                    }
                }
                //#####################end shift and fill intermediate#################
            } // o_d
        }     // w
    }

layer_4_pw_dw_shift_loop:
    for (int o_o_d = 0;
         o_o_d < layer_4_pw_num_fils / layer_4_pw_parallelism_out; o_o_d++)
    {
        int o_o_d_offset = o_o_d * layer_4_pw_parallelism_out;
    layer_3_shift_pipeline:
        for (int w = 0; w < layer_5_dw_ifm_width; w++)
        {
#pragma HLS UNROLL factor = 4
        //###################PW#######################
        layer_1_shift_loops:
            for (int o_d = 0;
                 o_d < layer_4_pw_parallelism_out; o_d++)
            {
#pragma HLS UNROLL
                upper[o_o_d_offset + o_d][w] =
                    lower[o_o_d_offset + o_d][layer_5_dw_strides - 1][w];
            }
        }
    }
}