#include "bottleneck.h"

void mob_v2_bottleneck_pipeline(fms_dt bottleneck_input[],
                                const int bottleneck_ifms_depth,
                                const int bottleneck_input_buffer_height,
                                const int bottleneck_ifms_width,
                                const int bottleneck_ofms_depth,
                                const int bottleneck_ofms_width,
                                const int expanded_ifms_depth,
                                const int dw_filter_dim,
                                const int strides,
                                fms_dt previous_pass_output[], // bottleneck_1_expanded_ifms_depth*bottleneck_1_ifms_width, height=1
                                fms_dt dw_input_buffer[max_of_bottlenecks_layers_depths][],
                                const weights_dt expansion_layer_weights[max_of_bottlenecks_layers_depths][],
                                const dw_weights_dt dw_weights[max_of_bottlenecks_layers_depths][],
                                const weights_dt projection_layer_weights[max_of_bottlenecks_projection_filters][],
                                int starting_h,
                                int starting_w,
                                const int bottleneck_expansion_parallelism_h,
                                const int bottleneck_expansion_parallelism_w)
{
    for (int d_in_out = 0; d_in_out < expanded_ifms_depth; d_in_out++)
    {
#pragma HLS PIPELINE
        fms_dt expansion_input_buffer[bottleneck_ifms_depth * bottleneck_expansion_parallelism_h *
                                      bottleneck_expansion_parallelism_w];
        pss_dt expansion_results_buffer[bottleneck_expansion_parallelism_h * bottleneck_expansion_parallelism_w];
        pss_dt projection_results_buffer[bottleneck_ofms_depth];

        for (int p_h = 0; p_h < bottleneck_expansion_parallelism_h; p_h++)
        {
#pragma HLS UNROLL
            for (int p_w = 0; p_w < bottleneck_expansion_parallelism_w; p_w++)
            {
#pragma HLS UNROLL
                fill_expansion_kernel_buffer(bottleneck_input, expansion_input_buffer, bottleneck_ifms_depth,
                                             bottleneck_ifms_width, p_h, starting_w + p_w);
                expansion_results_buffer[p_h * bottleneck_expansion_parallelism_w + p_w] =
                    expansion_kernel(expansion_input_buffer, bottleneck_ifms_depth, expansion_layer_weights, d_in_out);
            }
        }

        fill_dw_ifms_buffer_upper_part(dw_input_buffer, previous_pass_output, strides, dw_filter_dim, starting_w * strides,
                                       bottleneck_ifms_width, d_in_out);
        fill_dw_ifms_buffer_lower_part(dw_input_buffer,
                                       expansion_results_buffer, strides, dw_filter_dim, d_in_out);
        fms_dt dw_result = dw_kernel(dw_input_buffer, dw_weights, dw_filter_dim, d_in_out);

        shift_dw_ifms_buffer_horizontally(dw_input_buffer, strides, dw_filter_dim,
                                          d_in_out);

        projection_kernel(dw_result, bottleneck_ofms_depth, projection_layer_weights, projection_results_buffer, d_in_out);
    }
}

void mob_v2_bottleneck(fms_dt bottleneck_input[],
                       const int bottleneck_ifms_depth,
                       const int bottleneck_input_buffer_height,
                       const int bottleneck_ifms_width,
                       const int bottleneck_ofms_depth,
                       const int bottleneck_ofms_width,
                       const int expanded_ifms_depth,
                       const int dw_filter_dim,
                       const int strides,
                       fms_dt bottleneck_output[],    // bottleneck_1_ofms_depth*bottleneck_1_ofms_width
                       fms_dt previous_pass_output[], // bottleneck_1_expanded_ifms_depth*bottleneck_1_ifms_width, height=1
                       const weights_dt expansion_layer_weights[max_of_bottlenecks_layers_depths][],
                       const dw_weights_dt dw_weights[max_of_bottlenecks_layers_depths][],
                       const weights_dt projection_layer_weights[max_of_bottlenecks_projection_filters][],
                       int starting_h,
                       const int bottleneck_expansion_parallelism_h,
                       const int bottleneck_expansion_parallelism_w)
{

#pragma HLS INLINE off

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
            fms_dt dw_input_buffer[max_of_bottlenecks_layers_depths][dw_filter_dim * dw_filter_dim]; // depth = 1
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0
            mob_v2_bottleneck_pipeline(bottleneck_ifms_depth,
                                       bottleneck_input_buffer_height,
                                       bottleneck_ifms_width,
                                       bottleneck_ofms_depth,
                                       bottleneck_ofms_width,
                                       expanded_ifms_depth,
                                       dw_filter_dim,
                                       strides,
                                       previous_pass_output,
                                       dw_input_buffer,
                                       expansion_layer_weights,
                                       dw_weights,
                                       projection_layer_weights,
                                       starting_h,
                                       w);
        }
    }
}