#include "bottleneck.h"

void mob_v2_bottleneck(fms_dt bottleneck_input[],
                       fms_dt bottleneck_output[],
                       fms_dt previous_pass_dw_input[],
                       const weights_dt expansion_layer_weights[max_of_bottlenecks_layers_depths][],
                       const dw_weights_dt dw_weights[max_of_bottlenecks_layers_depths][],
                       const weights_dt projection_layer_weights[max_of_bottlenecks_projection_filters][],
                       const fused_scales_dt expansion_layer_fused_scales[],
                       const fused_scales_log_2_shifts_dt expansion_layer_fused_scales_log_2_shifts[],
                       const relu_6_fused_scales_dt expansion_layer_relu_6_fused_scales[],
                       const biases_dt expansion_layer_fused_zero_points[],
                       const fused_scales_dt dw_layer_fused_scales[],
                       const fused_scales_log_2_shifts_dt dw_layer_fused_scales_log_2_shifts[],
                       const relu_6_fused_scales_dt dw_layer_relu_6_fused_scales[],
                       const biases_dt dw_layer_fused_zero_points[],
                       const fused_scales_dt projection_layer_fused_scales[],
                       const fused_scales_log_2_shifts_dt projection_layer_fused_scales_log_2_shifts[],
                       const relu_6_fused_scales_dt projection_layer_relu_6_fused_scales[],
                       const biases_dt projection_layer_fused_zero_points[],
                       const int bottleneck_ifms_depth,
                       const int bottleneck_ifms_height,
                       const int bottleneck_ifms_width,
                       const int bottleneck_ofms_depth,
                       const int bottleneck_ofms_width,
                       const int expanded_ifms_depth,
                       const int dw_filter_dim,
                       const int strides,
                       int starting_h,
                       int starting_w,
                       const int bottleneck_expansion_parallelism_h,
                       const int bottleneck_expansion_parallelism_w,
                       const int bottleneck_expansion_layer_index,
                       const int bottleneck_dw_layer_index,
                       const int bottleneck_projection_layer_index,
                       const int expansion_layer_relu,
                       const int dw_layer_relu,
                       const int projection_layer_relu,
                       const int padding_left, const int padding_right, const int padding_top, const int padding_bottom)
{

#pragma HLS INLINE off

    fms_dt dw_input_buffer[max_of_bottlenecks_layers_depths][dw_filter_dim * dw_filter_dim]; // depth = 1
#pragma HLS ARRAY_PARTITION variable = intermediate_channels_buffer type = complete dim = 0

    const fms_dt expansion_layer_ofms_zero_point = conv_fms_zero_points[bottleneck_expansion_layer_index + 1];
    const rec_scales_dt expansion_layer_ofms_scale_rec = conv_fms_scales_rec[bottleneck_expansion_layer_index + 1];
    const rec_scales_dt expansion_layer_ofms_scale = conv_fms_scales[bottleneck_expansion_layer_index + 1];

    const fms_dt dw_layer_ofms_zero_point = conv_fms_zero_points[bottleneck_dw_layer_index + 1];
    const rec_scales_dt dw_layer_ofms_scale_rec = conv_fms_scales_rec[bottleneck_dw_layer_index + 1];
    const rec_scales_dt dw_layer_ofms_scale = conv_fms_scales[bottleneck_dw_layer_index + 1];
    const fms_dt current_dw_ifms_zero_point = conv_fms_zero_points[bottleneck_dw_layer_index];

    fms_quantization_scheme expansion_layer_normalization;
    expansion_layer_normalization.ofm_zero_point = expansion_layer_ofms_zero_point;
    expansion_layer_normalization.ofm_scale_rec = expansion_layer_ofms_scale_rec;
    expansion_layer_normalization.ofm_scale = expansion_layer_ofms_scale;

    fms_quantization_scheme dw_layer_normalization;
    dw_layer_normalization.ofm_zero_point = dw_layer_ofms_zero_point;
    dw_layer_normalization.ofm_scale_rec = dw_layer_ofms_scale_rec;
    dw_layer_normalization.ofm_scale = dw_layer_ofms_scale;

    pss_dt projection_results_buffer[bottleneck_ofms_depth];

    for (int d_in_out = 0; d_in_out < expanded_ifms_depth; d_in_out++)
    {
#pragma HLS PIPELINE
        fms_dt expansion_results_buffer[bottleneck_expansion_parallelism_h * bottleneck_expansion_parallelism_w];

        expansion_layer_normalization.fused_scales =
            expansion_layer_fused_scales[d_in_out];
        expansion_layer_normalization.fused_scales_log_2_shift =
            expansion_layer_fused_scales_log_2_shifts[d_in_out];
        expansion_layer_normalization.relu_6_fused_scale =
            expansion_layer_relu_6_fused_scales[d_in_out];
        expansion_layer_normalization.fused_zero_point =
            expansion_layer_fused_zero_points[d_in_out];

        for (int p_h = 0; p_h < bottleneck_expansion_parallelism_h; p_h++)
        {
#pragma HLS UNROLL
    //TODO first column, do not run dw
            for (int p_w = 0; p_w < bottleneck_expansion_parallelism_w; p_w++)
            {
#pragma HLS UNROLL
                if (starting_h + p_h < bottleneck_ifms_height + padding_top)
                {
                    expansion_results_buffer[p_h * bottleneck_expansion_parallelism_w + p_w] = pw_relu_norm(
                        expansion_kernel(bottleneck_input, bottleneck_ifms_depth,
                                         expansion_layer_weights, d_in_out),
                        expansion_layer_normalization, expansion_layer_relu);
                }
                else
                {
                    expansion_results_buffer[p_h * bottleneck_expansion_parallelism_w + p_w] = current_dw_ifms_zero_point;
                }
            }
        }

        fill_dw_ifms_buffer_upper_part(dw_input_buffer, previous_pass_dw_input, strides, dw_filter_dim, starting_w * strides,
                                       bottleneck_ifms_width, d_in_out, padding_left);
        fill_dw_ifms_buffer_lower_part(dw_input_buffer,
                                       expansion_results_buffer, strides, dw_filter_dim, d_in_out);
        update_dw_ifms_buffer_upper_part(previous_pass_dw_input, expansion_results_buffer, strides, dw_filter_dim,
                                         starting_w * strides, bottleneck_ifms_width, d_in_out, d_in_out, padding_left);

        dw_layer_normalization.fused_scales =
            dw_layer_fused_scales[d_in_out];
        dw_layer_normalization.fused_scales_log_2_shift =
            dw_layer_fused_scales_log_2_shifts[d_in_out];
        dw_layer_normalization.relu_6_fused_scale =
            dw_layer_relu_6_fused_scales[d_in_out];
        dw_layer_normalization.fused_zero_point =
            dw_layer_fused_zero_points[d_in_out];

        fms_dt dw_result = dw_relu_norm(dw_kernel(dw_input_buffer, dw_weights, dw_filter_dim, d_in_out),
                                        dw_layer_normalization, dw_layer_relu);

        shift_dw_ifms_buffer_horizontally(dw_input_buffer, strides, dw_filter_dim,
                                          d_in_out);

        projection_kernel(dw_result, bottleneck_ofms_depth, projection_layer_weights, projection_results_buffer, d_in_out);
    }
    normalize_projection_kernel_output(projection_results_buffer, bottleneck_output,
                                       projection_layer_fused_scales,
                                       projection_layer_fused_scales_log_2_shifts,
                                       projection_layer_relu_6_fused_scales,
                                       projection_layer_fused_zero_points, bottleneck_ofms_depth,
                                       projection_layer_relu, bottleneck_projection_layer_index);
}