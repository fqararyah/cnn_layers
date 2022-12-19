#ifndef BOTTLENECK_H
#define BOTTLENECK_H
#include "bottlenecks_sesl_specs.h"
#include "bottleneck_kernels.h"

void mob_v2_bottleneck_0(fms_dt bottleneck_input[],
                       fms_dt bottleneck_output[],
					   fms_dt r_previous_pass_dw_input[],
					   fms_dt w_previous_pass_dw_input[], int starting_h, int starting_w);

void mob_v2_bottleneck(fms_dt bottleneck_input[],
                       fms_dt bottleneck_output[],
					   fms_dt r_previous_pass_dw_input[],
					   fms_dt w_previous_pass_dw_input[],
                       const weights_dt expansion_layer_weights[][max_of_bottlenecks_expansion_layers_depths],
                       const dw_weights_dt dw_weights[][max_dw_filter_area_in_a_chain],
                       const weights_dt projection_layer_weights[][max_of_bottlenecks_layers_depths],
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
                       const int padding_left, const int padding_right, const int padding_top, const int padding_bottom, const int first_fill_from_left_offset);

#endif
