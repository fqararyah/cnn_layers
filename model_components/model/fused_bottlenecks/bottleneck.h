#ifndef BOTTLENECK_H
#define BOTTLENECK_H
#include "bottlenecks_sesl_specs.h"
#include "bottleneck_kernels.h"

void bottleneck_0_padding_top_right(fms_dt previous_pass_dw_input[][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
                                    fms_dt zero_point);

void bottleneck_0_do_padding_left(fms_dt previous_pass_dw_input[][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
                                  fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim], fms_dt zero_point);

void mob_v2_bottleneck_0(fms_dt bottleneck_input[],
                         pss_dt projection_kernel_output_buffer[bottleneck_0_ofms_depth],
						 pss_dt projection_kernel_output_buffer_prev[bottleneck_0_ofms_depth],
                         fms_dt bottleneck_0_1_communication_buffer[bottleneck_0_ofms_depth][bottleneck_0_rows_at_once][bottleneck_0_ofms_width],
                         fms_dt previous_pass_dw_input[][bottleneck_0_inter_pass_dw_input_height][bottleneck_0_inter_pass_dw_input_width],
                         fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim],
                         const int starting_h, const int h_in_communication_buffer, const int starting_w);

void mob_v2_bottleneck_1(fms_dt bottleneck_input[],
						 pss_dt projection_kernel_output_buffer[bottleneck_1_ofms_depth],
						 pss_dt projection_kernel_output_buffer_prev[bottleneck_1_ofms_depth],
						 fms_dt chain_seml_communication_buffer[bottleneck_1_ofms_depth][bottleneck_1_ofms_width],
						 fms_dt previous_pass_dw_input[][bottleneck_1_inter_pass_dw_input_width],
						 fms_dt dw_lower_buffer[][bottleneck_1_dw_filter_dim * bottleneck_1_dw_strides], 
						 const int starting_h,
						 const int starting_w);

#endif
