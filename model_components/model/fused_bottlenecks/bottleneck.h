#ifndef BOTTLENECK_H
#define BOTTLENECK_H
#include "bottlenecks_sesl_specs.h"
#include "bottleneck_kernels.h"

void bottleneck_0_padding_top_right(fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
                              fms_dt zero_point);

void bottleneck_0_do_padding_left(fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
                                  fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim], fms_dt zero_point);

void mob_v2_bottleneck_0(fms_dt bottleneck_input[],
                         fms_dt bottleneck_output[],
                         fms_dt previous_pass_dw_input[][bottlenck_0_inter_pass_dw_input_height][bottlenck_0_inter_pass_dw_input_width],
                         fms_dt dw_lower_buffer[][bottleneck_0_dw_filter_dim],
                         int starting_h, int starting_w);

void mob_v2_bottleneck_1(fms_dt bottleneck_input[],
                         fms_dt bottleneck_output[],
                         fms_dt r_previous_pass_dw_input[],
                         fms_dt w_previous_pass_dw_input[],
                         int starting_h,
                         int starting_w,
                         const int first_fill_from_left_offset);

#endif
