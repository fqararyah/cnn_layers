
#ifndef BOTTLENECK_PARALLELISM
#define BOTTLENECK_PARALLELISM
#include "bottlenecks_sesl_specs.h"

const int bottleneck_0_rows_at_once = 2;
const int bottleneck_0_expansion_parallelism_ifms = bottleneck_0_ifms_depth;
const int bottleneck_0_expansion_parallelism_ofms = 1;
const int bottleneck_0_expansion_parallelism_h = bottleneck_0_rows_at_once;
const int bottleneck_0_expansion_parallelism_w = bottleneck_0_rows_at_once;
//*****
const int bottleneck_0_dw_parallelism_ifms = bottleneck_0_expansion_parallelism_ofms;
//*****
const int bottleneck_0_projection_parallelism_ifms = 1;
const int bottleneck_0_projection_parallelism_ofms = bottleneck_0_ofms_depth;
const int bottleneck_0_projection_parallelism_h = 1;
const int bottleneck_0_projection_parallelism_w = 1;

const int bottlenck_0_input_buffer_height = layer_0_filter_dim + (bottleneck_0_expansion_parallelism_h - 1) * layer_0_strides;
const int bottlenck_0_input_buffer_width = layer_0_filter_dim + (bottleneck_0_expansion_parallelism_w - 1) * layer_0_strides;

const int bottleneck_0_fill_each_time = bottleneck_0_rows_at_once * layer_0_strides;

const int bottlenck_0_input_buffer_size = bottlenck_0_input_buffer_height * bottlenck_0_input_buffer_width *
                                          bottleneck_0_ifms_depth;
const int bottlenck_0_output_buffer_size = bottleneck_0_ofms_depth * bottleneck_0_rows_at_once * bottleneck_0_rows_at_once;
const int bottlenck_0_inter_pass_dw_input_size = bottleneck_0_expanded_ifms_depth * bottleneck_0_ifms_width *
                                                 (bottleneck_0_dw_filter_dim - bottleneck_0_dw_strides);
//**************************************************************************
const int bottleneck_1_rows_at_once = 1;
const int bottleneck_1_expansion_parallelism_ifms = bottleneck_1_ifms_depth;
const int bottleneck_1_expansion_parallelism_ofms = 1;
const int bottleneck_1_expansion_parallelism_h = 2;
const int bottleneck_1_expansion_parallelism_w = 2;
//*****
const int bottleneck_1_dw_parallelism_ifms = bottleneck_1_expansion_parallelism_ofms;
//*****
const int bottleneck_1_projection_parallelism_ifms = 1;
const int bottleneck_1_projection_parallelism_ofms = bottleneck_1_ofms_depth;
const int bottleneck_1_projection_parallelism_h = 1;
const int bottleneck_1_projection_parallelism_w = 1;

const int bottlenck_1_input_buffer_size = bottleneck_1_expansion_parallelism_h * bottleneck_1_expansion_parallelism_w * bottleneck_1_ifms_depth;
const int bottlenck_1_output_buffer_size = bottleneck_1_ofms_depth * bottleneck_1_ofms_width;
const int bottlenck_1_inter_pass_dw_input_size = bottleneck_1_expanded_ifms_depth * bottleneck_1_ifms_width *
                                                 (bottleneck_1_dw_filter_dim - bottleneck_1_dw_strides);

#endif
