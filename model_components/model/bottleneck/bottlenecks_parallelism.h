#include "bottlenecks_sesl_specs.h"

#ifndef BOTTLENECK_PARALLELISM
#define BOTTLENECK_PARALLELISM

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

const int bottleneck_1_rows_at_once = 1;

#endif
