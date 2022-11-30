#include "../headers/layers_specs.h"
#include "../../basic_defs/basic_defs_glue.h"

#ifndef BOTTLENECK_SPECS
#define BOTTLENECK_SPECS

const int max_of_bottlenecks_projection_filters = 24;
const int max_of_bottlenecks_layers_depths = 144;

const int bottleneck_1_ifms_depth = layer_4_pw_depth;
const int bottleneck_1_ifms_height = layer_4_pw_ifm_height;
const int bottleneck_1_ifms_width = layer_4_pw_ifm_width;
const int bottleneck_1_expanded_ifms_depth = layer_4_pw_num_fils;
const int bottleneck_1_dw_filter_dim = layer_5_dw_filter_size;
const int bottleneck_1_dw_padding_top = layer_5_dw_padding_top;
const int bottleneck_1_dw_padding_left = layer_5_dw_padding_left;
const int bottleneck_1_dw_strides = layer_5_dw_strides;
const int bottleneck_1_ofms_depth = layer_6_pw_num_fils;
const int bottleneck_1_rows_at_once = 1;
const int bottleneck_1_input_buffer_height = bottleneck_1_dw_strides * bottleneck_1_rows_at_once;
const int bottleneck_1_output_buffer_height = bottleneck_1_input_buffer_height / bottleneck_1_dw_strides;
const int bottleneck_1_ofms_width = bottleneck_1_ifms_width / bottleneck_1_dw_strides;
const int bottleneck_1_expansion_layer_index = 4;
const int bottleneck_1_dw_layer_index = bottleneck_1_expansion_layer_index + 1;
const int bottleneck_1_projection_layer_index = bottleneck_1_expansion_layer_index + 2;

#endif