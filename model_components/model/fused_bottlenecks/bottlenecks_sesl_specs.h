#ifndef BOTTLENECK_SPECS
#define BOTTLENECK_SPECS
#include "../../basic_defs/basic_defs_glue.h"
#include "../headers/model_glue.h"

const int max_of_bottlenecks_projection_filters = 24;
const int max_of_bottlenecks_layers_depths = 96;
const int max_of_bottlenecks_expansion_layers_depths = 24;

//******************************************
const int bottleneck_0_dw_filter_dim = layer_2_dw_filter_size;
const int bottleneck_0_dw_strides = layer_2_dw_strides;
const int bottleneck_0_dw_padding_left = layer_2_dw_padding_left;
const int bottleneck_0_dw_padding_right = layer_2_dw_padding_right;
const int bottleneck_0_dw_padding_top = layer_2_dw_padding_top;
const int bottleneck_0_dw_padding_bottom = layer_2_dw_padding_bottom;

const int bottleneck_0_ifms_depth = input_image_depth;
const int bottleneck_0_ifms_height = input_image_height;
const int bottleneck_0_ifms_width = input_image_width;
const int bottleneck_0_expanded_ifms_depth = layer_0_num_fils;
const int bottleneck_0_ofms_depth = layer_3_pw_num_fils;
const int bottleneck_0_ofms_height = bottleneck_0_ifms_height / layer_0_strides;
const int bottleneck_0_ofms_width = bottleneck_0_ifms_width / layer_0_strides;

const int bottleneck_0_expansion_layer_index = 0;
const int bottleneck_0_dw_layer_index = 2;
const int bottleneck_0_projection_layer_index = 3;

const int bottleneck_0_expansion_layer_relu = 6;
const int bottleneck_0_dw_layer_relu = 6;
const int bottleneck_0_projection_layer_relu = layer_3_relu;
//******************************************
const int bottleneck_1_dw_filter_dim = layer_5_dw_filter_size;
const int bottleneck_1_dw_strides = layer_5_dw_strides;
const int bottleneck_1_dw_padding_left = layer_5_dw_padding_left;
const int bottleneck_1_dw_padding_right = layer_5_dw_padding_right;
const int bottleneck_1_dw_padding_top = layer_5_dw_padding_top;
const int bottleneck_1_dw_padding_bottom = layer_5_dw_padding_bottom;

const int bottleneck_1_ifms_depth = layer_4_pw_depth;
const int bottleneck_1_ifms_height = layer_4_pw_ifm_height;
const int bottleneck_1_ifms_width = layer_4_pw_ifm_width;
const int bottleneck_1_expanded_ifms_depth = layer_4_pw_num_fils;
const int bottleneck_1_ofms_depth = layer_6_pw_num_fils;
const int bottleneck_1_ofms_height = bottleneck_1_ifms_height / bottleneck_1_dw_strides;
const int bottleneck_1_ofms_width = bottleneck_1_ifms_width / bottleneck_1_dw_strides;

const int bottleneck_1_expansion_layer_index = 4;
const int bottleneck_1_dw_layer_index = bottleneck_1_expansion_layer_index + 1;
const int bottleneck_1_projection_layer_index = bottleneck_1_expansion_layer_index + 2;

const int bottleneck_1_expansion_layer_relu = layer_4_relu;
const int bottleneck_1_dw_layer_relu = 6;
const int bottleneck_1_projection_layer_relu = layer_6_relu;
//************************************************

const int bottleneck_2_ifms_depth = layer_7_pw_depth;
const int bottleneck_2_ifms_height = layer_7_pw_ifm_height;
const int bottleneck_2_ifms_width = layer_7_pw_ifm_width;
const int bottleneck_2_expanded_ifms_depth = layer_7_pw_num_fils;
const int bottleneck_2_dw_filter_dim = layer_8_dw_filter_size;
const int bottleneck_2_dw_padding_top = layer_8_dw_padding_top;
const int bottleneck_2_dw_padding_left = layer_8_dw_padding_left;
const int bottleneck_2_dw_strides = layer_8_dw_strides;
const int bottleneck_2_ofms_depth = layer_9_pw_num_fils;
const int bottleneck_2_rows_at_once = 1;
const int bottleneck_2_input_buffer_height = bottleneck_2_dw_strides * bottleneck_2_rows_at_once;
const int bottleneck_2_output_buffer_height = bottleneck_2_input_buffer_height / bottleneck_2_dw_strides;
const int bottleneck_2_ofms_width = bottleneck_2_ifms_width / bottleneck_2_dw_strides;
const int bottleneck_2_expansion_layer_index = 7;
const int bottleneck_2_dw_layer_index = bottleneck_2_expansion_layer_index + 1;
const int bottleneck_2_projection_layer_index = bottleneck_2_expansion_layer_index + 2;

const int max_dw_filter_dim_in_a_chain = 3;
const int max_dw_filter_area_in_a_chain = max_dw_filter_dim_in_a_chain * max_dw_filter_dim_in_a_chain;

#endif
