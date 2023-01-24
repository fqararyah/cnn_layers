
#ifndef BOTTLENECK_PARALLELISM
#define BOTTLENECK_PARALLELISM
#include "bottlenecks_sesl_specs.h"

const int bottleneck_0_rows_at_once = 2;
const int bottleneck_0_expansion_parallelism_ifms = bottleneck_0_ifms_depth;
const int bottleneck_0_expansion_parallelism_ofms = 1;
const int bottleneck_0_expansion_parallelism_h = 1;
const int bottleneck_0_expansion_parallelism_w = 1;
//*****
const int bottleneck_0_dw_parallelism_ifms = bottleneck_0_expansion_parallelism_ofms;
//*****
const int bottleneck_0_projection_parallelism_ifms = 1;
const int bottleneck_0_projection_parallelism_ofms = bottleneck_0_ofms_depth;
const int bottleneck_0_projection_parallelism_h = 1;
const int bottleneck_0_projection_parallelism_w = 1;

const int bottlenck_0_input_buffer_height = layer_0_s_filter_dim + (bottleneck_0_expansion_parallelism_h - 1) * layer_0_s_strides;
const int bottlenck_0_input_buffer_width = layer_0_s_filter_dim + (bottleneck_0_expansion_parallelism_w - 1) * layer_0_s_strides;

const int bottleneck_0_fill_each_time = bottleneck_0_rows_at_once * layer_0_s_strides;

const int bottleneck_0_input_buffer_hw = bottlenck_0_input_buffer_height * bottlenck_0_input_buffer_width;
const int bottleneck_0_input_buffer_size = bottleneck_0_input_buffer_hw * bottleneck_0_ifms_depth;
const int bottleneck_0_output_buffer_size = bottleneck_0_ofms_depth * bottleneck_0_rows_at_once * bottleneck_0_rows_at_once;
const int bottleneck_0_inter_pass_dw_input_width = bottleneck_0_ifms_width / layer_0_s_strides +
                                                  bottleneck_0_dw_padding_left + bottleneck_0_dw_padding_right;
const int bottleneck_0_inter_pass_dw_input_height = (bottleneck_0_dw_filter_dim - bottleneck_0_dw_strides);
const int bottleneck_0_inter_pass_dw_input_size = bottleneck_0_expanded_ifms_depth * bottleneck_0_inter_pass_dw_input_width *
                                                 bottleneck_0_inter_pass_dw_input_height;
//**************************************************************************
const int bottleneck_1_rows_at_once = 1;
const int bottleneck_1_expansion_parallelism_ifms = bottleneck_1_ifms_depth;
const int bottleneck_1_expansion_parallelism_ofms = 1;
const int bottleneck_1_expansion_parallelism_h = bottleneck_1_dw_strides;
const int bottleneck_1_expansion_parallelism_w = bottleneck_1_dw_strides;
//*****
const int bottleneck_1_dw_parallelism_ifms = bottleneck_1_expansion_parallelism_ofms;
//*****
const int bottleneck_1_projection_parallelism_ifms = 1;
const int bottleneck_1_projection_parallelism_ofms = bottleneck_1_ofms_depth;
const int bottleneck_1_projection_parallelism_h = 1;
const int bottleneck_1_projection_parallelism_w = 1;

const int bottleneck_1_input_buffer_size = bottleneck_1_expansion_parallelism_h * bottleneck_1_expansion_parallelism_w * bottleneck_1_ifms_depth;
const int bottleneck_1_output_buffer_size = bottleneck_1_ofms_depth * bottleneck_1_ofms_width;
const int bottleneck_1_inter_pass_dw_input_width = bottleneck_1_ifms_width + bottleneck_1_dw_padding_left + bottleneck_1_dw_padding_right + 1;//+1 to make it even
const int bottleneck_1_inter_pass_dw_input_size = bottleneck_1_expanded_ifms_depth * bottleneck_1_inter_pass_dw_input_width *
                                                 (bottleneck_1_dw_filter_dim - bottleneck_1_dw_strides);
//**************************************************************************
const int bottleneck_2_rows_at_once = 1;
const int bottleneck_2_expansion_parallelism_ifms = bottleneck_2_ifms_depth;
const int bottleneck_2_expansion_parallelism_ofms = 1;
const int bottleneck_2_expansion_parallelism_h = 1;
const int bottleneck_2_expansion_parallelism_w = 1;
//*****
const int bottleneck_2_dw_parallelism_ifms = bottleneck_2_expansion_parallelism_ofms;
//*****
const int bottleneck_2_projection_parallelism_ifms = 1;
const int bottleneck_2_projection_parallelism_ofms = bottleneck_2_ofms_depth;
const int bottleneck_2_projection_parallelism_h = 1;
const int bottleneck_2_projection_parallelism_w = 1;

const int bottleneck_2_input_buffer_height = bottleneck_2_rows_at_once;
const int bottleneck_2_input_buffer_width = bottleneck_2_rows_at_once;

const int bottleneck_2_fill_each_time = bottleneck_2_rows_at_once * bottleneck_2_dw_strides;

const int bottleneck_2_input_buffer_hw = bottleneck_2_input_buffer_height * bottleneck_2_input_buffer_width;
const int bottleneck_2_input_buffer_size = bottleneck_2_input_buffer_hw * bottleneck_2_ifms_depth;
const int bottleneck_2_output_buffer_size = bottleneck_2_ofms_depth * bottleneck_2_rows_at_once * bottleneck_2_rows_at_once;
const int bottleneck_2_inter_pass_dw_input_width = bottleneck_2_ifms_width / bottleneck_2_dw_strides +
                                                  bottleneck_2_dw_padding_left + bottleneck_2_dw_padding_right;
const int bottleneck_2_inter_pass_dw_input_height = (bottleneck_2_dw_filter_dim - bottleneck_2_dw_strides);
const int bottleneck_2_inter_pass_dw_input_size = bottleneck_2_expanded_ifms_depth * bottleneck_2_inter_pass_dw_input_width *
                                                 bottleneck_2_inter_pass_dw_input_height;
#endif
