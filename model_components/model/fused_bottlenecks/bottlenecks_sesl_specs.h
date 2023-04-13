#ifndef BOTTLENECK_SPECS
#define BOTTLENECK_SPECS
#include "../../basic_defs/basic_defs_glue.h"
#include "../headers/model_glue.h"

//******************************************
const int bottleneck_0_dw_filter_dim = layer_2_dw_filter_dim;
const int bottleneck_0_dw_strides = 1;//todo
const int bottleneck_0_dw_padding_left = 1;//todo
const int bottleneck_0_dw_padding_right = 1;//todo
const int bottleneck_0_dw_padding_top = 1;//todo
const int bottleneck_0_dw_padding_bottom = 1;//todo

const int bottleneck_0_ifms_depth = input_image_depth;
const int bottleneck_0_ifms_height = input_image_height;
const int bottleneck_0_ifms_width = input_image_width;
const int bottleneck_0_expanded_ifms_depth = layer_1_s_num_fils;
const int bottleneck_0_ofms_depth = layer_3_pw_num_fils;
const int bottleneck_0_ofms_height = bottleneck_0_ifms_height / 2;//todo 2 is layer_1_s strides
const int bottleneck_0_ofms_width = bottleneck_0_ifms_width / 2;//todo 2 is layer_1_s strides

const int bottleneck_0_expansion_layer_relu = layer_1_s_specs.layer_activation;
const int bottleneck_0_dw_layer_relu = layer_2_dw_specs.layer_activation;
const int bottleneck_0_projection_layer_relu = layer_3_pw_specs.layer_activation;
//******************************************
#if MODEL_ID == 1
const int bottleneck_1_dw_1_filter_dim = layer_3_dw_filter_size;
const int bottleneck_1_dw_1_strides = layer_3_dw_strides;
const int bottleneck_1_dw_1_padding_left = layer_3_dw_padding_left;
const int bottleneck_1_dw_1_padding_right = layer_3_dw_padding_right;
const int bottleneck_1_dw_1_padding_top = layer_3_dw_padding_top;
const int bottleneck_1_dw_1_padding_bottom = layer_3_dw_padding_bottom;

const int bottleneck_1_dw_2_filter_dim = layer_5_dw_filter_size;
const int bottleneck_1_dw_2_strides = layer_5_dw_strides;
const int bottleneck_1_dw_2_padding_left = layer_5_dw_padding_left;
const int bottleneck_1_dw_2_padding_right = layer_5_dw_padding_right;
const int bottleneck_1_dw_2_padding_top = layer_5_dw_padding_top;
const int bottleneck_1_dw_2_padding_bottom = layer_5_dw_padding_bottom;

const int bottleneck_1_ifms_depth = layer_3_dw_depth;
const int bottleneck_1_ifms_height = layer_3_dw_ifm_height;
const int bottleneck_1_ifms_width = layer_3_dw_ifm_width;
const int bottleneck_1_ofms_depth = layer_5_dw_num_fils;

const int bottleneck_1_ofms_height = bottleneck_1_ifms_height / bottleneck_1_dw_1_strides;
const int bottleneck_1_ofms_width = bottleneck_1_ifms_width / bottleneck_1_dw_1_strides;

const int bottleneck_1_expansion_layer_relu = 6;
const int bottleneck_1_dw_layer_relu = 6;
const int bottleneck_1_projection_layer_relu = 6;
#else
const int bottleneck_1_dw_filter_dim = layer_6_dw_filter_dim;
const int bottleneck_1_dw_strides = 2;//todo
const int bottleneck_1_dw_padding_left = 0;//ToDO
const int bottleneck_1_dw_padding_right = 1;//ToDO
const int bottleneck_1_dw_padding_top = 0;//ToDO
const int bottleneck_1_dw_padding_bottom = 1;//ToDO

const int bottleneck_1_ifms_depth = layer_4_pw_depth;
const int bottleneck_1_ifms_height = layer_4_pw_ifm_width;
const int bottleneck_1_ifms_width = layer_4_pw_ifm_width;
const int bottleneck_1_expanded_ifms_depth = layer_4_pw_num_fils;
const int bottleneck_1_ofms_depth = layer_7_pw_num_fils;

const int bottleneck_1_ofms_height = bottleneck_1_ifms_height / bottleneck_1_dw_strides;
const int bottleneck_1_ofms_width = bottleneck_1_ifms_width / bottleneck_1_dw_strides;

const int bottleneck_1_expansion_layer_relu = layer_4_pw_specs.layer_activation;
const int bottleneck_1_dw_layer_relu = layer_6_dw_specs.layer_activation;
const int bottleneck_1_projection_layer_relu = layer_7_pw_specs.layer_activation;
#endif

//************************************************

const int bottleneck_2_dw_filter_dim = layer_9_dw_filter_dim;
const int bottleneck_2_dw_strides = 1; //todo
const int bottleneck_2_dw_padding_left = 1;//todo
const int bottleneck_2_dw_padding_right = 1;//todo
const int bottleneck_2_dw_padding_top = 1;//todo
const int bottleneck_2_dw_padding_bottom = 1;//todo

const int bottleneck_2_ifms_depth = bottleneck_1_ofms_depth;
const int bottleneck_2_ifms_height = bottleneck_1_ofms_height;
const int bottleneck_2_ifms_width = bottleneck_1_ofms_width;
const int bottleneck_2_expanded_ifms_depth = layer_8_pw_num_fils;
const int bottleneck_2_ofms_depth = layer_10_pw_num_fils;
const int bottleneck_2_ofms_height = bottleneck_2_ifms_height / bottleneck_2_dw_strides;
const int bottleneck_2_ofms_width = bottleneck_2_ifms_width / bottleneck_2_dw_strides;

const int bottleneck_2_expansion_layer_relu = layer_8_pw_specs.layer_activation;
const int bottleneck_2_dw_layer_relu = layer_9_dw_specs.layer_activation;
const int bottleneck_2_projection_layer_relu = layer_10_pw_specs.layer_activation;

const int max_dw_filter_dim_in_a_chain = 3;
//***************************************************************************************
//***************************************************************************************

#endif
