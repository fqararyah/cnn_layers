#include "../../basic_defs/basic_defs_glue.h"
#include "../../client/sesl_pw_weights.h"

#ifndef PIPELINE
#define PIPELINE

const int _4_stages_layer_0_rows_at_once = 1;
const int _5_stages_layer_0_rows_at_once = 2;
const int _6_stages_layer_0_rows_at_once = 2;
const int _7_stages_layer_0_rows_at_once = 2;

const int _4_stages_layer_2_rows_at_once = 1;
const int _5_stages_layer_2_rows_at_once = 1;
const int _6_stages_layer_2_rows_at_once = 2;
const int _7_stages_layer_2_rows_at_once = 2;

const int _4_stages_layer_3_rows_at_once = 1;
const int _5_stages_layer_3_rows_at_once = 1;
const int _6_stages_layer_3_rows_at_once = 2;
const int _7_stages_layer_3_rows_at_once = 2;

// l4, l5
const int _4_stages_layer_4_rows_at_once = 1;
const int _5_stages_layer_4_rows_at_once = 2;
const int _6_stages_layer_4_rows_at_once = 2;
const int _7_stages_layer_4_rows_at_once = 2;

// l6
const int _7_stages_layer_6_rows_at_once = 1;
const int _6_stages_layer_6_rows_at_once = 1;

//l7
const int _7_stages_layer_7_rows_at_once = 1;


void pw_conv_pipeline(fms_dt channels[max_fms_size],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		const int layer_num_fils, const int layer_conv_d,
		const int num_of_tiles_hw, const int num_of_tiles_w, int td_o,
		int td_i_o, int t_in_h, int t_in_w);


//void mobilenetv1_pipeline_6(
//		fms_dt channels[input_image_depth][input_image_height][input_image_width],
//		fms_dt result[max_fms_size]);
//
//void _5_layer_0_3x3_conv(
//		fms_dt channels_buffer[layer_0_depth][3 + _5_stages_layer_1_rows_at_once
//				- 1][layer_0_ifm_width],
//		layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][3][3],
//		fms_dt result[layer_2_dw_depth][layer_1_pw_ifm_width]);
//
//void _7_layer_0_3x3_conv(
//		fms_dt channels_buffer[input_image_depth][layer_0_filter_size
//				+ (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
//		layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
//		fms_dt result[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width]);
//
//void _4_layer_3_pw(
//		fms_dt channels_buffer[layer_3_pw_depth][layer_3_pw_ifm_width],
//		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
//		fms_dt result[max_fms_size], int starting_h);
//
//void _5_layer_3_pw(
//		fms_dt channels_buffer[layer_3_pw_depth][layer_3_pw_ifm_width],
//		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
//		fms_dt result[layer_3_pw_num_fils][layer_5_dw_ifm_width]);
//
//void _7_layer_3_pw(
//		fms_dt channels_buffer[layer_3_pw_depth][_7_stages_layer_1_rows_at_once][layer_3_pw_ifm_width],
//		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
//		fms_dt result[layer_3_pw_num_fils][_7_stages_layer_1_rows_at_once][layer_5_dw_ifm_width]);
//
//void _5_layer_3_pw(
//		fms_dt channels_buffer[layer_3_pw_depth][layer_3_pw_ifm_width],
//		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
//		fms_dt result[max_fms_size], int starting_h);
//
//void _7_layer_3_pw_5_dw(
//		fms_dt channels_buffer[layer_3_pw_depth][layer_5_dw_strides][layer_5_dw_ifm_width],
//		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
//		dw_weights_dt dw_weights[layer_5_dw_depth][layer_5_dw_filter_size][layer_5_dw_filter_size],
//		fms_dt upper[layer_5_dw_depth][layer_5_dw_ifm_width],
//		fms_dt lower[layer_5_dw_depth][layer_5_dw_strides][layer_5_dw_ifm_width],
//		fms_dt result[layer_4_pw_depth][layer_4_pw_ifm_width], int active_row);
//
//void _5_stages_fill_channels_buffer(
//		fms_dt channels[input_image_depth][input_image_height][input_image_width],
//		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size
//				+ (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
//		int starting_h);
//
//void _7_stages_fill_channels_buffer(
//		fms_dt channels[input_image_depth][input_image_height][input_image_width],
//		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size
//				+ (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
//		int starting_h);
//
//void _5_layer_1_pw_dw(
//		fms_dt channels_buffer[layer_1_pw_depth][layer_1_pw_ifm_width],
//		weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
//		dw_weights_dt dw_weights[layer_2_dw_depth][layer_2_dw_filter_size][layer_2_dw_filter_size],
//		fms_dt upper[layer_2_dw_depth][layer_2_dw_filter_size
//				- layer_2_dw_strides][layer_2_dw_ifm_width],
//		fms_dt lower[layer_2_dw_depth][layer_2_dw_strides][layer_2_dw_ifm_width],
//		fms_dt result[layer_3_pw_depth][layer_3_pw_ifm_width], int active_row);
//
//void _7_layer_1_pw_dw(
//		fms_dt channels_buffer[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width],
//		weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
//		dw_weights_dt dw_weights[layer_2_dw_depth][layer_2_dw_filter_size][layer_2_dw_filter_size],
//		fms_dt upper[layer_2_dw_depth][layer_2_dw_filter_size
//				- layer_2_dw_strides][layer_2_dw_ifm_width],
//		fms_dt lower[layer_2_dw_depth][_7_stages_layer_1_rows_at_once][layer_2_dw_ifm_width],
//		fms_dt result[layer_3_pw_depth][_7_stages_layer_1_rows_at_once][layer_3_pw_ifm_width],
//		int active_row);
//
//void _7_layer_0_3x3_conv(
//		fms_dt channels_buffer[input_image_depth][layer_0_filter_size
//				+ _7_stages_layer_1_rows_at_once - 1][input_image_width],
//		layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
//		fms_dt result[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width]);
//
//void _7_layer_4_pw(
//		fms_dt channels_buffer[layer_4_pw_depth][layer_4_pw_ifm_width],
//		weights_dt weights[layer_4_pw_num_fils][layer_4_pw_depth],
//		fms_dt result[max_fms_size], int starting_h);
//
//void _6_layer_3_pw_dw(
//		fms_dt channels_buffer[layer_3_pw_depth][layer_5_dw_strides][layer_5_dw_ifm_width],
//		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
//		dw_weights_dt dw_weights[layer_5_dw_depth][layer_5_dw_filter_size][layer_5_dw_filter_size],
//		fms_dt upper[layer_5_dw_depth][layer_5_dw_ifm_width],
//		fms_dt lower[layer_5_dw_depth][layer_5_dw_strides][layer_5_dw_ifm_width],
//		fms_dt result[max_fms_size], int starting_h, int active_row);
//
//void mobilenet_v2_pipeline_4(
//		fms_dt channels[input_image_depth][input_image_height][input_image_width],
//		fms_dt result[max_fms_size]);
//
//void mobilenet_v2_pipeline_5(
//		fms_dt channels[input_image_depth][input_image_height][input_image_width],
//		fms_dt result[max_fms_size]);
//
//void mobilenet_v2_pipeline_6(
//		fms_dt channels[input_image_depth][input_image_height][input_image_width],
//		fms_dt result[max_fms_size]);
//
//void mobilenet_v2_pipeline_7(
//		fms_dt channels[input_image_depth][input_image_height][input_image_width],
//		fms_dt result[max_fms_size]);

#endif
