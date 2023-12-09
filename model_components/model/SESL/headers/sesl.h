#include "../../../basic_defs/basic_defs_glue.h"

#include "../../../layers/headers/layers_glue.h"

#include "sesl_parallelism.h"

#ifndef SESL
#define SESL

const int _4_stages_layer_0_s_rows_at_once = 1;
const int _5_stages_layer_0_s_rows_at_once = 2;
const int _6_stages_layer_0_s_rows_at_once = 2;
const int _7_stages_layer_0_s_rows_at_once = 2;

const int _4_stages_layer_2_rows_at_once = 1;
const int _5_stages_layer_2_rows_at_once = 1;
const int _6_stages_layer_1_rows_at_once = 2;
const int _7_stages_layer_1_rows_at_once = 2;

const int _4_stages_layer_3_rows_at_once = 1;
const int _5_stages_layer_3_rows_at_once = 1;
const int _6_stages_layer_2_rows_at_once = 2;
const int _7_stages_layer_2_rows_at_once = 2;
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

const int _7_stages_layer_0_s_in_rows_at_once = first_conv_layer_specs.strides
		* _7_stages_layer_0_s_rows_at_once;
const int _7_stages_layer_0_s_in_buffer_height = first_conv_layer_filter_dim
		+ (_7_stages_layer_0_s_rows_at_once - 1) * first_conv_layer_specs.strides;

void pw_conv_pipeline(fms_dt channels[max_fms_size],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		const int layer_num_fils, const int layer_conv_d,
		const int num_of_tiles_hw, const int num_of_tiles_w, int td_o,
		int td_i_o, int t_in_h, int t_in_w);

void cnn_pipeline_7_mob_v2(
		fms_grp_dt input_image[input_image_depth
				* input_image_num_fms_groups_in_a_channel],
		fms_dt result[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size]);

void _6_layer_0_s_3x3_conv(
		fms_dt channels_buffer[input_image_depth][first_conv_layer_filter_dim
				+ (_6_stages_layer_0_s_rows_at_once - 1) * first_conv_layer_specs.strides][input_image_width],
		const layer_0_weights_dt weights[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim],
		fms_dt result[layer_1_dw_depth][_6_stages_layer_0_s_rows_at_once][layer_1_dw_ifm_width]);

void _6_layer_1_dw(
		fms_dt channels_buffer[layer_1_dw_depth][_6_stages_layer_1_rows_at_once][layer_1_dw_ifm_width],
		const dw_weights_dt dw_weights[layer_1_dw_depth][layer_2_dw_specs.filter_size*layer_2_dw_specs.filter_size],
		fms_dt upper[layer_1_dw_depth][layer_2_dw_specs.filter_size
				- layer_2_dw_specs.strides][layer_1_dw_ifm_width],
		fms_dt result[layer_2_pw_depth][_6_stages_layer_2_rows_at_once][layer_2_pw_ifm_width],
		int first_row);

void _6_layer_2_pw(
		fms_dt channels_buffer[layer_2_pw_depth][_6_stages_layer_2_rows_at_once][layer_2_pw_ifm_width],
		const weights_dt weights[layer_3_pw_specs.num_fils][layer_2_pw_depth],
		fms_dt result[layer_4_pw_depth][_6_stages_layer_3_rows_at_once][layer_4_pw_ifm_width]);

void _6_layer_3_pw_4_dw(
		fms_dt channels_buffer[layer_4_pw_depth][layer_6_dw_specs.strides][layer_4_dw_ifm_width],
		const weights_dt weights[layer_4_pw_num_fils][layer_4_pw_depth],
		const dw_weights_dt dw_weights[layer_4_dw_depth][layer_6_dw_specs.filter_size
				* layer_6_dw_specs.filter_size],
		fms_dt upper[layer_4_dw_depth][layer_4_dw_ifm_width],
		fms_dt lower[layer_4_dw_depth][layer_6_dw_specs.strides][layer_4_dw_ifm_width],
		fms_dt result[layer_5_pw_depth][layer_5_pw_ifm_width], int starting_h);

void _6_layer_5_pw(
		fms_dt channels_buffer[layer_5_pw_depth][layer_5_pw_ifm_width],
		const weights_dt weights[layer_7_pw_specs.layer_num_fils][layer_5_pw_depth],
		fms_dt result[max_fms_size], int starting_h);

#endif
