#include <math.h>
#include "general_specs.h"
#include "q_data_types.h"
#include "parallelism_and_tiling.h"
#include "layers_specs.h"
#include "sesl_specs.h"
#include "dw_weights.h"

// typedef ap_int<fms_dt_width*out_tile_h*out_tile_w> dw_pss_grp_dt;

void top_func(weights_grp_dt layer_1_weights_m[32],
		weights_grp_dt layer_2_weights_m[32], int &result_o);

void pw_conv(weights_grp_dt *weights, fms_dt channels[max_fms_size],
		fms_dt result[max_fms_size], int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w, fms_dt tmp_channels[max_tmp_fms_size],
		int read_write, const int num_of_weight_groups,
		const normalization_scheme normalization, const int direction, const int layer_weights_offset);

void dw_conv_3x3(dw_weights_dt weights[max_conv_d][max_conv_h][max_conv_w],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides,
		const normalization_scheme normalization);

void dw_conv_3x3_g(dw_weights_dt weights[max_conv_d][max_conv_h][max_conv_w],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int layer_width,const int layer_height,
		const int num_of_tiles_d, const int num_of_tiles_h,
		const int num_of_tiles_w, const int strides, const int padding_left,
		const normalization_scheme normalization, const int direction);

void dw_conv_5x5(dw_weights_dt weights[max_conv_d][5][5],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides,
		const normalization_scheme normalization);

void dw_conv_7x7(dw_weights_dt weights[max_conv_d][7][7],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		const int layer, const int layer_conv_d, const int num_of_tiles_d,
		const int num_of_tiles_h, const int num_of_tiles_w, const int strides,
		const normalization_scheme normalization);

void layer_0(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size],
		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3],
		const normalization_scheme normalization);

void layer_0_using_pw(
		weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3],
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w, const normalization_scheme normalization);

void avgpool(fms_dt channels[max_fms_size], fms_dt restult[max_fms_size]);

void fc_layer(const fc_weights_dt weights[fc_layer_input_size][fc_cols], fms_dt channels[fc_layer_input_size],
		fc_out_dt result[fc_cols]);

//############pipeline functions###########
void mobilenetv1_pipeline_6(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size]);

void _5_layer_0_3x3_conv(
		fms_dt channels_buffer[layer_0_depth][3 + _5_stages_layer_1_rows_at_once
				- 1][layer_0_ifm_width],
		layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][3][3],
		fms_dt result[layer_1_dw_depth][layer_1_pw_ifm_width]);

void _7_layer_0_3x3_conv(
		fms_dt channels_buffer[input_image_depth][layer_0_filter_size
				+ (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
		layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
		fms_dt result[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width]);

void _4_layer_2_pw(
		fms_dt channels_buffer[layer_2_pw_depth][layer_2_pw_ifm_width],
		weights_dt weights[layer_2_pw_num_fils][layer_2_pw_depth],
		fms_dt result[max_fms_size], int starting_h);

void _5_layer_2_pw(
		fms_dt channels_buffer[layer_2_pw_depth][layer_2_pw_ifm_width],
		weights_dt weights[layer_2_pw_num_fils][layer_2_pw_depth],
		fms_dt result[layer_3_pw_num_fils][layer_3_dw_ifm_width]);

void _7_layer_2_pw(
		fms_dt channels_buffer[layer_2_pw_depth][_7_stages_layer_1_rows_at_once][layer_2_pw_ifm_width],
		weights_dt weights[layer_2_pw_num_fils][layer_2_pw_depth],
		fms_dt result[layer_3_pw_depth][_7_stages_layer_1_rows_at_once][layer_3_dw_ifm_width]);

void _5_layer_3_pw(
		fms_dt channels_buffer[layer_3_pw_depth][layer_3_dw_ifm_width],
		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
		fms_dt result[max_fms_size], int starting_h);

void _7_layer_3_pw_dw(
		fms_dt channels_buffer[layer_3_pw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
		dw_weights_dt dw_weights[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size],
		fms_dt upper[layer_3_dw_depth][layer_3_dw_ifm_width],
		fms_dt lower[layer_3_dw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
		fms_dt result[layer_4_pw_depth][layer_4_pw_ifm_width], int active_row);

void _5_stages_fill_channels_buffer(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size
				+ (_5_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
		int starting_h);

void _7_stages_fill_channels_buffer(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt channels_buffer_0[input_image_depth][layer_0_filter_size
				+ (_7_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
		int starting_h);

void _5_layer_1_pw_dw(
		fms_dt channels_buffer[layer_1_pw_depth][layer_1_pw_ifm_width],
		weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
		dw_weights_dt dw_weights[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size],
		fms_dt upper[layer_1_dw_depth][layer_1_dw_filter_size
				- layer_1_dw_strides][layer_1_dw_ifm_width],
		fms_dt lower[layer_1_dw_depth][layer_1_dw_strides][layer_1_dw_ifm_width],
		fms_dt result[layer_2_pw_depth][layer_2_pw_ifm_width], int active_row);

void _7_layer_1_pw_dw(
		fms_dt channels_buffer[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width],
		weights_dt weights[layer_1_pw_num_fils][layer_1_pw_depth],
		dw_weights_dt dw_weights[layer_1_dw_depth][layer_1_dw_filter_size][layer_1_dw_filter_size],
		fms_dt upper[layer_1_dw_depth][layer_1_dw_filter_size
				- layer_1_dw_strides][layer_1_dw_ifm_width],
		fms_dt lower[layer_1_dw_depth][_7_stages_layer_1_rows_at_once][layer_1_dw_ifm_width],
		fms_dt result[layer_2_pw_depth][_7_stages_layer_1_rows_at_once][layer_2_pw_ifm_width],
		int active_row);

void _7_layer_0_3x3_conv(
		fms_dt channels_buffer[input_image_depth][layer_0_filter_size
				+ _7_stages_layer_1_rows_at_once - 1][input_image_width],
		layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
		fms_dt result[layer_1_pw_depth][_7_stages_layer_1_rows_at_once][layer_1_pw_ifm_width]);

void _7_layer_4_pw(
		fms_dt channels_buffer[layer_4_pw_depth][layer_4_pw_ifm_width],
		weights_dt weights[layer_4_pw_num_fils][layer_4_pw_depth],
		fms_dt result[max_fms_size], int starting_h);

void _6_layer_3_pw_dw(
		fms_dt channels_buffer[layer_3_pw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
		weights_dt weights[layer_3_pw_num_fils][layer_3_pw_depth],
		dw_weights_dt dw_weights[layer_3_dw_depth][layer_3_dw_filter_size][layer_3_dw_filter_size],
		fms_dt upper[layer_3_dw_depth][layer_3_dw_ifm_width],
		fms_dt lower[layer_3_dw_depth][layer_3_dw_strides][layer_3_dw_ifm_width],
		fms_dt result[max_fms_size], int starting_h, int active_row);

void mobilenet_v2_pipeline_4(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size]);

void mobilenet_v2_pipeline_5(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size]);

void mobilenet_v2_pipeline_6(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size]);

void mobilenet_v2_pipeline_7(
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size]);

void pw_conv_pipeline(fms_dt channels[max_fms_size],
		weights_dt weights_tile[pw_conv_parallelism_out][max_conv_d],
		pss_dt results_tile[pw_conv_parallelism_out][pw_tile_h][pw_tile_w],
		const int layer_num_fils, const int layer_conv_d,
		const int num_of_tiles_hw, const int num_of_tiles_w, int td_o,
		int td_i_o, int t_in_h, int t_in_w);
