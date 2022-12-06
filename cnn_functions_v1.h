#include "./basic_defs/basic_defs_glue.h"
#include "./model/model_glue.h"

#ifndef V1_FIRST
#define V1_FIRST
//*********UK*********
//*****layer_1 (first layer to apply UK from)*****
const int v1_layer_1_dw_num_fils = layer_0_num_fils * alpha;
const int v1_layer_1_dw_depth = v1_layer_1_dw_num_fils;
const int v1_layer_1_dw_strides = 1;
const int v1_layer_1_dw_ifm_height = layer_0_ofm_height;
const int v1_layer_1_dw_ifm_width = layer_0_ofm_width;
const int v1_layer_1_dw_ofm_height = v1_layer_1_dw_ifm_height / v1_layer_1_dw_strides;
const int v1_layer_1_dw_ofm_width = v1_layer_1_dw_ifm_width / v1_layer_1_dw_strides;
const int v1_layer_1_dw_padding_left = 1;
const int v1_layer_1_dw_padding_right = 1;
const int v1_layer_1_dw_filter_size = 3;

const int v1_layer_1_pw_num_fils = 192 / alpha;
const int v1_layer_1_pw_depth = v1_layer_1_dw_depth;
const int v1_layer_1_pw_ifm_height = v1_layer_1_dw_ofm_height;
const int v1_layer_1_pw_ifm_width = v1_layer_1_dw_ofm_width;
const int v1_layer_1_pw_ofm_height = v1_layer_1_pw_ifm_height;
const int v1_layer_1_pw_ofm_width = v1_layer_1_pw_ifm_width;
//*****end layer_1*****

//*****layer_2*****
const int v1_layer_2_dw_num_fils = v1_layer_1_pw_num_fils / alpha;
const int v1_layer_2_dw_depth = v1_layer_2_dw_num_fils;
const int v1_layer_2_dw_strides = 2;
const int v1_layer_2_dw_ifm_height = v1_layer_1_pw_ofm_height;
const int v1_layer_2_dw_ifm_width = v1_layer_1_pw_ofm_width;
const int v1_layer_2_dw_ofm_height = v1_layer_2_dw_ifm_height / v1_layer_2_dw_strides;
const int v1_layer_2_dw_ofm_width = v1_layer_2_dw_ifm_width / v1_layer_2_dw_strides;
const int v1_layer_2_dw_padding_left = 1;
const int v1_layer_2_dw_padding_right = 1;
const int v1_layer_2_dw_filter_size = 3;

const int v1_layer_2_pw_num_fils = 32 / alpha;
const int v1_layer_2_pw_depth = v1_layer_1_dw_depth;
const int v1_layer_2_pw_ifm_height = v1_layer_1_dw_ofm_height;
const int v1_layer_2_pw_ifm_width = v1_layer_1_dw_ofm_width;
const int v1_layer_2_pw_ofm_height = v1_layer_2_pw_ifm_height;
const int v1_layer_2_pw_ofm_width = v1_layer_2_pw_ifm_width;
//*****end layer_2*****

//*****layer_3*****
const int v1_layer_3_dw_num_fils = v1_layer_2_pw_num_fils / alpha;
const int v1_layer_3_dw_depth = v1_layer_3_dw_num_fils;
const int v1_layer_3_dw_strides = 1;
const int v1_layer_3_dw_ifm_height = v1_layer_2_pw_ofm_height;
const int v1_layer_3_dw_ifm_width = v1_layer_2_pw_ofm_width;
const int v1_layer_3_dw_ofm_height = v1_layer_3_dw_ifm_height / v1_layer_3_dw_strides;
const int v1_layer_3_dw_ofm_width = v1_layer_3_dw_ifm_width / v1_layer_3_dw_strides;
const int v1_layer_3_dw_padding_left = 1;
const int v1_layer_3_dw_padding_right = 1;
const int v1_layer_3_dw_filter_size = 3;

const int v1_layer_3_pw_num_fils = 32 / alpha;
const int v1_layer_3_pw_depth = v1_layer_2_dw_depth;
const int v1_layer_3_pw_ifm_height = v1_layer_2_dw_ofm_height;
const int v1_layer_3_pw_ifm_width = v1_layer_2_dw_ofm_width;
const int v1_layer_3_pw_ofm_height = v1_layer_3_pw_ifm_height;
const int v1_layer_3_pw_ofm_width = v1_layer_3_pw_ifm_width;
//*****end layer_3*****

//*****end variable based on experiment*****

//*********end UK*********

//*********DF*********

// l1 (second layer)
const int v1_3_stages_layer_1_rows_at_once = 1;
const int v1_4_stages_layer_1_rows_at_once = 2;
const int v1_7_stages_layer_1_rows_at_once = 2;



// l2 (third and fourth layers)
const int v1_4_stages_layer_2_rows_at_once = 1;
const int v1_7_stages_layer_2_rows_at_once = 1;



// l3 (fifth and sixth)
const int v1_7_stages_layer_3_rows_at_once = 1;



// l3 (seventh)
const int v1_layer_4_pw_num_fils = 128 / alpha;
const int v1_layer_4_pw_depth = v1_layer_3_dw_depth;
const int v1_layer_4_pw_ifm_width = v1_layer_3_dw_ofm_width;
const int v1_layer_4_pw_ofm_width = v1_layer_4_pw_ifm_width;

//*****variable based on experiment*****
const int v1_sesl_layer_0_parallelism_ofms = 1;
const int v1_layer_1_dw_parallelism = 2;

const int v1_layer_2_pw_parallelism_in = 32;
const int v1_layer_2_pw_parallelism_out = 2;
const int v1_layer_3_pw_parallelism_in = 64;
const int v1_layer_3_pw_parallelism_out = 1;
const int v1_layer_4_pw_parallelism_in = 128;
const int v1_layer_4_pw_parallelism_out = 1;

const int v1_sesl_layer_2_dw_parallelism = v1_layer_2_pw_parallelism_out;
const int v1_layer_3_dw_parallelism = v1_layer_3_pw_parallelism_out;
//*********end DF*********

const int v1_max_fms_size =
	DF ? switch_point_fms_width * switch_point_fms_height * switch_point_fms_depth * 8 / 7 : 112 * 112 * 96 * 8 / 7;

const int v1_max_tmp_fms_size = 2;//negligable, ther is actually no

#endif

void v1_3_layer_0_3x3_conv(
	fms_dt channels_buffer[input_image_depth][layer_0_filter_dim + (v1_3_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
	layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
	fms_dt result[v1_layer_2_pw_depth][v1_layer_2_pw_ifm_width]);

void v1_4_layer_1_dw(fms_dt upper[v1_layer_1_dw_depth][v1_layer_1_dw_filter_size - v1_layer_1_dw_strides][v1_layer_1_dw_ifm_width],
                     fms_dt lower[v1_layer_1_dw_depth][v1_4_stages_layer_1_rows_at_once][v1_layer_1_dw_ifm_width],
                     dw_weights_dt dw_weights[v1_layer_1_dw_depth][max_conv_h * max_conv_w],
                     fms_dt result[v1_layer_1_dw_num_fils][v1_4_stages_layer_1_rows_at_once][v1_layer_1_dw_ofm_width], int active_row);

void v1_4_layer_0_3x3_conv(
	fms_dt channels_buffer[input_image_depth][layer_0_filter_dim + (v1_4_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
	layer_0_weights_dt weights[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
	fms_dt result[v1_layer_1_dw_depth][v1_4_stages_layer_1_rows_at_once][v1_layer_1_dw_ifm_width]);

void v1_4_layer_2_pw_dw(
    fms_dt channels_buffer[v1_layer_2_pw_depth][v1_4_stages_layer_2_rows_at_once][v1_layer_2_dw_ifm_width],
    weights_dt weights[v1_layer_2_pw_num_fils][v1_layer_2_pw_depth],
    dw_weights_dt dw_weights[v1_layer_2_dw_depth][v1_layer_2_dw_filter_size][v1_layer_2_dw_filter_size],
    fms_dt upper[v1_layer_2_dw_depth][v1_layer_2_dw_ifm_width],
    fms_dt lower[v1_layer_2_dw_depth][v1_layer_2_dw_strides][v1_layer_2_dw_ifm_width],
	fms_dt result[max_fms_size], int starting_h, int active_row);

void v1_4_stages_fill_channels_buffer(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt channels_buffer_0[input_image_depth][layer_0_filter_dim + (v1_4_stages_layer_1_rows_at_once - 1) * layer_0_strides][input_image_width],
	int starting_h);

void mobilenet_v1_pipeline_3(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size]);

void mobilenet_v1_pipeline_4(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size]);

void mobilenet_v1_pipeline_7(
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size]);

