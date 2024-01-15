
#ifndef GENERAL_SPECS
#define GENERAL_SPECS

#define S_CONV 0
#define PW_CONV 1
#define DW_CONV 2

#define RELU 1
#define RELU6 6

#define MODEL_ACTIVATION RELU6
#define ADD_LAYER_ACTIVATION 0
typedef int conv_type;
//
static bool on_chip_weights_filled = false;
// MobileNetsV1, but could be useful in future
const int alpha = 1;

const int ON_CHIP_WEIGHTS_PORTS = 8;
const int fc_cols = 1000; // num classes
const int max_conv_layers = 100;

// skip connection
const int skip_connection_depth = 3;

// input specs
const int input_image_height = 224;
const int input_image_width = 224;
const int input_image_depth = 3;
const int input_image_hw = input_image_height * input_image_width;
const int input_image_num_fms_groups_in_width =
	(input_image_width % input_image_group_items) == 0 ? input_image_width / input_image_group_items : 1 + (input_image_width / input_image_group_items);
const int input_image_num_fms_groups_in_a_channel = input_image_num_fms_groups_in_width * input_image_height;

struct layer_specs
{
	int layer_index;
	conv_type conv_layer_type;
	int layer_num_fils;
	int strides;
	int filter_size;
	int padding_left;
	int padding_right;
	int padding_top;
	int padding_bottom;
	int layer_depth;
	int layer_ifm_height;
	int layer_ifm_width;
	int layer_ofm_height;
	int layer_ofm_width;
	int layer_activation;
	int layer_num_of_tiles_in_d;
	int layer_num_of_tiles_out_d;
	int layer_num_of_ifm_tiles_h;
	int layer_num_of_ifm_tiles_w;
	int layer_num_of_ofm_tiles_h;
	int layer_num_of_ofm_tiles_w;
	int layer_num_of_weight_groups_for_one_pass;
	int layer_weights_offset;
	int layer_weights_offset_on_chip;
	int dw_ifms_cumulative_width_offset;
	bool write_to_result_or_channels;
	bool write_to_tmp;
	bool fused_with_add;
	fms_dt layer_ifms_zero_point;
	scales_dt layer_ofms_scale;
	fms_dt layer_ofms_zero_point;
	rec_scales_dt add_layer_scale_reciprocal;
	biases_dt add_layer_zero_point;
	scales_dt skip_connection_other_layer_scale;
	biases_dt skip_connection_other_layer_zero_point;
};

struct pooling_layer_specs
{
	const int ifm_depth;
	const int ifm_height;
	const int ifm_width;
	const pooling_fused_scales_dt fused_scale;
	const biases_dt ifms_zero_point;
	const biases_dt ofms_zero_point;
};

struct Quantization_layer_specs
{
	const float fused_scale;
	const fms_dt ifms_zero_point;
	const biases_dt ofms_zero_point;
};

struct fc_layer_specs
{
	const fms_dt ifm_zero_point;
};

// switch point
#if ONLY_SEML
const int switch_point_fms_width = 112;
const int switch_point_fms_height = 112;
const int switch_point_fms_depth = 96; // not really, but the max of ...
#else
const int switch_point_fms_width = 56;
const int switch_point_fms_height = 56;
const int switch_point_fms_depth = 144; // not really, but the max of ...
#endif

const int max_fms_size = switch_point_fms_width * switch_point_fms_height * switch_point_fms_depth;
const int max_tmp_fms_size = 56 * 56 * 24;

const int CHANNELS_PIPELINE_DEPTH = 32;
const int CHANNELS_TILE_DEPTH = 1;
const int MIN_FMS_HEIGHT = 8;
const int MIN_FMS_WIDTH = 8;

const int CHANNELS_TILE_HEIGHT = PW_PARALLELISM_H;
const int CHANNELS_TILE_WIDTH = PW_PARALLELISM_W;

#include "mob_v2_general_specs_pipe_0.h"
#include "mob_v2_general_specs_pipe_6.h"

// assumptions
// CHANNELS_TILE_WIDTH = CHANNELS_TILE_HEIGHT
// CHANNELS_TILE_WIDTH is even
const int MAX_TILE_PADDING_TOP_LEFT = (MAX_FILTER_DIM_STRIDE_1 - 1) / 2;
const int MAX_TILE_PADDING_BOTTOM_RIGHT = MAX_FILTER_DIM_STRIDE_2 - 2;

const int CHANNELS_TILE_HEIGHT_PADDED = CHANNELS_TILE_HEIGHT + MAX_TILE_PADDING_TOP_LEFT + MAX_TILE_PADDING_BOTTOM_RIGHT;
const int CHANNELS_TILE_WIDTH_PADDED = CHANNELS_TILE_WIDTH + MAX_TILE_PADDING_TOP_LEFT + MAX_TILE_PADDING_BOTTOM_RIGHT;
const int pw_tile_d = pw_conv_parallelism_in;
#if FIBHA_VERSION == 1
const int pw_tile_h = 8;
const int pw_tile_w = 8;
#elif FIBHA_VERSION == 2
const int pw_tile_h = PW_PARALLELISM_H;
const int pw_tile_w = PW_PARALLELISM_W;
#endif
const int pw_tile_hw = pw_tile_h * pw_tile_w;
const int pw_tile_size = pw_tile_d * pw_tile_h * pw_tile_w;

const int dw_tile_d = pw_tile_d;
const int dw_tile_h = pw_tile_h;
const int dw_tile_w = pw_tile_w;
#if FIBHA_VERSION == 1
const int dw_pipeline_depth = 24;
#elif FIBHA_VERSION == 2
const int dw_pipeline_depth = CHANNELS_PIPELINE_DEPTH;
#endif

const int dw_tile_hw = dw_tile_h * dw_tile_w;
const int dw_tile_size = dw_tile_d * dw_tile_h * dw_tile_w;
const int dw_max_v2_buffer_height = dw_tile_h + (max_filter_hw_dim - 1); // where 3 is max conv kernel dim and 1 is mi strides
const int dw_max_v2_buffer_width = dw_max_v2_buffer_height;				 // where 3 is max conv kernel dim and 1 is mi strides

const int max_dw_input_width = 112 + 1 + 1; // where 1 is max padding left and right
const int max_tile_w = pw_tile_w;
const int max_tile_h = pw_tile_h;
const int max_tile_d = pw_tile_d > dw_tile_d ? pw_tile_d : dw_tile_d;

const int num_of_weights_in_the_same_filter_and_group_on_chip = weights_group_items / ON_CHIP_WEIGHTS_PORTS;
const int num_of_weights_in_the_same_filter_and_group = weights_group_items / pw_conv_parallelism_out;
const int num_of_weight_groups_in_the_largest_weight_tile = max_conv_d * pw_conv_parallelism_out / weights_group_items;
const int pw_weights_tile_partitioning_factor = num_of_weights_in_the_same_filter_and_group;

const int max_filter_area = max_filter_hw_dim * max_filter_hw_dim;

const int all_on_chip_pw_s_weights_groups = (all_on_chip_pw_s_weights + weights_group_items - 1) / weights_group_items;

#endif
