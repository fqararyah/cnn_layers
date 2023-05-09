
#ifndef GENERAL_SPECS
#define GENERAL_SPECS

#define S_CONV 0
#define PW_CONV 1
#define DW_CONV 2

#define RELU 1
#define RELU6 6

#define MODEL_ACTIVATION RELU
#define ADD_LAYER_ACTIVATION 0
typedef int conv_type;
//
const int first_quantization_arrays_num_elements = 8320;

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

#if MODEL_ID == MOB_V2
const int MAX_FMS_BUFFER_DEPTH = 192 * 16; // 192 * (28/7) * (28/7)
#elif MODEL_ID == RESNET50
const int MAX_FMS_BUFFER_DEPTH = 512 * 16; // 192 * (28/7) * (28/7)
#endif
const int MIN_FMS_HEIGHT = 8;
const int MIN_FMS_WIDTH = 8;
const int MAX_FILTER_DIM_STRIDE_1 = 3;
const int MAX_FILTER_DIM_STRIDE_2 = 3;
const int MAX_DW_LAYER_D = 960;

// assumptions
// CHANNELS_TILE_WIDTH = CHANNELS_TILE_HEIGHT
// CHANNELS_TILE_WIDTH is even
const int MAX_TILE_PADDING_TOP_LEFT = (MAX_FILTER_DIM_STRIDE_1 - 1) / 2;
const int MAX_TILE_PADDING_BOTTOM_RIGHT = MAX_FILTER_DIM_STRIDE_2 - 2;

// MobileNetsV1, but could be useful in future
const int alpha = 1;

// fc_layer
#if MODEL_ID == RESNET50
const int fc_layer_input_size = 2048;
#elif MODEL_ID == MOB_V2
const int fc_layer_input_size = 1280;
#endif
const int fc_cols = 1000;

// avg_pool_layer
const int avgpool_input_depth = 1280;
const int avgpool_input_height = 7;
const int avgpool_input_width = 7;

// skip connection
const int skip_connection_depth = 3;

// testing vars
const int DF = 1;
const int start_with_pw = 1;

// maxs for buffers
#if MODEL_ID == MOB_V2
const int max_conv_d = 1280 / alpha; // to_automate
#elif MODEL_ID == RESNET50
const int max_conv_d = 2048; // to_automate
#endif
const int min_strides = 1;
const int max_strides = 2;
#if MODEL_ID == MOB_V1 || MODEL_ID == MOB_V2 || MODEL_ID == RESNET50
const int max_filter_hw_dim = 3;
const int max_conv_filter_hw_dim = 3;
const int max_padding_lr = 2;
#elif MODEL_ID == 3 || MODEL_ID == 4
const int max_filter_hw_dim = 5;
#endif
const int max_padding = 1;
const int max_conv_h = max_conv_filter_hw_dim;
const int max_conv_w = max_conv_filter_hw_dim;
const int max_filter_area = max_conv_h * max_conv_w;

const int median_depth = 96;
const int median_width = 14;

// weights
const int all_pw_weights = 2109952 / weights_group_items;
const int max_num_of_weight_groups_for_one_pass = max_conv_d / weights_group_items;

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

struct pooling_layer_specs{
const pooling_fused_scales_dt fused_scale;
const biases_dt ifms_zero_point;
const biases_dt ofms_zero_point;
};

struct fc_layer_specs{
const fms_dt ifm_zero_point;
};


#endif
