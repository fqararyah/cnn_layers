
#ifndef GENERAL_SPECS
#define GENERAL_SPECS

// switch point
const int switch_point_fms_width = 56;
const int switch_point_fms_height = 56;
const int switch_point_fms_depth = 144; // not really, but the max of ...

//MobileNetsV1, but could be useful in future
const int alpha = 1;

//fc_layer
const int fc_layer_input_size = 1280;
const int fc_cols = 1000;

//avg_pool_layer
const int avgpool_input_depth = 1280;
const int avgpool_input_height = 7;
const int avgpool_input_width = 7;

//skip connection
const int skip_connection_depth = 3;

//testing vars
const int DF = 1;
const int start_with_pw = 1;

//maxs for buffers
const int max_conv_d = 1280 / alpha; //to_automate
#if MODEL_ID == 1 || MODEL_ID == 2
const int max_filter_hw_dim = 3;
#elif MODEL_ID == 3 || MODEL_ID == 4
const int max_filter_hw_dim = 5;
#endif
const int max_padding = 1;
const int max_conv_h = max_filter_hw_dim;
const int max_conv_w = max_filter_hw_dim;
const int max_fms_size =
	DF ? switch_point_fms_width * switch_point_fms_height * switch_point_fms_depth: 112 * 112 * 96;
const int max_tmp_fms_size = 56 * 56 * 24;

const int median_depth = 96;
const int median_width = 14;

//weights
const int all_pw_weights = 2125696/weights_group_items;
const int max_num_of_weight_groups_for_one_pass = max_conv_d / weights_group_items;

//input specs
const int input_image_height = 224;
const int input_image_width = 224;
const int input_image_depth = 3;
const int input_image_hw = input_image_height * input_image_width;
const int input_image_num_fms_groups_in_width =
			(input_image_width % input_image_group_items) == 0 ?
					input_image_width / input_image_group_items :
					1 + (input_image_width / input_image_group_items);
const int input_image_num_fms_groups_in_a_channel = input_image_num_fms_groups_in_width * input_image_height;

#endif
