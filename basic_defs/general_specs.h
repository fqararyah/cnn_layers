
#ifndef GENERAL_SPECS
#define GENERAL_SPECS
// switch point
const int switch_point_fms_width = 56;
const int switch_point_fms_height = 56;
const int switch_point_fms_depth = 24;

//MobileNetsV1, but could be useful in future
const int alpha = 1;

//fc_layer
const int fc_layer_input_size = 1280;
const int fc_cols = 1000;

//avg_pool_layer
const int avgpool_input_depth = 1280;
const int avgpool_input_height = 7;
const int avgpool_input_width = 7;

//testing vars
const int DF = 0;
const int start_with_pw = 1;

//maxs for buffers
const int max_conv_d = 1024 / alpha;
const int max_conv_h = 3;
const int max_conv_w = 3;
const int max_fms_size =
	DF ? switch_point_fms_width * switch_point_fms_height * switch_point_fms_depth: 112 * 112 * 96;
const int max_tmp_fms_size = 56 * 56 * 24 * 8 / 7;

const int median_depth = 96;
const int median_width = 14;

//weights
const int all_pw_weights = 2125696/weights_group_items;
const int max_num_of_weight_groups_for_one_pass = max_conv_d / weights_group_items;

//input specs
const int input_image_height = 224;
const int input_image_width = 224;
const int input_image_depth = 3;
const int input_image_group_items = input_image_width / 2;
const int num_input_image_tiles_w = input_image_width / input_image_group_items;

#endif
