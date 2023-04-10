#ifndef CONV
#define CONV

#include "../../basic_defs/basic_defs_glue.h"
#include "../../model/headers/model_glue.h"

void layer_0_s_using_pw(
		const layer_0_weights_dt weights_1[layer_1_s_num_fils][layer_1_s_depth][3][3],
		fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w);

void layer_0_s_3x3(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	const layer_0_weights_dt weights_1[layer_1_s_num_fils][layer_1_s_depth][layer_1_s_filter_dim][layer_1_s_filter_dim],
	fms_dt result[max_fms_size]);

#endif
