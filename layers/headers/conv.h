#ifndef CONV
#define CONV

#include "../../basic_defs/basic_defs_glue.h"
#include "../../model/model_glue.h"

void layer_0_using_pw(
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3],
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w);

void layer_0_3x3(
	const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
	fms_dt channels[input_image_depth][input_image_height][input_image_width],
	fms_dt result[max_fms_size]);

#endif
