#include "../../basic_defs/basic_defs_glue.h"
#include "../../model/model_glue.h"

#ifndef CONV
#define CONV

void layer_0_using_pw(
		weights_dt weights_0[layer_0_num_fils][layer_0_depth][3][3],
		fms_dt channels[input_image_depth][input_image_height][input_image_width],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w, const normalization_scheme normalization);

#endif
