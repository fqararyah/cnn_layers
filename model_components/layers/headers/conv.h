#ifndef CONV
#define CONV

#include "../../basic_defs/basic_defs_glue.h"
#include "../../model/headers/model_glue.h"

void layer_0_s_using_pw(
		const layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][3][3],
		fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
		fms_dt result[max_fms_size], const int layer, const int layer_conv_d,
		const int layer_num_fils, const int num_of_tiles_d_in,
		const int num_of_tiles_d_out, const int num_of_tiles_h,
		const int num_of_tiles_w);

void layer_0_s_3x3(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	fms_dt result[][MIN_FMS_HEIGHT][MIN_FMS_WIDTH]);

void layer_0_s_3x3(
	fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
	const layer_0_weights_dt weights_1[first_conv_layer_num_fils][first_conv_layer_depth][first_conv_layer_filter_dim][first_conv_layer_filter_dim],
	fms_dt result[max_fms_size]);

void pw_and_conv(weights_grp_dt *weights,
             fms_dt channels[][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
             fms_dt result[][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
             fms_dt tmp_channels[][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
             int layer, const layer_specs layer_specs_struct,
             const fused_scales_dt fused_scales[],
             const relu_6_fused_scales_dt relu_6_fused_scales[],
             const biases_dt fused_zero_points[],const int model_configs_list[2 * max_conv_layers]);

#endif
