#include "../layers/headers/layers_glue.h"

using namespace std;

#include "../model/model_glue.h"

#include "../pipeline/headers/pipeline_glue.h"

#include "../cnn_functions_v1.h"

#include "../utils/utils.h"
#include "dw_weights.h"
#include <iostream>
#include <math.h>

#ifndef SEML
#define SEML

void seml(fms_dt input_image[input_image_depth][input_image_height][input_image_width],
    weights_grp_dt off_chip_weights[all_pw_weights],
    fms_dt channels[max_fms_size],
	fms_dt result[max_fms_size],
	fms_dt result2[max_fms_size],
	fms_dt tmp_channels[max_tmp_fms_size],
	fms_dt tmp_channels_2[max_tmp_fms_size],
    layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
	dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w],
    fms_dt fc_input[fc_layer_input_size]){

	int read_write = 0;

    layer_0_using_pw(weights_0, input_image, result2, 0, 3, 32, 1,
				int(0.9 + ((float) layer_0_num_fils) / pw_conv_parallelism_out),
				224, 224 / 7, layers_0_normalization);
		 //begin_generation
		pw_conv(off_chip_weights, channels, result2, 1, layer_1_pw_depth,
				layer_1_pw_num_fils, layer_1_pw_num_of_tiles_in_d,
				layer_1_pw_num_of_tiles_out_d, layer_1_pw_num_of_tiles_h,
				layer_1_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_1_pw_num_of_weight_groups_in_depth,
				layer_1_pw_normalization, 1, layer_1_pw_weights_offset);
		fill_dw_layer_weights(dw_weights_1, dw_weights_buffer, layer_1_dw_depth, layer_1_dw_filter_size, layer_1_dw_filter_size);
		dw_conv_3x3(dw_weights_buffer, channels, result2, 1, layer_1_dw_depth,
				layer_1_dw_ifm_width, layer_1_dw_ifm_height, layer_1_dw_num_of_tiles_in_d,
				layer_1_dw_num_of_tiles_h, layer_1_dw_num_of_tiles_w,
				layer_1_dw_strides, layer_1_dw_padding_left,
				layer_1_dw_normalization, 0);
		 //***4
		pw_conv(off_chip_weights, channels, result2, 2, layer_2_pw_depth,
				layer_2_pw_num_fils, layer_2_pw_num_of_tiles_in_d,
				layer_2_pw_num_of_tiles_out_d, layer_2_pw_num_of_tiles_h,
				layer_2_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_2_pw_num_of_weight_groups_in_depth,
				layer_2_pw_normalization, 1, layer_2_pw_weights_offset);
		//***6
		pw_conv(off_chip_weights, channels, result2, 3, layer_3_pw_depth,
				layer_3_pw_num_fils, layer_3_pw_num_of_tiles_in_d,
				layer_3_pw_num_of_tiles_out_d, layer_3_pw_num_of_tiles_h,
				layer_3_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_3_pw_num_of_weight_groups_in_depth,
				layer_3_pw_normalization, 0, layer_3_pw_weights_offset);
		fill_dw_layer_weights(dw_weights_3, dw_weights_buffer, layer_1_dw_depth, layer_1_dw_filter_size, layer_1_dw_filter_size);
		dw_conv_3x3(dw_weights_buffer, channels, result2, 3, layer_3_dw_depth,
				layer_3_dw_ifm_width,layer_3_dw_ifm_height, layer_3_dw_num_of_tiles_in_d,
				layer_3_dw_num_of_tiles_h, layer_3_dw_num_of_tiles_w,
				layer_3_dw_strides, layer_1_dw_padding_left,
				layer_3_dw_normalization, 1);
		//***7
		pw_conv(off_chip_weights, channels, result2, 4, layer_4_pw_depth,
				layer_4_pw_num_fils, layer_4_pw_num_of_tiles_in_d,
				layer_4_pw_num_of_tiles_out_d, layer_4_pw_num_of_tiles_h,
				layer_4_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_4_pw_num_of_weight_groups_in_depth,
				layer_4_pw_normalization, 0, layer_4_pw_weights_offset);
		//***9
		pw_conv(off_chip_weights, channels, result2, 5, layer_5_pw_depth,
				layer_5_pw_num_fils, layer_5_pw_num_of_tiles_in_d,
				layer_5_pw_num_of_tiles_out_d, layer_5_pw_num_of_tiles_h,
				layer_5_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_5_pw_num_of_weight_groups_in_depth,
				layer_5_pw_normalization, 1, layer_5_pw_weights_offset);
		dw_conv_3x3(dw_weights_buffer, channels, result2, 5, layer_5_dw_depth,
				layer_5_dw_ifm_width,layer_5_dw_ifm_height, layer_5_dw_num_of_tiles_in_d,
				layer_5_dw_num_of_tiles_h, layer_5_dw_num_of_tiles_w,
				layer_5_dw_strides, layer_1_dw_padding_left,
				layer_5_dw_normalization, 0);
		//***10
		pw_conv(off_chip_weights, channels, result2, 6, layer_6_pw_depth,
				layer_6_pw_num_fils, layer_6_pw_num_of_tiles_in_d,
				layer_6_pw_num_of_tiles_out_d, layer_6_pw_num_of_tiles_h,
				layer_6_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_6_pw_num_of_weight_groups_in_depth,
				layer_6_pw_normalization, 1, layer_6_pw_weights_offset);
//		//***12
		pw_conv(off_chip_weights, channels, result2, 7, layer_7_pw_depth,
				layer_7_pw_num_fils, layer_7_pw_num_of_tiles_in_d,
				layer_7_pw_num_of_tiles_out_d, layer_7_pw_num_of_tiles_h,
				layer_7_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_7_pw_num_of_weight_groups_in_depth,
				layer_7_pw_normalization, 0, layer_7_pw_weights_offset);
		dw_conv_3x3(dw_weights_buffer, channels, result2, 7, layer_7_dw_depth,
				layer_7_dw_ifm_width, layer_7_dw_ifm_height, layer_7_dw_num_of_tiles_in_d,
				layer_7_dw_num_of_tiles_h, layer_7_dw_num_of_tiles_w,
				layer_7_dw_strides, layer_7_dw_padding_left,
				layer_7_dw_normalization, 1);
		//***13
		pw_conv(off_chip_weights, channels, result2, 8, layer_8_pw_depth,
				layer_8_pw_num_fils, layer_8_pw_num_of_tiles_in_d,
				layer_8_pw_num_of_tiles_out_d, layer_8_pw_num_of_tiles_h,
				layer_8_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_8_pw_num_of_weight_groups_in_depth,
				layer_8_pw_normalization, 0, layer_8_pw_weights_offset);
		//***15
		pw_conv(off_chip_weights, channels, result2, 9, layer_9_pw_depth,
				layer_9_pw_num_fils, layer_9_pw_num_of_tiles_in_d,
				layer_9_pw_num_of_tiles_out_d, layer_9_pw_num_of_tiles_h,
				layer_9_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_9_pw_num_of_weight_groups_in_depth,
				layer_9_pw_normalization, 1, layer_9_pw_weights_offset);
		dw_conv_3x3(dw_weights_buffer, channels, result2, 9, layer_9_dw_depth,
				layer_9_dw_ifm_width, layer_9_dw_ifm_height, layer_9_dw_num_of_tiles_in_d,
				layer_9_dw_num_of_tiles_h, layer_9_dw_num_of_tiles_w,
				layer_9_dw_strides, layer_9_dw_padding_left,
				layer_9_dw_normalization, 0);
		//		//***16
		pw_conv(off_chip_weights, channels, result2, 10, layer_10_pw_depth,
				layer_10_pw_num_fils, layer_10_pw_num_of_tiles_in_d,
				layer_10_pw_num_of_tiles_out_d, layer_10_pw_num_of_tiles_h,
				layer_10_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_10_pw_num_of_weight_groups_in_depth,
				layer_10_pw_normalization, 1, layer_10_pw_weights_offset);
		//***18
		pw_conv(off_chip_weights, channels, result2, 11, layer_11_pw_depth,
				layer_11_pw_num_fils, layer_11_pw_num_of_tiles_in_d,
				layer_11_pw_num_of_tiles_out_d, layer_11_pw_num_of_tiles_h,
				layer_11_pw_num_of_tiles_w, tmp_channels, read_write,
				layer_11_pw_num_of_weight_groups_in_depth,
				layer_11_pw_normalization, 0, layer_11_pw_weights_offset);
		dw_conv_3x3(dw_weights_buffer, channels, result2, 11, layer_11_dw_depth,
				layer_11_dw_ifm_width, layer_11_dw_ifm_height, layer_11_dw_num_of_tiles_in_d,
				layer_11_dw_num_of_tiles_h, layer_11_dw_num_of_tiles_w,
				layer_11_dw_strides, layer_11_dw_padding_left,
				layer_11_dw_normalization, 1);
				//***
		//end_generation
		avgpool(result2, fc_input);
		//fc_layer(fc_weights, fc_input, fc_output);

}

#endif