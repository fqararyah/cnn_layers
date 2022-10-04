
#include "../utils/utils.h"
#include "../layers/headers/layers_glue.h"

#include "../model/model_glue.h"

#include "../pipeline/headers/pipeline_glue.h"

#include "../cnn_functions_v1.h"

#include "dw_weights.h"
#include "quantization_and_biases.h"
#include <iostream>
#include <math.h>

using namespace std;

#ifndef SEML
#define SEML

void seml(
		fms_dt input_image[input_image_depth][input_image_height][input_image_width],
		weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt result2[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size],
		fms_dt tmp_channels_2[max_tmp_fms_size],
		layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_size][layer_0_filter_size],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w],
		fms_dt fc_input[fc_layer_input_size]) {

	layer_0_using_pw(weights_0, input_image, result2, 0, layer_0_depth,
			layer_0_num_fils, 1, layer_0_num_of_tiles_out_d,
			layer_0_num_of_tiles_h, layer_0_num_of_tiles_w,
			layers_0_normalization);
	//to generate layers within a range, rather than all layers, add, .e.g. [1:20]
	//where 1 is the first and 20 is the last layer to be generated
	//to the end of the previous layer
	//begin_code_generation[43:45]
pw_conv(off_chip_weights, channels, result2, 43, layer_43_pw_depth,
    layer_43_pw_num_fils, layer_43_pw_num_of_tiles_in_d,
    layer_43_pw_num_of_tiles_out_d, layer_43_pw_num_of_tiles_h,
    layer_43_pw_num_of_tiles_w, tmp_channels, 2,
    layer_43_pw_num_of_weight_groups_in_depth,
    layer_43_pw_normalization, 1, layer_43_pw_weights_offset);
fill_dw_layer_weights(dw_weights_44, dw_weights_buffer, layer_44_dw_depth, layer_44_dw_filter_size, layer_44_dw_filter_size);
    dw_conv_3x3(dw_weights_buffer, channels, result2, 44, layer_44_dw_depth,
    layer_44_dw_ifm_width, layer_44_dw_ifm_height, layer_44_dw_num_of_tiles_in_d,
    layer_44_dw_num_of_tiles_h, layer_44_dw_num_of_tiles_w,
    layer_44_dw_strides, layer_44_dw_padding_left,
    layer_44_dw_normalization, 0);
	//end_code_generation
	avgpool(result2, fc_input);
	//fc_layer(fc_weights, fc_input, fc_output);

}

#endif
