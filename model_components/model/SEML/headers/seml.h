#ifndef SEML
#define SEML
#include "../../../../tests/test_utils.h"
#include "../../../../client/prepare_weights_and_inputs.h"
#include "../../../utils/utils.h"
#include "../../../layers/headers/layers_glue.h"

//#include "../pipeline/headers/pipeline_glue.h"

//#include "../cnn_functions_v1.h"


#include <iostream>

using namespace std;

void seml(
		fms_grp_dt input_image[input_image_depth*input_image_height*input_image_width/input_image_group_items],
		weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt result2[max_fms_size], fms_dt tmp_channels[max_tmp_fms_size],
		fms_dt tmp_channels_2[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_num_fils][layer_0_depth][layer_0_filter_dim][layer_0_filter_dim],
		dw_weights_dt dw_weights_buffer[max_conv_d][max_conv_h][max_conv_w],
		fms_dt fc_input[fc_layer_input_size]);

#endif
