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

void seml(weights_grp_dt off_chip_weights[all_pw_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt tmp_channels[max_tmp_fms_size],
		const layer_0_weights_dt weights_0[layer_0_s_num_fils][layer_0_s_depth][layer_0_s_filter_dim][layer_0_s_filter_dim],
		fms_dt fc_input[fc_layer_input_size]);

#endif
