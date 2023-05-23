#ifndef SEML
#define SEML

#if HW == CPU
#if DEBUGGING
#include "../../../../tests/test_utils.h"
#endif
#include "../../../../client/prepare_weights_and_inputs.h"
#include <iostream>
#endif
#include "../../../utils/utils.h"
#include "../../../layers/headers/layers_glue.h"

//#include "../pipeline/headers/pipeline_glue.h"

//#include "../cnn_functions_v1.h"

using namespace std;

void seml(weights_grp_dt off_chip_weights[all_pw_s_weights],
		fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
		fms_dt tmp_channels[max_tmp_fms_size],
		fms_dt fc_input[fc_layer_input_size]);

void seml(weights_grp_dt off_chip_weights[all_pw_s_weights],
		fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
		fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
		fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
		fms_dt fc_input[fc_layer_input_size]);

#endif
