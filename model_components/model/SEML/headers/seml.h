#ifndef SEML
#define SEML

#if HW == CPU
#if 1
#include "../../../../tests/test_utils.h"
#include "../../../../client/prepare_weights_and_inputs.h"
#endif
#include <iostream>
#endif
#include "../../../utils/utils.h"
#include "../../../layers/headers/layers_glue.h"

// #include "../pipeline/headers/pipeline_glue.h"

// #include "../cnn_functions_v1.h"

using namespace std;

void seml(weights_grp_dt off_chip_weights[all_pw_s_weights],
          fms_dt channels[max_fms_size], fms_dt result[max_fms_size],
          fms_dt tmp_channels[max_tmp_fms_size],
          fms_dt fc_input[fc_layer_input_size]);

void seml(weights_grp_dt off_chip_weights[all_pw_s_weights],
          weights_dt off_chip_dw_weights[all_dw_off_chip_weights],
          fused_scales_dt off_chip_fused_scales[all_off_chip_fused_scales_zps],
          biases_dt off_chip_fused_zero_points[all_off_chip_fused_scales_zps],
          fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt tmp_channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
          fms_dt fc_input[fc_layer_input_size],
          int model_configs_list[2 * max_conv_layers]);

#endif
