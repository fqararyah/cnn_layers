#ifndef PIPELINE_MAIN
#define PIPELINE_MAIN

#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE

#if HW == CPU
#include "../../../client/prepare_weights_and_inputs.h"
#endif
#include "../../utils/utils.h"
#include "pipelined_engines_specs.h"
#include "pipelined_engines.h"

using namespace pipelined_engines;

void pre_first_pipeline_layers_mob_v2(fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
                                      fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                                                             [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                                                             [PRE_FIRST_PIPELINE_OUTPUT_WIDTH],
                                      weights_dt dw_layer_weights[layer_2_dw_num_fils][layer_2_dw_filter_dim * layer_2_dw_filter_dim],
                                      fms_quantization_scheme first_layer_quantization_params[first_conv_layer_num_fils],
                                      fms_quantization_scheme first_dw_layer_quantization_params[layer_2_dw_num_fils],
                                      fms_dt conv_dw_communication_buffer_inter[first_conv_layer_num_fils][layer_2_dw_filter_dim]
                                                                               [layer_2_dw_ifm_width],
                                      fms_dt first_layers_input[input_image_depth][PRE_FIRST_PIPELINE_INPUT_HEIGHT][input_image_width],
                                      const int starting_h,
                                      const int end_h);

void pipelined_engines_caller(fms_grp_dt input_image[input_image_depth * input_image_num_fms_groups_in_a_channel],
                              weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
                              fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH]);

#endif //#if FIRST_PART_IMPLEMENTATION == PIPELINED_ENGINES_MODE

#endif