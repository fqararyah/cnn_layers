#ifndef PIPELINE_MAIN
#define PIPELINE_MAIN

#if HW == CPU
#include "../../../client/prepare_weights_and_inputs.h"
#endif
#include "../../utils/utils.h"
#include "pipelined_engines_specs.h"
#include "pipelined_engines.h"

void pre_first_pipeline_layers_mob_v2(fms_grp_dt channels[input_image_depth * input_image_num_fms_groups_in_a_channel],
                               fms_dt pre_first_pipeline_layers_output[PRE_FIRST_PIPELINE_OUTPUT_DEPTH]
                                                                          [PRE_FIRST_PIPELINE_OUTPUT_HEIGHT]
                                                                          [PRE_FIRST_PIPELINE_OUTPUT_WIDTH]);

void pipelined_engines_caller(weights_dt on_chip_weights[][ON_CHIP_WEIGHTS_PORTS],
                              fms_dt pipelined_engines_input_buffer[pipelined_engines::MAX_PW_BUFFER_DEPTH]
                              [pipelined_engines::PW_BUFFER_HEIGHT][pipelined_engines::MAX_PW_BUFFER_WIDTH],
                              fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH]);

#endif