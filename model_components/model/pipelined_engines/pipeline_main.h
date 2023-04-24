#ifndef PIPELINE_MAIN
#define PIPELINE_MAIN

#if HW == CPU
#include "../../../client/prepare_weights_and_inputs.h"
#endif
#include "../../utils/utils.h"
#include "pipelined_engines_specs.h"
#include "pipelined_engines.h"

void pipelined_engines_caller(fms_dt result[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH]);

#endif