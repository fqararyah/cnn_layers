#ifndef POOLING
#define POOLING

#include "../../basic_defs/basic_defs_glue.h"
#include "../headers/norm_act.h"

void avgpool(fms_dt channels[max_fms_size], fms_dt restult[fc_layer_input_size]);
void avgpool(fms_dt channels[MAX_FMS_BUFFER_DEPTH][MIN_FMS_HEIGHT][MIN_FMS_WIDTH],
			 fms_dt result[fc_layer_input_size]);
void maxpool(fms_dt channels[max_fms_size], fms_dt restult[fc_layer_input_size]);

#endif
