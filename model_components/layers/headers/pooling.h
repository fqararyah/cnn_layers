#ifndef POOLING
#define POOLING

#include "../../basic_defs/basic_defs_glue.h"
#include "../headers/norm_act.h"

void avgpool(fms_dt channels[max_fms_size], fms_dt restult[fc_layer_input_size],
const pooling_layer_specs layer_specs_struct);
void avgpool(fms_dt channels[][CHANNELS_TILE_HEIGHT][CHANNELS_TILE_WIDTH],
			 fms_dt result[fc_layer_input_size], const pooling_layer_specs layer_specs_struct);
void maxpool(fms_dt channels[max_fms_size], fms_dt restult[fc_layer_input_size]);

#endif
