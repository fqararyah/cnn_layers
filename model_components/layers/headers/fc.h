#ifndef FC
#define FC

#include "../../basic_defs/basic_defs_glue.h"

void fc_layer(const fc_weights_dt weights[fc_layer_input_size][fc_cols], fms_dt channels[fc_layer_input_size],
		fc_out_dt result[fc_cols]);

#endif
