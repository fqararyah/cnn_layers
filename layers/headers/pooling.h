#ifndef POOLING
#define POOLING

#include "../../basic_defs/basic_defs_glue.h"

void avgpool(fms_dt channels[max_fms_size], fms_dt restult[fc_layer_input_size]);
void maxpool(fms_dt channels[max_fms_size], fms_dt restult[fc_layer_input_size]);

#endif
