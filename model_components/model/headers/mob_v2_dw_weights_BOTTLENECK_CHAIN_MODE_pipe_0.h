#include "../../basic_defs/basic_defs_glue.h"
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE && MODEL_ID == MOB_V2 && PIPELINE_LENGTH == 0
#ifndef DW_WEIGHTS
#define DW_WEIGHTS
static dw_weights_dt seml_dw_weights_3x3[96][9];
const static int dw_layers_weights_offsets[] ={0, 0, 0, 288, 288, 288, 288, 1152, 1152, 1152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
#endif
