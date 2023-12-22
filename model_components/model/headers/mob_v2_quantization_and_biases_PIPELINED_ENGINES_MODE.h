#include "../../basic_defs/basic_defs_glue.h"
#if FIRST_PART_IMPLEMENTATION ==PIPELINED_ENGINES_MODE && MODEL_ID == MOB_V2
#ifndef BIAS_QUANT
#define BIAS_QUANT
const static int layers_fused_parameters_offsets[] = { 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 224, 416, 448, 448, 640, 832, 864, 864, 1056, 1056, 1248, 1312, 1696, 2080, 2144, 2144, 2528, 2912, 2976, 2976, 3360, 3744, 3808, 3808, 4192, 4576, 4672, 5248, 5824, 5920, 5920, 6496, 7072, 7168, 7168, 7744, 7744, 8320, 8480, 9440, 10400, 10560, 10560, 11520, 12480, 12640, 12640, 13600, 14560, 14880, 16160, 16160, 16160, 16160, 16160};
const static int pipe_layers_fused_parameters_offsets[] = { 
0, 0, 32, 64, 80, 176, 176, 272, 296, 440, 584, 608, 608, 752, 752, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896, 896};
const static biases_dt pipe_fused_zero_points[] = 
{ };
const static fused_scales_dt pipe_fused_scales[] ={ };
const static fused_scales_log_2_shifts_dt pipe_fused_scales_log_2_shifts[] ={ };
const static relu_6_fused_scales_dt pipe_relu_6_fused_scales[] ={ };
static biases_dt seml_fused_zero_points_buffer[1280];
static fused_scales_dt seml_fused_scales_buffer[1280];
const static fused_scales_log_2_shifts_dt fused_scales_log_2_shifts[] ={ };
const static relu_6_fused_scales_dt relu_6_fused_scales[] ={ 0, 255, 255, 16, 255, 0, 255, 20, 255, 255, 14, 0, 255, 0, 255, 27, 255, 255, 27, 0, 255, 255, 21, 0, 255, 0, 255, 29, 255, 255, 31, 0, 255, 255, 39, 0, 255, 255, 33, 0, 255, 255, 36, 255, 255, 42, 0, 255, 255, 32, 0, 255, 0, 255, 44, 255, 255, 59, 0, 255, 255, 27, 0, 255, 255, 41, 255, 0, 0, 0, 0};
#endif
#endif
