
#ifndef SIMULATION_CONSTANTS
#define SIMULATION_CONSTANTS

#define CPU 1
#define _FPGA 0
#define HW CPU
#define PIPELINE_LENGTH 6
#define CHAIN_LENGTH PIPELINE_LENGTH
#define SOFT_PIPELINE 0
#define MOB_V1 1
#define MOB_V2 2
#define MOB_V2_0_5 25
#define MOB_V2_0_25 225
#define MOB_V2_0_75 275
#define MNAS 3
#define PROX 4
#define RESNET50 5
#define MODEL_ID MOB_V2
#define ONLY_SESL 0
#if PIPELINE_LENGTH == 0
#define ONLY_SEML 1
#else
#define ONLY_SEML 0
#endif
#define USE_FIRB 1
#define DEBUGGING 0
#define TESTING 1
#define FIBHA_VERSION 2
#define BOTTLENECK_CHAIN_MODE 1
#define PIPELINED_ENGINES_MODE 2
#define FIRST_PART_IMPLEMENTATION BOTTLENECK_CHAIN_MODE 
#define DW_WEIGHTS_OFF_CHIP 1

#endif