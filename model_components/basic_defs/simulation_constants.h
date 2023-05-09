
#ifndef SIMULATION_CONSTANTS
#define SIMULATION_CONSTANTS

#define CPU 1
#define _FPGA 0
#define CHAIN_LENGTH 11
#define MOB_V1 1
#define MOB_V2 2
#define MNAS 3
#define PROX 4
#define RESNET50 5
#define MODEL_ID MOB_V2//1: mob_v1, 2: mob_v2, 3: mnasnet, 4: proxylessnas
#define HW CPU//0 is
#define ONLY_SESL 0
#define ONLY_SEML 0
#define USE_FIRB 1
#define DEBUGGING 1
#define TESTING 1
#define FIBHA_VERSION 2
#define BOTTLENECK_CHAIN_MODE 1
#define PIPELINED_ENGINES_MODE 2
#define FIRST_PART_IMPLEMENTATION PIPELINED_ENGINES_MODE 
#endif