#include "bottlenecks_sesl_specs.h"

#ifndef BOTTLENECK_PARALLELISM
#define BOTTLENECK_PARALLELISM

const int bottleneck_1_expansion_parallelism_ifms = bottleneck_1_ifms_depth;
const int bottleneck_1_expansion_parallelism_ofms = 1;
const int bottleneck_1_expansion_parallelism_h = 2;
const int bottleneck_1_expansion_parallelism_w = 2;
//*****
const int bottleneck_1_dw_parallelism_ifms = bottleneck_1_expansion_parallelism_ofms;
//*****
const int bottleneck_1_projection_parallelism_ifms = 1;
const int bottleneck_1_projection_parallelism_ofms = bottleneck_1_ofms_depth;
const int bottleneck_1_projection_parallelism_h = 1;
const int bottleneck_1_projection_parallelism_w = 1;

#endif