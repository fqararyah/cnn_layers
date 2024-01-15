#include "ap_int.h"
#include "ap_fixed.h"

#include "model_components/basic_defs/basic_defs_glue.h"
#include "model_components/model/headers/model_glue.h"
#if MODEL_ID == MNAS
#include "dep_model/layers_specs_mnas.h"
#elif MODEL_ID == MOB_V2
#include "model_components/model/headers/mob_v2_layers_specs.h"
#elif MODEL_ID == MOB_V1
#include "dep_model/layers_specs_mob_v1.h"
#elif MODEL_ID == PROX
#include "dep_model/layers_specs_prox.h"
#endif
#if ONLY_SEML == 0 && FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
//#include "dep_model/sesl.h"
#include "model_components/model/fused_bottlenecks/bottlenecks_chain.h"
#endif
