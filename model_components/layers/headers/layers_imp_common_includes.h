#include "norm_act.h"
#if MODEL_ID == MOB_V2
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "../../model/headers/mob_v2_quantization_and_biases.h"
#else
#include "../../model/headers/mob_v2_quantization_and_biases_v2.h"
#endif
#elif MODEL_ID == RESNET50
#include "../../model/headers/resnet50_quantization_and_biases_v2.h"
#endif
#include "../../utils/utils.h"
