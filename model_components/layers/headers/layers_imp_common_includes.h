#include "norm_act.h"
#if MODEL_ID == MOB_V2
#include "../../model/headers/quantization_and_biases.h"
#include "../../model/headers/mob_v2_quantization_and_biases_v2.h"
#elif MODEL_ID == RESNET50
#include "../../model/headers/resnet50_quantization_and_biases_v2.h"
#endif
#include "../../utils/utils.h"
