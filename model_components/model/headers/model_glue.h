
#if MODEL_ID == MOB_V2
#include "mob_v2_layers_specs.h"
#elif MODEL_ID == MOB_V2_0_5
#include "mob_v2_0_5_layers_specs.h"
#elif MODEL_ID == MOB_V2_0_25
#include "mob_v2_0_25_layers_specs.h"
#elif MODEL_ID == MOB_V2_0_75
#include "mob_v2_0_75_layers_specs.h"
#elif MODEL_ID == RESNET50
#include "resnet50_layers_specs.h"
#endif
#if MODEL_ID == MOB_V2
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "mob_v2_dw_weights.h"
#include "../../model/headers/mob_v2_on_chip_weights.h"
#include "../../model/headers/mob_v2_quantization_and_biases.h"
#else
#include "../../model/headers/mob_v2_on_chip_weights_v2.h"
#include "../../model/headers/mob_v2_quantization_and_biases_v2.h"
#include "mob_v2_dw_weights_v2.h"
#endif
#elif MODEL_ID == MOB_V2_0_5
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "mob_v2_0_5_dw_weights.h"
#include "../../model/headers/mob_v2_0_5_on_chip_weights.h"
#include "../../model/headers/mob_v2_0_5_quantization_and_biases.h"
#else
#include "mob_v2_0_5_dw_weights_v2.h"
#include "../../model/headers/mob_v2_0_5_on_chip_weights_v2.h"
#include "../../model/headers/mob_v2_0_5_quantization_and_biases_v2.h"
#endif
#elif MODEL_ID == MOB_V2_0_25
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "mob_v2_0_25_dw_weights.h"
#include "../../model/headers/mob_v2_0_25_on_chip_weights.h"
#include "../../model/headers/mob_v2_0_25_quantization_and_biases.h"
#else
#include "mob_v2_0_25_dw_weights_v2.h"
#include "../../model/headers/mob_v2_0_25_on_chip_weights_v2.h"
#include "../../model/headers/mob_v2_0_25_quantization_and_biases_v2.h"
#endif
#elif MODEL_ID == MOB_V2_0_75
#if FIRST_PART_IMPLEMENTATION == BOTTLENECK_CHAIN_MODE
#include "mob_v2_0_75_dw_weights.h"
#include "../../model/headers/mob_v2_0_75_on_chip_weights.h"
#include "../../model/headers/mob_v2_0_75_quantization_and_biases.h"
#else
#include "mob_v2_0_75_dw_weights_v2.h"
#include "../../model/headers/mob_v2_0_75_on_chip_weights_v2.h"
#include "../../model/headers/mob_v2_0_75_quantization_and_biases_v2.h"
#endif
#elif MODEL_ID == RESNET50
#include "../../model/headers/resnet50_quantization_and_biases_v2.h"
#endif
#include "../pipelined_engines/pipelined_engines_specs.h"
#include "../pipelined_engines/pipeline_main.h"

#include "../SEML/headers/seml.h"